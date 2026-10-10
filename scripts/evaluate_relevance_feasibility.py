"""Read-only comparison of relevance and feasibility scoring designs.

This script evaluates existing persisted jobs only. It never writes to AWS and does not
change the production matcher, filters, role policy, or eligibility evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import boto3

from job_intelligence.ai import AIProviderError, build_ai_providers, to_job_analysis
from job_intelligence.aws.dynamodb import (
    DynamoCandidateProfileRepository,
    DynamoJobRepository,
)
from job_intelligence.config import Settings
from job_intelligence.eligibility import EligibilityConfig, EligibilityEvaluator
from job_intelligence.evaluation import ndcg_at_k, precision_at_k
from job_intelligence.filtering import DeterministicJobFilter, FilteringConfig
from job_intelligence.matching import HybridJobMatcher, MatchingConfig
from job_intelligence.models import CandidateProfile
from job_intelligence.role_policy import (
    RoleFamilyClass,
    RoleFamilyClassifier,
    default_ai_data_role_family_policy,
)

DEFAULT_JOB_IDS = (
    "825b9857-f45e-49dd-bd33-720d9c323cf8",  # Data Scientist, Corporate
    "4a5f7ee1-0a80-46a2-bffa-c4d14e2da01b",  # Data Product Analyst, Corporate
    "d0ac376e-d779-428b-bd8c-49999d4ce9ce",  # Data QA Associate 1
    "94494862-a402-4be2-88ff-1d2240d6d88d",  # Data QA Associate 2
    "eb235dad-b127-4d03-a490-fc94bf495cae",  # Investor Research Analyst
    "d10e7304-1359-4c6a-afc4-04683c9319b1",  # Product Manager, AI
)

LABEL_PATH = Path("evaluation/results/profile-impact/profile-labels.json")
RELEVANCE_COMPONENTS = ("role", "skill", "responsibility", "domain")
FAMILY_PRIOR = {
    RoleFamilyClass.CORE.value: 1.0,
    RoleFamilyClass.ADJACENT.value: 0.75,
}
ELIGIBILITY_ORDER = {
    "eligible": 0,
    "unknown": 1,
    "stretch": 2,
    "severe_stretch": 3,
    "ineligible": 4,
}


def main() -> None:
    arguments = _parse_args()
    settings = Settings.from_environment()
    table = _table(settings)
    profile = _profile(table, arguments.profile_key)
    states = {
        str(state.job.id): state
        for state in DynamoJobRepository(table).list()
        if str(state.job.id) in DEFAULT_JOB_IDS
    }
    missing = [job_id for job_id in DEFAULT_JOB_IDS if job_id not in states]
    if missing:
        raise LookupError(f"Persisted evaluation jobs were not found: {missing}")

    providers = build_ai_providers(settings)
    matcher = HybridJobMatcher(
        MatchingConfig.from_settings(settings),
        embedding_provider=providers.embeddings,
    )
    filtering_config = FilteringConfig.from_settings(settings)
    role_classifier = RoleFamilyClassifier(
        filtering_config.role_policy or default_ai_data_role_family_policy()
    )
    eligibility = EligibilityEvaluator(EligibilityConfig.from_settings(settings))
    job_filter = DeterministicJobFilter(filtering_config)
    labels = _load_labels()
    rows: list[dict[str, Any]] = []

    for job_id in DEFAULT_JOB_IDS:
        state = states[job_id]
        version = max(state.versions, key=lambda item: item.version_number, default=None)
        if version is None:
            raise LookupError(f"No job version exists for {job_id}")
        try:
            extracted = providers.llm.extract_job_requirements(state.job.description)
            analysis = to_job_analysis(
                extracted,
                job_id=state.job.id,
                job_version_id=version.id,
            )
        except (AIProviderError, ValueError) as exc:
            raise RuntimeError(f"Could not extract analysis for {job_id}") from exc

        match = matcher.match(state.job, profile, analysis=analysis)
        family = role_classifier.classify(state.job.title)
        stage1 = job_filter.evaluate(state.job, profile)
        stage2 = eligibility.evaluate(
            profile,
            state.job,
            analysis,
            role_family_class=family.classification,
        )
        rows.append(
            _row(
                state.job.title,
                state.job.id,
                analysis,
                match,
                family.classification.value if family.classification else None,
                stage1,
                stage2,
                labels.get(job_id),
            )
        )

    result = {
        "configuration": {
            "candidate_profile_key": arguments.profile_key,
            "job_ids": list(DEFAULT_JOB_IDS),
            "llm_provider": settings.llm_provider.value,
            "llm_model_id": settings.mantle_model_id,
            "embedding_provider": settings.embedding_provider.value,
            "embedding_model_id": settings.local_embedding_model_id,
            "production_behavior_changed": False,
            "family_prior": FAMILY_PRIOR,
            "family_prior_blends": [0.10, 0.20],
        },
        "rows": rows,
        "models": _model_results(rows),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "relevance-feasibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (arguments.output_dir / "relevance-feasibility.md").write_text(
        _markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"evaluated": len(rows), "output_dir": str(arguments.output_dir)}))


def _row(
    title: str,
    job_id: Any,
    analysis: Any,
    match: Any,
    family: str | None,
    stage1: Any,
    stage2: Any,
    label: int | None,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    numerator = float(match.evidence["aggregation"]["numerator"])
    for name, score in match.evidence["scores"].items():
        contribution = float(match.evidence["weighted_scores"][name])
        components[name] = {
            "raw_score": float(score),
            "weight": float(match.evidence["weights"][name]),
            "weighted_contribution": contribution,
            "available": bool(match.evidence["components"][name]["available"]),
            "numerator_percentage": contribution / numerator if numerator else None,
        }
    location_rule = next(
        (rule for rule in stage1.rules if rule.rule.value == "location"),
        None,
    )
    relevance = _available_weighted_mean(components, RELEVANCE_COMPONENTS)
    prior = FAMILY_PRIOR.get(family, 0.5)
    return {
        "job_id": str(job_id),
        "title": title,
        "label": label,
        "family": family,
        "category": _category(title),
        "minimum_experience_years": analysis.min_experience_years,
        "seniority": analysis.seniority.value,
        "components": components,
        "current_final_score": float(match.final_score),
        "current_numerator": numerator,
        "current_denominator": float(match.evidence["aggregation"]["denominator"]),
        "relevance_score": relevance,
        "family_prior": prior,
        "stage1_passed": stage1.passed,
        "stage1_location_status": location_rule.status.value if location_rule else None,
        "stage2_status": stage2.status.value,
        "experience_status": stage2.experience_status.value,
        "experience_gap_years": stage2.experience_gap_years,
        "location_eligible": bool(location_rule.passed) if location_rule else None,
    }


def _available_weighted_mean(
    components: dict[str, Any],
    names: tuple[str, ...],
) -> float:
    available = [name for name in names if components[name]["available"]]
    denominator = sum(components[name]["weight"] for name in available)
    if denominator == 0:
        return 0.0
    return sum(components[name]["weighted_contribution"] for name in available) / denominator


def _model_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, list[dict[str, Any]]] = {
        "A_current": _sort(rows, lambda row: row["current_final_score"]),
        "B_relevance_only": _sort(rows, lambda row: row["relevance_score"]),
    }
    for blend in (0.10, 0.20):
        key = f"C_role_family_prior_{int(blend * 100)}pct"
        models[key] = _sort(
            rows,
            lambda row, blend=blend: (
                (1.0 - blend) * row["relevance_score"]
                + blend * row["family_prior"]
            ),
        )
    models["D_relevance_then_feasibility"] = sorted(
        rows,
        key=lambda row: (
            -row["relevance_score"],
            ELIGIBILITY_ORDER.get(row["stage2_status"], 99),
            row["title"].casefold(),
            row["job_id"],
        ),
    )
    models["D_family_then_relevance"] = sorted(
        rows,
        key=lambda row: (
            0 if row["family"] == RoleFamilyClass.CORE.value else 1,
            -row["relevance_score"],
            ELIGIBILITY_ORDER.get(row["stage2_status"], 99),
            row["title"].casefold(),
            row["job_id"],
        ),
    )
    return {
        name: {
            "order": [row["title"] for row in ordered],
            "job_ids": [row["job_id"] for row in ordered],
            "metrics": _metrics(ordered),
        }
        for name, ordered in models.items()
    }


def _sort(rows: list[dict[str, Any]], score: Any) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-score(row), row["title"].casefold(), row["job_id"]),
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = [int(row["label"]) for row in rows if row["label"] is not None]
    if len(labels) != len(rows):
        return {"precision_at_5": None, "ndcg_at_5": None, "ndcg_at_10": None}
    return {
        "precision_at_5": precision_at_k(labels, 5, relevance_threshold=2),
        "ndcg_at_5": ndcg_at_k(labels, 5, relevance_threshold=2),
        "ndcg_at_10": ndcg_at_k(labels, 10, relevance_threshold=2),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Relevance and Feasibility Diagnostic",
        "",
        (
            "Read-only evaluation. Production ranking, filters, role policy, and Stage-2 "
            "eligibility were not changed."
        ),
        "",
        "## Current component decomposition",
        "",
        "| Job | Component | Raw | Weight | Contribution | Available | % numerator |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in result["rows"]:
        for name, component in row["components"].items():
            percentage = component["numerator_percentage"]
            lines.append(
                f"| {row['title']} | {name} | {component['raw_score']:.3f} | "
                f"{component['weight']:.3f} | {component['weighted_contribution']:.3f} | "
                f"{component['available']} | "
                f"{'n/a' if percentage is None else f'{percentage:.1%}'} |"
            )
    lines.extend(["", "## Model comparisons", ""])
    lines.extend(
        [
            "| Model | Ordering | Precision@5 | NDCG@5 | NDCG@10 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, model in result["models"].items():
        metrics = model["metrics"]
        lines.append(
            f"| {name} | {' > '.join(model['order'])} | "
            f"{_metric(metrics['precision_at_5'])} | {_metric(metrics['ndcg_at_5'])} | "
            f"{_metric(metrics['ndcg_at_10'])} |"
        )
    lines.extend(["", "## Per-job relevance and feasibility", ""])
    lines.extend(
        [
            "| Job | Family | Relevance | Stage-2 | Gap | Location | Label |",
            "|---|---|---:|---|---:|---|---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['title']} | {row['family']} | {row['relevance_score']:.3f} | "
            f"{row['stage2_status']} | {_number(row['experience_gap_years'])} | "
            f"{row['location_eligible']} | {row['label']} |"
        )
    lines.extend(
        [
            "",
            "Model C uses a diagnostic-only bounded family prior: CORE=1.00, ADJACENT=0.75, "
            "blended at 10% and 20%. These values are sensitivity probes, not production settings.",
            "",
        ]
    )
    return "\n".join(lines)


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _category(title: str) -> str:
    normalized = title.casefold()
    if "data scientist" in normalized:
        return "data_science"
    if "product manager" in normalized and "ai" in normalized:
        return "product_ai"
    if "analyst" in normalized or "analytics" in normalized:
        return "analytics"
    return "other"


def _table(settings: Settings) -> Any:
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(
        settings.dynamodb_table_name,
        "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME",
    )
    return boto3.resource("dynamodb", region_name=region).Table(table_name)


def _profile(table: Any, profile_key: str) -> CandidateProfile:
    profile = DynamoCandidateProfileRepository(table).find_by_key(profile_key)
    if profile is None:
        raise LookupError(f"Candidate profile was not found: {profile_key}")
    return profile


def _load_labels() -> dict[str, int]:
    if not LABEL_PATH.exists():
        return {}
    values = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    return {
        str(item["job_id"]): int(item["provisional_label"])
        for item in values
        if item.get("provisional_label") is not None
    }


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-key", default="candidate_001")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/relevance-feasibility"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
