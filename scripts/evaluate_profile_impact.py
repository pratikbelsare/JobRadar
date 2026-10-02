"""Compare the unchanged matcher with sparse and locally enriched profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import boto3

from job_intelligence.ai import (
    AIProviderError,
    build_ai_providers,
    to_job_analysis,
)
from job_intelligence.aws.dynamodb import (
    DynamoCompanyRepository,
    DynamoJobRepository,
)
from job_intelligence.config import Settings
from job_intelligence.explanations import (
    ExplanationGenerationError,
    ExplanationGroundingError,
    ExplanationService,
)
from job_intelligence.matching import HybridJobMatcher, MatchingConfig
from job_intelligence.models import CandidateProfile

LABEL_NAMES = {
    3: "highly_relevant",
    2: "relevant_stretch",
    1: "adjacent",
    0: "irrelevant",
}


def load_profile(path: Path) -> CandidateProfile:
    return CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))


def reference_job_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(row["job_id"]) for row in payload["results"]]
    if len(ids) != len(set(ids)):
        raise ValueError("The reference evaluation contains duplicate job IDs")
    return ids


def select_reference_states(states: list[Any], job_ids: list[str]) -> list[Any]:
    by_id = {str(state.job.id): state for state in states}
    missing = [job_id for job_id in job_ids if job_id not in by_id]
    if missing:
        raise LookupError(f"Reference jobs are missing from DynamoDB: {missing}")
    return [by_id[job_id] for job_id in job_ids]


def compare_profiles(
    states: list[Any],
    before: CandidateProfile,
    after: CandidateProfile,
    llm_provider: Any,
    matcher: HybridJobMatcher,
    explanation_service: ExplanationService,
    *,
    generate_explanations: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in states:
        job = state.job
        version = max(state.versions, key=lambda item: item.version_number, default=None)
        if version is None:
            rows.append({"job_id": str(job.id), "title": job.title, "status": "failed"})
            continue
        try:
            analysis_result = llm_provider.extract_job_requirements(job.description)
            analysis = to_job_analysis(
                analysis_result,
                job_id=job.id,
                job_version_id=version.id,
            )
            before_match = matcher.match(job, before, analysis=analysis)
            after_match = matcher.match(job, after, analysis=analysis)
            explanation_status = "not_requested"
            if generate_explanations:
                try:
                    explanation_service.generate(job, after, after_match, analysis=analysis)
                    explanation_status = "generated"
                except ExplanationGroundingError:
                    explanation_status = "grounding_failed"
                except ExplanationGenerationError:
                    explanation_status = "generation_failed"
            rows.append(
                {
                    "job_id": str(job.id),
                    "title": job.title,
                    "category": _category(job.title),
                    "minimum_experience_years": analysis.min_experience_years,
                    "maximum_experience_years": analysis.max_experience_years,
                    "status": "ok",
                    "role": after_match.role_match,
                    "before_final": before_match.final_score,
                    "after_final": after_match.final_score,
                    "before_skills": before_match.skill_match,
                    "after_skills": after_match.skill_match,
                    "before_experience": before_match.experience_match,
                    "after_experience": after_match.experience_match,
                    "before_responsibility": before_match.responsibility_match,
                    "after_responsibility": after_match.responsibility_match,
                    "before_domain": before_match.domain_match,
                    "after_domain": after_match.domain_match,
                    "location": after_match.location_match,
                    "role_method": after_match.evidence["components"]["role"].get("method"),
                    "extracted_domain": analysis.domain,
                    "extracted_seniority": analysis.seniority.value,
                    "explanation_status": explanation_status,
                }
            )
        except AIProviderError as exc:
            rows.append(
                {
                    "job_id": str(job.id),
                    "title": job.title,
                    "category": _category(job.title),
                    "status": "failed",
                    "issue": _safe_text(str(exc)),
                }
            )
    return _add_ranks(rows)


def provisional_label(row: dict[str, object]) -> tuple[int, str]:
    """Assign a reviewable title/category label without reading any score."""

    title = str(row["title"]).casefold()
    category = str(row.get("category", ""))
    minimum = row.get("minimum_experience_years")
    if "data scientist" in title:
        return 2, (
            "Direct target role, but the stated experience requirement is above "
            "the controlled 1.2-year profile value."
        )
    if "data product analyst" in title or "product manager" in title and "ai" in title:
        return 2, (
            "Relevant data/AI-adjacent role family, but the function or stated "
            "experience requirement makes it a stretch."
        )
    if "data analyst" in title:
        return 1, (
            "Data-adjacent role, but not one of the configured ML Engineer, AI "
            "Engineer, or Data Scientist targets."
        )
    if category == "analytics" or "research analyst" in title or "insights" in title:
        return 1, "Adjacent analytics or research role rather than a configured target role."
    if "data qa" in title:
        return 1, "Data-adjacent quality role, but not a configured target role."
    if minimum is not None and _as_float(minimum) > 4 and any(
        term in title for term in ("product", "engineer", "director")
    ):
        return 0, (
            "Unrelated or substantially senior function for the configured "
            "target roles and experience."
        )
    return 0, (
        "No direct AI, ML, Data Scientist, or closely adjacent target-role "
        "evidence in the title."
    )


def build_labels(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    for row in rows:
        label, rationale = provisional_label(row)
        labels.append(
            {
                "job_id": row["job_id"],
                "title": row["title"],
                "extracted_category": row.get("category"),
                "minimum_experience_years": row.get("minimum_experience_years"),
                "provisional_label": label,
                "provisional_label_name": LABEL_NAMES[label],
                "rationale": rationale,
                "human_label": None,
            }
        )
    return labels


def ranking_metrics(
    rows: list[dict[str, object]], labels: list[dict[str, object]]
) -> dict[str, float]:
    labels_by_id = {
        str(row["job_id"]): _as_int(row["provisional_label"]) for row in labels
    }
    ranked = sorted(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: -_as_float(row["after_final"]),
    )
    relevant_total = sum(label >= 2 for label in labels_by_id.values())
    return {
        "precision_at_5": _precision(ranked, labels_by_id, 5),
        "precision_at_10": _precision(ranked, labels_by_id, 10),
        "recall_at_10": _recall(ranked, labels_by_id, 10, relevant_total),
        "ndcg_at_5": _ndcg(ranked, labels_by_id, 5),
        "ndcg_at_10": _ndcg(ranked, labels_by_id, 10),
    }


def write_artifacts(
    rows: list[dict[str, object]],
    labels: list[dict[str, object]],
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "results": rows}
    (output_dir / "profile-impact.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "profile-labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8"
    )

    label_fields = [
        "job_id",
        "title",
        "extracted_category",
        "minimum_experience_years",
        "provisional_label",
        "provisional_label_name",
        "rationale",
        "human_label",
    ]
    with (output_dir / "profile-labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=label_fields)
        writer.writeheader()
        writer.writerows(labels)

    lines = [
        "# Candidate Profile Impact Evaluation",
        "",
        "The before and after rows use one extraction per job and the unchanged "
        "matcher configuration.",
        "Scores and provisional labels are not ground truth.",
        "",
        "| After rank | Before rank | Title | Before | After | Role | Skills | "
        "Experience | Responsibility | Domain | Location |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: _as_int(item.get("after_rank"), default=999)):
        lines.append(
            (
                "| {after_rank} | {before_rank} | {title} | {before} | {after} | "
                "{role} | {skills} | {experience} | {responsibility} | "
                "{domain} | {location} |"
            ).format(
                after_rank=row.get("after_rank", "n/a"),
                before_rank=row.get("before_rank", "n/a"),
                title=_md(row.get("title")),
                before=_score(row.get("before_final")),
                after=_score(row.get("after_final")),
                role=_score(row.get("role")),
                skills=_score(row.get("after_skills")),
                experience=_score(row.get("after_experience")),
                responsibility=_score(row.get("after_responsibility")),
                domain=_score(row.get("after_domain")),
                location=_score(row.get("location")),
            )
        )
    lines.extend(["", "## Provisional metrics", ""])
    for name, value in metrics.items():
        lines.append(f"- {name}: {value:.4f}")
    (output_dir / "profile-impact.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = _parse_args()
    settings = Settings.from_environment()
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(settings.dynamodb_table_name, "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME")
    if settings.llm_provider.value != "bedrock_mantle":
        raise ValueError("Set JOB_INTELLIGENCE_LLM_PROVIDER=bedrock_mantle")
    if settings.embedding_provider.value != "huggingface_local":
        raise ValueError("Set JOB_INTELLIGENCE_EMBEDDING_PROVIDER=huggingface_local")

    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    companies = DynamoCompanyRepository(table).list()
    yipitdata = next(
        (company for company in companies if company.name.casefold() == "yipitdata"),
        None,
    )
    if yipitdata is None:
        raise LookupError("YipitData company was not found in DynamoDB")
    all_states = [
        state for state in DynamoJobRepository(table).list() if state.job.company_id == yipitdata.id
    ]
    states = select_reference_states(all_states, reference_job_ids(Path(arguments.reference)))
    before = load_profile(Path(arguments.baseline_profile))
    after = load_profile(Path(arguments.profile))
    providers = build_ai_providers(settings)
    matcher = HybridJobMatcher(
        MatchingConfig.from_settings(settings),
        embedding_provider=providers.embeddings,
    )
    rows = compare_profiles(
        states,
        before,
        after,
        providers.llm,
        matcher,
        ExplanationService(providers.llm),
        generate_explanations=not arguments.skip_explanations,
    )
    labels = build_labels(rows)
    metrics = ranking_metrics(rows, labels)
    write_artifacts(rows, labels, metrics, Path(arguments.output_dir))
    print(f"evaluated={len(rows)}")
    print(f"output_dir={arguments.output_dir}")
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")


def _add_ranks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    successful = [row for row in rows if row.get("status") == "ok"]
    before_order = sorted(successful, key=lambda row: -_as_float(row["before_final"]))
    after_order = sorted(successful, key=lambda row: -_as_float(row["after_final"]))
    before_ranks = {str(row["job_id"]): index for index, row in enumerate(before_order, 1)}
    after_ranks = {str(row["job_id"]): index for index, row in enumerate(after_order, 1)}
    for row in rows:
        key = str(row["job_id"])
        row["before_rank"] = before_ranks.get(key)
        row["after_rank"] = after_ranks.get(key)
        if row.get("before_rank") is not None and row.get("after_rank") is not None:
            row["rank_movement"] = _as_int(row["before_rank"]) - _as_int(row["after_rank"])
    return rows


def _precision(ranked: list[dict[str, object]], labels: dict[str, int], k: int) -> float:
    top = ranked[:k]
    return sum(labels.get(str(row["job_id"]), 0) >= 2 for row in top) / len(top) if top else 0.0


def _recall(
    ranked: list[dict[str, object]], labels: dict[str, int], k: int, relevant_total: int
) -> float:
    if relevant_total == 0:
        return 0.0
    return sum(labels.get(str(row["job_id"]), 0) >= 2 for row in ranked[:k]) / relevant_total


def _ndcg(ranked: list[dict[str, object]], labels: dict[str, int], k: int) -> float:
    def gain(index: int, label: int) -> float:
        return (2**label - 1) / math.log2(index + 2)

    actual = sum(
        gain(index, labels.get(str(row["job_id"]), 0))
        for index, row in enumerate(ranked[:k])
    )
    ideal_labels = sorted(labels.values(), reverse=True)[:k]
    ideal = sum(gain(index, label) for index, label in enumerate(ideal_labels))
    return actual / ideal if ideal else 0.0


def _category(title: str) -> str:
    normalized = title.casefold()
    if "data scientist" in normalized:
        return "data_science"
    if "product" in normalized and " ai" in f" {normalized}":
        return "product_ai"
    if any(term in normalized for term in ("machine learning", "artificial intelligence")):
        return "ml_ai"
    if any(term in normalized for term in ("analyst", "analytics", "insights", "research")):
        return "analytics"
    if any(
        term in normalized
        for term in ("sales", "account executive", "business development", "client success")
    ):
        return "unsuitable_sales"
    if any(term in normalized for term in ("payroll", "security", "sdet", "it delivery")):
        return "unsuitable_other"
    return "other"


def _score(value: object) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.3f}"


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric value, got {value!r}")
    return float(value)


def _as_int(value: object, *, default: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected an integer value, got {value!r}")
    if isinstance(value, int):
        return value
    if default is not None and value is None:
        return default
    raise ValueError(f"Expected an integer value, got {value!r}")


def _md(value: object) -> str:
    return str(value if value is not None else "n/a").replace("|", "\\|")


def _safe_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:240]


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="evaluation/profile/pratik-enriched.json")
    parser.add_argument("--baseline-profile", default="examples/candidate_profile.example.json")
    parser.add_argument("--reference", default="evaluation/results/persisted-job-ranking.json")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--skip-explanations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
