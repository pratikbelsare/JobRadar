"""Evaluate the configured local AI fallback against persisted YipitData jobs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3

from job_intelligence.ai import AIProviderError, build_ai_providers, to_job_analysis
from job_intelligence.aws.dynamodb import (
    DynamoCandidateProfileRepository,
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

CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("data_science", ("data scientist",)),
    ("product_ai", ("product", " ai")),
    ("ml_ai", ("machine learning", " ai", "artificial intelligence")),
    ("analytics", ("analyst", "analytics", "insights", "research")),
    (
        "unsuitable_sales",
        ("sales", "account executive", "business development", "client success"),
    ),
    (
        "unsuitable_other",
        ("payroll", "workplace", "security", "sdet", "software engineer", "it delivery"),
    ),
)


def main() -> None:
    arguments = _parse_args()
    settings = Settings.from_environment()
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(
        settings.dynamodb_table_name,
        "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME",
    )
    profile_key = _required(
        settings.candidate_profile_key,
        "JOB_INTELLIGENCE_CANDIDATE_PROFILE_KEY",
    )
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
    profile = DynamoCandidateProfileRepository(table).find_by_key(profile_key)
    if profile is None:
        raise LookupError(f"Candidate profile was not found: {profile_key}")

    states = [
        state
        for state in DynamoJobRepository(table).list()
        if state.job.company_id == yipitdata.id
    ]
    selected = select_sample(states, arguments.sample_size)
    providers = build_ai_providers(settings)
    matching_config = MatchingConfig.from_settings(settings)
    matcher = HybridJobMatcher(matching_config, embedding_provider=providers.embeddings)
    explanation_service = ExplanationService(providers.llm)
    rows = evaluate_jobs(
        selected,
        profile,
        providers.llm,
        matcher,
        explanation_service,
        generate_explanations=not arguments.skip_explanations,
    )
    output_dir = Path(arguments.output_dir)
    paths = write_artifacts(
        rows,
        output_dir,
        configuration={
            "company": yipitdata.name,
            "candidate_profile_key": profile_key,
            "sample_size_requested": arguments.sample_size,
            "sample_size_evaluated": len(rows),
            "llm_provider": settings.llm_provider.value,
            "llm_model_id": settings.mantle_model_id,
            "embedding_provider": settings.embedding_provider.value,
            "embedding_model_id": settings.local_embedding_model_id,
            "ranking_version": matching_config.ranking_version,
            "explanations_requested": not arguments.skip_explanations,
        },
    )
    print(f"evaluated={len(rows)}")
    print(f"json={paths['json']}")
    print(f"csv={paths['csv']}")
    print(f"markdown={paths['markdown']}")


def select_sample(states: list[Any], sample_size: int) -> list[Any]:
    """Select a deterministic, category-balanced sample without changing job data."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for state in states:
        grouped[category_for_title(state.job.title)].append(state)
    for values in grouped.values():
        values.sort(key=lambda state: (state.job.title.casefold(), str(state.job.id)))

    selected: list[Any] = []
    categories = [name for name, _ in CATEGORY_PATTERNS] + ["other"]
    while len(selected) < min(sample_size, len(states)):
        added = False
        for category in categories:
            values = grouped[category]
            if values:
                selected.append(values.pop(0))
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
    return selected


def category_for_title(title: str) -> str:
    normalized = f" {title.casefold()} "
    for category, terms in CATEGORY_PATTERNS:
        if all(term in normalized for term in terms) if category == "product_ai" else any(
            term in normalized for term in terms
        ):
            return category
    return "other"


def evaluate_jobs(
    states: list[Any],
    profile: CandidateProfile,
    llm_provider: Any,
    matcher: HybridJobMatcher,
    explanation_service: ExplanationService,
    *,
    generate_explanations: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in states:
        job = state.job
        base: dict[str, object] = {
            "job_id": str(job.id),
            "title": job.title,
            "category": category_for_title(job.title),
            "source_url": str(job.source_url),
            "job_min_experience_years": job.min_experience_years,
            "job_max_experience_years": job.max_experience_years,
            "analysis_min_experience_years": None,
            "analysis_max_experience_years": None,
            "role_score": None,
            "role_method": None,
            "skill_score": None,
            "experience_score": None,
            "location_score": None,
            "responsibility_score": None,
            "domain_score": None,
            "final_score": None,
            "extracted_seniority": None,
            "extracted_domain": None,
            "analysis_provider": None,
            "analysis_model_id": None,
            "explanation_status": "not_requested",
            "status": "ok",
            "issue_type": None,
            "issue": None,
        }
        version = max(state.versions, key=lambda item: item.version_number, default=None)
        if version is None:
            rows.append(_failure(base, "extraction", "missing_job_version", "No job version"))
            continue
        try:
            result = llm_provider.extract_job_requirements(job.description)
            analysis = to_job_analysis(
                result,
                job_id=job.id,
                job_version_id=version.id,
            )
            match = matcher.match(job, profile, analysis=analysis)
            role_evidence = match.evidence["components"]["role"]
            base.update(
                {
                    "extracted_seniority": analysis.seniority.value,
                    "extracted_domain": analysis.domain,
                    "analysis_min_experience_years": analysis.min_experience_years,
                    "analysis_max_experience_years": analysis.max_experience_years,
                    "analysis_provider": analysis.provider,
                    "analysis_model_id": analysis.model_id,
                    "role_score": match.role_match,
                    "role_method": role_evidence.get("method"),
                    "skill_score": match.skill_match,
                    "experience_score": match.experience_match,
                    "location_score": match.location_match,
                    "responsibility_score": match.responsibility_match,
                    "domain_score": match.domain_match,
                    "final_score": match.final_score,
                    "signal_summary": _signal_summary(match),
                }
            )
            if generate_explanations:
                try:
                    explanation_service.generate(job, profile, match, analysis=analysis)
                    base["explanation_status"] = "generated"
                except ExplanationGroundingError as exc:
                    base["explanation_status"] = "grounding_failed"
                    base["issue_type"] = "explanation"
                    base["issue"] = _safe_error(exc)
                except ExplanationGenerationError as exc:
                    base["explanation_status"] = "generation_failed"
                    base["issue_type"] = "explanation"
                    base["issue"] = _safe_error(exc.__cause__ or exc)
        except AIProviderError as exc:
            rows.append(_failure(base, "extraction", type(exc).__name__, str(exc)))
            continue
        except Exception as exc:
            rows.append(_failure(base, "matching", type(exc).__name__, str(exc)))
            continue
        rows.append(base)
    return sorted(
        rows,
        key=lambda row: (
            -(_numeric_score(row.get("final_score")) or -1.0),
            str(row["title"]).casefold(),
        ),
    )


def write_artifacts(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    configuration: dict[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "persisted-job-ranking.json"
    csv_path = output_dir / "persisted-job-ranking.csv"
    markdown_path = output_dir / "persisted-job-ranking.md"
    payload = {"configuration": configuration, "results": rows}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fields = [
        "title",
        "category",
        "extracted_seniority",
        "extracted_domain",
        "analysis_min_experience_years",
        "analysis_max_experience_years",
        "job_min_experience_years",
        "job_max_experience_years",
        "role_score",
        "role_method",
        "skill_score",
        "experience_score",
        "location_score",
        "responsibility_score",
        "domain_score",
        "final_score",
        "explanation_status",
        "status",
        "issue_type",
        "issue",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Persisted Job Ranking Evaluation",
        "",
        "Scores are existing deterministic matcher outputs, not ground-truth labels.",
        "",
        "| Title | Category | Min Exp | Role | Skills | Experience | Location | "
        "Responsibility | Domain | Final | Explanation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {title} | {category} | {min_experience} | {role} ({method}) | {skill} | "
            "{experience} | {location} | {responsibility} | {domain} | {final} | "
            "{explanation} |".format(
                title=_md(row["title"]),
                category=_md(row["category"]),
                min_experience=_score(row["analysis_min_experience_years"]),
                role=_score(row["role_score"]),
                method=_md(row["role_method"] or "n/a"),
                skill=_score(row["skill_score"]),
                experience=_score(row["experience_score"]),
                location=_score(row["location_score"]),
                responsibility=_score(row["responsibility_score"]),
                domain=_score(row["domain_score"]),
                final=_score(row["final_score"]),
                explanation=_md(row["explanation_status"]),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _signal_summary(match: Any) -> str:
    if match.final_score >= 0.7:
        return "strong aggregate score"
    if match.final_score < 0.35:
        return "weak aggregate score"
    return "mixed aggregate signals"


def _failure(
    row: dict[str, object],
    stage: str,
    issue_type: str,
    issue: str,
) -> dict[str, object]:
    row.update(
        {
            "status": "failed",
            "issue_type": f"{stage}:{issue_type}",
            "issue": _safe_text(issue),
            "explanation_status": "not_attempted",
        }
    )
    return row


def _safe_error(error: Exception) -> str:
    return _safe_text(str(error))


def _safe_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:240]


def _score(value: object) -> str:
    numeric = _numeric_score(value)
    return "n/a" if numeric is None else f"{numeric:.3f}"


def _numeric_score(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _md(value: object) -> str:
    return str(value if value is not None else "n/a").replace("|", "\\|")


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=18,
        help="Number of existing YipitData jobs to evaluate (default: 18).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory for JSON, CSV, and Markdown artifacts.",
    )
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Skip optional explanation calls while evaluating ranking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
