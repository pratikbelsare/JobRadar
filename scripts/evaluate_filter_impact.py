"""Evaluate the existing deterministic filter before unchanged ranking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import boto3

from job_intelligence.ai import AIProviderError, build_ai_providers, to_job_analysis
from job_intelligence.aws.dynamodb import DynamoCompanyRepository, DynamoJobRepository
from job_intelligence.config import Settings
from job_intelligence.filtering import DeterministicJobFilter, FilteringConfig
from job_intelligence.matching import HybridJobMatcher, MatchingConfig
from job_intelligence.models import CandidateProfile

try:
    from scripts.evaluate_profile_impact import (
        _category,
        _ndcg,
        _precision,
        _recall,
        reference_job_ids,
        select_reference_states,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from evaluate_profile_impact import (
        _category,
        _ndcg,
        _precision,
        _recall,
        reference_job_ids,
        select_reference_states,
    )


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_filter(
    states: list[Any],
    profile: CandidateProfile,
    filter_engine: DeterministicJobFilter,
    previous_rows: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    llm_provider: Any,
    matcher: HybridJobMatcher,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in states:
        job = state.job
        job_id = str(job.id)
        previous = previous_rows[job_id]
        label = labels[job_id]
        result = filter_engine.evaluate(job, profile)
        row: dict[str, object] = {
            "job_id": job_id,
            "title": job.title,
            "category": previous.get("category", _category(job.title)),
            "provisional_label": label["provisional_label"],
            "provisional_label_name": label["provisional_label_name"],
            "filter_passed": result.passed,
            "filter_outcome": result.outcome.value,
            "filter_reasons": result.reasons,
            "filter_rules": [rule.model_dump(mode="json") for rule in result.rules],
            "job_minimum_experience_years": job.min_experience_years,
            "analysis_minimum_experience_years": previous.get("minimum_experience_years"),
            "previous_unfiltered_rank": previous.get("after_rank"),
            "previous_unfiltered_score": previous.get("after_final"),
            "status": "filtered_out" if not result.passed else "retained",
            "ranking_error": None,
        }
        role_rule = next((rule for rule in result.rules if rule.rule.value == "role"), None)
        row["role_filter_status"] = role_rule.status.value if role_rule else None
        row["role_filter_details"] = role_rule.details if role_rule else {}
        if result.passed:
            try:
                version = max(state.versions, key=lambda item: item.version_number, default=None)
                if version is None:
                    raise ValueError("No job version is available")
                analysis_result = llm_provider.extract_job_requirements(job.description)
                analysis = to_job_analysis(
                    analysis_result,
                    job_id=job.id,
                    job_version_id=version.id,
                )
                match = matcher.match(job, profile, analysis=analysis)
                row.update(
                    {
                        "extracted_domain": analysis.domain,
                        "extracted_seniority": analysis.seniority.value,
                        "minimum_experience_years": analysis.min_experience_years,
                        "role_score": match.role_match,
                        "skill_score": match.skill_match,
                        "experience_score": match.experience_match,
                        "responsibility_score": match.responsibility_match,
                        "domain_score": match.domain_match,
                        "location_score": match.location_match,
                        "filtered_score": match.final_score,
                        "status": "ranked",
                    }
                )
            except (AIProviderError, ValueError) as exc:
                row["status"] = "ranking_failed"
                row["ranking_error"] = _safe_text(str(exc))
        rows.append(row)
    return add_filtered_ranks(rows)


def add_filtered_ranks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(
        (row for row in rows if row.get("status") == "ranked"),
        key=lambda row: -_as_float(row["filtered_score"]),
    )
    ranks = {str(row["job_id"]): index for index, row in enumerate(ranked, 1)}
    for row in rows:
        row["filtered_rank"] = ranks.get(str(row["job_id"]))
    return rows


def compare_metrics(
    rows: list[dict[str, object]],
    previous_rows: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, object]:
    previous_ranked = [
        row for row in previous_rows.values() if row.get("status") == "ok"
    ]
    previous_ranked.sort(key=lambda row: -_as_float(row["after_final"]))
    retained = [row for row in rows if row.get("status") == "ranked"]
    retained.sort(key=lambda row: -_as_float(row["filtered_score"]))
    label_values = {
        job_id: int(value["provisional_label"]) for job_id, value in labels.items()
    }
    before = {
        "precision_at_5": _precision(previous_ranked, label_values, 5),
        "precision_at_10": _precision(previous_ranked, label_values, 10),
        "recall_at_10": _recall(
            previous_ranked,
            label_values,
            10,
            sum(label >= 2 for label in label_values.values()),
        ),
        "ndcg_at_5": _ndcg(previous_ranked, label_values, 5),
        "ndcg_at_10": _ndcg(previous_ranked, label_values, 10),
    }
    relevant_total = sum(label >= 2 for label in label_values.values())
    retained_relevant = sum(
        label_values.get(str(row["job_id"]), 0) >= 2 for row in retained
    )
    retained_labels = {
        str(row["job_id"]): label_values[str(row["job_id"])] for row in retained
    }
    after = {
        "precision_at_5": _precision(retained, label_values, 5),
        "precision_at_10": _precision(retained, label_values, 10),
        "recall_relevant_retained": retained_relevant / relevant_total
        if relevant_total
        else 0.0,
        "ndcg_at_5": _ndcg(retained, retained_labels, 5),
        "ndcg_at_10": _ndcg(retained, retained_labels, 10),
    }
    removed = [row for row in rows if row["status"] == "filtered_out"]
    return {
        "before_filtering": before,
        "after_filtering": after,
        "jobs_total": len(rows),
        "jobs_removed": len(removed),
        "jobs_retained": len([row for row in rows if row["filter_passed"]]),
        "irrelevant_removed": sum(
            label_values.get(str(row["job_id"]), 0) < 2 for row in removed
        ),
        "relevant_or_stretch_removed": sum(
            label_values.get(str(row["job_id"]), 0) >= 2 for row in removed
        ),
    }


def write_artifacts(
    rows: list[dict[str, object]], metrics: dict[str, object], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "filter-impact.json").write_text(
        json.dumps({"metrics": metrics, "results": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fields = [
        "job_id",
        "title",
        "category",
        "provisional_label",
        "provisional_label_name",
        "filter_passed",
        "filter_outcome",
        "filter_reasons",
        "analysis_minimum_experience_years",
        "role_filter_status",
        "previous_unfiltered_rank",
        "filtered_rank",
        "role_score",
        "skill_score",
        "experience_score",
        "responsibility_score",
        "domain_score",
        "location_score",
        "filtered_score",
        "status",
        "ranking_error",
    ]
    with (output_dir / "filter-impact.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["filter_reasons"] = " ; ".join(_as_reasons(row["filter_reasons"]))
            writer.writerow(output)

    lines = [
        "# Existing Filter Impact Evaluation",
        "",
        "The deterministic filter is evaluated against canonical Job fields using the "
        "existing configured rules. No filter or ranking logic was changed.",
        "",
        "| Filtered rank | Previous rank | Title | Label | Outcome | Min exp | Role | "
        "Skills | Exp | Responsibility | Domain | Location | Final |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: _rank_key(value.get("filtered_rank"))):
        lines.append(
            "| {rank} | {previous} | {title} | {label} | {outcome} | {minimum} | "
            "{role} | {skills} | {experience} | {responsibility} | {domain} | "
            "{location} | {final} |".format(
                rank=row.get("filtered_rank") or "-",
                previous=row.get("previous_unfiltered_rank") or "-",
                title=_md(row["title"]),
                label=row["provisional_label"],
                outcome=row["filter_outcome"],
                minimum=_score(row.get("minimum_experience_years")),
                role=_score(row.get("role_score")),
                skills=_score(row.get("skill_score")),
                experience=_score(row.get("experience_score")),
                responsibility=_score(row.get("responsibility_score")),
                domain=_score(row.get("domain_score")),
                location=_score(row.get("location_score")),
                final=_score(row.get("filtered_score")),
            )
        )
    lines.extend(["", "## Metric comparison", "", "```json", json.dumps(metrics, indent=2), "```"])
    lines.extend(["", "## Filter reasons", ""])
    for row in rows:
        lines.append(
            f"- **{_md(row['title'])}** — {row['filter_outcome']}: "
            f"{' '.join(_as_reasons(row['filter_reasons']))}"
        )
    (output_dir / "filter-impact.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = _parse_args()
    settings = Settings.from_environment()
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(settings.dynamodb_table_name, "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME")
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    company = next(
        company
        for company in DynamoCompanyRepository(table).list()
        if company.name.casefold() == "yipitdata"
    )
    all_states = [
        state for state in DynamoJobRepository(table).list() if state.job.company_id == company.id
    ]
    ids = reference_job_ids(Path(arguments.reference))
    states = select_reference_states(all_states, ids)
    previous_payload = load_json(Path(arguments.unfiltered_artifact))
    label_payload = load_json(Path(arguments.labels))
    if not isinstance(previous_payload, dict) or not isinstance(label_payload, list):
        raise ValueError("Evaluation artifacts have unexpected shapes")
    previous_rows = {str(row["job_id"]): row for row in previous_payload["results"]}
    labels = {str(row["job_id"]): row for row in label_payload}
    profile = CandidateProfile.model_validate_json(
        Path(arguments.profile).read_text(encoding="utf-8")
    )
    filter_engine = DeterministicJobFilter(FilteringConfig.from_settings(settings))
    providers = build_ai_providers(settings)
    matcher = HybridJobMatcher(
        MatchingConfig.from_settings(settings),
        embedding_provider=providers.embeddings,
    )
    rows = evaluate_filter(
        states,
        profile,
        filter_engine,
        previous_rows,
        labels,
        providers.llm,
        matcher,
    )
    metrics = compare_metrics(rows, previous_rows, labels)
    write_artifacts(rows, metrics, Path(arguments.output_dir))
    print(f"evaluated={len(rows)}")
    print(json.dumps(metrics, indent=2))


def _rank_key(value: object) -> int:
    return int(value) if isinstance(value, int) else 999


def _score(value: object) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:.3f}"


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric value, got {value!r}")
    return float(value)


def _as_reasons(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected filter reasons, got {value!r}")
    return value


def _md(value: object) -> str:
    return str(value if value is not None else "-").replace("|", "\\|")


def _safe_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:240]


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="evaluation/profile/pratik-enriched.json")
    parser.add_argument("--reference", default="evaluation/results/persisted-job-ranking.json")
    parser.add_argument(
        "--unfiltered-artifact",
        default="evaluation/results/profile-impact/profile-impact.json",
    )
    parser.add_argument(
        "--labels",
        default="evaluation/results/profile-impact/profile-labels.json",
    )
    parser.add_argument("--output-dir", default="evaluation/results/filter-impact")
    return parser.parse_args()


if __name__ == "__main__":
    main()
