"""Run two local Moloco scans and write product-validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastapi.testclient import TestClient

from job_intelligence.ai import build_ai_providers
from job_intelligence.api import build_local_application, create_app
from job_intelligence.config import EmbeddingProvider, LLMProvider, Settings
from job_intelligence.explanations import ExplanationService
from job_intelligence.local_runtime import (
    LocalRepositories,
    build_local_pipeline,
    build_local_repositories,
)
from job_intelligence.models import CandidateProfile, Company, ConnectorType, JobStatus
from job_intelligence.pipeline import PipelineRunResult


def main() -> None:
    arguments = _arguments()
    settings = Settings.from_environment().model_copy(
        update={"application_data_directory": arguments.data_dir}
    )
    _validate_settings(settings)
    profile = CandidateProfile.model_validate_json(arguments.profile.read_text(encoding="utf-8"))
    settings = settings.model_copy(update={"candidate_profile_id": profile.id})
    repositories = build_local_repositories(settings)
    repositories.profiles.save(profile)
    company = _load_or_create_moloco(repositories, arguments.board_token)
    providers = build_ai_providers(settings)
    pipeline = build_local_pipeline(settings, repositories, providers.llm, providers.embeddings)

    results = [pipeline.run_company(company, profile) for _ in range(arguments.runs)]
    api_validation = _validate_api_workflow(settings, providers.llm)
    _write_artifacts(
        arguments.output_dir,
        results,
        repositories,
        profile,
        api_validation,
    )
    print(json.dumps([result.summary.model_dump(mode="json") for result in results], indent=2))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("evaluation/benchmark/v2/candidate_profile.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/moloco-e2e-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/moloco-e2e-v1"),
    )
    parser.add_argument("--board-token", default="moloco")
    parser.add_argument("--runs", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.runs not in {1, 2}:
        parser.error("local Moloco validation supports one or two runs")
    return arguments


def _validate_settings(settings: Settings) -> None:
    required = {
        "JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS": settings.http_timeout_seconds,
        "JOB_INTELLIGENCE_HTTP_USER_AGENT": settings.http_user_agent,
        "JOB_INTELLIGENCE_HTTP_MAX_RETRIES": settings.http_max_retries,
        "JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS": (
            settings.filter_acceptable_experience_gap_years
        ),
        "JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS": (
            settings.filter_max_stretch_experience_gap_years
        ),
    }
    if settings.llm_provider in {LLMProvider.BEDROCK, LLMProvider.BEDROCK_MANTLE}:
        required["JOB_INTELLIGENCE_AWS_REGION"] = settings.aws_region
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError("Missing required configuration: " + ", ".join(missing))
    if settings.embedding_provider != EmbeddingProvider.HUGGINGFACE_LOCAL:
        raise RuntimeError("Set JOB_INTELLIGENCE_EMBEDDING_PROVIDER=huggingface_local")


def _load_or_create_moloco(
    repositories: LocalRepositories,
    board_token: str,
) -> Company:
    existing = next(
        (company for company in repositories.companies.list() if company.name == "Moloco"),
        None,
    )
    if existing is not None:
        return existing
    career_url = f"https://job-boards.greenhouse.io/{board_token}"
    company = Company.model_validate(
        {
            "id": uuid5(NAMESPACE_URL, career_url),
            "name": "Moloco",
            "career_url": career_url,
            "connector_type": ConnectorType.GREENHOUSE,
            "connector_config": {"board_token": board_token},
        }
    )
    return repositories.companies.save(company)


def _validate_api_workflow(settings: Settings, llm: Any) -> dict[str, object]:
    explanation_service = ExplanationService(llm)
    application = build_local_application(settings, explanation_service=explanation_service)
    client = TestClient(create_app(application))
    health = client.get("/health")
    jobs_response = client.get("/jobs", params={"company": "Moloco", "sort_by": "score"})
    jobs_response.raise_for_status()
    jobs = jobs_response.json()
    matched = [item for item in jobs if item.get("match") is not None]
    detail_ok = False
    explained = next((item for item in matched if item.get("explanation") is not None), None)
    if explained is not None:
        detail = client.get(f"/jobs/{explained['job']['id']}")
        detail.raise_for_status()
        payload = detail.json()
        match = payload.get("match") or {}
        detail_ok = all(
            (
                payload.get("company") is not None,
                match.get("relevance_score") is not None,
                match.get("eligibility") is not None,
                match.get("role_family_class") is not None,
                payload.get("explanation") is not None,
            )
        )

    actions = [
        ("shortlist", JobStatus.SHORTLISTED.value),
        ("dismiss", JobStatus.DISMISSED.value),
        ("applied", JobStatus.APPLIED.value),
    ]
    action_results: dict[str, bool] = {}
    for item, (endpoint, expected) in zip(matched[:3], actions, strict=False):
        job_id = item["job"]["id"]
        response = client.post(f"/jobs/{job_id}/{endpoint}")
        action_results[endpoint] = (
            response.status_code == 200 and response.json()["job"]["status"] == expected
        )

    restarted = TestClient(
        create_app(build_local_application(settings, explanation_service=explanation_service))
    )
    persisted_actions: dict[str, bool] = {}
    for item, (endpoint, expected) in zip(matched[:3], actions, strict=False):
        job_id = item["job"]["id"]
        response = restarted.get(f"/jobs/{job_id}")
        persisted_actions[endpoint] = (
            response.status_code == 200 and response.json()["job"]["status"] == expected
        )

    ui_contract_ok = bool(matched) and all(
        item["job"].get("title")
        and item.get("company")
        and item["match"].get("relevance_score") is not None
        and item["match"].get("role_family_class")
        and item["match"].get("eligibility")
        and item["job"].get("status")
        for item in matched
    )
    return {
        "health": health.status_code == 200 and health.json().get("status") == "ok",
        "job_list_count": len(jobs),
        "matched_job_count": len(matched),
        "job_detail_contract": detail_ok,
        "actions": action_results,
        "actions_persisted_after_restart": persisted_actions,
        "streamlit_data_contract": ui_contract_ok,
    }


def _write_artifacts(
    output_directory: Path,
    results: list[PipelineRunResult],
    repositories: LocalRepositories,
    profile: CandidateProfile,
    api_validation: dict[str, object],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results, start=1):
        _write_json(
            output_directory / f"run-{index}-summary.json",
            result.model_dump(mode="json"),
        )
    _write_json(
        output_directory / "failures.json",
        [
            {"run": index, **failure.model_dump(mode="json")}
            for index, result in enumerate(results, start=1)
            for failure in result.failures
        ],
    )
    _write_json(output_directory / "api-ui-validation.json", api_validation)
    latest = results[-1]
    outcome_by_id = {outcome.job_id: outcome for outcome in latest.jobs if outcome.job_id}
    ranked_rows: list[dict[str, object]] = []
    for match in repositories.matches.list(profile.id):
        state = repositories.jobs.find_by_id(match.job_id)
        if state is None or state.job.company_id != latest.summary.company_id:
            continue
        outcome = outcome_by_id.get(match.job_id)
        evidence = match.relevance_evidence
        ranked_rows.append(
            {
                "title": state.job.title,
                "location": state.job.location or "",
                "source_url": str(state.job.source_url),
                "role_family": match.role_family_class or "unknown",
                "eligibility": match.eligibility.status if match.eligibility else "unknown",
                "experience_gap_years": (
                    match.eligibility.experience_gap_years if match.eligibility else None
                ),
                "relevance_score": match.relevance_score,
                "role_score": match.role_match,
                "skill_score": match.skill_match,
                "responsibility_score": match.responsibility_match,
                "domain_score": match.domain_match,
                "available_components": (
                    ",".join(evidence.included_components) if evidence else ""
                ),
                "explanation_status": (
                    outcome.explanation_status.value if outcome is not None else "unknown"
                ),
                "workflow_status": state.job.status.value,
            }
        )
    ranked_rows.sort(
        key=lambda row: _numeric(row["relevance_score"]),
        reverse=True,
    )
    _write_csv(output_directory / "ranked-jobs.csv", ranked_rows)
    rejected_rows: list[dict[str, object]] = []
    for outcome in latest.jobs:
        if outcome.stage1 is None or outcome.stage1.passed:
            continue
        rejected_rows.append(
            {
                "title": outcome.title,
                "source_job_id": outcome.source_job_id,
                "role_family": outcome.role_family_class or "unknown",
                "outcome": outcome.stage1.outcome.value,
                "reasons": " | ".join(outcome.stage1.reasons),
            }
        )
    _write_csv(output_directory / "rejected-jobs.csv", rejected_rows)
    (output_directory / "e2e-report.md").write_text(
        _markdown_report(results, ranked_rows, api_validation),
        encoding="utf-8",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    results: list[PipelineRunResult],
    ranked_rows: list[dict[str, object]],
    api_validation: dict[str, object],
) -> str:
    lines = [
        "# Moloco end-to-end validation",
        "",
        "Local product validation using matching v0.1. This is not a ranking benchmark.",
        "",
        "| Run | Discovered | New | Changed | Unchanged | Retained | Rejected | Failures |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, result in enumerate(results, start=1):
        summary = result.summary
        lines.append(
            f"| {index} | {summary.jobs_discovered} | {summary.new_jobs} | "
            f"{summary.changed_jobs} | {summary.unchanged_jobs} | "
            f"{summary.stage1_retained} | {summary.stage1_rejected} | "
            f"{summary.failures} |"
        )
    lines.extend(
        [
            "",
            "## Ranked retained jobs",
            "",
            "| Rank | Title | Relevance | Role family | Eligibility | Gap |",
            "|---:|---|---:|---|---|---:|",
        ]
    )
    for rank, row in enumerate(ranked_rows, start=1):
        relevance = row["relevance_score"]
        relevance_text = f"{_numeric(relevance):.4f}" if relevance is not None else "n/a"
        lines.append(
            f"| {rank} | {row['title']} | {relevance_text} | {row['role_family']} | "
            f"{row['eligibility']} | {row['experience_gap_years'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## API and Streamlit contract validation",
            "",
            "```json",
            json.dumps(api_validation, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "Relevance is presented independently from feasibility. A high relevance score "
            "can therefore appear alongside a severe experience stretch without contradiction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _numeric(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()
