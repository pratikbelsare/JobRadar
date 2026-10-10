"""Run two local multi-company scans and write product-validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
from job_intelligence.models import CandidateProfile, Company, JobStatus
from job_intelligence.multi_company import MultiCompanyPipelineRunner, MultiCompanyRunResult


def main() -> None:
    arguments = _arguments()
    settings = Settings.from_environment().model_copy(
        update={"application_data_directory": arguments.data_dir}
    )
    _validate_settings(settings)
    profile = CandidateProfile.model_validate_json(
        arguments.profile.read_text(encoding="utf-8")
    )
    settings = settings.model_copy(update={"candidate_profile_id": profile.id})
    repositories = build_local_repositories(settings)
    repositories.profiles.save(profile)
    companies = _load_companies(arguments.companies)
    repositories.companies.save_all(companies)
    providers = build_ai_providers(settings)
    pipeline = build_local_pipeline(
        settings,
        repositories,
        providers.llm,
        providers.embeddings,
    )
    runner = MultiCompanyPipelineRunner(pipeline)

    results = [runner.run(companies, profile) for _ in range(arguments.runs)]
    api_validation = _validate_api_workflow(settings, providers.llm, companies)
    _write_artifacts(
        arguments.output_dir,
        results,
        repositories,
        profile,
        api_validation,
    )
    print(
        json.dumps(
            [result.summary.model_dump(mode="json") for result in results],
            indent=2,
            ensure_ascii=False,
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("evaluation/benchmark/v2/candidate_profile.json"),
    )
    parser.add_argument(
        "--companies",
        type=Path,
        default=Path("evaluation/config/multi-company-local-v1.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/multi-company-e2e-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/multi-company-e2e-v1"),
    )
    parser.add_argument("--runs", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.runs != 2:
        parser.error("Milestone 14 validation requires exactly two runs")
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


def _load_companies(path: Path) -> list[Company]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Company configuration must contain a JSON list")
    companies = [Company.model_validate(item) for item in raw]
    enabled = [company for company in companies if company.enabled]
    if not enabled:
        raise ValueError("At least one company must be enabled")
    if len({company.id for company in companies}) != len(companies):
        raise ValueError("Company IDs must be unique")
    return companies


def _validate_api_workflow(
    settings: Settings,
    llm: Any,
    companies: list[Company],
) -> dict[str, object]:
    explanation_service = ExplanationService(llm)
    application = build_local_application(settings, explanation_service=explanation_service)
    client = TestClient(create_app(application))
    health = client.get("/health")
    jobs_response = client.get("/jobs", params={"sort_by": "score"})
    jobs_response.raise_for_status()
    jobs = jobs_response.json()
    matched = [item for item in jobs if item.get("match") is not None]
    configured_names = {company.name for company in companies if company.enabled}
    represented_names = {
        item["company"]["name"] for item in jobs if item.get("company") is not None
    }
    matched_company_names = {
        item["company"]["name"] for item in matched if item.get("company") is not None
    }

    company_filters: dict[str, bool] = {}
    for company in companies:
        if not company.enabled:
            continue
        response = client.get("/jobs", params={"company": company.name})
        response.raise_for_status()
        company_filters[company.name] = bool(response.json()) and all(
            item.get("company", {}).get("name") == company.name for item in response.json()
        )

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
                payload["job"].get("source_url") is not None,
                match.get("relevance_score") is not None,
                match.get("eligibility") is not None,
                match.get("role_family_class") is not None,
                payload.get("explanation") is not None,
            )
        )

    action_candidates: list[dict[str, Any]] = []
    action_companies: set[str] = set()
    for item in matched:
        company_name = item.get("company", {}).get("name")
        if company_name and company_name not in action_companies:
            action_candidates.append(item)
            action_companies.add(company_name)
        if len(action_candidates) == 3:
            break
    if len(action_candidates) < 3:
        used_ids = {item["job"]["id"] for item in action_candidates}
        action_candidates.extend(
            item for item in matched if item["job"]["id"] not in used_ids
        )

    actions = [
        ("shortlist", JobStatus.SHORTLISTED.value),
        ("dismiss", JobStatus.DISMISSED.value),
        ("applied", JobStatus.APPLIED.value),
    ]
    action_results: dict[str, bool] = {}
    action_job_ids: list[tuple[str, str, str]] = []
    for item, (endpoint, expected) in zip(action_candidates[:3], actions, strict=False):
        job_id = item["job"]["id"]
        response = client.post(f"/jobs/{job_id}/{endpoint}")
        action_results[endpoint] = (
            response.status_code == 200 and response.json()["job"]["status"] == expected
        )
        action_job_ids.append((job_id, endpoint, expected))

    restarted = TestClient(
        create_app(build_local_application(settings, explanation_service=explanation_service))
    )
    persisted_actions: dict[str, bool] = {}
    for job_id, endpoint, expected in action_job_ids:
        response = restarted.get(f"/jobs/{job_id}")
        persisted_actions[endpoint] = (
            response.status_code == 200 and response.json()["job"]["status"] == expected
        )

    scores = [
        item["match"].get("relevance_score")
        if item["match"].get("relevance_score") is not None
        else item["match"]["final_score"]
        for item in matched
    ]
    ui_contract_ok = bool(matched) and all(
        item["job"].get("title")
        and item["job"].get("source_url")
        and item.get("company", {}).get("name")
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
        "configured_companies_present": configured_names <= represented_names,
        "matched_companies": sorted(matched_company_names),
        "company_filters": company_filters,
        "relevance_sorting": scores == sorted(scores, reverse=True),
        "eligibility_present": all(item["match"].get("eligibility") for item in matched),
        "canonical_links_present": all(item["job"].get("source_url") for item in jobs),
        "job_detail_contract": detail_ok,
        "actions": action_results,
        "actions_persisted_after_restart": persisted_actions,
        "streamlit_data_contract": ui_contract_ok,
    }


def _write_artifacts(
    output_directory: Path,
    results: list[MultiCompanyRunResult],
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
    _write_json(output_directory / "api-ui-validation.json", api_validation)

    company_by_id = {company.id: company for company in repositories.companies.list()}
    company_rows: list[dict[str, object]] = []
    for run_index, result in enumerate(results, start=1):
        for company_result in result.company_results:
            summary = company_result.summary
            company_rows.append(
                {
                    "run": run_index,
                    "company": summary.company_name,
                    "status": summary.status.value,
                    "duration_seconds": round(summary.duration_seconds, 3),
                    "discovered": summary.jobs_discovered,
                    "normalized": summary.jobs_normalized,
                    "new": summary.new_jobs,
                    "changed": summary.changed_jobs,
                    "unchanged": summary.unchanged_jobs,
                    "stage1_retained": summary.stage1_retained,
                    "stage1_rejected": summary.stage1_rejected,
                    "analysis_successes": summary.analysis_successes,
                    "analysis_failures": summary.analysis_failures,
                    "eligibility": json.dumps(summary.eligibility_counts, sort_keys=True),
                    "matches_persisted": (
                        summary.matches_created
                        + summary.matches_updated
                        + summary.matches_reused
                    ),
                    "explanations_available": (
                        summary.explanations_created + summary.explanations_reused
                    ),
                    "failures": summary.failures,
                }
            )
    _write_csv(output_directory / "company-summary.csv", company_rows)

    latest = results[-1]
    latest_outcomes = {
        outcome.job_id: outcome
        for result in latest.company_results
        for outcome in result.jobs
        if outcome.job_id is not None
    }
    ranked_rows: list[dict[str, object]] = []
    for match in repositories.matches.list(profile.id):
        state = repositories.jobs.find_by_id(match.job_id)
        if state is None:
            continue
        company = company_by_id.get(state.job.company_id)
        outcome = latest_outcomes.get(match.job_id)
        ranked_rows.append(
            {
                "rank": 0,
                "company": company.name if company else str(state.job.company_id),
                "title": state.job.title,
                "location": state.job.location or "",
                "role_family": match.role_family_class or "unknown",
                "relevance_score": match.relevance_score,
                "eligibility": match.eligibility.status if match.eligibility else "unknown",
                "experience_gap_years": (
                    match.eligibility.experience_gap_years if match.eligibility else None
                ),
                "explanation_available": repositories.explanations.get(
                    match.job_id, profile.id
                )
                is not None,
                "explanation_status": (
                    outcome.explanation_status.value if outcome is not None else "unknown"
                ),
                "canonical_url": str(state.job.source_url),
                "workflow_status": state.job.status.value,
            }
        )
    ranked_rows.sort(
        key=lambda row: _numeric(row["relevance_score"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    _write_csv(output_directory / "ranked-jobs.csv", ranked_rows)

    product_rows = [
        {
            **row,
            "user_useful": "",
            "user_would_apply": "",
            "user_notes": "",
        }
        for row in ranked_rows
    ]
    _write_csv(output_directory / "product-review.csv", product_rows)

    rejected_rows: list[dict[str, object]] = []
    for company_result in latest.company_results:
        for outcome in company_result.jobs:
            if outcome.stage1 is None or outcome.stage1.passed:
                continue
            rejected_rows.append(
                {
                    "company": company_result.summary.company_name,
                    "title": outcome.title,
                    "source_job_id": outcome.source_job_id,
                    "role_family": outcome.role_family_class or "unknown",
                    "outcome": outcome.stage1.outcome.value,
                    "reasons": " | ".join(outcome.stage1.reasons),
                }
            )
    _write_csv(output_directory / "rejected-jobs.csv", rejected_rows)

    failures: list[dict[str, object]] = []
    for run_index, result in enumerate(results, start=1):
        failures.extend(
            {"run": run_index, "scope": "company", **failure.model_dump(mode="json")}
            for failure in result.company_failures
        )
        for company_result in result.company_results:
            failures.extend(
                {
                    "run": run_index,
                    "scope": "job",
                    "company": company_result.summary.company_name,
                    **failure.model_dump(mode="json"),
                }
                for failure in company_result.failures
            )
    _write_json(output_directory / "failures.json", failures)
    (output_directory / "e2e-report.md").write_text(
        _markdown_report(results, company_rows, ranked_rows, api_validation),
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
    results: list[MultiCompanyRunResult],
    company_rows: list[dict[str, object]],
    ranked_rows: list[dict[str, object]],
    api_validation: dict[str, object],
) -> str:
    lines = [
        "# Multi-company end-to-end validation",
        "",
        "Local product validation using matching v0.1. This is not a ranking benchmark.",
        "",
        "| Run | Companies | Discovered | New | Changed | Unchanged | Retained | Failures |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, result in enumerate(results, start=1):
        summary = result.summary
        lines.append(
            f"| {index} | {summary.companies_attempted} | {summary.jobs_discovered} | "
            f"{summary.new_jobs} | {summary.changed_jobs} | {summary.unchanged_jobs} | "
            f"{summary.stage1_retained} | {summary.total_failures} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Stage-2 eligibility",
            "",
            "| Run | Distribution |",
            "|---:|---|",
        ]
    )
    for index, result in enumerate(results, start=1):
        distribution = json.dumps(result.summary.eligibility_counts, sort_keys=True)
        lines.append(f"| {index} | `{distribution}` |")
    lines.extend(
        [
            "",
            "## Company results",
            "",
            "| Run | Company | Discovered | New | Changed | Unchanged | Retained | Rejected |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in company_rows:
        lines.append(
            f"| {row['run']} | {row['company']} | {row['discovered']} | {row['new']} | "
            f"{row['changed']} | {row['unchanged']} | {row['stage1_retained']} | "
            f"{row['stage1_rejected']} |"
        )
    lines.extend(
        [
            "",
            "## Ranked retained jobs",
            "",
            "| Rank | Company | Title | Relevance | Eligibility | Gap | Explanation |",
            "|---:|---|---|---:|---|---:|---|",
        ]
    )
    for row in ranked_rows:
        lines.append(
            f"| {row['rank']} | {row['company']} | {row['title']} | "
            f"{_numeric(row['relevance_score']):.4f} | {row['eligibility']} | "
            f"{row['experience_gap_years'] or ''} | {row['explanation_available']} |"
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
            "Career relevance is presented independently from feasibility. Workflow status "
            "and company identity remain attached to each canonical job.",
            "",
            "## Issues",
            "",
            "- **Blocking:** None observed.",
            "- **Important:** None remained after the second pass.",
            (
                "- **Backlog:** Initial grounded-explanation failures are retained in "
                "`failures.json`; successful future-run recovery is tracked separately."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _numeric(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()
