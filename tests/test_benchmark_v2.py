from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import AnyHttpUrl

from job_intelligence.benchmark_v2 import (
    BenchmarkSource,
    BenchmarkSourcePlan,
    SelectionBucket,
    build_source_plan_jobs,
    classify_role_category,
    count_independent_jobs,
    duplicate_groups,
    selection_bucket,
)
from job_intelligence.connectors import JobReference
from job_intelligence.models import Job

COMPANY_A = UUID("11111111-1111-1111-1111-111111111111")
COMPANY_B = UUID("22222222-2222-2222-2222-222222222222")


def test_selection_uses_title_strata_and_round_robins_companies() -> None:
    sources = [_source("A", COMPANY_A), _source("B", COMPANY_B)]
    listings = {
        COMPANY_A: [
            _reference(COMPANY_A, "1", "Machine Learning Engineer"),
            _reference(COMPANY_A, "2", "Data Analyst"),
            _reference(COMPANY_A, "3", "Security Engineer"),
            _reference(COMPANY_A, "7", "Senior Machine Learning Engineer"),
        ],
        COMPANY_B: [
            _reference(COMPANY_B, "4", "Data Scientist"),
            _reference(COMPANY_B, "5", "Decision Scientist"),
            _reference(COMPANY_B, "6", "QA Engineer"),
        ],
    }
    composition = {
        SelectionBucket.TARGET_REGULAR: 2,
        SelectionBucket.TARGET_STRETCH: 1,
        SelectionBucket.ADJACENT: 2,
        SelectionBucket.IRRELEVANT: 2,
    }

    jobs = build_source_plan_jobs(
        sources,
        listings,
        target_composition=composition,
        max_jobs_per_company=4,
    )
    plan = BenchmarkSourcePlan(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        target_size=7,
        max_jobs_per_company=4,
        target_composition=composition,
        sources=sources,
        jobs=jobs,
    )

    assert len(plan.jobs) == 7
    assert sum(job.company_id == COMPANY_A for job in jobs) == 4
    assert sum(job.company_id == COMPANY_B for job in jobs) == 3


def test_interest_form_is_not_selected_as_a_target_job() -> None:
    assert selection_bucket("Expression of Interest: Machine Learning Engineer") == (
        SelectionBucket.IRRELEVANT
    )
    assert selection_bucket("Senior Machine Learning Engineer") == (
        SelectionBucket.TARGET_STRETCH
    )
    assert selection_bucket("Machine Learning Engineer") == SelectionBucket.TARGET_REGULAR


def test_reporting_categories_capture_target_adjacent_and_hard_negative_titles() -> None:
    assert classify_role_category("Generative AI Engineer") == "genai_llm"
    assert classify_role_category("Applied Scientist II") == "applied_scientist"
    assert classify_role_category("Product Manager, AI Infrastructure") == (
        "analytics_data_adjacent"
    )
    assert classify_role_category("Sales Engineer, AI Solutions") == (
        "ambiguous_hard_negative"
    )


def test_duplicate_grouping_is_company_scoped_and_description_based() -> None:
    jobs = [
        _job(COMPANY_A, "1", "Data Scientist", "Build ML models with Python and SQL."),
        _job(COMPANY_A, "2", "Data Scientist", "Build ML models with Python and SQL!"),
        _job(COMPANY_B, "3", "Data Scientist", "Build ML models with Python and SQL."),
        _job(COMPANY_A, "4", "Data Scientist", "Manage enterprise payroll operations."),
    ]

    groups = duplicate_groups(jobs)

    assert groups[jobs[0].id] is not None
    assert groups[jobs[0].id] == groups[jobs[1].id]
    assert groups[jobs[2].id] is None
    assert groups[jobs[3].id] is None
    assert count_independent_jobs(groups.values()) == 3


def _source(name: str, company_id: UUID) -> BenchmarkSource:
    return BenchmarkSource(
        company_name=name,
        board_token=name.casefold(),
        career_url=AnyHttpUrl(f"https://job-boards.greenhouse.io/{name.casefold()}"),
        company_id=company_id,
    )


def _reference(company_id: UUID, source_id: str, title: str) -> JobReference:
    return JobReference(
        company_id=company_id,
        source_job_id=source_id,
        title=title,
        location="Remote",
        source_url=AnyHttpUrl(f"https://example.com/jobs/{source_id}"),
        details_url=AnyHttpUrl(f"https://api.example.com/jobs/{source_id}"),
    )


def _job(company_id: UUID, source_id: str, title: str, description: str) -> Job:
    return Job(
        id=UUID(int=int(source_id)),
        source_job_id=source_id,
        company_id=company_id,
        title=title,
        description=description,
        source_url=AnyHttpUrl(f"https://example.com/jobs/{source_id}"),
    )
