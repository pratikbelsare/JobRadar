from __future__ import annotations

from typing import Any

from job_intelligence.ui_presenters import (
    dashboard_summary,
    filter_jobs,
    sort_jobs,
    table_rows,
)


def _job(
    title: str,
    company: str,
    *,
    score: float,
    eligibility: str = "eligible",
    family: str = "core",
    status: str = "new",
    location: str = "London",
    first_seen: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "job": {
            "id": f"{company}-{title}",
            "title": title,
            "location": location,
            "status": status,
            "first_seen_at": first_seen,
        },
        "company": {"name": company},
        "match": {
            "relevance_score": score,
            "role_family_class": family,
            "eligibility": {
                "status": eligibility,
                "experience_gap_years": 2.0 if eligibility == "stretch" else 0.0,
            },
        },
    }


def test_recommended_view_excludes_applied_and_dismissed_jobs() -> None:
    jobs = [
        _job("Data Scientist", "A", score=0.8),
        _job("ML Engineer", "B", score=0.7, status="shortlisted"),
        _job("Applied Scientist", "C", score=0.6, status="applied"),
        _job("Data Analyst", "D", score=0.5, status="dismissed"),
    ]

    recommended = filter_jobs(jobs, view="Recommended")
    all_matches = filter_jobs(jobs, view="All matches")

    assert [item["job"]["title"] for item in recommended] == [
        "Data Scientist",
        "ML Engineer",
    ]
    assert len(all_matches) == 4


def test_filters_combine_company_family_eligibility_status_location_and_search() -> None:
    jobs = [
        _job(
            "Data Scientist",
            "PhysicsX",
            score=0.8,
            status="shortlisted",
            location="London, United Kingdom",
        ),
        _job(
            "Data Scientist",
            "Moloco",
            score=0.7,
            eligibility="stretch",
            location="New York",
        ),
    ]

    result = filter_jobs(
        jobs,
        view="All matches",
        companies={"PhysicsX"},
        role_families={"core"},
        eligibility={"eligible"},
        statuses={"shortlisted"},
        location_query="london",
        search_query="physics",
    )

    assert len(result) == 1
    assert result[0]["company"]["name"] == "PhysicsX"


def test_sort_options_are_deterministic() -> None:
    jobs = [
        _job(
            "B role",
            "Zulu",
            score=0.6,
            eligibility="stretch",
            first_seen="2026-01-02T00:00:00Z",
        ),
        _job(
            "A role",
            "Alpha",
            score=0.9,
            eligibility="severe_stretch",
            first_seen="2026-01-01T00:00:00Z",
        ),
        _job(
            "C role",
            "Beta",
            score=0.7,
            eligibility="eligible",
            first_seen="2026-01-03T00:00:00Z",
        ),
    ]

    assert [item["job"]["title"] for item in sort_jobs(jobs, "Career relevance")] == [
        "A role",
        "C role",
        "B role",
    ]
    assert [item["job"]["title"] for item in sort_jobs(jobs, "Newest discovered")] == [
        "C role",
        "B role",
        "A role",
    ]
    assert [item["job"]["title"] for item in sort_jobs(jobs, "Eligibility")] == [
        "C role",
        "B role",
        "A role",
    ]
    assert [item["company"]["name"] for item in sort_jobs(jobs, "Company")] == [
        "Alpha",
        "Beta",
        "Zulu",
    ]


def test_dashboard_and_table_rows_use_product_facing_values() -> None:
    jobs = [
        _job("Data Scientist", "A", score=0.82),
        _job(
            "ML Engineer",
            "B",
            score=0.7,
            eligibility="stretch",
            status="shortlisted",
        ),
        _job("Applied Scientist", "B", score=0.65, status="applied"),
    ]

    summary = dashboard_summary(jobs)
    rows = table_rows(jobs)

    assert summary == {
        "matching_jobs": 3,
        "new_jobs": 1,
        "shortlisted": 1,
        "applied": 1,
        "companies": 2,
    }
    assert rows[0]["Relevance"] == "82%"
    assert rows[0]["Family"] == "Core role"
    assert rows[1]["Eligibility"] == "Stretch"
    assert rows[1]["Gap"] == "+2.0 yrs"
    assert rows[1]["Status"] == "Shortlisted"
