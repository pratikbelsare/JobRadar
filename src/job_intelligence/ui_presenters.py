"""Pure presentation helpers for the Streamlit client."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Literal

JobPayload = dict[str, Any]
JobViewName = Literal[
    "Recommended",
    "All matches",
    "Shortlisted",
    "Applied",
    "Dismissed",
]
JobSortName = Literal[
    "Career relevance",
    "Newest discovered",
    "Eligibility",
    "Company",
    "Job title",
]

VIEW_OPTIONS: tuple[JobViewName, ...] = (
    "Recommended",
    "All matches",
    "Shortlisted",
    "Applied",
    "Dismissed",
)
SORT_OPTIONS: tuple[JobSortName, ...] = (
    "Career relevance",
    "Newest discovered",
    "Eligibility",
    "Company",
    "Job title",
)

_ELIGIBILITY_ORDER = {
    "eligible": 0,
    "stretch": 1,
    "unknown": 2,
    "severe_stretch": 3,
    "ineligible": 4,
}
_VIEW_STATUS = {
    "Shortlisted": "shortlisted",
    "Applied": "applied",
    "Dismissed": "dismissed",
}


def primary_score(item: JobPayload) -> float:
    match = item.get("match") or {}
    value = match.get("relevance_score")
    if value is None:
        value = match.get("final_score", 0.0)
    return float(value or 0.0)


def company_name(item: JobPayload) -> str:
    company = item.get("company") or {}
    return str(company.get("name") or "Unknown company")


def eligibility_status(item: JobPayload) -> str:
    match = item.get("match") or {}
    eligibility = match.get("eligibility") or {}
    return str(eligibility.get("status") or "unknown")


def role_family(item: JobPayload) -> str:
    match = item.get("match") or {}
    return str(match.get("role_family_class") or "unknown")


def filter_jobs(
    jobs: Iterable[JobPayload],
    *,
    view: JobViewName,
    companies: set[str] | None = None,
    role_families: set[str] | None = None,
    eligibility: set[str] | None = None,
    statuses: set[str] | None = None,
    location_query: str = "",
    search_query: str = "",
) -> list[JobPayload]:
    """Apply product-facing filters without changing backend ranking semantics."""

    filtered: list[JobPayload] = []
    search = search_query.strip().casefold()
    location = location_query.strip().casefold()
    for item in jobs:
        match = item.get("match")
        if match is None:
            continue
        job = item.get("job") or {}
        status = str(job.get("status") or "new")
        if view == "Recommended" and status in {"applied", "dismissed"}:
            continue
        expected_status = _VIEW_STATUS.get(view)
        if expected_status is not None and status != expected_status:
            continue
        if companies and company_name(item) not in companies:
            continue
        if role_families and role_family(item) not in role_families:
            continue
        if eligibility and eligibility_status(item) not in eligibility:
            continue
        if statuses and status not in statuses:
            continue
        job_location = str(job.get("location") or "")
        if location and location not in job_location.casefold():
            continue
        if search:
            haystack = " ".join(
                (
                    str(job.get("title") or ""),
                    company_name(item),
                    job_location,
                )
            ).casefold()
            if search not in haystack:
                continue
        filtered.append(item)
    return filtered


def sort_jobs(jobs: Iterable[JobPayload], sort_by: JobSortName) -> list[JobPayload]:
    items = list(jobs)
    if sort_by == "Career relevance":
        return sorted(items, key=primary_score, reverse=True)
    if sort_by == "Newest discovered":
        return sorted(items, key=_discovered_at, reverse=True)
    if sort_by == "Eligibility":
        return sorted(
            items,
            key=lambda item: (
                _ELIGIBILITY_ORDER.get(eligibility_status(item), 99),
                -primary_score(item),
            ),
        )
    if sort_by == "Company":
        return sorted(
            items,
            key=lambda item: (
                company_name(item).casefold(),
                -primary_score(item),
            ),
        )
    return sorted(
        items,
        key=lambda item: (
            str((item.get("job") or {}).get("title") or "").casefold(),
            company_name(item).casefold(),
        ),
    )


def dashboard_summary(jobs: Iterable[JobPayload]) -> dict[str, int]:
    items = [item for item in jobs if item.get("match") is not None]
    statuses = [str((item.get("job") or {}).get("status") or "new") for item in items]
    return {
        "matching_jobs": len(items),
        "new_jobs": statuses.count("new"),
        "shortlisted": statuses.count("shortlisted"),
        "applied": statuses.count("applied"),
        "companies": len({company_name(item) for item in items}),
    }


def table_rows(jobs: Iterable[JobPayload]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in jobs:
        job = item.get("job") or {}
        gap = ((item.get("match") or {}).get("eligibility") or {}).get(
            "experience_gap_years"
        )
        rows.append(
            {
                "job_id": str(job.get("id") or ""),
                "Job": str(job.get("title") or "Untitled role"),
                "Company": company_name(item),
                "Location": str(job.get("location") or "Not specified"),
                "Relevance": f"{primary_score(item):.0%}",
                "Eligibility": humanize(eligibility_status(item)),
                "Gap": f"+{float(gap):.1f} yrs" if gap is not None and float(gap) > 0 else "—",
                "Family": role_family_label(role_family(item)),
                "Status": humanize(str(job.get("status") or "new")),
            }
        )
    return rows


def humanize(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Unknown"


def role_family_label(value: str) -> str:
    labels = {
        "core": "Core role",
        "adjacent": "Adjacent role",
        "exclude": "Excluded",
        "unknown": "Unknown",
    }
    return labels.get(value, humanize(value))


def _discovered_at(item: JobPayload) -> datetime:
    job = item.get("job") or {}
    raw = job.get("first_seen_at") or job.get("posted_at") or ""
    if not isinstance(raw, str) or not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


__all__ = [
    "JobPayload",
    "JobSortName",
    "JobViewName",
    "SORT_OPTIONS",
    "VIEW_OPTIONS",
    "company_name",
    "dashboard_summary",
    "eligibility_status",
    "filter_jobs",
    "humanize",
    "primary_score",
    "role_family",
    "role_family_label",
    "sort_jobs",
    "table_rows",
]
