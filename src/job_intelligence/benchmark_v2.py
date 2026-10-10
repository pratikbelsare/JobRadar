"""Evaluation-only planning and duplicate grouping for frozen Benchmark v2."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Iterable, Sequence
from uuid import UUID

from pydantic import AnyHttpUrl, Field, PositiveInt, model_validator

from .connectors.base import JobReference
from .models import Job, Model, NonEmptyText
from .normalization import normalize_text


class SelectionBucket(str, Enum):
    TARGET_REGULAR = "target_regular"
    TARGET_STRETCH = "target_stretch"
    ADJACENT = "adjacent"
    IRRELEVANT = "irrelevant"


class BenchmarkSource(Model):
    company_name: NonEmptyText
    board_token: NonEmptyText
    career_url: AnyHttpUrl
    company_id: UUID


class PlannedJob(Model):
    company_name: NonEmptyText
    company_id: UUID
    board_token: NonEmptyText
    source_job_id: NonEmptyText
    title: NonEmptyText
    location: NonEmptyText | None = None
    canonical_url: AnyHttpUrl
    details_url: AnyHttpUrl
    role_category: NonEmptyText
    selection_bucket: SelectionBucket


class BenchmarkSourcePlan(Model):
    schema_version: str = "2"
    benchmark_version: NonEmptyText = "v2"
    created_at: datetime
    target_size: PositiveInt
    max_jobs_per_company: PositiveInt
    target_composition: dict[SelectionBucket, PositiveInt]
    sources: list[BenchmarkSource] = Field(min_length=2)
    jobs: list[PlannedJob] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> BenchmarkSourcePlan:
        if len(self.jobs) != self.target_size:
            raise ValueError("source plan job count must equal target_size")
        identities = [(job.company_id, job.source_job_id) for job in self.jobs]
        if len(identities) != len(set(identities)):
            raise ValueError("source plan contains duplicate source identities")
        counts: dict[SelectionBucket, int] = defaultdict(int)
        companies: dict[UUID, int] = defaultdict(int)
        for job in self.jobs:
            counts[job.selection_bucket] += 1
            companies[job.company_id] += 1
        if dict(counts) != self.target_composition:
            raise ValueError("source plan does not match target composition")
        if any(count > self.max_jobs_per_company for count in companies.values()):
            raise ValueError("source plan exceeds per-company cap")
        if set(companies) != {source.company_id for source in self.sources}:
            raise ValueError("every configured source must contribute jobs")
        return self


def classify_role_category(title: str) -> str:
    value = _key(title)
    if re.search(r"\b(?:product manager|product management)\b", value) and re.search(
        r"\b(?:ai|ml|data|genai|generative)\b", value
    ):
        return "analytics_data_adjacent"
    if re.search(r"\bai transformation\b", value):
        return "analytics_data_adjacent"
    if re.search(r"\b(?:generative ai|genai|llm)\b", value):
        return "genai_llm"
    if re.search(r"\b(?:applied|research) scientist\b", value):
        return "applied_scientist"
    if re.search(r"\b(?:data scientist|data science)\b", value):
        return "data_scientist"
    if re.search(r"\b(?:machine learning|ml)\b", value):
        return "ml_engineer"
    if re.search(r"\b(?:applied ai|ai engineer|ai applied scientist|sde[^,]* ai)\b", value):
        return "ai_engineer"
    if re.search(
        r"\b(?:data analyst|analytics|insights?|research analyst|decision scientist|"
        r"lead analyst|data engineer|data platform|data quality|data qa|ai product|"
        r"product manager[^,]*ai|"
        r"ai transformation|mlops)\b",
        value,
    ):
        return "analytics_data_adjacent"
    if re.search(r"\b(?:data|ai|analytics|engineer)\b", value):
        return "ambiguous_hard_negative"
    return "irrelevant"


def selection_bucket(title: str) -> SelectionBucket:
    if re.search(r"\b(?:expression of interest|submit your interest)\b", _key(title)):
        return SelectionBucket.IRRELEVANT
    category = classify_role_category(title)
    if category in {
        "ml_engineer",
        "ai_engineer",
        "data_scientist",
        "applied_scientist",
        "genai_llm",
    }:
        return (
            SelectionBucket.TARGET_STRETCH
            if _senior_title(title)
            else SelectionBucket.TARGET_REGULAR
        )
    if category == "analytics_data_adjacent":
        return SelectionBucket.ADJACENT
    return SelectionBucket.IRRELEVANT


def build_source_plan_jobs(
    sources: Sequence[BenchmarkSource],
    listings: Mapping[UUID, Sequence[JobReference]],
    *,
    target_composition: dict[SelectionBucket, int],
    max_jobs_per_company: int,
) -> list[PlannedJob]:
    """Select without using ranking scores, round-robin across official sources."""

    candidates: dict[SelectionBucket, dict[UUID, list[PlannedJob]]] = {
        bucket: defaultdict(list) for bucket in SelectionBucket
    }
    source_by_id = {source.company_id: source for source in sources}
    for company_id, references in listings.items():
        source = source_by_id[company_id]
        seen_titles: set[str] = set()
        for reference in sorted(
            references,
            key=lambda item: (
                _selection_priority(item.title),
                _key(item.title),
                item.source_job_id,
            ),
        ):
            title_key = _key(reference.title)
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            bucket = selection_bucket(reference.title)
            candidates[bucket][company_id].append(
                PlannedJob(
                    company_name=source.company_name,
                    company_id=company_id,
                    board_token=source.board_token,
                    source_job_id=reference.source_job_id,
                    title=reference.title,
                    location=reference.location,
                    canonical_url=reference.source_url,
                    details_url=reference.details_url,
                    role_category=classify_role_category(reference.title),
                    selection_bucket=bucket,
                )
            )

    selected: list[PlannedJob] = []
    company_counts: dict[UUID, int] = defaultdict(int)
    company_order = [source.company_id for source in sources]
    for bucket in SelectionBucket:
        needed = target_composition[bucket]
        while needed:
            progress = False
            for company_id in company_order:
                if needed == 0:
                    break
                available = candidates[bucket][company_id]
                if not available or company_counts[company_id] >= max_jobs_per_company:
                    continue
                selected.append(available.pop(0))
                company_counts[company_id] += 1
                needed -= 1
                progress = True
            if not progress:
                raise ValueError(f"not enough diverse jobs for selection bucket {bucket.value}")
    return selected


def duplicate_groups(jobs: Sequence[Job]) -> dict[UUID, str | None]:
    """Group same-company, same-title descriptions with at least 90% token overlap."""

    parents = {job.id: job.id for job in jobs}

    def find(job_id: UUID) -> UUID:
        while parents[job_id] != job_id:
            parents[job_id] = parents[parents[job_id]]
            job_id = parents[job_id]
        return job_id

    def union(left: UUID, right: UUID) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root, key=str)] = min(left_root, right_root, key=str)

    by_company_title: dict[tuple[UUID, str], list[Job]] = defaultdict(list)
    for job in jobs:
        by_company_title[(job.company_id, _key(job.title))].append(job)
    for candidates in by_company_title.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if _description_similarity(left.description, right.description) >= 0.9:
                    union(left.id, right.id)

    grouped: dict[UUID, list[UUID]] = defaultdict(list)
    for job in jobs:
        grouped[find(job.id)].append(job.id)
    result: dict[UUID, str | None] = {job.id: None for job in jobs}
    for members in grouped.values():
        if len(members) < 2:
            continue
        serialized = "|".join(sorted(str(member) for member in members))
        group_id = "dup-" + hashlib.sha256(serialized.encode()).hexdigest()[:12]
        for member in members:
            result[member] = group_id
    return result


def count_independent_jobs(group_ids: Iterable[str | None]) -> int:
    values = list(group_ids)
    duplicates = {value for value in values if value is not None}
    singles = sum(value is None for value in values)
    return singles + len(duplicates)


def _senior_title(title: str) -> bool:
    return re.search(
        r"\b(?:senior|sr\.?|staff|principal|lead|manager|director|head|vp|iii|iv)\b",
        _key(title),
    ) is not None


def _selection_priority(title: str) -> tuple[int, int]:
    value = _key(title)
    hard_edge = int(not bool(re.search(r"\b(?:data|ai|ml|analytics|engineer)\b", value)))
    return hard_edge, len(value)


def _description_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _key(value: str) -> str:
    return normalize_text(value).casefold()


__all__ = [
    "BenchmarkSource",
    "BenchmarkSourcePlan",
    "PlannedJob",
    "SelectionBucket",
    "build_source_plan_jobs",
    "classify_role_category",
    "count_independent_jobs",
    "duplicate_groups",
    "selection_bucket",
]
