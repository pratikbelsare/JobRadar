"""Frozen, reproducible ranking benchmark support.

This module contains no AWS or LLM integration. A benchmark can therefore be loaded
and evaluated from local files without queue, database, or extraction-provider access.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence
from uuid import UUID

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from .ai import TextEmbeddingProvider
from .eligibility import EligibilityConfig, EligibilityEvaluator, EligibilityResult
from .evaluation import ndcg_at_k, precision_at_k
from .filtering import DeterministicJobFilter, FilteringConfig, FilterResult
from .matching import HybridJobMatcher, MatchingConfig
from .models import CandidateProfile, Job, JobAnalysis, Model, NonEmptyText, Score
from .role_policy import RoleFamilyClassifier, RoleFamilyMatch

IMMUTABLE_BENCHMARK_FILES = (
    "jobs.json",
    "job_analyses.json",
    "candidate_profile.json",
)
V2_IMMUTABLE_BENCHMARK_FILES = (
    *IMMUTABLE_BENCHMARK_FILES,
    "source_plan.json",
    "stage1_results.json",
    "stage2_results.json",
)
LABEL_NAMES = {
    0: "irrelevant",
    1: "adjacent",
    2: "relevant_stretch",
    3: "strong_target",
}


class FrozenJob(Model):
    """Canonical job and the exact historical version used for extraction."""

    job: Job
    job_version_id: UUID
    category: NonEmptyText
    role_family: RoleFamilyMatch
    stage1_filter: FilterResult
    eligibility: EligibilityResult
    company_name: NonEmptyText | None = None
    source_updated_at: datetime | None = None
    role_category: NonEmptyText | None = None
    selection_bucket: NonEmptyText | None = None
    duplicate_group_id: NonEmptyText | None = None
    raw_description: NonEmptyText | None = None


class FrozenJobSet(Model):
    jobs: list[FrozenJob] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_jobs(self) -> FrozenJobSet:
        identifiers = [item.job.id for item in self.jobs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("frozen benchmark contains duplicate job IDs")
        return self


class FrozenAnalysisSet(Model):
    analyses: list[JobAnalysis] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_analyses(self) -> FrozenAnalysisSet:
        identifiers = [item.job_id for item in self.analyses]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("frozen benchmark contains duplicate job analyses")
        return self


class BenchmarkManifest(Model):
    """Immutable benchmark metadata and exact business-logic configuration."""

    schema_version: Literal["1", "2"] = "1"
    benchmark_version: NonEmptyText
    created_at: datetime
    company_name: NonEmptyText
    job_count: PositiveInt
    profile_version: NonEmptyText
    profile_sha256: NonEmptyText
    file_sha256: dict[str, NonEmptyText]
    analysis_provider: NonEmptyText
    analysis_model_id: NonEmptyText
    analysis_prompt_version: NonEmptyText
    analyses_reused_local: NonNegativeInt = 0
    analyses_reused_repository: NonNegativeInt = 0
    analyses_extracted: NonNegativeInt = 0
    embedding_provider: NonEmptyText
    embedding_model_id: NonEmptyText
    embedding_device: NonEmptyText
    matching_config: MatchingConfig
    filtering_config: FilteringConfig
    eligibility_config: EligibilityConfig
    companies: list[NonEmptyText] = Field(default_factory=list)
    independent_job_count: PositiveInt | None = None
    duplicate_group_count: NonNegativeInt = 0
    semantic_fingerprint: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_counts_and_hashes(self) -> BenchmarkManifest:
        expected_files = (
            V2_IMMUTABLE_BENCHMARK_FILES
            if self.schema_version == "2"
            else IMMUTABLE_BENCHMARK_FILES
        )
        if set(self.file_sha256) != set(expected_files):
            raise ValueError("manifest must hash every immutable benchmark file")
        source_total = (
            self.analyses_reused_local + self.analyses_reused_repository + self.analyses_extracted
        )
        if source_total != self.job_count:
            raise ValueError("analysis source counts must equal job_count")
        if self.file_sha256["candidate_profile.json"] != self.profile_sha256:
            raise ValueError("profile hash must match candidate_profile.json hash")
        if self.schema_version == "2":
            if not self.companies:
                raise ValueError("v2 manifest must list represented companies")
            if self.independent_job_count is None:
                raise ValueError("v2 manifest must record independent_job_count")
            expected_fingerprint = semantic_benchmark_fingerprint(self.file_sha256)
            if self.semantic_fingerprint != expected_fingerprint:
                raise ValueError("v2 semantic fingerprint does not match immutable files")
        return self


class FrozenStage1Result(Model):
    job_id: UUID
    role_family: RoleFamilyMatch
    passed: bool
    outcome: NonEmptyText
    reason_codes: list[NonEmptyText] = Field(default_factory=list)
    result: FilterResult


class FrozenStage1Set(Model):
    results: list[FrozenStage1Result] = Field(min_length=1)


class FrozenStage2Result(Model):
    job_id: UUID
    result: EligibilityResult


class FrozenStage2Set(Model):
    results: list[FrozenStage2Result] = Field(min_length=1)


@dataclass(frozen=True)
class FrozenBenchmark:
    directory: Path
    manifest: BenchmarkManifest
    jobs: tuple[FrozenJob, ...]
    analyses: dict[UUID, JobAnalysis]
    profile: CandidateProfile


class BenchmarkIntegrityError(ValueError):
    """Raised when frozen input files no longer match their manifest."""


class BenchmarkRow(Model):
    job_id: UUID
    title: NonEmptyText
    category: NonEmptyText
    role_family: NonEmptyText
    stage1_passed: bool
    stage1_outcome: NonEmptyText
    eligibility_status: NonEmptyText
    experience_gap_years: float | None = Field(default=None, ge=0)
    experience_status: NonEmptyText
    seniority_status: NonEmptyText
    relevance_score: Score
    relevance_rank: PositiveInt | None = None
    legacy_blended_score: Score
    legacy_rank: PositiveInt | None = None
    role_score: Score
    skill_score: Score
    responsibility_score: Score
    domain_score: Score
    location_score: Score
    relevance_included_components: list[NonEmptyText] = Field(default_factory=list)
    relevance_excluded_components: list[NonEmptyText] = Field(default_factory=list)
    company_name: NonEmptyText | None = None
    duplicate_group_id: NonEmptyText | None = None
    selection_bucket: NonEmptyText | None = None


class BenchmarkMetrics(Model):
    subset: NonEmptyText
    label_source: NonEmptyText
    evaluated_count: NonNegativeInt
    relevant_count: NonNegativeInt
    precision_at_5: Score
    precision_at_10: Score
    recall_at_10: Score
    ndcg_at_5: Score
    ndcg_at_10: Score


class BenchmarkLabel(Model):
    job_id: UUID
    provisional_label: int | None = Field(default=None, ge=0, le=3)
    human_label: int | None = Field(default=None, ge=0, le=3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def semantic_benchmark_fingerprint(file_hashes: dict[str, str]) -> str:
    serialized = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_frozen_benchmark(directory: Path) -> FrozenBenchmark:
    manifest_path = directory / "manifest.json"
    try:
        manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkIntegrityError(f"could not read {manifest_path}") from exc
    except ValueError as exc:
        raise BenchmarkIntegrityError("benchmark manifest is invalid") from exc

    immutable_files = (
        V2_IMMUTABLE_BENCHMARK_FILES
        if manifest.schema_version == "2"
        else IMMUTABLE_BENCHMARK_FILES
    )
    for name in immutable_files:
        path = directory / name
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise BenchmarkIntegrityError(f"could not read immutable file {name}") from exc
        expected = manifest.file_sha256[name]
        if not _secure_equal(actual, expected):
            raise BenchmarkIntegrityError(f"immutable file hash mismatch: {name}")

    try:
        jobs = FrozenJobSet.model_validate_json(
            (directory / "jobs.json").read_text(encoding="utf-8")
        )
        analysis_set = FrozenAnalysisSet.model_validate_json(
            (directory / "job_analyses.json").read_text(encoding="utf-8")
        )
        profile = CandidateProfile.model_validate_json(
            (directory / "candidate_profile.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise BenchmarkIntegrityError("frozen benchmark data is invalid") from exc

    analyses = {analysis.job_id: analysis for analysis in analysis_set.analyses}
    if len(jobs.jobs) != manifest.job_count:
        raise BenchmarkIntegrityError("manifest job_count does not match jobs.json")
    if set(analyses) != {item.job.id for item in jobs.jobs}:
        raise BenchmarkIntegrityError("jobs and analyses do not contain identical job IDs")
    for item in jobs.jobs:
        analysis = analyses[item.job.id]
        if analysis.job_version_id != item.job_version_id:
            raise BenchmarkIntegrityError(
                f"analysis version does not match frozen job: {item.job.id}"
            )
    if manifest.schema_version == "2":
        _validate_v2_snapshots(directory, jobs)

    return FrozenBenchmark(
        directory=directory,
        manifest=manifest,
        jobs=tuple(jobs.jobs),
        analyses=analyses,
        profile=profile,
    )


def evaluate_frozen_benchmark(
    benchmark: FrozenBenchmark,
    embedding_provider: TextEmbeddingProvider,
    *,
    verify_frozen_stages: bool = True,
) -> list[BenchmarkRow]:
    """Run only existing deterministic policy, eligibility, and matching code."""

    manifest = benchmark.manifest
    role_policy = manifest.filtering_config.role_policy
    if role_policy is None:
        raise BenchmarkIntegrityError("frozen filtering configuration has no role policy")
    job_filter = DeterministicJobFilter(manifest.filtering_config)
    classifier = RoleFamilyClassifier(role_policy)
    eligibility = EligibilityEvaluator(manifest.eligibility_config)
    matcher = HybridJobMatcher(
        manifest.matching_config,
        embedding_provider=embedding_provider,
    )
    rows: list[BenchmarkRow] = []
    for item in benchmark.jobs:
        job = item.job
        analysis = benchmark.analyses[job.id]
        family = classifier.classify(job.title)
        stage1 = job_filter.evaluate(job, benchmark.profile)
        stage2 = eligibility.evaluate(
            benchmark.profile,
            job,
            analysis,
            role_family_class=family.classification,
        )
        if verify_frozen_stages and family != item.role_family:
            raise BenchmarkIntegrityError(
                f"Stage-1 role-family output changed for frozen job: {job.id}"
            )
        if verify_frozen_stages and stage1 != item.stage1_filter:
            raise BenchmarkIntegrityError(
                f"Stage-1 filter output changed for frozen job: {job.id}"
            )
        if verify_frozen_stages and stage2 != item.eligibility:
            raise BenchmarkIntegrityError(
                f"Stage-2 eligibility output changed for frozen job: {job.id}"
            )
        match = matcher.match(job, benchmark.profile, analysis=analysis)
        if match.relevance_score is None or match.relevance_evidence is None:
            raise BenchmarkIntegrityError("matcher did not return relevance evidence")
        rows.append(
            BenchmarkRow(
                job_id=job.id,
                title=job.title,
                category=item.category,
                role_family=(
                    family.classification.value if family.classification is not None else "unknown"
                ),
                stage1_passed=stage1.passed,
                stage1_outcome=stage1.outcome.value,
                eligibility_status=stage2.status.value,
                experience_gap_years=stage2.experience_gap_years,
                experience_status=stage2.experience_status.value,
                seniority_status=stage2.seniority_status.value,
                relevance_score=match.relevance_score,
                legacy_blended_score=match.final_score,
                role_score=match.role_match,
                skill_score=match.skill_match,
                responsibility_score=match.responsibility_match,
                domain_score=match.domain_match,
                location_score=match.location_match,
                relevance_included_components=match.relevance_evidence.included_components,
                relevance_excluded_components=match.relevance_evidence.excluded_components,
                company_name=item.company_name,
                duplicate_group_id=item.duplicate_group_id,
                selection_bucket=item.selection_bucket,
            )
        )
    return _add_ranks(rows)


def load_benchmark_labels(path: Path) -> list[BenchmarkLabel]:
    labels: list[BenchmarkLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for raw in csv.DictReader(file):
            human = (raw.get("human_label") or "").strip()
            provisional = (raw.get("provisional_label") or "").strip()
            labels.append(
                BenchmarkLabel(
                    job_id=UUID(raw["job_id"]),
                    provisional_label=int(provisional) if provisional else None,
                    human_label=int(human) if human else None,
                )
            )
    if len(labels) != len({label.job_id for label in labels}):
        raise ValueError("label file contains duplicate job IDs")
    return labels


def calculate_benchmark_metrics(
    rows: Sequence[BenchmarkRow],
    labels: Sequence[BenchmarkLabel],
    *,
    label_source: Literal["provisional", "human", "auto"] = "auto",
    subset: str = "all",
    relevance_threshold: int = 2,
) -> BenchmarkMetrics:
    selected_rows = [row for row in rows if _in_subset(row, subset)]
    labels_by_id = {label.job_id: label for label in labels}
    graded: list[int] = []
    effective_source = label_source
    for row in selected_rows:
        label = labels_by_id.get(row.job_id)
        if label is None:
            continue
        value = _label_value(label, label_source)
        if value is not None:
            graded.append(value)
    relevant_count = sum(value >= relevance_threshold for value in graded)
    recall = (
        sum(value >= relevance_threshold for value in graded[:10]) / relevant_count
        if relevant_count
        else 0.0
    )
    return BenchmarkMetrics(
        subset=subset,
        label_source=effective_source,
        evaluated_count=len(graded),
        relevant_count=relevant_count,
        precision_at_5=precision_at_k(graded, 5, relevance_threshold=relevance_threshold),
        precision_at_10=precision_at_k(graded, 10, relevance_threshold=relevance_threshold),
        recall_at_10=recall,
        ndcg_at_5=ndcg_at_k(graded, 5, relevance_threshold=relevance_threshold),
        ndcg_at_10=ndcg_at_k(graded, 10, relevance_threshold=relevance_threshold),
    )


def benchmark_fingerprint(rows: Sequence[BenchmarkRow]) -> str:
    payload = [row.model_dump(mode="json") for row in rows]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_v2_snapshots(directory: Path, jobs: FrozenJobSet) -> None:
    try:
        stage1 = FrozenStage1Set.model_validate_json(
            (directory / "stage1_results.json").read_text(encoding="utf-8")
        )
        stage2 = FrozenStage2Set.model_validate_json(
            (directory / "stage2_results.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise BenchmarkIntegrityError("v2 Stage-1 or Stage-2 snapshot is invalid") from exc
    stage1_by_id = {item.job_id: item for item in stage1.results}
    stage2_by_id = {item.job_id: item for item in stage2.results}
    expected_ids = {item.job.id for item in jobs.jobs}
    if set(stage1_by_id) != expected_ids or set(stage2_by_id) != expected_ids:
        raise BenchmarkIntegrityError("v2 snapshots do not match frozen job IDs")
    for item in jobs.jobs:
        first = stage1_by_id[item.job.id]
        second = stage2_by_id[item.job.id]
        if first.role_family != item.role_family or first.result != item.stage1_filter:
            raise BenchmarkIntegrityError(f"v2 Stage-1 snapshot mismatch: {item.job.id}")
        if second.result != item.eligibility:
            raise BenchmarkIntegrityError(f"v2 Stage-2 snapshot mismatch: {item.job.id}")


def categorize_job(title: str) -> str:
    """Coarse reporting category; never used by filtering or ranking."""

    normalized = title.casefold()
    if "data scientist" in normalized or "machine learning" in normalized:
        return "core_ai_data"
    if "data" in normalized and any(term in normalized for term in ("analyst", "qa", "quality")):
        return "data_adjacent"
    if any(term in normalized for term in ("analytics", "insights", "research analyst")):
        return "analytics_research"
    if "product manager" in normalized and "ai" in normalized:
        return "ai_product_adjacent"
    return "clearly_unsuitable_or_other"


def _add_ranks(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    relevance = sorted(
        rows,
        key=lambda row: (-row.relevance_score, row.title.casefold(), str(row.job_id)),
    )
    relevance_rank = {row.job_id: index for index, row in enumerate(relevance, 1)}
    legacy = sorted(
        rows,
        key=lambda row: (-row.legacy_blended_score, row.title.casefold(), str(row.job_id)),
    )
    legacy_rank = {row.job_id: index for index, row in enumerate(legacy, 1)}
    return [
        row.model_copy(
            update={
                "relevance_rank": relevance_rank[row.job_id],
                "legacy_rank": legacy_rank[row.job_id],
            }
        )
        for row in relevance
    ]


def _label_value(
    label: BenchmarkLabel,
    source: Literal["provisional", "human", "auto"],
) -> int | None:
    if source == "provisional":
        return label.provisional_label
    if source == "human":
        return label.human_label
    return label.human_label if label.human_label is not None else label.provisional_label


def _in_subset(row: BenchmarkRow, subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "core":
        return row.role_family == "core"
    if subset == "core_adjacent":
        return row.role_family in {"core", "adjacent"}
    if subset == "eligible":
        return row.eligibility_status == "eligible"
    if subset == "stretch":
        return row.eligibility_status in {"stretch", "severe_stretch"}
    if subset == "retained":
        return row.eligibility_status != "ineligible"
    raise ValueError(f"unsupported benchmark subset: {subset}")


def _secure_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


__all__ = [
    "BenchmarkIntegrityError",
    "BenchmarkLabel",
    "BenchmarkManifest",
    "BenchmarkMetrics",
    "BenchmarkRow",
    "FrozenAnalysisSet",
    "FrozenBenchmark",
    "FrozenJob",
    "FrozenJobSet",
    "FrozenStage1Result",
    "FrozenStage1Set",
    "FrozenStage2Result",
    "FrozenStage2Set",
    "IMMUTABLE_BENCHMARK_FILES",
    "LABEL_NAMES",
    "V2_IMMUTABLE_BENCHMARK_FILES",
    "benchmark_fingerprint",
    "calculate_benchmark_metrics",
    "categorize_job",
    "evaluate_frozen_benchmark",
    "load_benchmark_labels",
    "load_frozen_benchmark",
    "sha256_file",
    "semantic_benchmark_fingerprint",
    "write_json",
]
