"""Reusable local-first orchestration for one complete company scan.

The pipeline coordinates existing domain services. It deliberately contains no
normalization, filtering, extraction, eligibility, matching, or explanation rules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from time import monotonic
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from .ai.prompts import JOB_REQUIREMENTS_PROMPT_VERSION
from .analysis_repository import JobAnalysisRepository
from .aws.sqs import JobAnalysisMessage
from .aws.workers import JobAnalysisWorker
from .change_detection import JobChangeType
from .connectors.base import ConnectorStatus, JobSourceConnector
from .deduplication import ExactDuplicateDetector
from .explanation_repository import ExplanationRepository
from .explanations import ExplanationService
from .filtering import DeterministicJobFilter, FilterResult
from .ingestion import JobIngestionService
from .job_repository import JobRepository
from .match_repository import JobMatchRepository
from .models import (
    CandidateProfile,
    Company,
    JobAnalysis,
    JobMatch,
    Model,
    NonEmptyText,
    ScanRun,
    ScanRunStatus,
)
from .normalization import JobNormalizer, job_identity
from .role_title_normalization import RoleTitleNormalizer
from .run_repository import ScanRunRepository


class ConnectorResolver(Protocol):
    def for_company(self, company: Company) -> JobSourceConnector:
        """Return the configured connector for a company."""


class PipelineStage(str, Enum):
    LISTING = "listing"
    DETAIL_FETCH = "detail_fetch"
    INGESTION = "ingestion"
    STAGE_1 = "stage_1"
    ANALYSIS_MATCHING = "analysis_matching"
    MATCH_ENRICHMENT = "match_enrichment"
    EXPLANATION = "explanation"
    RUN_PERSISTENCE = "run_persistence"


class ExplanationStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    REUSED = "reused"
    CREATED = "created"
    FAILED = "failed"


class PipelineFailure(Model):
    stage: PipelineStage
    error_type: NonEmptyText
    message: NonEmptyText
    source_job_id: NonEmptyText | None = None
    job_id: UUID | None = None
    title: NonEmptyText | None = None


class PipelineJobOutcome(Model):
    source_job_id: NonEmptyText
    title: NonEmptyText
    job_id: UUID | None = None
    change_type: JobChangeType | None = None
    normalized_title: NonEmptyText | None = None
    base_title: NonEmptyText | None = None
    role_family_class: NonEmptyText | None = None
    stage1: FilterResult | None = None
    analysis_succeeded: bool = False
    analysis_reused: bool = False
    match_id: UUID | None = None
    match_reused: bool = False
    relevance_score: NonNegativeFloat | None = None
    legacy_score: NonNegativeFloat | None = None
    eligibility_status: NonEmptyText | None = None
    experience_gap_years: NonNegativeFloat | None = None
    explanation_status: ExplanationStatus = ExplanationStatus.NOT_APPLICABLE


class PipelineRunSummary(Model):
    run_id: UUID
    company_id: UUID
    company_name: NonEmptyText
    started_at: datetime
    completed_at: datetime
    duration_seconds: NonNegativeFloat
    status: ScanRunStatus
    jobs_discovered: NonNegativeInt = 0
    jobs_normalized: NonNegativeInt = 0
    duplicates: NonNegativeInt = 0
    new_jobs: NonNegativeInt = 0
    changed_jobs: NonNegativeInt = 0
    unchanged_jobs: NonNegativeInt = 0
    stage1_retained: NonNegativeInt = 0
    stage1_rejected: NonNegativeInt = 0
    analysis_successes: NonNegativeInt = 0
    analysis_failures: NonNegativeInt = 0
    eligibility_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    matches_created: NonNegativeInt = 0
    matches_updated: NonNegativeInt = 0
    matches_reused: NonNegativeInt = 0
    explanations_created: NonNegativeInt = 0
    explanations_reused: NonNegativeInt = 0
    explanation_failures: NonNegativeInt = 0
    extraction_calls: NonNegativeInt = 0
    explanation_calls: NonNegativeInt = 0
    extraction_cache_hits: NonNegativeInt = 0
    extraction_cache_misses: NonNegativeInt = 0
    embeddings_reused: NonNegativeInt = 0
    failures: NonNegativeInt = 0
    duplicate_jobs_created: NonNegativeInt = 0
    persistent_records_created: dict[str, NonNegativeInt] = Field(default_factory=dict)


class PipelineRunResult(Model):
    summary: PipelineRunSummary
    jobs: list[PipelineJobOutcome] = Field(default_factory=list)
    failures: list[PipelineFailure] = Field(default_factory=list)


class JobIntelligencePipeline:
    """Coordinate one company through the existing JobRadar services."""

    def __init__(
        self,
        *,
        connectors: ConnectorResolver,
        ingestion: JobIngestionService,
        filtering: DeterministicJobFilter,
        analysis_worker: JobAnalysisWorker,
        jobs: JobRepository,
        analyses: JobAnalysisRepository,
        matches: JobMatchRepository,
        explanations: ExplanationRepository,
        explanation_service: ExplanationService,
        runs: ScanRunRepository,
        analysis_model_id: str,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
        normalizer: JobNormalizer | None = None,
        duplicate_detector: ExactDuplicateDetector | None = None,
    ) -> None:
        if not analysis_model_id.strip():
            raise ValueError("analysis_model_id must not be blank")
        self._connectors = connectors
        self._ingestion = ingestion
        self._filtering = filtering
        self._analysis_worker = analysis_worker
        self._jobs = jobs
        self._analyses = analyses
        self._matches = matches
        self._explanations = explanations
        self._explanation_service = explanation_service
        self._runs = runs
        self._analysis_model_id = analysis_model_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timer = timer or monotonic
        self._title_normalizer = RoleTitleNormalizer()
        self._normalizer = normalizer or JobNormalizer()
        self._duplicate_detector = duplicate_detector or ExactDuplicateDetector()

    def run_company(
        self,
        company: Company,
        profile: CandidateProfile,
    ) -> PipelineRunResult:
        run_id = uuid4()
        started_at = _utc(self._clock())
        started_timer = self._timer()
        self._runs.save(
            ScanRun(
                id=run_id,
                status=ScanRunStatus.RUNNING,
                started_at=started_at,
                companies_requested=1,
            )
        )
        before_jobs = len(self._jobs.list())
        counters: Counter[str] = Counter()
        eligibility_counts: Counter[str] = Counter()
        outcomes: list[PipelineJobOutcome] = []
        failures: list[PipelineFailure] = []

        try:
            connector = self._connectors.for_company(company)
            listing = connector.list_jobs(company)
        except Exception as exc:
            failures.append(_failure(PipelineStage.LISTING, exc))
            return self._finish(
                run_id,
                company,
                started_at,
                started_timer,
                before_jobs,
                counters,
                eligibility_counts,
                outcomes,
                failures,
            )
        if listing.status not in {ConnectorStatus.SUCCESS, ConnectorStatus.NO_JOBS}:
            failures.append(
                PipelineFailure(
                    stage=PipelineStage.LISTING,
                    error_type="ConnectorError",
                    message=listing.error or f"Connector returned {listing.status.value}",
                )
            )
            return self._finish(
                run_id,
                company,
                started_at,
                started_timer,
                before_jobs,
                counters,
                eligibility_counts,
                outcomes,
                failures,
            )

        counters["jobs_discovered"] = len(listing.jobs)
        seen_source_ids: set[str] = set()
        seen_source_urls: set[str] = set()
        for reference in listing.jobs:
            source_url = str(reference.source_url)
            if (
                reference.source_job_id in seen_source_ids
                or source_url in seen_source_urls
            ):
                counters["duplicates"] += 1
                continue
            seen_source_ids.add(reference.source_job_id)
            seen_source_urls.add(source_url)
            outcome = PipelineJobOutcome(
                source_job_id=reference.source_job_id,
                title=reference.title,
            )
            try:
                details = connector.get_job(reference)
                if details.status != ConnectorStatus.SUCCESS or details.job is None:
                    raise RuntimeError(
                        details.error or f"Connector returned {details.status.value}"
                    )
            except Exception as exc:
                failures.append(
                    _failure(
                        PipelineStage.DETAIL_FETCH,
                        exc,
                        source_job_id=reference.source_job_id,
                        title=reference.title,
                    )
                )
                outcomes.append(outcome)
                continue

            try:
                observed_at = self._clock()
                candidate = self._normalizer.normalize(details.job, observed_at=observed_at)
                counters["jobs_normalized"] += 1
                duplicate = self._duplicate_detector.find_duplicate(
                    candidate,
                    (state.job for state in self._jobs.list()),
                )
                if duplicate is not None and not job_identity(duplicate).matches(
                    job_identity(candidate)
                ):
                    counters["duplicates"] += 1
                    outcome.job_id = duplicate.id
                    outcomes.append(outcome)
                    continue
                change = self._ingestion.ingest(details.job, observed_at=observed_at)
                counters[f"{change.classification.value}_jobs"] += 1
                counters["raw_snapshots"] += 1
                outcome.job_id = change.job.id
                outcome.change_type = change.classification
                normalized_title = self._title_normalizer.normalize(change.job.title)
                outcome.normalized_title = normalized_title.normalized_title
                outcome.base_title = normalized_title.base_title
            except Exception as exc:
                failures.append(
                    _failure(
                        PipelineStage.INGESTION,
                        exc,
                        source_job_id=reference.source_job_id,
                        title=reference.title,
                    )
                )
                outcomes.append(outcome)
                continue

            try:
                filter_result = self._filtering.evaluate(change.job, profile)
                outcome.stage1 = filter_result
            except Exception as exc:
                failures.append(
                    _failure(
                        PipelineStage.STAGE_1,
                        exc,
                        source_job_id=reference.source_job_id,
                        job_id=change.job.id,
                        title=change.job.title,
                    )
                )
                outcomes.append(outcome)
                continue
            if not filter_result.passed:
                counters["stage1_rejected"] += 1
                role_details = filter_result.rules[0].details if filter_result.rules else {}
                family = role_details.get("role_family_class")
                outcome.role_family_class = str(family) if family is not None else "unknown"
                outcomes.append(outcome)
                continue

            counters["stage1_retained"] += 1
            self._process_retained(
                run_id,
                change.classification,
                change.job.id,
                profile,
                filter_result,
                outcome,
                counters,
                eligibility_counts,
                failures,
            )
            outcomes.append(outcome)

        return self._finish(
            run_id,
            company,
            started_at,
            started_timer,
            before_jobs,
            counters,
            eligibility_counts,
            outcomes,
            failures,
        )

    def _process_retained(
        self,
        run_id: UUID,
        change_type: JobChangeType,
        job_id: UUID,
        profile: CandidateProfile,
        filter_result: FilterResult,
        outcome: PipelineJobOutcome,
        counters: Counter[str],
        eligibility_counts: Counter[str],
        failures: list[PipelineFailure],
    ) -> None:
        state = self._jobs.find_by_id(job_id)
        if state is None or not state.versions:
            failures.append(
                PipelineFailure(
                    stage=PipelineStage.ANALYSIS_MATCHING,
                    error_type="MissingJobVersion",
                    message="Persisted job has no version history",
                    source_job_id=outcome.source_job_id,
                    job_id=job_id,
                    title=outcome.title,
                )
            )
            return
        version = max(state.versions, key=lambda item: item.version_number)
        try:
            cached_analysis = self._analyses.get(
                job_id,
                version.id,
                self._analysis_model_id,
                JOB_REQUIREMENTS_PROMPT_VERSION,
            )
            existing_match = self._matches.get(job_id, profile.id)
        except Exception as exc:
            failures.append(
                _failure(
                    PipelineStage.ANALYSIS_MATCHING,
                    exc,
                    source_job_id=outcome.source_job_id,
                    job_id=job_id,
                    title=outcome.title,
                )
            )
            return

        can_reuse_match = (
            change_type == JobChangeType.UNCHANGED
            and cached_analysis is not None
            and existing_match is not None
        )
        analysis: JobAnalysis
        match: JobMatch
        if can_reuse_match:
            assert cached_analysis is not None
            assert existing_match is not None
            analysis = cached_analysis
            match = existing_match
            outcome.analysis_reused = True
            outcome.match_reused = True
            counters["extraction_cache_hits"] += 1
            counters["matches_reused"] += 1
            counters["embeddings_reused"] += 1
        else:
            if cached_analysis is None:
                counters["extraction_cache_misses"] += 1
                counters["extraction_calls"] += 1
            else:
                counters["extraction_cache_hits"] += 1
                outcome.analysis_reused = True
            try:
                worker_result = self._analysis_worker.process(
                    JobAnalysisMessage(
                        run_id=run_id,
                        job_id=job_id,
                        candidate_profile_id=profile.id,
                    )
                )
                analysis = worker_result.analysis
                if cached_analysis is None:
                    counters["analyses_created"] += 1
                persisted_match = self._matches.get(job_id, profile.id)
                if persisted_match is None:
                    raise RuntimeError("Analysis worker did not persist a match")
                match = persisted_match
                if existing_match is None:
                    counters["matches_created"] += 1
                else:
                    counters["matches_updated"] += 1
            except Exception as exc:
                if cached_analysis is None:
                    try:
                        persisted_analysis = self._analyses.get(
                            job_id,
                            version.id,
                            self._analysis_model_id,
                            JOB_REQUIREMENTS_PROMPT_VERSION,
                        )
                    except Exception:
                        persisted_analysis = None
                    if persisted_analysis is not None:
                        counters["analyses_created"] += 1
                counters["analysis_failures"] += 1
                failures.append(
                    _failure(
                        PipelineStage.ANALYSIS_MATCHING,
                        exc,
                        source_job_id=outcome.source_job_id,
                        job_id=job_id,
                        title=outcome.title,
                    )
                )
                return

        try:
            evidence = dict(match.evidence)
            evidence["stage1"] = filter_result.model_dump(mode="json")
            enriched_match = match.model_copy(update={"evidence": evidence})
            self._matches.save(enriched_match)
            match = enriched_match
        except Exception as exc:
            failures.append(
                _failure(
                    PipelineStage.MATCH_ENRICHMENT,
                    exc,
                    source_job_id=outcome.source_job_id,
                    job_id=job_id,
                    title=outcome.title,
                )
            )
            return

        outcome.analysis_succeeded = True
        counters["analysis_successes"] += 1
        outcome.match_id = match.id
        outcome.relevance_score = match.relevance_score
        outcome.legacy_score = match.legacy_final_score or match.final_score
        outcome.role_family_class = match.role_family_class or "unknown"
        if match.eligibility is not None:
            outcome.eligibility_status = match.eligibility.status
            outcome.experience_gap_years = match.eligibility.experience_gap_years
            eligibility_counts[match.eligibility.status] += 1

        existing_explanation = self._explanations.get(job_id, profile.id)
        if change_type == JobChangeType.UNCHANGED and existing_explanation is not None:
            outcome.explanation_status = ExplanationStatus.REUSED
            counters["explanations_reused"] += 1
            return
        try:
            explanation = self._explanation_service.generate(
                state.job,
                profile,
                match,
                analysis=analysis,
            )
            self._explanations.save(job_id, profile.id, explanation.value)
            outcome.explanation_status = ExplanationStatus.CREATED
            counters["explanations_created"] += 1
            counters["explanation_calls"] += 1
        except Exception as exc:
            outcome.explanation_status = ExplanationStatus.FAILED
            counters["explanation_failures"] += 1
            counters["explanation_calls"] += 1
            failures.append(
                _failure(
                    PipelineStage.EXPLANATION,
                    exc,
                    source_job_id=outcome.source_job_id,
                    job_id=job_id,
                    title=outcome.title,
                )
            )

    def _finish(
        self,
        run_id: UUID,
        company: Company,
        started_at: datetime,
        started_timer: float,
        before_jobs: int,
        counters: Counter[str],
        eligibility_counts: Counter[str],
        outcomes: list[PipelineJobOutcome],
        failures: list[PipelineFailure],
    ) -> PipelineRunResult:
        completed_at = _utc(self._clock())
        duration = max(0.0, self._timer() - started_timer)
        after_jobs = len(self._jobs.list())
        jobs_created = max(0, after_jobs - before_jobs)
        duplicate_jobs_created = max(0, jobs_created - counters["new_jobs"])
        status = ScanRunStatus.SUCCEEDED
        if failures and outcomes:
            status = ScanRunStatus.PARTIAL
        elif failures:
            status = ScanRunStatus.FAILED
        run = ScanRun(
            id=run_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            companies_requested=1,
            companies_succeeded=1 if status != ScanRunStatus.FAILED else 0,
            companies_failed=1 if status == ScanRunStatus.FAILED else 0,
            jobs_discovered=counters["jobs_discovered"],
            processed_company_ids=[company.id],
        )
        try:
            self._runs.save(run)
        except Exception as exc:
            failures.append(_failure(PipelineStage.RUN_PERSISTENCE, exc))
            status = ScanRunStatus.PARTIAL

        summary = PipelineRunSummary(
            run_id=run_id,
            company_id=company.id,
            company_name=company.name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            status=status,
            jobs_discovered=counters["jobs_discovered"],
            jobs_normalized=counters["jobs_normalized"],
            duplicates=counters["duplicates"],
            new_jobs=counters["new_jobs"],
            changed_jobs=counters["changed_jobs"],
            unchanged_jobs=counters["unchanged_jobs"],
            stage1_retained=counters["stage1_retained"],
            stage1_rejected=counters["stage1_rejected"],
            analysis_successes=counters["analysis_successes"],
            analysis_failures=counters["analysis_failures"],
            eligibility_counts=dict(eligibility_counts),
            matches_created=counters["matches_created"],
            matches_updated=counters["matches_updated"],
            matches_reused=counters["matches_reused"],
            explanations_created=counters["explanations_created"],
            explanations_reused=counters["explanations_reused"],
            explanation_failures=counters["explanation_failures"],
            extraction_calls=counters["extraction_calls"],
            explanation_calls=counters["explanation_calls"],
            extraction_cache_hits=counters["extraction_cache_hits"],
            extraction_cache_misses=counters["extraction_cache_misses"],
            embeddings_reused=counters["embeddings_reused"],
            failures=len(failures),
            duplicate_jobs_created=duplicate_jobs_created,
            persistent_records_created={
                "canonical_jobs": jobs_created,
                "job_versions": counters["new_jobs"] + counters["changed_jobs"],
                "raw_snapshots": counters["raw_snapshots"],
                "analyses": counters["analyses_created"],
                "matches": counters["matches_created"],
                "explanations": counters["explanations_created"],
            },
        )
        return PipelineRunResult(summary=summary, jobs=outcomes, failures=failures)


def _failure(
    stage: PipelineStage,
    error: Exception,
    *,
    source_job_id: str | None = None,
    job_id: UUID | None = None,
    title: str | None = None,
) -> PipelineFailure:
    message = str(error).strip() or "Unexpected processing failure"
    return PipelineFailure(
        stage=stage,
        error_type=type(error).__name__,
        message=message[:500],
        source_job_id=source_job_id,
        job_id=job_id,
        title=title,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "ConnectorResolver",
    "ExplanationStatus",
    "JobIntelligencePipeline",
    "PipelineFailure",
    "PipelineJobOutcome",
    "PipelineRunResult",
    "PipelineRunSummary",
    "PipelineStage",
]
