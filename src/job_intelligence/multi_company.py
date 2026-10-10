"""Lightweight sequential coordination for local multi-company validation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from time import monotonic
from typing import Callable
from uuid import UUID, uuid4

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from .models import CandidateProfile, Company, Model, NonEmptyText, ScanRunStatus
from .pipeline import (
    ExplanationStatus,
    JobIntelligencePipeline,
    PipelineRunResult,
    PipelineRunSummary,
)


class CompanyRunFailure(Model):
    """Unexpected company-level failure outside the per-job pipeline boundary."""

    company_id: UUID
    company_name: NonEmptyText
    error_type: NonEmptyText
    message: NonEmptyText


class MultiCompanyRunSummary(Model):
    """Aggregate observability for one pass over a configured company catalog."""

    run_id: UUID
    started_at: datetime
    completed_at: datetime
    duration_seconds: NonNegativeFloat
    status: ScanRunStatus
    companies_attempted: NonNegativeInt
    companies_succeeded: NonNegativeInt
    companies_partial: NonNegativeInt
    companies_failed: NonNegativeInt
    failed_company_names: list[NonEmptyText] = Field(default_factory=list)
    company_summaries: list[PipelineRunSummary] = Field(default_factory=list)
    jobs_discovered: NonNegativeInt = 0
    jobs_normalized: NonNegativeInt = 0
    new_jobs: NonNegativeInt = 0
    changed_jobs: NonNegativeInt = 0
    unchanged_jobs: NonNegativeInt = 0
    job_versions_created: NonNegativeInt = 0
    duplicate_jobs_created: NonNegativeInt = 0
    stage1_retained: NonNegativeInt = 0
    stage1_rejected: NonNegativeInt = 0
    eligibility_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    analyses_generated: NonNegativeInt = 0
    analyses_reused: NonNegativeInt = 0
    analysis_failures: NonNegativeInt = 0
    embeddings_generated: NonNegativeInt = 0
    embeddings_reused: NonNegativeInt = 0
    matches_generated: NonNegativeInt = 0
    matches_reused: NonNegativeInt = 0
    explanations_generated: NonNegativeInt = 0
    explanations_reused: NonNegativeInt = 0
    explanation_failures: NonNegativeInt = 0
    explanation_recoveries: NonNegativeInt = 0
    extraction_calls: NonNegativeInt = 0
    extraction_cache_hits: NonNegativeInt = 0
    extraction_cache_misses: NonNegativeInt = 0
    explanation_calls: NonNegativeInt = 0
    failed_jobs: NonNegativeInt = 0
    total_failures: NonNegativeInt = 0


class MultiCompanyRunResult(Model):
    summary: MultiCompanyRunSummary
    company_results: list[PipelineRunResult] = Field(default_factory=list)
    company_failures: list[CompanyRunFailure] = Field(default_factory=list)


class MultiCompanyPipelineRunner:
    """Run the existing one-company pipeline sequentially with failure isolation."""

    def __init__(
        self,
        pipeline: JobIntelligencePipeline,
        *,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timer = timer or monotonic

    def run(
        self,
        companies: list[Company],
        profile: CandidateProfile,
    ) -> MultiCompanyRunResult:
        run_id = uuid4()
        started_at = _utc(self._clock())
        started_timer = self._timer()
        enabled_companies = [company for company in companies if company.enabled]
        company_results: list[PipelineRunResult] = []
        company_failures: list[CompanyRunFailure] = []

        for company in enabled_companies:
            try:
                company_results.append(self._pipeline.run_company(company, profile))
            except Exception as exc:
                company_failures.append(
                    CompanyRunFailure(
                        company_id=company.id,
                        company_name=company.name,
                        error_type=type(exc).__name__,
                        message=(str(exc).strip() or "Unexpected company failure")[:500],
                    )
                )

        completed_at = _utc(self._clock())
        duration_seconds = max(0.0, self._timer() - started_timer)
        summaries = [result.summary for result in company_results]
        failed_summaries = [
            summary for summary in summaries if summary.status == ScanRunStatus.FAILED
        ]
        partial_summaries = [
            summary for summary in summaries if summary.status == ScanRunStatus.PARTIAL
        ]
        failed_names = [summary.company_name for summary in failed_summaries]
        failed_names.extend(failure.company_name for failure in company_failures)
        companies_failed = len(failed_names)
        companies_partial = len(partial_summaries)
        companies_succeeded = len(enabled_companies) - companies_failed
        status = ScanRunStatus.SUCCEEDED
        if companies_failed == len(enabled_companies) and enabled_companies:
            status = ScanRunStatus.FAILED
        elif companies_failed or companies_partial:
            status = ScanRunStatus.PARTIAL

        analyses_reused = sum(
            int(outcome.analysis_reused)
            for result in company_results
            for outcome in result.jobs
        )
        failed_job_keys = {
            (result.summary.company_id, failure.source_job_id or str(failure.job_id))
            for result in company_results
            for failure in result.failures
            if failure.source_job_id is not None or failure.job_id is not None
        }
        analysis_successes = _sum(summaries, "analysis_successes")
        embeddings_reused = _sum(summaries, "embeddings_reused")
        eligibility_counts: Counter[str] = Counter()
        for company_summary in summaries:
            eligibility_counts.update(company_summary.eligibility_counts)
        summary = MultiCompanyRunSummary(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            status=status,
            companies_attempted=len(enabled_companies),
            companies_succeeded=companies_succeeded,
            companies_partial=companies_partial,
            companies_failed=companies_failed,
            failed_company_names=failed_names,
            company_summaries=summaries,
            jobs_discovered=_sum(summaries, "jobs_discovered"),
            jobs_normalized=_sum(summaries, "jobs_normalized"),
            new_jobs=_sum(summaries, "new_jobs"),
            changed_jobs=_sum(summaries, "changed_jobs"),
            unchanged_jobs=_sum(summaries, "unchanged_jobs"),
            job_versions_created=sum(
                item.persistent_records_created.get("job_versions", 0)
                for item in summaries
            ),
            duplicate_jobs_created=_sum(summaries, "duplicate_jobs_created"),
            stage1_retained=_sum(summaries, "stage1_retained"),
            stage1_rejected=_sum(summaries, "stage1_rejected"),
            eligibility_counts=dict(eligibility_counts),
            analyses_generated=sum(
                item.persistent_records_created.get("analyses", 0) for item in summaries
            ),
            analyses_reused=analyses_reused,
            analysis_failures=_sum(summaries, "analysis_failures"),
            embeddings_generated=max(0, analysis_successes - embeddings_reused),
            embeddings_reused=embeddings_reused,
            matches_generated=(
                _sum(summaries, "matches_created")
                + _sum(summaries, "matches_updated")
            ),
            matches_reused=_sum(summaries, "matches_reused"),
            explanations_generated=_sum(summaries, "explanations_created"),
            explanations_reused=_sum(summaries, "explanations_reused"),
            explanation_failures=_sum(summaries, "explanation_failures"),
            explanation_recoveries=sum(
                int(
                    outcome.change_type is not None
                    and outcome.change_type.value == "unchanged"
                    and outcome.explanation_status == ExplanationStatus.CREATED
                )
                for result in company_results
                for outcome in result.jobs
            ),
            extraction_calls=_sum(summaries, "extraction_calls"),
            extraction_cache_hits=_sum(summaries, "extraction_cache_hits"),
            extraction_cache_misses=_sum(summaries, "extraction_cache_misses"),
            explanation_calls=_sum(summaries, "explanation_calls"),
            failed_jobs=len(failed_job_keys),
            total_failures=(
                sum(len(result.failures) for result in company_results)
                + len(company_failures)
            ),
        )
        return MultiCompanyRunResult(
            summary=summary,
            company_results=company_results,
            company_failures=company_failures,
        )


def _sum(summaries: list[PipelineRunSummary], field: str) -> int:
    return sum(int(getattr(summary, field)) for summary in summaries)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "CompanyRunFailure",
    "MultiCompanyPipelineRunner",
    "MultiCompanyRunResult",
    "MultiCompanyRunSummary",
]
