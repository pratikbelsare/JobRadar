"""AWS-independent coordinator, scanner, and analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, TypeVar
from uuid import UUID, uuid4

from ..ai import AIProvider, TextEmbeddingProvider, to_job_analysis
from ..ai.prompts import JOB_REQUIREMENTS_PROMPT_VERSION
from ..analysis_repository import JobAnalysisRepository
from ..change_detection import JobChangeType
from ..company_repository import CompanyRepository
from ..connectors import ConnectorRegistry, ConnectorStatus
from ..connectors.base import ConnectorError
from ..filtering import DeterministicJobFilter
from ..ingestion import JobIngestionService
from ..job_repository import JobRepository
from ..match_repository import JobMatchRepository
from ..matching import HybridJobMatcher, MatchingConfig
from ..models import CandidateProfile, Company, JobAnalysis, ScanRun, ScanRunStatus
from ..profile_repository import CandidateProfileRepository
from ..run_repository import ScanRunRepository
from .diagnostics import log_processing_failure
from .sqs import CompanyScanMessage, JobAnalysisMessage, QueuePublisher


class AWSWorkerError(RuntimeError):
    """Raised when a worker should allow the queue retry/DLQ policy to act."""


@dataclass(frozen=True)
class CoordinatorResult:
    run: ScanRun
    selected_company_ids: list[UUID]


class RunCoordinator:
    """Select a fair company batch and enqueue one small message per company."""

    def __init__(
        self,
        companies: CompanyRepository,
        runs: ScanRunRepository,
        queue: QueuePublisher,
        *,
        batch_size: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._companies = companies
        self._runs = runs
        self._queue = queue
        self._batch_size = batch_size
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def start(self) -> CoordinatorResult:
        now = _utc(self._clock())
        enabled = [company for company in self._companies.list() if company.enabled]
        enabled.sort(
            key=lambda company: company.last_checked_at
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        selected = enabled[: self._batch_size]
        run = ScanRun(
            id=uuid4(),
            status=ScanRunStatus.SUCCEEDED if not selected else ScanRunStatus.RUNNING,
            started_at=now,
            completed_at=now if not selected else None,
            companies_requested=len(selected),
        )
        self._runs.save(run)
        for company in selected:
            self._queue.publish(CompanyScanMessage(run_id=run.id, company_id=company.id))
        return CoordinatorResult(run=run, selected_company_ids=[company.id for company in selected])


@dataclass(frozen=True)
class ScannerResult:
    company_id: UUID
    jobs_discovered: int
    jobs_queued: int


class CompanyScanner:
    """Process one company message by composing existing connector/domain services."""

    def __init__(
        self,
        *,
        companies: CompanyRepository,
        profiles: CandidateProfileRepository,
        runs: ScanRunRepository,
        connectors: ConnectorRegistry,
        ingestion: JobIngestionService,
        filtering: DeterministicJobFilter,
        analysis_queue: QueuePublisher,
        candidate_profile_id: UUID | None = None,
        candidate_profile_key: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._companies = companies
        self._profiles = profiles
        self._runs = runs
        self._connectors = connectors
        self._ingestion = ingestion
        self._filtering = filtering
        self._analysis_queue = analysis_queue
        self._candidate_profile_id = candidate_profile_id
        self._candidate_profile_key = candidate_profile_key
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def process(self, message: CompanyScanMessage) -> ScannerResult:
        company = self._companies.get(message.company_id)
        if company is None:
            raise AWSWorkerError(f"Company not found: {message.company_id}")
        profile = self._load_profile()
        discovered = 0
        try:
            connector = self._connectors.for_company(company)
            listing = connector.list_jobs(company)
            if listing.status not in (ConnectorStatus.SUCCESS, ConnectorStatus.NO_JOBS):
                raise AWSWorkerError(listing.error or f"Connector failed: {listing.status.value}")

            discovered = len(listing.jobs)
            queued = 0
            errors: list[str] = []
            for reference in listing.jobs:
                details = connector.get_job(reference)
                if details.status != ConnectorStatus.SUCCESS or details.job is None:
                    errors.append(details.error or f"Could not fetch {reference.source_job_id}")
                    continue
                result = self._ingestion.ingest(details.job, observed_at=self._clock())
                if result.classification in (JobChangeType.NEW, JobChangeType.CHANGED):
                    filter_result = self._filtering.evaluate(result.job, profile)
                    if filter_result.passed:
                        self._analysis_queue.publish(
                            JobAnalysisMessage(
                                run_id=message.run_id,
                                job_id=result.job.id,
                                candidate_profile_id=profile.id,
                            )
                        )
                        queued += 1
            if errors:
                raise AWSWorkerError("; ".join(errors))
            self._record_company_result(
                message.run_id,
                company,
                succeeded=True,
                jobs_discovered=discovered,
            )
            return ScannerResult(
                company_id=company.id,
                jobs_discovered=len(listing.jobs),
                jobs_queued=queued,
            )
        except Exception as exc:
            self._record_company_result(
                message.run_id,
                company,
                succeeded=False,
                jobs_discovered=discovered,
            )
            if isinstance(exc, AWSWorkerError):
                raise
            if isinstance(exc, ConnectorError):
                raise AWSWorkerError(str(exc)) from exc
            raise AWSWorkerError(f"Company scan failed: {company.id}") from exc

    def _load_profile(self) -> CandidateProfile:
        if self._candidate_profile_id is not None:
            profile = self._profiles.load(self._candidate_profile_id)
            if profile is None:
                raise AWSWorkerError(f"Candidate profile not found: {self._candidate_profile_id}")
            return profile
        if self._candidate_profile_key is not None:
            profile = self._profiles.find_by_key(self._candidate_profile_key)
            if profile is None:
                raise AWSWorkerError(
                    f"Candidate profile not found: {self._candidate_profile_key}"
                )
            return profile
        profiles = self._profiles.list()
        if not profiles:
            raise AWSWorkerError("No candidate profile is configured")
        return profiles[0]

    def _record_company_result(
        self,
        run_id: UUID,
        company: Company,
        *,
        succeeded: bool,
        jobs_discovered: int,
    ) -> None:
        now = _utc(self._clock())
        if succeeded:
            updated_company = Company.model_validate(
                {
                    **company.model_dump(mode="python"),
                    "last_checked_at": now,
                    "last_success_at": now,
                    "failure_count": 0,
                }
            )
        else:
            updated_company = Company.model_validate(
                {
                    **company.model_dump(mode="python"),
                    "last_checked_at": now,
                    "failure_count": company.failure_count + 1,
                }
            )
        self._companies.save(updated_company)
        run = self._runs.get(run_id)
        if run is None or company.id in run.processed_company_ids:
            return
        processed = [*run.processed_company_ids, company.id]
        values = {
            **run.model_dump(mode="python"),
            "processed_company_ids": processed,
            "companies_succeeded": run.companies_succeeded + (1 if succeeded else 0),
            "companies_failed": run.companies_failed + (0 if succeeded else 1),
            "jobs_discovered": run.jobs_discovered + jobs_discovered,
        }
        if len(processed) >= run.companies_requested:
            values["completed_at"] = now
            values["status"] = (
                ScanRunStatus.SUCCEEDED
                if succeeded and values["companies_failed"] == 0
                else ScanRunStatus.PARTIAL
            )
        self._runs.save(ScanRun.model_validate(values))


@dataclass(frozen=True)
class AnalysisResult:
    analysis: JobAnalysis
    match_id: UUID


class JobAnalysisWorker:
    """Cache analysis by job version, then persist deterministic matching output."""

    def __init__(
        self,
        *,
        jobs: JobRepository,
        profiles: CandidateProfileRepository,
        analyses: JobAnalysisRepository,
        matches: JobMatchRepository,
        provider: AIProvider,
        model_id: str,
        embedding_provider: TextEmbeddingProvider | None = None,
        matcher: HybridJobMatcher | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        self._jobs = jobs
        self._profiles = profiles
        self._analyses = analyses
        self._matches = matches
        self._provider = provider
        self._model_id = model_id
        self._matcher = matcher or HybridJobMatcher(
            MatchingConfig(),
            embedding_provider=embedding_provider or provider,
        )

    def process(self, message: JobAnalysisMessage) -> AnalysisResult:
        identifiers: dict[str, object] = {
            "run_id": message.run_id,
            "job_id": message.job_id,
            "candidate_profile_id": message.candidate_profile_id,
        }
        state = _analysis_stage(
            "job_load",
            identifiers,
            lambda: self._jobs.find_by_id(message.job_id),
        )
        if state is None:
            raise AWSWorkerError(f"Job not found: {message.job_id}")
        profile = _analysis_stage(
            "profile_load",
            identifiers,
            lambda: self._profiles.load(message.candidate_profile_id),
        )
        if profile is None:
            raise AWSWorkerError(f"Candidate profile not found: {message.candidate_profile_id}")
        identifiers["candidate_profile_key"] = profile.profile_key
        version = _analysis_stage(
            "version_selection",
            identifiers,
            lambda: max(state.versions, key=lambda item: item.version_number, default=None),
        )
        if version is None:
            raise AWSWorkerError(f"Job has no version history: {message.job_id}")
        identifiers["job_version_id"] = version.id
        prompt_version = JOB_REQUIREMENTS_PROMPT_VERSION
        analysis = _analysis_stage(
            "analysis_cache_lookup",
            identifiers,
            lambda: self._analyses.get(
                state.job.id,
                version.id,
                self._model_id,
                prompt_version,
            ),
        )
        if analysis is None:
            result = _analysis_stage(
                "bedrock_structured_extraction",
                identifiers,
                lambda: self._provider.extract_job_requirements(state.job.description),
            )
            prompt_version = result.metadata.prompt_version or JOB_REQUIREMENTS_PROMPT_VERSION
            analysis = _analysis_stage(
                "analysis_cache_lookup_after_extraction",
                identifiers,
                lambda: self._analyses.get(
                    state.job.id,
                    version.id,
                    result.metadata.model_id,
                    prompt_version,
                ),
            )
            if analysis is None:
                analysis = to_job_analysis(
                    result,
                    job_id=state.job.id,
                    job_version_id=version.id,
                )
                _analysis_stage(
                    "analysis_persistence",
                    identifiers,
                    lambda: self._analyses.save(analysis),
                )

        match = _analysis_stage(
            "matching_and_embedding",
            identifiers,
            lambda: self._matcher.match(state.job, profile, analysis=analysis),
        )
        existing = _analysis_stage(
            "match_cache_lookup",
            identifiers,
            lambda: self._matches.get(state.job.id, profile.id),
        )
        if existing is not None:
            match = match.model_copy(update={"id": existing.id, "created_at": existing.created_at})
        _analysis_stage(
            "match_persistence",
            identifiers,
            lambda: self._matches.save(match),
        )
        return AnalysisResult(analysis=analysis, match_id=match.id)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


_AnalysisValue = TypeVar("_AnalysisValue")


def _analysis_stage(
    stage: str,
    identifiers: dict[str, object],
    operation: Callable[[], _AnalysisValue],
) -> _AnalysisValue:
    try:
        return operation()
    except Exception as exc:
        log_processing_failure(
            event="analysis_stage_failed",
            message_id="not_available_in_worker",
            stage=stage,
            error=exc,
            identifiers=identifiers,
        )
        raise


__all__ = [
    "AWSWorkerError",
    "AnalysisResult",
    "CompanyScanner",
    "CoordinatorResult",
    "JobAnalysisWorker",
    "RunCoordinator",
    "ScannerResult",
]
