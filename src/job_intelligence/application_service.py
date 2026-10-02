"""Application services consumed by the API and local UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .ai.models import JobMatchExplanation
from .company_repository import CompanyRepository
from .explanation_repository import ExplanationRepository
from .explanations import ExplanationService
from .feedback import UserFeedbackService
from .job_repository import JobRepository
from .match_repository import JobMatchRepository
from .models import (
    CandidateProfile,
    Company,
    FeedbackLabel,
    FeedbackReason,
    Job,
    JobMatch,
    JobStatus,
    ScanRun,
    UserFeedback,
)
from .profile_service import CandidateProfileService, ProfileNotFoundError
from .run_repository import ScanRunRepository


class ApplicationNotFoundError(LookupError):
    """Raised when an API resource does not exist."""


class ApplicationValidationError(ValueError):
    """Raised when an application action violates a domain invariant."""


class ExplanationUnavailableError(RuntimeError):
    """Raised when explanation generation has not been configured."""


@dataclass(frozen=True)
class JobView:
    job: Job
    company: Company | None
    match: JobMatch | None
    explanation: JobMatchExplanation | None
    feedback: list[UserFeedback]


class JobRadarApplication:
    """Coordinate existing domain services without putting business rules in routes."""

    def __init__(
        self,
        *,
        jobs: JobRepository,
        profiles: CandidateProfileService,
        companies: CompanyRepository,
        runs: ScanRunRepository,
        matches: JobMatchRepository,
        feedback: UserFeedbackService,
        explanations: ExplanationRepository,
        profile_id: UUID,
        explanation_service: ExplanationService | None = None,
    ) -> None:
        self._jobs = jobs
        self._profiles = profiles
        self._companies = companies
        self._runs = runs
        self._matches = matches
        self._feedback = feedback
        self._explanations = explanations
        self.profile_id = profile_id
        self._explanation_service = explanation_service

    def get_profile(self) -> CandidateProfile:
        profile = self._profiles.load(self.profile_id)
        if profile is None:
            raise ApplicationNotFoundError(f"Profile not found: {self.profile_id}")
        return profile

    def update_profile(self, profile: CandidateProfile) -> CandidateProfile:
        try:
            updated = self._profiles.update(self.profile_id, profile)
        except ProfileNotFoundError as exc:
            raise ApplicationNotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise ApplicationValidationError(str(exc)) from exc
        return updated

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        company: str | None = None,
        min_match_score: float | None = None,
        location: str | None = None,
        title: str | None = None,
        sort_by: Literal["score", "date"] = "score",
        descending: bool = True,
    ) -> list[JobView]:
        matches = {item.job_id: item for item in self._matches.list(self.profile_id)}
        feedback = self._feedback.list()
        companies = {item.id: item for item in self._companies.list()}
        views = []
        for state in self._jobs.list():
            job = state.job
            match = matches.get(job.id)
            if status is not None and job.status != status:
                continue
            company_record = companies.get(job.company_id)
            if company and not _company_matches(company_record, job.company_id, company):
                continue
            if location and location.casefold() not in (job.location or "").casefold():
                continue
            if title and title.casefold() not in job.title.casefold():
                continue
            if min_match_score is not None and (
                match is None or match.final_score < min_match_score
            ):
                continue
            views.append(self._view(job, company_record, match, feedback))

        if sort_by == "score":
            views.sort(
                key=lambda item: item.match.final_score if item.match else 0.0,
                reverse=descending,
            )
        else:
            views.sort(
                key=lambda item: item.job.posted_at or item.job.last_seen_at,
                reverse=descending,
            )
        return views

    def get_job(self, job_id: UUID) -> JobView:
        state = self._jobs.find_by_id(job_id)
        if state is None:
            raise ApplicationNotFoundError(f"Job not found: {job_id}")
        company = self._companies.get(state.job.company_id)
        match = self._matches.get(job_id, self.profile_id)
        return self._view(state.job, company, match, self._feedback.list())

    def list_companies(self) -> list[Company]:
        return self._companies.list()

    def list_runs(self) -> list[ScanRun]:
        return self._runs.list()

    def set_job_status(self, job_id: UUID, status: JobStatus) -> JobView:
        state = self._jobs.find_by_id(job_id)
        if state is None:
            raise ApplicationNotFoundError(f"Job not found: {job_id}")
        updated = Job.model_validate({**state.job.model_dump(mode="python"), "status": status})
        self._jobs.save(state.model_copy(update={"job": updated}))
        return self.get_job(job_id)

    def submit_feedback(
        self,
        job_id: UUID,
        *,
        label: FeedbackLabel,
        reason: FeedbackReason | None,
        notes: str | None,
    ) -> UserFeedback:
        if self._jobs.find_by_id(job_id) is None:
            raise ApplicationNotFoundError(f"Job not found: {job_id}")
        return self._feedback.create(
            job_id=job_id,
            candidate_profile_id=self.profile_id,
            label=label,
            reason=reason,
            notes=notes,
        )

    def request_explanation(self, job_id: UUID, *, force: bool = False) -> JobMatchExplanation:
        current = self.get_job(job_id)
        if current.match is None:
            raise ExplanationUnavailableError("A persisted match is required for an explanation")
        if not force:
            existing = self._explanations.get(job_id, self.profile_id)
            if existing is not None:
                return existing
        if self._explanation_service is None:
            raise ExplanationUnavailableError("Explanation generation is not configured")
        result = self._explanation_service.generate(
            current.job,
            self.get_profile(),
            current.match,
        )
        self._explanations.save(job_id, self.profile_id, result.value)
        return result.value

    def _view(
        self,
        job: Job,
        company: Company | None,
        match: JobMatch | None,
        feedback: list[UserFeedback],
    ) -> JobView:
        return JobView(
            job=job,
            company=company,
            match=match,
            explanation=self._explanations.get(job.id, self.profile_id),
            feedback=[item for item in feedback if item.job_id == job.id],
        )


__all__ = [
    "ApplicationNotFoundError",
    "ApplicationValidationError",
    "ExplanationUnavailableError",
    "JobRadarApplication",
    "JobView",
]


def _company_matches(company: Company | None, company_id: UUID, query: str) -> bool:
    normalized = query.casefold()
    return normalized in str(company_id).casefold() or (
        company is not None and normalized in company.name.casefold()
    )
