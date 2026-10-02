"""FastAPI application for the single-user local JobRadar workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from .ai.models import JobMatchExplanation
from .application_service import (
    ApplicationNotFoundError,
    ApplicationValidationError,
    ExplanationUnavailableError,
    JobRadarApplication,
    JobView,
)
from .company_repository import JsonCompanyRepository
from .config import Settings
from .explanation_repository import JsonExplanationRepository
from .explanations import (
    ExplanationGenerationError,
    ExplanationGroundingError,
    ExplanationService,
)
from .feedback import JsonUserFeedbackRepository, UserFeedbackService
from .job_repository import JsonJobRepository
from .match_repository import JsonJobMatchRepository
from .models import (
    CandidateProfile,
    Company,
    FeedbackLabel,
    FeedbackReason,
    Job,
    JobMatch,
    JobStatus,
    Model,
    ScanRun,
    Score,
    UserFeedback,
)
from .profile_repository import JsonCandidateProfileRepository
from .profile_service import CandidateProfileService
from .run_repository import JsonScanRunRepository


class HealthResponse(Model):
    status: Literal["ok"] = "ok"
    service: str = "job-intelligence"


class JobResponse(Model):
    job: Job
    company: Company | None = None
    match: JobMatch | None = None
    explanation: JobMatchExplanation | None = None
    feedback: list[UserFeedback] = Field(default_factory=list)


class FeedbackRequest(Model):
    label: FeedbackLabel
    reason: FeedbackReason | None = None
    notes: str | None = None


class ExplanationRequest(Model):
    force: bool = False


def create_app(
    application: JobRadarApplication | None = None,
    *,
    settings: Settings | None = None,
    explanation_service: ExplanationService | None = None,
) -> FastAPI:
    """Create an app with injected services, or a local JSON-backed application."""

    application = application or build_local_application(
        settings or Settings.from_environment(),
        explanation_service=explanation_service,
    )
    app = FastAPI(title="JobRadar API", version="0.1.0")
    app.state.application = application

    @app.exception_handler(ApplicationNotFoundError)
    async def not_found_handler(_: Request, exc: ApplicationNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ExplanationUnavailableError)
    async def explanation_handler(_: Request, exc: ExplanationUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ApplicationValidationError)
    async def validation_handler(_: Request, exc: ApplicationValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ExplanationGenerationError)
    async def generation_handler(_: Request, exc: ExplanationGenerationError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ExplanationGroundingError)
    async def grounding_handler(_: Request, exc: ExplanationGroundingError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/jobs", response_model=list[JobResponse])
    async def list_jobs(
        status: JobStatus | None = None,
        company: str | None = Query(default=None, min_length=1),
        min_match_score: Score | None = Query(default=None),
        location: str | None = Query(default=None, min_length=1),
        title: str | None = Query(default=None, min_length=1),
        sort_by: Literal["score", "date"] = Query(default="score"),
        descending: bool = True,
    ) -> list[JobResponse]:
        views = application.list_jobs(
            status=status,
            company=company,
            min_match_score=min_match_score,
            location=location,
            title=title,
            sort_by=sort_by,
            descending=descending,
        )
        return [_job_response(view) for view in views]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: UUID) -> JobResponse:
        return _job_response(application.get_job(job_id))

    @app.get("/companies", response_model=list[Company])
    async def list_companies() -> list[Company]:
        return application.list_companies()

    @app.get("/runs", response_model=list[ScanRun])
    async def list_runs() -> list[ScanRun]:
        return application.list_runs()

    @app.get("/profile", response_model=CandidateProfile)
    async def get_profile() -> CandidateProfile:
        return application.get_profile()

    @app.put("/profile", response_model=CandidateProfile)
    async def update_profile(profile: CandidateProfile = Body(...)) -> CandidateProfile:
        return application.update_profile(profile)

    @app.post("/jobs/{job_id}/shortlist", response_model=JobResponse)
    async def shortlist(job_id: UUID) -> JobResponse:
        return _job_response(application.set_job_status(job_id, JobStatus.SHORTLISTED))

    @app.post("/jobs/{job_id}/dismiss", response_model=JobResponse)
    async def dismiss(job_id: UUID) -> JobResponse:
        return _job_response(application.set_job_status(job_id, JobStatus.DISMISSED))

    @app.post("/jobs/{job_id}/applied", response_model=JobResponse)
    async def applied(job_id: UUID) -> JobResponse:
        return _job_response(application.set_job_status(job_id, JobStatus.APPLIED))

    @app.post("/jobs/{job_id}/feedback", response_model=UserFeedback, status_code=201)
    async def submit_feedback(job_id: UUID, payload: FeedbackRequest = Body(...)) -> UserFeedback:
        return application.submit_feedback(
            job_id,
            label=payload.label,
            reason=payload.reason,
            notes=payload.notes,
        )

    @app.post("/jobs/{job_id}/explanation", response_model=JobMatchExplanation)
    async def request_explanation(
        job_id: UUID,
        payload: ExplanationRequest | None = Body(default=None),
    ) -> JobMatchExplanation:
        return application.request_explanation(job_id, force=payload.force if payload else False)

    return app


def build_local_application(
    settings: Settings,
    *,
    explanation_service: ExplanationService | None = None,
) -> JobRadarApplication:
    data_directory = settings.application_data_directory or Path("data")
    profile_directory = settings.candidate_profile_directory or data_directory / "profiles"
    profile_repository = JsonCandidateProfileRepository(profile_directory)
    profiles = CandidateProfileService(profile_repository)
    profile_id = settings.candidate_profile_id
    if profile_id is None:
        existing = profile_repository.list()
        profile_id = existing[0].id if existing else profiles.create(CandidateProfile()).id
    return JobRadarApplication(
        jobs=JsonJobRepository(data_directory / "jobs"),
        profiles=profiles,
        companies=JsonCompanyRepository(data_directory / "companies.json"),
        runs=JsonScanRunRepository(data_directory / "runs.json"),
        matches=JsonJobMatchRepository(data_directory / "matches.json"),
        feedback=UserFeedbackService(JsonUserFeedbackRepository(data_directory / "feedback.json")),
        explanations=JsonExplanationRepository(data_directory / "explanations.json"),
        profile_id=profile_id,
        explanation_service=explanation_service,
    )


def _job_response(view: JobView) -> JobResponse:
    return JobResponse(
        job=view.job,
        company=view.company,
        match=view.match,
        explanation=view.explanation,
        feedback=view.feedback,
    )


__all__ = ["FeedbackRequest", "JobResponse", "build_local_application", "create_app"]
