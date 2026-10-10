"""API tests use only local JSON repositories and a fake explanation provider."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from job_intelligence.ai.models import AIResponseMetadata, AIResult, JobMatchExplanation
from job_intelligence.api import create_app
from job_intelligence.application_service import JobRadarApplication
from job_intelligence.change_detection import StoredJobState
from job_intelligence.company_repository import JsonCompanyRepository
from job_intelligence.explanation_repository import JsonExplanationRepository
from job_intelligence.explanations import ExplanationService
from job_intelligence.feedback import JsonUserFeedbackRepository, UserFeedbackService
from job_intelligence.job_repository import JsonJobRepository
from job_intelligence.match_repository import JsonJobMatchRepository
from job_intelligence.models import (
    CandidateProfile,
    Company,
    ConnectorType,
    FeedbackLabel,
    Job,
    JobMatch,
    JobStatus,
    WorkMode,
)
from job_intelligence.profile_repository import JsonCandidateProfileRepository
from job_intelligence.profile_service import CandidateProfileService
from job_intelligence.run_repository import JsonScanRunRepository


class FakeExplanationProvider:
    def generate_explanation(self, context: object) -> AIResult[JobMatchExplanation]:
        explanation = JobMatchExplanation(experience_alignment="Evidence is available.")
        return AIResult(
            value=explanation,
            metadata=AIResponseMetadata(provider="test", model_id="test-model"),
        )


def _application(tmp_path: Path) -> tuple[JobRadarApplication, Job, CandidateProfile]:
    profile = CandidateProfile(target_roles=["Software Engineer"])
    company = Company(
        name="Example",
        career_url="https://example.com/careers",
        connector_type=ConnectorType.GENERIC,
    )
    job = Job(
        source_job_id="job-1",
        company_id=company.id,
        title="Software Engineer",
        location="Remote",
        work_mode=WorkMode.REMOTE,
        description="Build reliable systems.",
        source_url="https://example.com/jobs/job-1",
        posted_at=datetime.now(timezone.utc),
    )
    jobs = JsonJobRepository(tmp_path / "jobs")
    jobs.save(StoredJobState(job=job))
    companies = JsonCompanyRepository(tmp_path / "companies.json")
    companies.save_all([company])
    matches = JsonJobMatchRepository(tmp_path / "matches.json")
    matches.save_all(
        [
            JobMatch(
                job_id=job.id,
                candidate_profile_id=profile.id,
                role_match=0.9,
                skill_match=0.8,
                experience_match=0.8,
                responsibility_match=0.7,
                domain_match=0.6,
                location_match=1.0,
                preference_match=0.8,
                final_score=0.85,
                ranking_version="test",
            )
        ]
    )
    profiles = CandidateProfileService(JsonCandidateProfileRepository(tmp_path / "profiles"))
    profiles.create(profile)
    application = JobRadarApplication(
        jobs=jobs,
        profiles=profiles,
        companies=companies,
        runs=JsonScanRunRepository(tmp_path / "runs.json"),
        matches=matches,
        feedback=UserFeedbackService(JsonUserFeedbackRepository(tmp_path / "feedback.json")),
        explanations=JsonExplanationRepository(tmp_path / "explanations.json"),
        profile_id=profile.id,
        explanation_service=ExplanationService(FakeExplanationProvider()),  # type: ignore[arg-type]
    )
    return application, job, profile


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, Job, CandidateProfile]:
    application, job, profile = _application(tmp_path)
    return TestClient(create_app(application)), job, profile


def test_health_and_job_listing(client: tuple[TestClient, Job, CandidateProfile]) -> None:
    http, _, _ = client
    assert http.get("/health").json()["status"] == "ok"
    response = http.get("/jobs", params={"min_match_score": 0.8})
    assert response.status_code == 200
    assert response.json()[0]["match"]["final_score"] == 0.85


def test_job_filters_and_missing_detail(client: tuple[TestClient, Job, CandidateProfile]) -> None:
    http, job, _ = client
    assert http.get("/jobs", params={"title": "unrelated"}).json() == []
    assert len(http.get("/jobs", params={"company": "example"}).json()) == 1
    assert http.get(f"/jobs/{job.id}").status_code == 200
    assert http.get(f"/jobs/{uuid4()}").status_code == 404


def test_profile_update_and_invalid_payload(
    client: tuple[TestClient, Job, CandidateProfile],
) -> None:
    http, _, profile = client
    payload = profile.model_dump(mode="json")
    payload["target_roles"] = ["Data Engineer"]
    assert http.put("/profile", json=payload).json()["target_roles"] == ["Data Engineer"]
    invalid = {**payload, "experience_years": -1}
    assert http.put("/profile", json=invalid).status_code == 422


def test_job_actions_feedback_and_explanation(
    client: tuple[TestClient, Job, CandidateProfile],
) -> None:
    http, job, _ = client
    assert http.post(f"/jobs/{job.id}/shortlist").json()["job"]["status"] == JobStatus.SHORTLISTED
    assert http.post(f"/jobs/{job.id}/dismiss").json()["job"]["status"] == JobStatus.DISMISSED
    assert http.post(f"/jobs/{job.id}/applied").json()["job"]["status"] == JobStatus.APPLIED
    feedback = http.post(
        f"/jobs/{job.id}/feedback",
        json={"label": FeedbackLabel.STRONG_FIT.value},
    )
    assert feedback.status_code == 201
    explanation = http.post(f"/jobs/{job.id}/explanation", json={})
    assert explanation.status_code == 200
    assert explanation.json()["experience_alignment"] == "Evidence is available."
