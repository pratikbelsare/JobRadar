from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from job_intelligence.models import (
    CandidatePreferences,
    CandidateProfile,
    Company,
    FeedbackLabel,
    Job,
    JobAnalysis,
    JobMatch,
    JobVersion,
    ScanRun,
    UserFeedback,
    WorkExperience,
    WorkMode,
)


def test_core_models_construct_with_valid_values() -> None:
    company = Company(name="Example Co", career_url="https://example.com/careers")
    profile = CandidateProfile(
        target_roles=["Machine Learning Engineer"],
        experience_years=2.5,
        skills=["Python"],
        preferences=CandidatePreferences(preferred_work_modes=[WorkMode.REMOTE]),
    )
    job = Job(
        source_job_id="job-123",
        company_id=company.id,
        title="ML Engineer",
        location="Bengaluru",
        description="Build machine learning systems.",
        source_url="https://example.com/jobs/job-123",
        min_experience_years=1,
        max_experience_years=4,
    )
    version = JobVersion(
        job_id=job.id,
        version_number=1,
        content_hash="abc123",
        description=job.description,
    )
    analysis = JobAnalysis(
        job_id=job.id,
        job_version_id=version.id,
        model_id="provider/model-id",
        prompt_version="job-analysis-v1",
    )
    match = JobMatch(
        job_id=job.id,
        candidate_profile_id=profile.id,
        role_match=0.9,
        skill_match=0.8,
        experience_match=0.7,
        responsibility_match=0.6,
        domain_match=0.5,
        location_match=0.4,
        preference_match=0.3,
        final_score=0.7,
        ranking_version="baseline-v1",
    )
    feedback = UserFeedback(
        job_id=job.id,
        candidate_profile_id=profile.id,
        label=FeedbackLabel.WORTH_REVIEWING,
    )
    run = ScanRun(companies_requested=1, companies_succeeded=1)

    assert company.enabled
    assert profile.experience_years == 2.5
    assert profile.preferences.preferred_work_modes == [WorkMode.REMOTE]
    assert job.source_job_id == "job-123"
    assert version.version_number == 1
    assert analysis.model_id == "provider/model-id"
    assert match.final_score == 0.7
    assert feedback.label == FeedbackLabel.WORTH_REVIEWING
    assert run.companies_succeeded == 1


def test_company_rejects_blank_name_and_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Company(name="  ", career_url="https://example.com/careers")

    with pytest.raises(ValidationError):
        Company(name="Example Co", career_url="not-a-url")


def test_candidate_rejects_negative_experience() -> None:
    with pytest.raises(ValidationError):
        CandidateProfile(experience_years=-0.1)


def test_job_rejects_inverted_experience_range_and_dates() -> None:
    now = datetime.now()
    with pytest.raises(ValidationError):
        Job(
            source_job_id="job-123",
            company_id=uuid4(),
            title="Engineer",
            location="Remote",
            description="A role.",
            source_url="https://example.com/jobs/job-123",
            min_experience_years=5,
            max_experience_years=2,
        )

    with pytest.raises(ValidationError):
        Job(
            source_job_id="job-123",
            company_id=uuid4(),
            title="Engineer",
            location="Remote",
            description="A role.",
            source_url="https://example.com/jobs/job-123",
            first_seen_at=now,
            last_seen_at=now - timedelta(seconds=1),
        )


def test_other_models_reject_invalid_ranges_and_counts() -> None:
    with pytest.raises(ValidationError):
        JobMatch(
            job_id=uuid4(),
            candidate_profile_id=uuid4(),
            role_match=1.1,
            skill_match=0.0,
            experience_match=0.0,
            responsibility_match=0.0,
            domain_match=0.0,
            location_match=0.0,
            preference_match=0.0,
            final_score=0.0,
            ranking_version="v1",
        )

    with pytest.raises(ValidationError):
        ScanRun(companies_requested=1, companies_failed=2)

    with pytest.raises(ValidationError):
        WorkExperience(
            employer="Example Co",
            role="Engineer",
            start_date="2024-01-01",
            end_date="2023-01-01",
        )
