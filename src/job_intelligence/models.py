"""Validated domain models for the first implementation milestone."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum, IntEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class Seniority(str, Enum):
    INTERN = "intern"
    ENTRY = "entry"
    ASSOCIATE = "associate"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    DIRECTOR = "director"
    UNKNOWN = "unknown"


class ConnectorType(str, Enum):
    GENERIC = "generic"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    CUSTOM = "custom"


class JobStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    SHORTLISTED = "shortlisted"
    MAYBE = "maybe"
    APPLIED = "applied"
    DISMISSED = "dismissed"


class FeedbackLabel(IntEnum):
    IRRELEVANT = 0
    WEAK_FIT = 1
    WORTH_REVIEWING = 2
    STRONG_FIT = 3


class FeedbackReason(str, Enum):
    WRONG_ROLE = "wrong_role"
    WRONG_LOCATION = "wrong_location"
    TOO_SENIOR = "too_senior"
    SKILL_MISMATCH = "skill_mismatch"
    STRONG_FIT = "strong_fit"
    OTHER = "other"


class ScanRunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class Model(BaseModel):
    """Shared model configuration for predictable serialization and assignment checks."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        protected_namespaces=(),
    )


class Evidence(Model):
    text: NonEmptyText
    source: NonEmptyText | None = None


class WorkExperience(Model):
    employer: NonEmptyText
    role: NonEmptyText
    start_date: date | None = None
    end_date: date | None = None
    description: NonEmptyText | None = None
    skills: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dates(self) -> WorkExperience:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


class Project(Model):
    name: NonEmptyText
    description: NonEmptyText
    skills: list[NonEmptyText] = Field(default_factory=list)


class Education(Model):
    institution: NonEmptyText
    degree: NonEmptyText
    field_of_study: NonEmptyText | None = None
    completion_year: int | None = Field(default=None, ge=1900, le=2200)


class CandidatePreferences(Model):
    preferred_employment_types: list[EmploymentType] = Field(default_factory=list)
    preferred_domains: list[NonEmptyText] = Field(default_factory=list)


class HardConstraints(Model):
    excluded_locations: list[NonEmptyText] = Field(default_factory=list)
    excluded_employment_types: list[EmploymentType] = Field(default_factory=list)
    work_authorization_required: bool | None = None


class Company(Model):
    id: UUID = Field(default_factory=uuid4)
    name: NonEmptyText
    career_url: AnyHttpUrl
    connector_type: ConnectorType = ConnectorType.GENERIC
    enabled: bool = True
    last_checked_at: datetime | None = None
    last_success_at: datetime | None = None
    failure_count: NonNegativeInt = 0


class CandidateProfile(Model):
    id: UUID = Field(default_factory=uuid4)
    target_roles: list[NonEmptyText] = Field(default_factory=list)
    experience_years: NonNegativeFloat = 0
    preferred_locations: list[NonEmptyText] = Field(default_factory=list)
    work_mode: WorkMode | None = None
    skills: list[NonEmptyText] = Field(default_factory=list)
    skill_evidence: dict[str, list[Evidence]] = Field(default_factory=dict)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    domains: list[NonEmptyText] = Field(default_factory=list)
    preferences: CandidatePreferences = Field(default_factory=CandidatePreferences)
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)


class Job(Model):
    id: UUID = Field(default_factory=uuid4)
    source_job_id: NonEmptyText
    company_id: UUID
    title: NonEmptyText
    location: NonEmptyText
    work_mode: WorkMode = WorkMode.UNKNOWN
    description: NonEmptyText
    min_experience_years: NonNegativeFloat | None = None
    max_experience_years: NonNegativeFloat | None = None
    seniority: Seniority = Seniority.UNKNOWN
    required_skills: list[NonEmptyText] = Field(default_factory=list)
    preferred_skills: list[NonEmptyText] = Field(default_factory=list)
    responsibilities: list[NonEmptyText] = Field(default_factory=list)
    education: list[NonEmptyText] = Field(default_factory=list)
    domain: NonEmptyText | None = None
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    posted_at: datetime | None = None
    source_url: AnyHttpUrl
    first_seen_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: NonEmptyText | None = None
    status: JobStatus = JobStatus.NEW

    @model_validator(mode="after")
    def validate_experience_range(self) -> Job:
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.max_experience_years < self.min_experience_years
        ):
            raise ValueError("max_experience_years must not be below min_experience_years")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at must not be before first_seen_at")
        return self


class JobVersion(Model):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    version_number: PositiveInt
    content_hash: NonEmptyText
    description: NonEmptyText
    captured_at: datetime = Field(default_factory=datetime.utcnow)
    is_current: bool = True


class JobAnalysis(Model):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    job_version_id: UUID
    model_id: NonEmptyText
    prompt_version: NonEmptyText
    required_skills: list[NonEmptyText] = Field(default_factory=list)
    preferred_skills: list[NonEmptyText] = Field(default_factory=list)
    min_experience_years: NonNegativeFloat | None = None
    max_experience_years: NonNegativeFloat | None = None
    seniority: Seniority = Seniority.UNKNOWN
    location_constraints: list[NonEmptyText] = Field(default_factory=list)
    responsibilities: list[NonEmptyText] = Field(default_factory=list)
    education: list[NonEmptyText] = Field(default_factory=list)
    domain: NonEmptyText | None = None
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_experience_range(self) -> JobAnalysis:
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.max_experience_years < self.min_experience_years
        ):
            raise ValueError("max_experience_years must not be below min_experience_years")
        return self


class JobMatch(Model):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    candidate_profile_id: UUID
    role_match: Score
    skill_match: Score
    experience_match: Score
    responsibility_match: Score
    domain_match: Score
    location_match: Score
    preference_match: Score
    final_score: Score
    ranking_version: NonEmptyText
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserFeedback(Model):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    candidate_profile_id: UUID
    label: FeedbackLabel
    reason: FeedbackReason | None = None
    notes: NonEmptyText | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ScanRun(Model):
    id: UUID = Field(default_factory=uuid4)
    status: ScanRunStatus = ScanRunStatus.CREATED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    companies_requested: NonNegativeInt = 0
    companies_succeeded: NonNegativeInt = 0
    companies_failed: NonNegativeInt = 0
    jobs_discovered: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_run(self) -> ScanRun:
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        if self.companies_succeeded + self.companies_failed > self.companies_requested:
            raise ValueError("company outcomes cannot exceed companies_requested")
        return self


__all__ = [
    "CandidatePreferences",
    "CandidateProfile",
    "Company",
    "ConnectorType",
    "Education",
    "EmploymentType",
    "Evidence",
    "FeedbackLabel",
    "FeedbackReason",
    "HardConstraints",
    "Job",
    "JobAnalysis",
    "JobMatch",
    "JobStatus",
    "JobVersion",
    "Model",
    "NonEmptyText",
    "Project",
    "ScanRun",
    "ScanRunStatus",
    "Score",
    "Seniority",
    "UserFeedback",
    "WorkExperience",
    "WorkMode",
]
