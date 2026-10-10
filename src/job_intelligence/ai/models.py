"""Validated provider-independent schemas for AI results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import (
    Field,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from ..models import (
    Education,
    EmploymentType,
    Evidence,
    Model,
    NonEmptyText,
    Project,
    Seniority,
    WorkExperience,
    WorkMode,
)

T = TypeVar("T")


class AIResponseMetadata(Model):
    provider: NonEmptyText
    model_id: NonEmptyText
    prompt_version: NonEmptyText | None = None
    status: NonEmptyText = "succeeded"
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: NonEmptyText | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None


class AIResult(Model, Generic[T]):
    value: T
    metadata: AIResponseMetadata


class ExplanationSource(str, Enum):
    JOB = "job"
    CANDIDATE = "candidate"
    MATCH = "match"


class ExplanationReference(Model):
    source: ExplanationSource
    claim: NonEmptyText
    evidence: NonEmptyText
    reference_id: NonEmptyText | None = None


class JobMatchExplanation(Model):
    why_fit: list[NonEmptyText] = Field(default_factory=list)
    strongest_matches: list[NonEmptyText] = Field(default_factory=list)
    potential_gaps: list[NonEmptyText] = Field(default_factory=list)
    experience_alignment: NonEmptyText
    important_considerations: list[NonEmptyText] = Field(default_factory=list)
    evidence_references: list[ExplanationReference] = Field(default_factory=list)


class ExplanationContext(Model):
    job_id: UUID
    candidate_profile_id: UUID
    job_title: NonEmptyText
    job_requirements: dict[str, object] = Field(default_factory=dict)
    candidate_evidence: dict[str, object] = Field(default_factory=dict)
    match_evidence: dict[str, object] = Field(default_factory=dict)


class JobRequirements(Model):
    """Facts extracted from a job description; suitability is not included."""

    min_experience_years: NonNegativeFloat | None = None
    max_experience_years: NonNegativeFloat | None = None
    seniority: Seniority = Seniority.UNKNOWN
    required_skills: list[NonEmptyText] = Field(default_factory=list)
    preferred_skills: list[NonEmptyText] = Field(default_factory=list)
    responsibilities: list[NonEmptyText] = Field(default_factory=list)
    education: list[NonEmptyText] = Field(default_factory=list)
    location_constraints: list[NonEmptyText] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    domain: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_experience_range(self) -> JobRequirements:
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.max_experience_years < self.min_experience_years
        ):
            raise ValueError("max_experience_years must not be below min_experience_years")
        return self


class CandidateProfileProposal(Model):
    """Optional resume-derived values that may be proposed to a profile."""

    target_roles: list[NonEmptyText] | None = None
    experience_years: NonNegativeFloat | None = None
    preferred_locations: list[NonEmptyText] | None = None
    work_mode: WorkMode | None = None
    skills: list[NonEmptyText] | None = None
    skill_evidence: dict[str, list[Evidence]] | None = None
    work_experience: list[WorkExperience] | None = None
    projects: list[Project] | None = None
    education: list[Education] | None = None
    domains: list[NonEmptyText] | None = None


class EmbeddingVector(Model):
    values: list[FiniteFloat] = Field(min_length=1)

    @property
    def dimensions(self) -> PositiveInt:
        return len(self.values)


__all__ = [
    "AIResponseMetadata",
    "AIResult",
    "CandidateProfileProposal",
    "ExplanationContext",
    "ExplanationReference",
    "ExplanationSource",
    "EmbeddingVector",
    "JobRequirements",
    "JobMatchExplanation",
]
