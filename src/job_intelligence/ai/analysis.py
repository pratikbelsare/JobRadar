"""Conversion from provider output to the existing persisted analysis model."""

from __future__ import annotations

from uuid import UUID

from ..models import JobAnalysis
from .models import AIResult, JobRequirements


def to_job_analysis(
    result: AIResult[JobRequirements],
    *,
    job_id: UUID,
    job_version_id: UUID,
) -> JobAnalysis:
    """Create a traceable domain analysis without adding suitability decisions."""

    requirements = result.value
    metadata = result.metadata
    usage = {
        name: value
        for name, value in {
            "input_tokens": metadata.input_tokens,
            "output_tokens": metadata.output_tokens,
        }.items()
        if value is not None
    }
    return JobAnalysis(
        job_id=job_id,
        job_version_id=job_version_id,
        model_id=metadata.model_id,
        prompt_version=metadata.prompt_version or "unknown",
        required_skills=requirements.required_skills,
        preferred_skills=requirements.preferred_skills,
        min_experience_years=requirements.min_experience_years,
        max_experience_years=requirements.max_experience_years,
        seniority=requirements.seniority,
        location_constraints=requirements.location_constraints,
        work_modes=requirements.work_modes,
        employment_type=requirements.employment_type,
        responsibilities=requirements.responsibilities,
        education=requirements.education,
        domain=requirements.domain,
        provider=metadata.provider,
        status=metadata.status,
        error=metadata.error,
        usage=usage,
    )


__all__ = ["to_job_analysis"]
