"""On-demand, evidence-grounded job-match explanations."""

from __future__ import annotations

import json
import re

from .ai import (
    AIProvider,
    AIProviderError,
    AIResult,
    ExplanationContext,
    ExplanationSource,
    JobMatchExplanation,
)
from .models import CandidateProfile, Job, JobAnalysis, JobMatch
from .normalization import normalize_text


class ExplanationGenerationError(RuntimeError):
    """Raised when an explanation provider call fails."""


class ExplanationGroundingError(ValueError):
    """Raised when an explanation reference is not supported by supplied evidence."""


class ExplanationService:
    """Generate explanations only when explicitly requested by a caller."""

    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    def generate(
        self,
        job: Job,
        profile: CandidateProfile,
        match: JobMatch,
        *,
        analysis: JobAnalysis | None = None,
    ) -> AIResult[JobMatchExplanation]:
        context = build_explanation_context(job, profile, match, analysis=analysis)
        try:
            result = self._provider.generate_explanation(context)
        except AIProviderError as exc:
            raise ExplanationGenerationError("Could not generate job explanation") from exc
        validate_explanation_grounding(result.value, context)
        return result


def build_explanation_context(
    job: Job,
    profile: CandidateProfile,
    match: JobMatch,
    *,
    analysis: JobAnalysis | None = None,
) -> ExplanationContext:
    """Build the bounded evidence packet given to an explanation provider."""

    job_requirements = (
        analysis.model_dump(mode="json")
        if analysis is not None
        else {
            "title": job.title,
            "description": job.description,
            "location": job.location,
            "work_mode": job.work_mode.value,
            "required_skills": job.required_skills,
            "preferred_skills": job.preferred_skills,
            "responsibilities": job.responsibilities,
            "min_experience_years": job.min_experience_years,
            "max_experience_years": job.max_experience_years,
            "domain": job.domain,
        }
    )
    candidate_evidence = {
        "target_roles": profile.target_roles,
        "experience_years": profile.experience_years,
        "skills": profile.skills,
        "skill_evidence": profile.skill_evidence,
        "work_experience": profile.work_experience,
        "projects": profile.projects,
        "education": profile.education,
        "domains": profile.domains,
    }
    return ExplanationContext(
        job_id=job.id,
        candidate_profile_id=profile.id,
        job_title=job.title,
        job_requirements=job_requirements,
        candidate_evidence=candidate_evidence,
        match_evidence=match.model_dump(mode="json"),
    )


def validate_explanation_grounding(
    explanation: JobMatchExplanation,
    context: ExplanationContext,
) -> None:
    """Validate references and reject unsupported claims about missing skills."""

    all_text = [
        *explanation.why_fit,
        *explanation.strongest_matches,
        *explanation.potential_gaps,
        explanation.experience_alignment,
        *explanation.important_considerations,
    ]
    forbidden_absence_claim = re.compile(
        r"\b(?:candidate|applicant|profile)\b.*\b(?:lacks|does not know|has no)\b",
        re.IGNORECASE,
    )
    if any(forbidden_absence_claim.search(text) for text in all_text):
        raise ExplanationGroundingError(
            "Explanations must say 'no evidence found' instead of asserting a skill is absent"
        )

    evidence_by_source = {
        ExplanationSource.JOB: _serialized(context.job_requirements),
        ExplanationSource.CANDIDATE: _serialized(context.candidate_evidence),
        ExplanationSource.MATCH: _serialized(context.match_evidence),
    }
    for reference in explanation.evidence_references:
        evidence = normalize_text(reference.evidence).casefold()
        if evidence not in evidence_by_source[reference.source]:
            raise ExplanationGroundingError(
                f"Explanation evidence is not present in the {reference.source.value} context"
            )


def _serialized(value: dict[str, object]) -> str:
    return normalize_text(json.dumps(value, sort_keys=True, default=str)).casefold()


__all__ = [
    "ExplanationGenerationError",
    "ExplanationGroundingError",
    "ExplanationService",
    "build_explanation_context",
    "validate_explanation_grounding",
]
