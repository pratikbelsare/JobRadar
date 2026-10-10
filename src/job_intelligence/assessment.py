"""Additive relevance, eligibility, and role-family assessment orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .eligibility import EligibilityEvaluator, EligibilityResult
from .filtering import FilterResult
from .matching import HybridJobMatcher
from .models import CandidateProfile, EligibilitySummary, Job, JobAnalysis, JobMatch
from .role_policy import RoleFamilyClassifier, RoleFamilyMatch


@dataclass(frozen=True)
class JobAssessment:
    """The three independent dimensions attached to one evaluated job."""

    match: JobMatch
    eligibility: EligibilityResult
    role_family: RoleFamilyMatch


class JobAssessmentService:
    """Combine existing match, eligibility, and role-family services without new scoring."""

    def __init__(
        self,
        *,
        matcher: HybridJobMatcher,
        eligibility: EligibilityEvaluator,
        role_families: RoleFamilyClassifier,
    ) -> None:
        self._matcher = matcher
        self._eligibility = eligibility
        self._role_families = role_families

    def evaluate(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis,
        *,
        filter_result: FilterResult | None = None,
    ) -> JobAssessment:
        role_family = self._role_families.classify(job.title)
        eligibility = self._eligibility.evaluate(
            profile,
            job,
            analysis,
            role_family_class=role_family.classification,
        )
        match = self._matcher.match(
            job,
            profile,
            analysis=analysis,
            filter_result=filter_result,
        )
        role_family_class = (
            role_family.classification.value
            if role_family.classification is not None
            else "unknown"
        )
        enriched_match = match.model_copy(
            update={
                "eligibility": _eligibility_summary(eligibility),
                "role_family_class": role_family_class,
            }
        )
        return JobAssessment(
            match=enriched_match,
            eligibility=eligibility,
            role_family=role_family,
        )


def _eligibility_summary(result: EligibilityResult) -> EligibilitySummary:
    return EligibilitySummary(
        status=result.status.value,
        experience_gap_years=result.experience_gap_years,
        experience_status=result.experience_status.value,
        seniority_status=result.seniority_status.value,
        hard_block_reasons=result.hard_block_reasons,
        reason_codes=result.reason_codes,
        visible=result.visible,
        recommended_to_retain=result.recommended_to_retain,
    )


__all__ = ["JobAssessment", "JobAssessmentService"]
