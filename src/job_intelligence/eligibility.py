"""Pure post-analysis eligibility evaluation.

This module deliberately does not classify role families or calculate ranking scores.
It evaluates structured job requirements after analysis and is not wired into the
scanner or analysis worker yet.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import Field, NonNegativeFloat, model_validator

from .config import Settings
from .models import CandidateProfile, EmploymentType, Job, JobAnalysis, Model, Seniority, WorkMode
from .normalization import normalize_text
from .role_policy import RoleFamilyClass


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    STRETCH = "stretch"
    SEVERE_STRETCH = "severe_stretch"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class EligibilityExperienceStatus(str, Enum):
    ALIGNED = "aligned"
    STRETCH = "stretch"
    SEVERE_STRETCH = "severe_stretch"
    UNKNOWN = "unknown"


class SeniorityStatus(str, Enum):
    ALIGNED = "aligned"
    ELEVATED = "elevated"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class EligibilityConfig(Model):
    """Thresholds and conservative hard-constraint policy for Stage 2."""

    acceptable_gap_years: NonNegativeFloat
    max_stretch_gap_years: NonNegativeFloat
    seniority_min_experience_years: dict[Seniority, NonNegativeFloat] = Field(
        default_factory=dict
    )
    elevated_seniority_levels: list[Seniority] = Field(
        default_factory=lambda: [Seniority.SENIOR, Seniority.LEAD, Seniority.MANAGER]
    )
    severe_seniority_levels: list[Seniority] = Field(
        default_factory=lambda: [Seniority.PRINCIPAL, Seniority.DIRECTOR]
    )
    location_aliases: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gap_order(self) -> EligibilityConfig:
        if self.max_stretch_gap_years < self.acceptable_gap_years:
            raise ValueError("max_stretch_gap_years must not be below acceptable_gap_years")
        overlap = set(self.elevated_seniority_levels) & set(self.severe_seniority_levels)
        if overlap:
            raise ValueError(f"seniority levels cannot be both elevated and severe: {overlap}")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> EligibilityConfig:
        acceptable = settings.filter_acceptable_experience_gap_years
        maximum = settings.filter_max_stretch_experience_gap_years
        if acceptable is None or maximum is None:
            raise ValueError("Eligibility experience thresholds must be configured")
        return cls(
            acceptable_gap_years=acceptable,
            max_stretch_gap_years=maximum,
        )


class EligibilityResult(Model):
    """Explainable Stage-2 output; no final ranking score is included."""

    status: EligibilityStatus
    role_family_class: RoleFamilyClass | None = None
    experience_gap_years: NonNegativeFloat | None = None
    experience_status: EligibilityExperienceStatus
    seniority_status: SeniorityStatus
    reason_codes: list[str] = Field(default_factory=list)
    hard_block_reasons: list[str] = Field(default_factory=list)
    visible: bool
    recommended_to_retain: bool
    details: dict[str, Any] = Field(default_factory=dict)


class EligibilityEvaluator:
    """Evaluate structured requirements independently from role filtering and ranking."""

    def __init__(self, config: EligibilityConfig) -> None:
        self._config = config

    def evaluate(
        self,
        profile: CandidateProfile,
        job: Job,
        analysis: JobAnalysis,
        *,
        role_family_class: RoleFamilyClass | None = None,
    ) -> EligibilityResult:
        experience_status, gap = self._experience(profile, analysis)
        seniority_status = self._seniority(profile, analysis)
        hard_blocks, hard_reasons = self._hard_blockers(profile, job, analysis)
        reason_codes = [*hard_reasons]

        if hard_blocks:
            status = EligibilityStatus.INELIGIBLE
            reason_codes.extend(hard_blocks)
        else:
            status = _combine_status(
                experience_status,
                seniority_status,
                analysis.seniority,
                self._config,
            )
            reason_codes.extend(
                _experience_reasons(experience_status, seniority_status, analysis.seniority)
            )
            if profile.hard_constraints.work_authorization_required:
                reason_codes.append("authorization_unknown")
            if analysis.education and not _education_is_evidenced(profile, analysis):
                reason_codes.append("education_not_verifiable_as_mandatory")
            if analysis.required_skills:
                reason_codes.append("skills_are_not_hard_blockers")

        visible = not hard_blocks
        return EligibilityResult(
            status=status,
            role_family_class=role_family_class,
            experience_gap_years=gap,
            experience_status=experience_status,
            seniority_status=seniority_status,
            reason_codes=_deduplicate(reason_codes),
            hard_block_reasons=hard_blocks,
            visible=visible,
            recommended_to_retain=visible,
            details={
                "candidate_experience_years": profile.experience_years,
                "minimum_experience_years": analysis.min_experience_years,
                "maximum_experience_years": analysis.max_experience_years,
                "seniority": analysis.seniority.value,
                "employment_type": analysis.employment_type.value,
            },
        )

    def _experience(
        self,
        profile: CandidateProfile,
        analysis: JobAnalysis,
    ) -> tuple[EligibilityExperienceStatus, float | None]:
        minimum = analysis.min_experience_years
        maximum = analysis.max_experience_years
        if minimum is None and maximum is None:
            return EligibilityExperienceStatus.UNKNOWN, None
        if minimum is not None and profile.experience_years < minimum:
            gap = minimum - profile.experience_years
            if gap <= self._config.acceptable_gap_years:
                return EligibilityExperienceStatus.ALIGNED, gap
            if gap <= self._config.max_stretch_gap_years:
                return EligibilityExperienceStatus.STRETCH, gap
            return EligibilityExperienceStatus.SEVERE_STRETCH, gap
        if maximum is not None and profile.experience_years > maximum:
            return EligibilityExperienceStatus.ALIGNED, profile.experience_years - maximum
        return EligibilityExperienceStatus.ALIGNED, 0.0

    def _seniority(
        self,
        profile: CandidateProfile,
        analysis: JobAnalysis,
    ) -> SeniorityStatus:
        seniority = analysis.seniority
        if seniority == Seniority.UNKNOWN:
            return SeniorityStatus.UNKNOWN
        minimum = self._config.seniority_min_experience_years.get(seniority)
        if minimum is not None and profile.experience_years < minimum:
            return SeniorityStatus.MISMATCH
        if seniority in self._config.severe_seniority_levels:
            return SeniorityStatus.ELEVATED
        if seniority in self._config.elevated_seniority_levels:
            return SeniorityStatus.ELEVATED
        return SeniorityStatus.ALIGNED

    def _hard_blockers(
        self,
        profile: CandidateProfile,
        job: Job,
        analysis: JobAnalysis,
    ) -> tuple[list[str], list[str]]:
        blockers: list[str] = []
        reasons: list[str] = []
        constraints = profile.hard_constraints

        if (
            analysis.employment_type != EmploymentType.UNKNOWN
            and analysis.employment_type in constraints.excluded_employment_types
        ):
            blockers.append("employment_type_excluded")
            reasons.append("hard_employment_type_mismatch")

        modes = [mode for mode in analysis.work_modes if mode != WorkMode.UNKNOWN]
        if not modes and job.work_mode != WorkMode.UNKNOWN:
            modes = [job.work_mode]
        if constraints.remote_only and modes and WorkMode.REMOTE not in modes:
            blockers.append("remote_only_mismatch")
            reasons.append("hard_remote_only_mismatch")

        locations = [*analysis.location_constraints]
        if job.location is not None:
            locations.append(job.location)
        excluded = _first_location_match(locations, constraints.excluded_locations, self._config)
        if excluded is not None:
            blockers.append("excluded_location")
            reasons.append("hard_location_mismatch")

        searchable = _key(" ".join(filter(None, [job.title, job.location, job.description])))
        excluded_term = _first_term_match(searchable, constraints.excluded_terms)
        if excluded_term is not None:
            blockers.append("excluded_term")
            reasons.append("hard_excluded_term")
        return _deduplicate(blockers), _deduplicate(reasons)


def _combine_status(
    experience: EligibilityExperienceStatus,
    seniority: SeniorityStatus,
    job_seniority: Seniority,
    config: EligibilityConfig,
) -> EligibilityStatus:
    if experience == EligibilityExperienceStatus.SEVERE_STRETCH:
        return EligibilityStatus.SEVERE_STRETCH
    if experience == EligibilityExperienceStatus.STRETCH:
        if job_seniority in config.severe_seniority_levels:
            return EligibilityStatus.SEVERE_STRETCH
        return EligibilityStatus.STRETCH
    if experience == EligibilityExperienceStatus.UNKNOWN:
        if job_seniority in config.severe_seniority_levels:
            return EligibilityStatus.SEVERE_STRETCH
        if seniority in {SeniorityStatus.ELEVATED, SeniorityStatus.MISMATCH}:
            return EligibilityStatus.STRETCH
        return EligibilityStatus.UNKNOWN
    if job_seniority in config.severe_seniority_levels:
        return EligibilityStatus.SEVERE_STRETCH
    if seniority in {SeniorityStatus.ELEVATED, SeniorityStatus.MISMATCH}:
        return EligibilityStatus.STRETCH
    return EligibilityStatus.ELIGIBLE


def _experience_reasons(
    experience: EligibilityExperienceStatus,
    seniority: SeniorityStatus,
    job_seniority: Seniority,
) -> list[str]:
    reasons = {
        EligibilityExperienceStatus.ALIGNED: "experience_aligned",
        EligibilityExperienceStatus.STRETCH: "experience_stretch",
        EligibilityExperienceStatus.SEVERE_STRETCH: "experience_severe_stretch",
        EligibilityExperienceStatus.UNKNOWN: "experience_unknown",
    }
    seniority_reason = {
        SeniorityStatus.ALIGNED: "seniority_aligned",
        SeniorityStatus.ELEVATED: "seniority_elevated",
        SeniorityStatus.MISMATCH: "seniority_mismatch",
        SeniorityStatus.UNKNOWN: "seniority_unknown",
    }
    return [reasons[experience], seniority_reason[seniority]]


def _education_is_evidenced(profile: CandidateProfile, analysis: JobAnalysis) -> bool:
    candidate_text = " ".join(
        [
            *[education.degree for education in profile.education],
            *[education.field_of_study or "" for education in profile.education],
        ]
    )
    return any(_key(item) in _key(candidate_text) for item in analysis.education)


def _first_location_match(
    locations: list[str],
    excluded: list[str],
    config: EligibilityConfig,
) -> str | None:
    for excluded_location in excluded:
        aliases = config.location_aliases.get(excluded_location, [])
        terms = [_key(excluded_location), *(_key(item) for item in aliases)]
        if any(_first_term_match(_key(location), terms) is not None for location in locations):
            return excluded_location
    return None


def _first_term_match(text: str, terms: list[str]) -> str | None:
    for term in terms:
        normalized = _key(term)
        if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text):
            return term
    return None


def _key(value: str) -> str:
    return normalize_text(value).casefold()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "EligibilityConfig",
    "EligibilityEvaluator",
    "EligibilityExperienceStatus",
    "EligibilityResult",
    "EligibilityStatus",
    "SeniorityStatus",
]
