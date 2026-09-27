"""Pure, deterministic filtering of normalized jobs against candidate preferences."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import Field, NonNegativeFloat, model_validator

from .config import Settings
from .models import CandidateProfile, EmploymentType, Job, Model, WorkMode
from .normalization import normalize_text


class FilterOutcome(str, Enum):
    PASS = "pass"
    STRETCH = "stretch"
    REJECT_ROLE = "reject_role"
    REJECT_LOCATION = "reject_location"
    REJECT_EXPERIENCE = "reject_experience"
    REJECT_CONSTRAINT = "reject_constraint"


class FilterRule(str, Enum):
    ROLE = "role"
    LOCATION = "location"
    EXPERIENCE = "experience"
    CONSTRAINT = "constraint"


class RuleStatus(str, Enum):
    PASS = "pass"
    ACCEPTABLE = "acceptable"
    STRETCH = "stretch"
    REJECT = "reject"


class ExperienceFilterConfig(Model):
    """Configurable gaps used to classify under-qualified roles."""

    acceptable_gap_years: NonNegativeFloat
    max_stretch_gap_years: NonNegativeFloat

    @model_validator(mode="after")
    def validate_gap_order(self) -> ExperienceFilterConfig:
        if self.max_stretch_gap_years < self.acceptable_gap_years:
            raise ValueError("max_stretch_gap_years must not be below acceptable_gap_years")
        return self


class FilteringConfig(Model):
    """Role/location aliases and thresholds supplied by the application."""

    experience: ExperienceFilterConfig
    role_aliases: dict[str, list[str]] = Field(default_factory=dict)
    location_aliases: dict[str, list[str]] = Field(default_factory=dict)
    excluded_role_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        role_aliases: dict[str, list[str]] | None = None,
        location_aliases: dict[str, list[str]] | None = None,
        excluded_role_terms: list[str] | None = None,
        excluded_terms: list[str] | None = None,
    ) -> FilteringConfig:
        acceptable_gap = settings.filter_acceptable_experience_gap_years
        max_stretch_gap = settings.filter_max_stretch_experience_gap_years
        if (
            acceptable_gap is None
            or max_stretch_gap is None
        ):
            raise ValueError(
                "Filtering experience thresholds must be configured before filtering"
            )
        return cls(
            experience=ExperienceFilterConfig(
                acceptable_gap_years=acceptable_gap,
                max_stretch_gap_years=max_stretch_gap,
            ),
            role_aliases=role_aliases or {},
            location_aliases=location_aliases or {},
            excluded_role_terms=excluded_role_terms or [],
            excluded_terms=excluded_terms or [],
        )


class FilterRuleResult(Model):
    rule: FilterRule
    status: RuleStatus
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class FilterResult(Model):
    passed: bool
    outcome: FilterOutcome
    reasons: list[str] = Field(default_factory=list)
    rules: list[FilterRuleResult] = Field(default_factory=list)


class DeterministicJobFilter:
    """Run role, location, experience, then hard-constraint rules in order."""

    def __init__(self, config: FilteringConfig) -> None:
        self._config = config

    def evaluate(self, job: Job, profile: CandidateProfile) -> FilterResult:
        rules: list[FilterRuleResult] = []

        role_result = self._evaluate_role(job, profile)
        rules.append(role_result)
        if not role_result.passed:
            return _rejected(FilterOutcome.REJECT_ROLE, rules)

        location_result = self._evaluate_location(job, profile)
        rules.append(location_result)
        if not location_result.passed:
            return _rejected(FilterOutcome.REJECT_LOCATION, rules)

        experience_result = self._evaluate_experience(job, profile)
        rules.append(experience_result)
        if not experience_result.passed:
            return _rejected(FilterOutcome.REJECT_EXPERIENCE, rules)

        constraint_result = self._evaluate_constraints(job, profile)
        rules.append(constraint_result)
        if not constraint_result.passed:
            return _rejected(FilterOutcome.REJECT_CONSTRAINT, rules)

        outcome = (
            FilterOutcome.STRETCH
            if experience_result.status == RuleStatus.STRETCH
            else FilterOutcome.PASS
        )
        return FilterResult(
            passed=True,
            outcome=outcome,
            reasons=[rule.reason for rule in rules],
            rules=rules,
        )

    def _evaluate_role(self, job: Job, profile: CandidateProfile) -> FilterRuleResult:
        title = _key(job.title)
        excluded_terms = _terms(self._config.excluded_role_terms)
        matched_exclusion = _first_term_match(title, excluded_terms)
        if matched_exclusion:
            return _rule(
                FilterRule.ROLE,
                RuleStatus.REJECT,
                False,
                f"Job title contains excluded role term '{matched_exclusion}'.",
                title=job.title,
                excluded_term=matched_exclusion,
            )

        if not profile.target_roles:
            return _rule(
                FilterRule.ROLE,
                RuleStatus.PASS,
                True,
                "No target roles are configured; role filtering is not restrictive.",
                title=job.title,
            )

        aliases = _normalized_aliases(self._config.role_aliases)
        matched_role: str | None = None
        matched_term: str | None = None
        for target_role in profile.target_roles:
            target = _key(target_role)
            terms = [target, *aliases.get(target, [])]
            matched_term = _first_term_match(title, terms)
            if matched_term:
                matched_role = target_role
                break
        if matched_role is not None and matched_term is not None:
            return _rule(
                FilterRule.ROLE,
                RuleStatus.PASS,
                True,
                f"Job title matches configured target role '{matched_role}'.",
                title=job.title,
                matched_term=matched_term,
            )
        return _rule(
            FilterRule.ROLE,
            RuleStatus.REJECT,
            False,
            "Job title does not match any configured target role or alias.",
            title=job.title,
            target_roles=list(profile.target_roles),
        )

    def _evaluate_location(self, job: Job, profile: CandidateProfile) -> FilterRuleResult:
        location = _key(job.location)
        excluded_locations = [*profile.hard_constraints.excluded_locations]
        matched_exclusion = _first_location_match(
            location,
            excluded_locations,
            self._config.location_aliases,
        )
        if matched_exclusion:
            return _rule(
                FilterRule.LOCATION,
                RuleStatus.REJECT,
                False,
                f"Job location matches excluded location '{matched_exclusion}'.",
                job_location=job.location,
                excluded_location=matched_exclusion,
            )

        preferred_locations = profile.preferred_locations
        preferred_modes = _preferred_work_modes(profile)
        remote_job = _is_remote(job)
        if not preferred_locations and not preferred_modes:
            return _rule(
                FilterRule.LOCATION,
                RuleStatus.PASS,
                True,
                "No location or work-mode preference is configured.",
                job_location=job.location,
            )
        location_match = _first_location_match(
            location,
            preferred_locations,
            self._config.location_aliases,
        )
        mode_match = bool(preferred_modes and job.work_mode in preferred_modes)
        remote_preference = WorkMode.REMOTE in preferred_modes
        if location_match or (remote_preference and remote_job) or mode_match:
            return _rule(
                FilterRule.LOCATION,
                RuleStatus.PASS,
                True,
                "Job location or work mode matches configured preferences.",
                job_location=job.location,
                matched_location=location_match,
                matched_work_mode=job.work_mode.value if mode_match else None,
            )
        if job.location is None and not remote_job:
            return _rule(
                FilterRule.LOCATION,
                RuleStatus.ACCEPTABLE,
                True,
                "Job location is unknown; location filtering cannot be conclusive.",
                job_location=None,
            )

        return _rule(
            FilterRule.LOCATION,
            RuleStatus.REJECT,
            False,
            "Job location does not match configured location or work-mode preferences.",
            job_location=job.location,
            preferred_locations=list(preferred_locations),
            preferred_work_modes=[mode.value for mode in preferred_modes],
        )

    def _evaluate_experience(self, job: Job, profile: CandidateProfile) -> FilterRuleResult:
        candidate_years = profile.experience_years
        minimum = job.min_experience_years
        maximum = job.max_experience_years
        details = {
            "candidate_years": candidate_years,
            "job_min_years": minimum,
            "job_max_years": maximum,
        }
        if minimum is None and maximum is None:
            return _rule(
                FilterRule.EXPERIENCE,
                RuleStatus.PASS,
                True,
                "Job experience requirements are unknown; experience filtering is not restrictive.",
                **details,
            )
        if minimum is not None and candidate_years < minimum:
            gap = minimum - candidate_years
            details["experience_gap_years"] = gap
            if gap <= self._config.experience.acceptable_gap_years:
                return _rule(
                    FilterRule.EXPERIENCE,
                    RuleStatus.ACCEPTABLE,
                    True,
                    "Candidate is slightly below the stated minimum.",
                    **details,
                )
            if gap <= self._config.experience.max_stretch_gap_years:
                return _rule(
                    FilterRule.EXPERIENCE,
                    RuleStatus.STRETCH,
                    True,
                    "Candidate is below the stated minimum but within the configured "
                    "stretch range.",
                    **details,
                )
            return _rule(
                FilterRule.EXPERIENCE,
                RuleStatus.REJECT,
                False,
                "Candidate is beyond the configured maximum experience gap.",
                **details,
            )
        if maximum is not None and candidate_years > maximum:
            return _rule(
                FilterRule.EXPERIENCE,
                RuleStatus.ACCEPTABLE,
                True,
                "Candidate exceeds the stated maximum; this is not configured as a hard rejection.",
                **details,
            )
        return _rule(
            FilterRule.EXPERIENCE,
            RuleStatus.PASS,
            True,
            "Candidate experience fits the stated job range.",
            **details,
        )

    def _evaluate_constraints(self, job: Job, profile: CandidateProfile) -> FilterRuleResult:
        hard = profile.hard_constraints
        all_excluded_terms = [*self._config.excluded_terms, *hard.excluded_terms]
        searchable_text = _key(" ".join(filter(None, [job.title, job.location, job.description])))
        matched_term = _first_term_match(searchable_text, _terms(all_excluded_terms))
        if matched_term:
            return _rule(
                FilterRule.CONSTRAINT,
                RuleStatus.REJECT,
                False,
                f"Job text contains explicitly excluded term '{matched_term}'.",
                excluded_term=matched_term,
            )
        if job.employment_type in hard.excluded_employment_types:
            return _rule(
                FilterRule.CONSTRAINT,
                RuleStatus.REJECT,
                False,
                f"Job employment type '{job.employment_type.value}' is explicitly excluded.",
                employment_type=job.employment_type.value,
            )
        if hard.remote_only and not _is_remote(job):
            return _rule(
                FilterRule.CONSTRAINT,
                RuleStatus.REJECT,
                False,
                "Remote-only is configured but the job is not identified as remote.",
                work_mode=job.work_mode.value,
                job_location=job.location,
            )
        if (
            profile.preferences.preferred_employment_types
            and job.employment_type not in profile.preferences.preferred_employment_types
        ):
            return _rule(
                FilterRule.CONSTRAINT,
                RuleStatus.ACCEPTABLE,
                True,
                "Job employment type is outside the preferred types but is not a hard constraint.",
                employment_type=job.employment_type.value,
                preferred_employment_types=[
                    value.value for value in profile.preferences.preferred_employment_types
                ],
            )
        if hard.work_authorization_required:
            return _rule(
                FilterRule.CONSTRAINT,
                RuleStatus.ACCEPTABLE,
                True,
                "Work authorization is required, but the canonical job model has no "
                "authorization field to evaluate.",
                evaluated=False,
            )
        return _rule(
            FilterRule.CONSTRAINT,
            RuleStatus.PASS,
            True,
            "No configured hard constraint excludes this job.",
        )


def _rule(
    rule: FilterRule,
    status: RuleStatus,
    passed: bool,
    reason: str,
    **details: Any,
) -> FilterRuleResult:
    return FilterRuleResult(
        rule=rule,
        status=status,
        passed=passed,
        reason=reason,
        details=details,
    )


def _rejected(outcome: FilterOutcome, rules: list[FilterRuleResult]) -> FilterResult:
    return FilterResult(
        passed=False,
        outcome=outcome,
        reasons=[rule.reason for rule in rules],
        rules=rules,
    )


def _key(value: str | None) -> str:
    return normalize_text(value or "").casefold()


def _terms(values: list[str]) -> list[str]:
    return [_key(value) for value in values if _key(value)]


def _first_term_match(text: str, terms: list[str]) -> str | None:
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            return term
    return None


def _normalized_aliases(aliases: dict[str, list[str]]) -> dict[str, list[str]]:
    return {_key(key): _terms(values) for key, values in aliases.items()}


def _location_variants(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,/|]", value) if part.strip()]


def _first_location_match(
    job_location: str,
    preferred_locations: list[str],
    aliases: dict[str, list[str]],
) -> str | None:
    normalized_aliases = _normalized_aliases(aliases)
    variants = [_key(job_location), *[_key(value) for value in _location_variants(job_location)]]
    for preferred in preferred_locations:
        preferred_key = _key(preferred)
        terms = [preferred_key, *normalized_aliases.get(preferred_key, [])]
        for term in terms:
            if any(
                re.search(rf"(?<!\w){re.escape(term)}(?!\w)", variant)
                for variant in variants
                if variant
            ):
                return preferred
    return None


def _preferred_work_modes(profile: CandidateProfile) -> list[WorkMode]:
    modes = [
        mode
        for mode in profile.preferences.preferred_work_modes
        if mode != WorkMode.UNKNOWN
    ]
    if (
        profile.work_mode is not None
        and profile.work_mode != WorkMode.UNKNOWN
        and profile.work_mode not in modes
    ):
        modes.append(profile.work_mode)
    return modes


def _is_remote(job: Job) -> bool:
    return job.work_mode == WorkMode.REMOTE or "remote" in _key(job.location)


__all__ = [
    "DeterministicJobFilter",
    "ExperienceFilterConfig",
    "FilterOutcome",
    "FilterResult",
    "FilterRule",
    "FilterRuleResult",
    "FilteringConfig",
    "RuleStatus",
]
