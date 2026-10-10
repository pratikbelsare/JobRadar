"""Interpretable structured and semantic candidate-job matching."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from .ai import TextEmbeddingProvider
from .config import Settings
from .filtering import FilterResult
from .models import (
    CandidateProfile,
    Job,
    JobAnalysis,
    JobMatch,
    Model,
    NonEmptyText,
    RelevanceComponentEvidence,
    RelevanceEvidence,
    WorkMode,
)
from .normalization import normalize_text


class MatchComponent(str, Enum):
    ROLE = "role"
    SKILL = "skill"
    EXPERIENCE = "experience"
    RESPONSIBILITY = "responsibility"
    DOMAIN = "domain"
    LOCATION = "location"
    PREFERENCE = "preference"


RELEVANCE_COMPONENTS = (
    MatchComponent.ROLE.value,
    MatchComponent.SKILL.value,
    MatchComponent.RESPONSIBILITY.value,
    MatchComponent.DOMAIN.value,
)


@dataclass(frozen=True)
class ComponentScore:
    """Internal score plus explicit evidence availability."""

    score: float
    available: bool
    evidence: dict[str, Any]


class MatchingWeights(Model):
    """Initial weights from the product plan; callers can replace every value."""

    role: NonNegativeFloat = 0.25
    skill: NonNegativeFloat = 0.25
    experience: NonNegativeFloat = 0.20
    responsibility: NonNegativeFloat = 0.15
    domain: NonNegativeFloat = 0.05
    location: NonNegativeFloat = 0.05
    preference: NonNegativeFloat = 0.05

    @model_validator(mode="after")
    def validate_total(self) -> MatchingWeights:
        total = sum(
            (
                self.role,
                self.skill,
                self.experience,
                self.responsibility,
                self.domain,
                self.location,
                self.preference,
            )
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("matching weights must sum to 1.0")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> MatchingWeights:
        defaults = cls()
        default_values = defaults.model_dump()
        values = {
            field: getattr(settings, f"matching_{field}_weight")
            for field in (
                "role",
                "skill",
                "experience",
                "responsibility",
                "domain",
                "location",
                "preference",
            )
        }
        return cls(
            **{
                field: default_values[field] if value is None else value
                for field, value in values.items()
            }
        )


class MatchingConfig(Model):
    weights: MatchingWeights = Field(default_factory=MatchingWeights)
    experience_gap_scale_years: PositiveFloat = 2.0
    role_aliases: dict[str, list[str]] = Field(default_factory=dict)
    location_aliases: dict[str, list[str]] = Field(default_factory=dict)
    required_skill_weight: NonNegativeFloat = 0.7
    preferred_skill_weight: NonNegativeFloat = 0.3
    ranking_version: NonEmptyText = "hybrid-v1"

    @model_validator(mode="after")
    def validate_skill_weights(self) -> MatchingConfig:
        if not math.isclose(
            self.required_skill_weight + self.preferred_skill_weight,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("skill weights must sum to 1.0")
        if self.required_skill_weight < self.preferred_skill_weight:
            raise ValueError("required skill weight must not be below preferred skill weight")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> MatchingConfig:
        return cls(
            weights=MatchingWeights.from_settings(settings),
            experience_gap_scale_years=(
                settings.matching_experience_gap_scale_years
                if settings.matching_experience_gap_scale_years is not None
                else 2.0
            ),
            required_skill_weight=(
                settings.matching_required_skill_weight
                if settings.matching_required_skill_weight is not None
                else 0.7
            ),
            preferred_skill_weight=(
                settings.matching_preferred_skill_weight
                if settings.matching_preferred_skill_weight is not None
                else 0.3
            ),
        )


class RejectedJobError(ValueError):
    """Raised when a hard-filtered job is sent to the ranking layer."""


class HybridJobMatcher:
    """Calculate component scores without asking an LLM to invent a final score."""

    def __init__(
        self,
        config: MatchingConfig | None = None,
        *,
        embedding_provider: TextEmbeddingProvider | None = None,
    ) -> None:
        self._config = config or MatchingConfig()
        self._embedding_provider = embedding_provider

    def match(
        self,
        job: Job,
        profile: CandidateProfile,
        *,
        analysis: JobAnalysis | None = None,
        filter_result: FilterResult | None = None,
    ) -> JobMatch:
        if filter_result is not None and not filter_result.passed:
            raise RejectedJobError("A rejected job cannot enter matching")

        role_result = self._role_match(job, profile)
        skill_result = self._skill_match(job, profile, analysis)
        experience_result = self._experience_match(
            job,
            profile,
            analysis,
        )
        responsibility_result = self._responsibility_match(
            job,
            profile,
            analysis,
        )
        domain_result = self._domain_match(job, profile, analysis)
        location_result = self._location_match(job, profile)
        preference_result = self._preference_match(job, profile, analysis)

        component_results = {
            MatchComponent.ROLE.value: role_result,
            MatchComponent.SKILL.value: skill_result,
            MatchComponent.EXPERIENCE.value: experience_result,
            MatchComponent.RESPONSIBILITY.value: responsibility_result,
            MatchComponent.DOMAIN.value: domain_result,
            MatchComponent.LOCATION.value: location_result,
            MatchComponent.PREFERENCE.value: preference_result,
        }
        scores = {name: result.score for name, result in component_results.items()}
        weights = self._config.weights.model_dump(mode="json")
        included_components = [
            name for name, result in component_results.items() if result.available
        ]
        excluded_components = [
            name for name, result in component_results.items() if not result.available
        ]
        weighted_scores = {
            name: (
                scores[name] * weights[name]
                if component_results[name].available
                else 0.0
            )
            for name in scores
        }
        numerator = sum(weighted_scores.values())
        denominator = sum(weights[name] for name in included_components)
        final_score = numerator / denominator if denominator else 0.0
        relevance_evidence = compute_relevance_evidence(component_results, weights)
        evidence = {
            "components": {
                name: {
                    **result.evidence,
                    "available": result.available,
                    "availability": "available" if result.available else "unknown",
                }
                for name, result in component_results.items()
            },
            "scores": scores,
            "weights": weights,
            "weighted_scores": weighted_scores,
            "aggregation": {
                "method": "available_weighted_mean",
                "numerator": numerator,
                "denominator": denominator,
                "included_components": included_components,
                "excluded_components": excluded_components,
            },
        }
        return JobMatch(
            job_id=job.id,
            candidate_profile_id=profile.id,
            role_match=role_result.score,
            skill_match=skill_result.score,
            experience_match=experience_result.score,
            responsibility_match=responsibility_result.score,
            domain_match=domain_result.score,
            location_match=location_result.score,
            preference_match=preference_result.score,
            final_score=max(0.0, min(1.0, final_score)),
            ranking_version=self._config.ranking_version,
            legacy_final_score=max(0.0, min(1.0, final_score)),
            relevance_score=relevance_evidence.score,
            relevance_evidence=relevance_evidence,
            evidence=evidence,
        )

    def _role_match(
        self,
        job: Job,
        profile: CandidateProfile,
    ) -> ComponentScore:
        title = _key(job.title)
        if not profile.target_roles:
            return ComponentScore(
                0.5,
                False,
                {"status": "unknown", "reason": "no target roles configured"},
            )
        matched = _matching_role(title, profile.target_roles, self._config.role_aliases)
        if matched is not None:
            return ComponentScore(
                1.0,
                True,
                {"matched_role": matched, "method": "deterministic"},
            )
        if self._embedding_provider is None:
            return ComponentScore(
                0.0,
                True,
                {"matched_role": None, "method": "deterministic"},
            )
        job_vector = self._embedding_provider.embed_text(job.title).value.values
        semantic = max(
            (
                _embedding_similarity(
                    job_vector,
                    self._embedding_provider.embed_text(role).value.values,
                )
                for role in profile.target_roles
            ),
            default=0.0,
        )
        return ComponentScore(
            semantic,
            True,
            {"matched_role": None, "method": "embedding", "similarity": semantic},
        )

    def _skill_match(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis | None,
    ) -> ComponentScore:
        required = analysis.required_skills if analysis else job.required_skills
        preferred = analysis.preferred_skills if analysis else job.preferred_skills
        candidate_skills = {_key(skill) for skill in profile.skills}
        evidenced_skills = {_key(skill) for skill in profile.skill_evidence}
        for experience in profile.work_experience:
            evidenced_skills.update(_key(skill) for skill in experience.skills)
        for project in profile.projects:
            evidenced_skills.update(_key(skill) for skill in project.skills)
        required_values = [
            _skill_value(skill, candidate_skills, evidenced_skills)
            for skill in required
        ]
        preferred_values = [
            _skill_value(skill, candidate_skills, evidenced_skills) for skill in preferred
        ]
        required_score = _average_or_neutral(required_values)
        preferred_score = _average_or_neutral(preferred_values)
        if required and preferred:
            score = (
                self._config.required_skill_weight * required_score
                + self._config.preferred_skill_weight * preferred_score
            )
        elif required:
            score = required_score
        elif preferred:
            score = preferred_score
        else:
            score = 0.5
        required_without_evidence = [
            skill for skill in required if _key(skill) not in evidenced_skills
        ]
        preferred_without_evidence = [
            skill for skill in preferred if _key(skill) not in evidenced_skills
        ]
        return ComponentScore(
            score,
            bool(required or preferred),
            {
                "matched_required_skills": [
                    skill for skill in required if _key(skill) in candidate_skills
                ],
                "required_skills_without_evidence": required_without_evidence,
                "matched_preferred_skills": [
                    skill for skill in preferred if _key(skill) in candidate_skills
                ],
                "preferred_skills_without_evidence": preferred_without_evidence,
                "requirements_present": bool(required or preferred),
            },
        )

    def _experience_match(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis | None,
    ) -> ComponentScore:
        minimum = job.min_experience_years if analysis is None else analysis.min_experience_years
        maximum = job.max_experience_years if analysis is None else analysis.max_experience_years
        candidate_years = profile.experience_years
        details = {
            "candidate_years": candidate_years,
            "job_min_years": minimum,
            "job_max_years": maximum,
        }
        if minimum is None and maximum is None:
            return ComponentScore(
                0.5,
                False,
                {**details, "reason": "experience requirement is unknown"},
            )
        if minimum is not None and candidate_years < minimum:
            gap = minimum - candidate_years
            score = max(0.0, 1.0 - gap / self._config.experience_gap_scale_years)
            return ComponentScore(score, True, {**details, "gap_years": gap})
        if maximum is not None and candidate_years > maximum:
            gap = candidate_years - maximum
            score = max(0.0, 1.0 - gap / self._config.experience_gap_scale_years)
            return ComponentScore(score, True, {**details, "gap_years": gap})
        return ComponentScore(1.0, True, details)

    def _responsibility_match(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis | None,
    ) -> ComponentScore:
        responsibilities = analysis.responsibilities if analysis else job.responsibilities
        candidate_texts = [
            item.description
            for item in profile.work_experience
            if item.description is not None
        ] + [project.description for project in profile.projects]
        if not responsibilities or not candidate_texts:
            return ComponentScore(
                0.5,
                False,
                {"method": "neutral", "reason": "insufficient evidence"},
            )
        if self._embedding_provider is not None:
            candidate_vectors = [
                self._embedding_provider.embed_text(candidate_text).value.values
                for candidate_text in candidate_texts
            ]
            similarities = [
                max(
                    _embedding_similarity(
                        self._embedding_provider.embed_text(responsibility).value.values,
                        candidate_vector,
                    )
                    for candidate_vector in candidate_vectors
                )
                for responsibility in responsibilities
            ]
            return ComponentScore(
                _average_or_neutral(similarities),
                True,
                {"method": "embedding", "similarities": similarities},
            )
        similarities = [
            max(_token_similarity(responsibility, candidate) for candidate in candidate_texts)
            for responsibility in responsibilities
        ]
        return ComponentScore(
            _average_or_neutral(similarities),
            True,
            {"method": "token_overlap", "similarities": similarities},
        )

    def _domain_match(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis | None,
    ) -> ComponentScore:
        domain = analysis.domain if analysis else job.domain
        if domain is None or not profile.domains:
            return ComponentScore(
                0.5,
                False,
                {"reason": "domain information is incomplete"},
            )
        matched = next((item for item in profile.domains if _key(item) == _key(domain)), None)
        return ComponentScore(
            1.0 if matched else 0.0,
            True,
            {"job_domain": domain, "matched_domain": matched},
        )

    def _location_match(
        self,
        job: Job,
        profile: CandidateProfile,
    ) -> ComponentScore:
        preferred = profile.preferred_locations
        modes = set(profile.preferences.preferred_work_modes)
        if profile.work_mode is not None:
            modes.add(profile.work_mode)
        if not preferred and not modes:
            return ComponentScore(
                0.5,
                False,
                {"reason": "no location preference configured"},
            )
        if job.location is None and job.work_mode == WorkMode.UNKNOWN:
            return ComponentScore(
                0.5,
                False,
                {"reason": "job location is unknown"},
            )
        location_match = _location_match(
            job.location,
            preferred,
            self._config.location_aliases,
        )
        mode_match = job.work_mode in modes and job.work_mode != WorkMode.UNKNOWN
        remote_match = WorkMode.REMOTE in modes and _is_remote(job)
        score = 1.0 if location_match or mode_match or remote_match else 0.0
        return ComponentScore(
            score,
            True,
            {
                "matched_location": location_match,
                "matched_work_mode": job.work_mode.value if mode_match else None,
            },
        )

    def _preference_match(
        self,
        job: Job,
        profile: CandidateProfile,
        analysis: JobAnalysis | None,
    ) -> ComponentScore:
        values: list[float] = []
        details: dict[str, Any] = {}
        preferred_types = profile.preferences.preferred_employment_types
        if preferred_types:
            if job.employment_type.value != "unknown":
                type_score = float(job.employment_type in preferred_types)
                values.append(type_score)
                details["employment_type_score"] = type_score
            else:
                details["employment_type_status"] = "unknown"
        preferred_domains = profile.preferences.preferred_domains
        job_domain = analysis.domain if analysis else job.domain
        if preferred_domains:
            if job_domain is not None:
                domain_score = float(
                    any(_key(item) == _key(job_domain) for item in preferred_domains)
                )
                values.append(domain_score)
                details["preferred_domain_score"] = domain_score
            else:
                details["preferred_domain_status"] = "unknown"
        if not values:
            return ComponentScore(
                0.5,
                False,
                {**details, "reason": "no optional preference evidence available"},
            )
        return ComponentScore(sum(values) / len(values), True, details)


def _matching_role(title: str, roles: list[str], aliases: dict[str, list[str]]) -> str | None:
    normalized_aliases = {
        _key(key): [_key(item) for item in values]
        for key, values in aliases.items()
    }
    for role in roles:
        role_key = _key(role)
        terms = [role_key, *normalized_aliases.get(role_key, [])]
        if any(_term_match(title, term) for term in terms):
            return role
    return None


def compute_relevance_evidence(
    component_results: Mapping[str, ComponentScore],
    weights: Mapping[str, float],
) -> RelevanceEvidence:
    """Aggregate role, skill, responsibility, and domain evidence only."""

    included = [
        name for name in RELEVANCE_COMPONENTS if component_results[name].available
    ]
    excluded = [
        name for name in RELEVANCE_COMPONENTS if not component_results[name].available
    ]
    numerator = sum(
        component_results[name].score * weights[name]
        for name in included
    )
    denominator = sum(weights[name] for name in included)
    score = numerator / denominator if denominator else 0.0
    components = {
        name: RelevanceComponentEvidence(
            raw_score=component_results[name].score,
            weight=weights[name],
            weighted_contribution=(
                component_results[name].score * weights[name]
                if component_results[name].available
                else 0.0
            ),
            available=component_results[name].available,
            numerator_share=(
                component_results[name].score * weights[name] / numerator
                if component_results[name].available and numerator
                else None
            ),
        )
        for name in RELEVANCE_COMPONENTS
    }
    return RelevanceEvidence(
        score=score,
        numerator=numerator,
        denominator=denominator,
        included_components=included,
        excluded_components=excluded,
        components=components,
    )


def _skill_value(
    skill: str,
    candidate_skills: set[str],
    evidenced_skills: set[str],
) -> float:
    key = _key(skill)
    if key in evidenced_skills:
        return 1.0
    if key in candidate_skills:
        return 0.75
    return 0.0


def _average_or_neutral(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.5


def _embedding_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", _key(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+", _key(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _key(value: str) -> str:
    return normalize_text(value).casefold()


def _term_match(text: str, term: str) -> bool:
    return bool(term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text))


def _location_match(
    location: str | None,
    preferred: list[str],
    aliases: dict[str, list[str]],
) -> bool:
    if location is None:
        return False
    location_key = _key(location)
    normalized_aliases = {
        _key(key): [_key(item) for item in values] for key, values in aliases.items()
    }
    return any(
        any(
            _term_match(location_key, term)
            for term in [_key(item), *normalized_aliases.get(_key(item), [])]
        )
        for item in preferred
    )


def _is_remote(job: Job) -> bool:
    return job.work_mode == WorkMode.REMOTE or (
        job.location is not None and "remote" in _key(job.location)
    )


__all__ = [
    "ComponentScore",
    "compute_relevance_evidence",
    "HybridJobMatcher",
    "MatchComponent",
    "MatchingConfig",
    "MatchingWeights",
    "RejectedJobError",
]
