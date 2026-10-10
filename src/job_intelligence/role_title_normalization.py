"""Deterministic, profession-agnostic normalization for role-title experiments.

The existing production classifier remains unchanged. This module exposes an
explainable counterfactual classifier that first honors an original-title match and
only then tries a structurally normalized base title against the same policy.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import Field

from .models import Model, NonEmptyText
from .normalization import normalize_text
from .role_policy import RoleFamilyClassifier, RoleFamilyMatch, RoleFamilyPolicy


class TitleSeniority(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    ASSOCIATE = "associate"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"


class NormalizedRoleTitle(Model):
    original_title: NonEmptyText
    normalized_title: NonEmptyText
    base_title: NonEmptyText
    seniority: list[TitleSeniority] = Field(default_factory=list)
    level: NonEmptyText | None = None
    specializations: list[NonEmptyText] = Field(default_factory=list)
    annotations: list[NonEmptyText] = Field(default_factory=list)
    transformations: list[NonEmptyText] = Field(default_factory=list)


class ExplainableRoleClassification(Model):
    title: NormalizedRoleTitle
    original_match: RoleFamilyMatch
    match: RoleFamilyMatch
    title_used_for_match: NonEmptyText
    used_normalized_base: bool
    reason: NonEmptyText


_SENIORITY_ALIASES = {
    "intern": TitleSeniority.INTERN,
    "junior": TitleSeniority.JUNIOR,
    "jr": TitleSeniority.JUNIOR,
    "associate": TitleSeniority.ASSOCIATE,
    "senior": TitleSeniority.SENIOR,
    "sr": TitleSeniority.SENIOR,
    "staff": TitleSeniority.STAFF,
    "principal": TitleSeniority.PRINCIPAL,
    "lead": TitleSeniority.LEAD,
}
_SENIORITY_TOKEN = r"(?:intern|junior|jr\.?|associate|senior|sr\.?|staff|principal|lead)"
_PREFIX_RE = re.compile(
    rf"^(?P<prefix>{_SENIORITY_TOKEN}"
    rf"(?:\s*(?:/|\||&|or)\s*{_SENIORITY_TOKEN})*)\s+",
    flags=re.IGNORECASE,
)
_SUFFIX_SENIORITY_RE = re.compile(r"\s+(?P<suffix>intern|junior|jr\.?)$", re.IGNORECASE)
_ANNOTATION_RE = re.compile(r"\s*\((?P<annotation>[^()]*)\)\s*")
_LEVEL_RE = re.compile(
    r"\s+(?P<level>(?:i{1,4}|v|vi|[1-9]|all\s+levels?))$",
    flags=re.IGNORECASE,
)
_SPECIALIZATION_RE = re.compile(
    r"\s*(?:,|[–—]|\||\s/\s|\s-\s|(?<=[A-Za-z])-(?=[A-Z]))\s*"
)


class RoleTitleNormalizer:
    """Extract structural title metadata without interpreting AI/Data keywords."""

    def normalize(self, title: str) -> NormalizedRoleTitle:
        normalized = _normalize_punctuation(title)
        working = normalized
        transformations: list[str] = []

        annotations = [
            normalize_text(match.group("annotation"))
            for match in _ANNOTATION_RE.finditer(working)
            if match.group("annotation").strip()
        ]
        if annotations:
            working = normalize_text(_ANNOTATION_RE.sub(" ", working))
            transformations.append("parenthetical_annotation_removed")

        seniority: list[TitleSeniority] = []
        prefix = _PREFIX_RE.match(working)
        if prefix is not None:
            seniority = _seniority_values(prefix.group("prefix"))
            working = normalize_text(working[prefix.end() :])
            transformations.append(
                "mixed_seniority_prefix_removed"
                if len(seniority) > 1
                else "seniority_prefix_removed"
            )

        parts = [part for part in _SPECIALIZATION_RE.split(working) if part.strip()]
        base = normalize_text(parts[0]) if parts else working
        specializations = [normalize_text(part) for part in parts[1:]]
        specializations.extend(
            annotation
            for annotation in annotations
            if _is_ascii_specialization(annotation)
        )
        if specializations:
            transformations.append("specialization_suffix_removed")

        level_match = _LEVEL_RE.search(base)
        level = None
        if level_match is not None:
            level = normalize_text(level_match.group("level"))
            base = normalize_text(base[: level_match.start()])
            transformations.append("level_suffix_removed")

        suffix_seniority = _SUFFIX_SENIORITY_RE.search(base)
        if suffix_seniority is not None:
            suffix = _seniority_values(suffix_seniority.group("suffix"))
            seniority.extend(value for value in suffix if value not in seniority)
            base = normalize_text(base[: suffix_seniority.start()])
            transformations.append("seniority_suffix_removed")

        if not base:
            base = normalized
            transformations.append("normalization_fallback_to_original")
        return NormalizedRoleTitle(
            original_title=title,
            normalized_title=normalized,
            base_title=base,
            seniority=seniority,
            level=level,
            specializations=specializations,
            annotations=annotations,
            transformations=transformations,
        )


class NormalizedRoleFamilyClassifier:
    """Compatibility wrapper around the production normalized classifier."""

    def __init__(
        self,
        policy: RoleFamilyPolicy,
        *,
        normalizer: RoleTitleNormalizer | None = None,
    ) -> None:
        self._classifier = RoleFamilyClassifier(policy)
        self._normalizer = normalizer

    def classify(self, title: str) -> ExplainableRoleClassification:
        return self._classifier.classify_with_evidence(
            title,
            normalizer=self._normalizer,
        )


def _normalize_punctuation(value: str) -> str:
    return normalize_text(value.replace("‐", "-").replace("‑", "-").replace("−", "-"))


def _seniority_values(value: str) -> list[TitleSeniority]:
    matches = re.findall(_SENIORITY_TOKEN, value, flags=re.IGNORECASE)
    results: list[TitleSeniority] = []
    for match in matches:
        key = match.casefold().rstrip(".")
        seniority = _SENIORITY_ALIASES[key]
        if seniority not in results:
            results.append(seniority)
    return results


def _is_ascii_specialization(value: str) -> bool:
    letters = [character for character in value if character.isalpha()]
    return bool(letters) and all(character.isascii() for character in letters)


__all__ = [
    "ExplainableRoleClassification",
    "NormalizedRoleFamilyClassifier",
    "NormalizedRoleTitle",
    "RoleTitleNormalizer",
    "TitleSeniority",
]
