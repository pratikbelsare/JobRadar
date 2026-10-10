"""Evaluation-only deterministic policies for Research Scientist titles.

Nothing in the production role classifier imports this module. It exists to compare
candidate taxonomy policies before one is approved for production use.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import Field

from .models import Model, NonEmptyText
from .normalization import normalize_text
from .role_policy import RoleFamilyClass
from .role_title_normalization import NormalizedRoleTitle, RoleTitleNormalizer


class ResearchScientistPolicy(str, Enum):
    GENERIC_CORE = "generic_core"
    GENERIC_ADJACENT = "generic_adjacent"
    CONTEXT_QUALIFIED = "context_qualified"


class ResearchScientistPolicyConfig(Model):
    """Configurable title evidence; no company or description terms are allowed."""

    core_qualifiers: list[NonEmptyText] = Field(
        default_factory=lambda: [
            "ai",
            "artificial intelligence",
            "machine learning",
            "ml",
            "human-ai",
            "foundation model",
            "foundation models",
            "language model",
            "language models",
            "llm",
            "nlp",
            "computer vision",
            "multimodal",
            "multi-modal",
            "reinforcement learning",
            "deep learning",
            "generative model",
            "generative models",
            "inference",
        ]
    )
    adjacent_qualifiers: list[NonEmptyText] = Field(
        default_factory=lambda: [
            "benchmark",
            "benchmarks",
            "model",
            "models",
        ]
    )
    bare_title_class: RoleFamilyClass = RoleFamilyClass.ADJACENT


class ResearchScientistDecision(Model):
    applicable: bool
    policy: ResearchScientistPolicy
    classification: RoleFamilyClass | None = None
    normalized_title: NormalizedRoleTitle
    title_evidence: list[NonEmptyText] = Field(default_factory=list)
    matched_qualifiers: list[NonEmptyText] = Field(default_factory=list)
    reason: NonEmptyText


class ResearchScientistPolicyEvaluator:
    """Compare Research Scientist policies using title evidence only."""

    def __init__(self, config: ResearchScientistPolicyConfig | None = None) -> None:
        self._config = config or ResearchScientistPolicyConfig()
        self._normalizer = RoleTitleNormalizer()

    def evaluate(
        self,
        title: str,
        policy: ResearchScientistPolicy,
    ) -> ResearchScientistDecision:
        normalized = self._normalizer.normalize(title)
        context = _research_scientist_context(normalized)
        if context is None:
            return ResearchScientistDecision(
                applicable=False,
                policy=policy,
                normalized_title=normalized,
                reason="The normalized title is not an anchored Research Scientist role.",
            )
        evidence = context
        if policy == ResearchScientistPolicy.GENERIC_CORE:
            return _decision(
                policy,
                normalized,
                RoleFamilyClass.CORE,
                evidence,
                [],
                "Any anchored Research Scientist title is classified CORE.",
            )
        if policy == ResearchScientistPolicy.GENERIC_ADJACENT:
            return _decision(
                policy,
                normalized,
                RoleFamilyClass.ADJACENT,
                evidence,
                [],
                "Any anchored Research Scientist title is classified ADJACENT.",
            )

        core_matches = _matched_qualifiers(evidence, self._config.core_qualifiers)
        if core_matches:
            return _decision(
                policy,
                normalized,
                RoleFamilyClass.CORE,
                evidence,
                core_matches,
                "Strong configured AI/ML title evidence qualifies the role as CORE.",
            )
        adjacent_matches = _matched_qualifiers(
            evidence,
            self._config.adjacent_qualifiers,
        )
        if adjacent_matches:
            return _decision(
                policy,
                normalized,
                RoleFamilyClass.ADJACENT,
                evidence,
                adjacent_matches,
                "Weak computational research evidence retains the role as ADJACENT.",
            )
        if not evidence:
            return _decision(
                policy,
                normalized,
                self._config.bare_title_class,
                evidence,
                [],
                "A bare Research Scientist title is retained conservatively as ADJACENT.",
            )
        return _decision(
            policy,
            normalized,
            None,
            evidence,
            [],
            "The specialization has no configured AI/ML or computational qualifier.",
        )


def _research_scientist_context(title: NormalizedRoleTitle) -> list[str] | None:
    base = _key(title.base_title)
    if base == "research scientist":
        return [*title.specializations]
    prefix = re.fullmatch(r"(?P<context>.+)\s+research scientist", base)
    if prefix is not None:
        return [normalize_text(prefix.group("context")), *title.specializations]
    return None


def _matched_qualifiers(evidence: list[str], qualifiers: list[str]) -> list[str]:
    haystack = _key(" ".join(evidence))
    return [
        qualifier
        for qualifier in qualifiers
        if re.search(rf"(?<!\w){re.escape(_key(qualifier))}(?!\w)", haystack)
    ]


def _decision(
    policy: ResearchScientistPolicy,
    title: NormalizedRoleTitle,
    classification: RoleFamilyClass | None,
    evidence: list[str],
    qualifiers: list[str],
    reason: str,
) -> ResearchScientistDecision:
    return ResearchScientistDecision(
        applicable=True,
        policy=policy,
        classification=classification,
        normalized_title=title,
        title_evidence=evidence,
        matched_qualifiers=qualifiers,
        reason=reason,
    )


def _key(value: str) -> str:
    return normalize_text(value).casefold()


__all__ = [
    "ResearchScientistDecision",
    "ResearchScientistPolicy",
    "ResearchScientistPolicyConfig",
    "ResearchScientistPolicyEvaluator",
]
