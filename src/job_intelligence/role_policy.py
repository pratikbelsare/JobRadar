"""Configurable, deterministic role-family classification.

The classifier knows only about the policy supplied to it.  The initial policy is
provided as structured data for the first AI/ML/Data profile, while callers can
replace it with a policy for any other profession.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from .models import Model, NonEmptyText
from .normalization import normalize_text

if TYPE_CHECKING:
    from .role_title_normalization import ExplainableRoleClassification, RoleTitleNormalizer


class RoleFamilyClass(str, Enum):
    CORE = "core"
    ADJACENT = "adjacent"
    EXCLUDE = "exclude"


class UnknownRoleAction(str, Enum):
    REJECT = "reject"
    PASS = "pass"


class RoleMatchType(str, Enum):
    EXACT_TITLE = "exact_title"
    ALIAS = "alias"
    INCLUDE_PATTERN = "include_pattern"


class RoleFamily(Model):
    """One configured role family and its deterministic title vocabulary."""

    id: NonEmptyText
    canonical_name: NonEmptyText
    classification: RoleFamilyClass = Field(alias="class")
    exact_titles: list[NonEmptyText] = Field(default_factory=list)
    aliases: list[NonEmptyText] = Field(default_factory=list)
    include_patterns: list[NonEmptyText] = Field(default_factory=list)
    exclude_patterns: list[NonEmptyText] = Field(default_factory=list)

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "validate_assignment": True,
        "protected_namespaces": (),
    }

    @model_validator(mode="after")
    def require_match_rule(self) -> RoleFamily:
        if not (self.exact_titles or self.aliases or self.include_patterns):
            raise ValueError("a role family must define at least one title matching rule")
        return self

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def validate_patterns(cls, values: list[str]) -> list[str]:
        for pattern in values:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid role-family regular expression: {pattern}") from exc
        return values


class RoleFamilyPolicy(Model):
    """Role-family taxonomy and eligibility configuration for Stage 1 filtering."""

    families: list[RoleFamily] = Field(min_length=1)
    target_family_ids: list[NonEmptyText] = Field(default_factory=list)
    adjacent_family_ids: list[NonEmptyText] = Field(default_factory=list)
    unknown_role_action: UnknownRoleAction = UnknownRoleAction.REJECT

    @model_validator(mode="after")
    def validate_family_ids(self) -> RoleFamilyPolicy:
        ids = {family.id for family in self.families}
        configured = [*self.target_family_ids, *self.adjacent_family_ids]
        unknown = sorted(set(configured) - ids)
        if unknown:
            raise ValueError(f"role-family eligibility references unknown family IDs: {unknown}")
        duplicates = set(self.target_family_ids) & set(self.adjacent_family_ids)
        if duplicates:
            raise ValueError(
                f"a family cannot be both target and adjacent: {sorted(duplicates)}"
            )
        return self

    def family_by_id(self, family_id: str) -> RoleFamily:
        for family in self.families:
            if family.id == family_id:
                return family
        raise KeyError(family_id)

    def is_eligible(self, family: RoleFamily) -> bool:
        if family.classification == RoleFamilyClass.EXCLUDE:
            return False
        if self.target_family_ids or self.adjacent_family_ids:
            return family.id in (*self.target_family_ids, *self.adjacent_family_ids)
        return family.classification in {
            RoleFamilyClass.CORE,
            RoleFamilyClass.ADJACENT,
        }


class RoleFamilyMatch(Model):
    """The explainable result of classifying one job title."""

    family_id: NonEmptyText | None = None
    canonical_name: NonEmptyText | None = None
    classification: RoleFamilyClass | None = None
    matched_by: RoleMatchType | None = None
    matched_term: NonEmptyText | None = None


class RoleFamilyClassifier:
    """Classify titles using only the supplied role-family policy."""

    def __init__(self, policy: RoleFamilyPolicy) -> None:
        self._policy = policy

    def classify(self, title: str) -> RoleFamilyMatch:
        """Classify an original title, then its structural base-title fallback."""

        return self.classify_with_evidence(title).match

    def classify_with_evidence(
        self,
        title: str,
        *,
        normalizer: RoleTitleNormalizer | None = None,
    ) -> ExplainableRoleClassification:
        """Return normalization and matching evidence without keyword promotion."""

        # The lazy import keeps the pure title-normalization module reusable while
        # avoiding a module-import cycle with its compatibility wrapper.
        from .role_title_normalization import (
            ExplainableRoleClassification,
            RoleTitleNormalizer,
        )

        normalized = (normalizer or RoleTitleNormalizer()).normalize(title)
        original_match = self._classify_configured_title(title)
        if original_match.classification is not None:
            return ExplainableRoleClassification(
                title=normalized,
                original_match=original_match,
                match=original_match,
                title_used_for_match=title,
                used_normalized_base=False,
                reason="The original title matched the configured taxonomy.",
            )
        normalized_match = self._classify_configured_title(normalized.base_title)
        if normalized_match.classification is not None:
            return ExplainableRoleClassification(
                title=normalized,
                original_match=original_match,
                match=normalized_match,
                title_used_for_match=normalized.base_title,
                used_normalized_base=True,
                reason=(
                    "The original title was unknown; its structurally normalized base "
                    "title matched the configured taxonomy."
                ),
            )
        return ExplainableRoleClassification(
            title=normalized,
            original_match=original_match,
            match=normalized_match,
            title_used_for_match=normalized.base_title,
            used_normalized_base=normalized.base_title.casefold() != title.casefold(),
            reason=(
                "Neither the original title nor its structurally normalized base title "
                "matched the configured taxonomy."
            ),
        )

    def _classify_configured_title(self, title: str) -> RoleFamilyMatch:
        normalized_title = _key(title)
        for family in self._policy.families:
            match = _match_family(normalized_title, family)
            if match is not None:
                matched_by, matched_term = match
                return RoleFamilyMatch(
                    family_id=family.id,
                    canonical_name=family.canonical_name,
                    classification=family.classification,
                    matched_by=matched_by,
                    matched_term=matched_term,
                )
        return RoleFamilyMatch()


def default_ai_data_role_family_policy() -> RoleFamilyPolicy:
    """Return the initial structured policy used when no override is configured."""

    return RoleFamilyPolicy.model_validate(
        {
            "families": [
                {
                    "id": "machine_learning_engineering",
                    "canonical_name": "Machine Learning Engineering",
                    "class": "core",
                    "exact_titles": ["Machine Learning Engineer", "ML Engineer"],
                    "aliases": [
                        "Applied AI Engineer",
                        "GenAI Engineer",
                        "Generative AI Engineer",
                        "LLM Engineer",
                    ],
                    "include_patterns": [
                        r"^(?:machine learning|ml) engineer(?:\s+(?:i{1,3}|iv|v))?$",
                    ],
                },
                {
                    "id": "ai_engineering",
                    "canonical_name": "AI Engineering",
                    "class": "core",
                    "exact_titles": ["AI Engineer"],
                    "include_patterns": [r"^(?:applied |generative |genai |llm )?ai engineer$"],
                },
                {
                    "id": "data_science",
                    "canonical_name": "Data Science",
                    "class": "core",
                    "exact_titles": [
                        "Data Scientist",
                        "Applied Scientist",
                        "ML Scientist",
                        "Machine Learning Scientist",
                    ],
                    "include_patterns": [
                        r"^data scientist(?:\s*[,\-].*)?$",
                        r"^applied scientist(?:\s+(?:i{1,3}|iv|v))?$",
                        r"^(?:ml|machine learning) scientist$",
                    ],
                },
                {
                    "id": "data_analytics",
                    "canonical_name": "Data Analytics",
                    "class": "adjacent",
                    "exact_titles": [
                        "Data Analyst",
                        "Data Product Analyst",
                        "Product Data Analyst",
                        "Decision Scientist",
                    ],
                    "include_patterns": [
                        r"^data product analyst(?:\s*[,\-].*)?$",
                        r"^(?:data|product data) analyst(?:\s*[,\-].*)?$",
                        r"^decision scientist(?:\s*[,\-].*)?$",
                    ],
                },
                {
                    "id": "data_quality",
                    "canonical_name": "Data Quality",
                    "class": "adjacent",
                    "exact_titles": ["Data QA Associate", "Data Quality Analyst"],
                    "include_patterns": [r"^(?:data qa|data quality)\b.*$"],
                },
                {
                    "id": "research_analytics",
                    "canonical_name": "Research Analytics",
                    "class": "adjacent",
                    "exact_titles": ["Investor Research Analyst"],
                    "include_patterns": [r"^(?:investor )?research analyst(?:\s*[,\-].*)?$"],
                },
                {
                    "id": "insights_analytics",
                    "canonical_name": "Insights and Analytics",
                    "class": "adjacent",
                    "exact_titles": ["Insights Associate"],
                    "include_patterns": [r"^insights?\s+(?:associate|analyst)\b.*$"],
                },
                {
                    "id": "ai_product",
                    "canonical_name": "AI and Data Product",
                    "class": "adjacent",
                    "exact_titles": ["Product Manager, AI"],
                    "include_patterns": [
                        (
                            r"^product manager(?:\s*,\s*|\s+-\s*)"
                            r"(?:ai|artificial intelligence|machine learning|data)"
                            r"(?:\s+product)?$"
                        ),
                    ],
                },
                {
                    "id": "sales",
                    "canonical_name": "Sales",
                    "class": "exclude",
                    "exact_titles": ["Sales Representative"],
                    "include_patterns": [r"\b(?:sales|account executive)\b"],
                },
                {
                    "id": "business_development",
                    "canonical_name": "Business Development",
                    "class": "exclude",
                    "exact_titles": ["Business Development Associate"],
                    "include_patterns": [r"\bbusiness development\b"],
                },
                {
                    "id": "account_management",
                    "canonical_name": "Account Management",
                    "class": "exclude",
                    "exact_titles": ["Client Account Management Associate"],
                    "include_patterns": [r"\baccount management\b"],
                },
                {
                    "id": "payroll_hr",
                    "canonical_name": "Payroll and Human Resources",
                    "class": "exclude",
                    "exact_titles": ["Senior Payroll and Benefits Administrator"],
                    "include_patterns": [r"\b(?:payroll|human resources|hr)\b"],
                },
                {
                    "id": "security_engineering",
                    "canonical_name": "Security Engineering",
                    "class": "exclude",
                    "exact_titles": ["Product Security Engineer"],
                    "include_patterns": [r"\bsecurity engineer(?:ing)?\b"],
                },
                {
                    "id": "qa_sdet",
                    "canonical_name": "Quality Assurance and SDET",
                    "class": "exclude",
                    "exact_titles": ["Senior SDET"],
                    "include_patterns": [r"\b(?:sdet|quality assurance|qa engineer)\b"],
                },
                {
                    "id": "it_delivery",
                    "canonical_name": "IT Delivery",
                    "class": "exclude",
                    "exact_titles": ["IT Delivery Engineer"],
                    "include_patterns": [r"\bit delivery\b"],
                },
                {
                    "id": "product_development",
                    "canonical_name": "Unrelated Product Development",
                    "class": "exclude",
                    "exact_titles": ["Associate Director of Product Development"],
                    "include_patterns": [r"^associate director of product development$"],
                },
            ],
            "target_family_ids": [
                "machine_learning_engineering",
                "ai_engineering",
                "data_science",
            ],
            "adjacent_family_ids": [
                "data_analytics",
                "data_quality",
                "research_analytics",
                "insights_analytics",
                "ai_product",
            ],
            "unknown_role_action": "reject",
        }
    )


def _match_family(title: str, family: RoleFamily) -> tuple[RoleMatchType, str] | None:
    if any(
        re.search(exclusion, title, flags=re.IGNORECASE)
        for exclusion in family.exclude_patterns
    ):
        return None
    for exact_title in family.exact_titles:
        if title == _key(exact_title):
            return RoleMatchType.EXACT_TITLE, exact_title
    for alias in family.aliases:
        alias_key = _key(alias)
        if _phrase_matches(title, alias_key):
            return RoleMatchType.ALIAS, alias
    for pattern in family.include_patterns:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return RoleMatchType.INCLUDE_PATTERN, pattern
    return None


def _phrase_matches(title: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", title) is not None


def _key(value: str) -> str:
    return normalize_text(value).casefold()


__all__ = [
    "RoleFamily",
    "RoleFamilyClass",
    "RoleFamilyClassifier",
    "RoleFamilyMatch",
    "RoleFamilyPolicy",
    "RoleMatchType",
    "UnknownRoleAction",
    "default_ai_data_role_family_policy",
]
