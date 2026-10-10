from __future__ import annotations

import pytest

from job_intelligence.role_policy import (
    RoleFamily,
    RoleFamilyClass,
    RoleFamilyPolicy,
    default_ai_data_role_family_policy,
)
from job_intelligence.role_title_normalization import (
    NormalizedRoleFamilyClassifier,
    RoleTitleNormalizer,
    TitleSeniority,
)


@pytest.mark.parametrize(
    ("title", "base", "seniority", "specialization", "level"),
    [
        (
            "Principal Machine Learning Engineer",
            "Machine Learning Engineer",
            [TitleSeniority.PRINCIPAL],
            [],
            None,
        ),
        (
            "Senior Machine Learning Engineer, Voice AI",
            "Machine Learning Engineer",
            [TitleSeniority.SENIOR],
            ["Voice AI"],
            None,
        ),
        (
            "Machine Learning Engineer - Inference",
            "Machine Learning Engineer",
            [],
            ["Inference"],
            None,
        ),
        ("Applied Scientist III", "Applied Scientist", [], [], "III"),
        (
            "Principal Data Scientist / Algorithm Engineer",
            "Data Scientist",
            [TitleSeniority.PRINCIPAL],
            ["Algorithm Engineer"],
            None,
        ),
        (
            "Machine Learning Engineer (머신러닝 엔지니어)",
            "Machine Learning Engineer",
            [],
            [],
            None,
        ),
        (
            "Sr. AI Engineer (AI/ML Inference)",
            "AI Engineer",
            [TitleSeniority.SENIOR],
            ["AI/ML Inference"],
            None,
        ),
        (
            "Senior / Staff AI Engineer",
            "AI Engineer",
            [TitleSeniority.SENIOR, TitleSeniority.STAFF],
            [],
            None,
        ),
    ],
)
def test_title_normalization_extracts_structure(
    title: str,
    base: str,
    seniority: list[TitleSeniority],
    specialization: list[str],
    level: str | None,
) -> None:
    result = RoleTitleNormalizer().normalize(title)

    assert result.base_title == base
    assert result.seniority == seniority
    assert result.specializations == specialization
    assert result.level == level


@pytest.mark.parametrize(
    "title",
    [
        "Principal Machine Learning Engineer",
        "Senior Machine Learning Engineer, Voice AI",
        "Machine Learning Engineer - Inference",
        "Applied Scientist III",
        "Principal Data Scientist / Algorithm Engineer",
        "Machine Learning Engineer (머신러닝 엔지니어)",
        "Sr. AI Engineer (AI/ML Inference)",
        "Senior / Staff AI Engineer",
    ],
)
def test_structural_variants_match_existing_core_taxonomy(title: str) -> None:
    result = NormalizedRoleFamilyClassifier(
        default_ai_data_role_family_policy()
    ).classify(title)

    assert result.match.classification == RoleFamilyClass.CORE
    assert result.match.matched_by is not None


@pytest.mark.parametrize(
    "title",
    [
        "Product Manager, AI Infrastructure",
        "Staff Engineer 1 – Salesforce AI",
        "Senior Data Engineer",
        "Machine Learning Operations Engineer",
        "AI Transformation Architect",
        "AI Transformation Engineer",
        "Software Engineer – AI/ML",
        "Data Engineer",
    ],
)
def test_misleading_ai_ml_data_tokens_do_not_become_core(title: str) -> None:
    result = NormalizedRoleFamilyClassifier(
        default_ai_data_role_family_policy()
    ).classify(title)

    assert result.match.classification != RoleFamilyClass.CORE


def test_security_exclusion_is_preserved_before_normalization() -> None:
    result = NormalizedRoleFamilyClassifier(
        default_ai_data_role_family_policy()
    ).classify("Staff Security Engineer")

    assert result.match.classification == RoleFamilyClass.EXCLUDE
    assert result.used_normalized_base is False


def test_senior_data_analyst_maps_to_existing_adjacent_family() -> None:
    result = NormalizedRoleFamilyClassifier(
        default_ai_data_role_family_policy()
    ).classify("Senior Data Analyst, Reporting")

    assert result.match.classification == RoleFamilyClass.ADJACENT


def test_normalization_works_with_an_unrelated_configurable_taxonomy() -> None:
    policy = RoleFamilyPolicy(
        families=[
            RoleFamily(
                id="backend",
                canonical_name="Backend Engineering",
                classification=RoleFamilyClass.CORE,
                exact_titles=["Backend Engineer"],
            )
        ],
        target_family_ids=["backend"],
    )

    result = NormalizedRoleFamilyClassifier(policy).classify(
        "Senior Backend Engineer - Payments"
    )

    assert result.match.family_id == "backend"
    assert result.title.base_title == "Backend Engineer"
