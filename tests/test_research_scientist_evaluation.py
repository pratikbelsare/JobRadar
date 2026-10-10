from __future__ import annotations

import pytest

from job_intelligence.research_scientist_evaluation import (
    ResearchScientistPolicy,
    ResearchScientistPolicyEvaluator,
)
from job_intelligence.role_policy import RoleFamilyClass

NEGATIVE_TITLES = (
    "Research Scientist - Chemistry",
    "Research Scientist - Biology",
    "Research Scientist - Materials Science",
    "Research Scientist - Drug Discovery",
    "Research Scientist - Clinical Research",
    "Research Scientist - Agriculture",
    "Research Scientist - Battery Materials",
)
POSITIVE_TITLES = (
    "AI Research Scientist",
    "Machine Learning Research Scientist",
    "Research Scientist - NLP",
    "Research Scientist - Computer Vision",
    "Research Scientist - Foundation Models",
)


@pytest.mark.parametrize("title", NEGATIVE_TITLES)
def test_generic_core_promotes_unrelated_research_titles(title: str) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.GENERIC_CORE,
    )

    assert decision.classification == RoleFamilyClass.CORE


@pytest.mark.parametrize("title", NEGATIVE_TITLES)
def test_generic_adjacent_retains_but_does_not_promote_unrelated_titles(title: str) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.GENERIC_ADJACENT,
    )

    assert decision.classification == RoleFamilyClass.ADJACENT


@pytest.mark.parametrize("title", NEGATIVE_TITLES)
def test_context_policy_rejects_unrelated_research_specializations(title: str) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.CONTEXT_QUALIFIED,
    )

    assert decision.classification is None
    assert decision.matched_qualifiers == []


@pytest.mark.parametrize("title", POSITIVE_TITLES)
def test_context_policy_promotes_explicit_ai_ml_research_titles(title: str) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.CONTEXT_QUALIFIED,
    )

    assert decision.classification == RoleFamilyClass.CORE
    assert decision.matched_qualifiers


def test_context_policy_retains_bare_research_scientist_as_adjacent() -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        "Research Scientist",
        ResearchScientistPolicy.CONTEXT_QUALIFIED,
    )

    assert decision.classification == RoleFamilyClass.ADJACENT
    assert decision.title_evidence == []


@pytest.mark.parametrize(
    ("title", "classification"),
    [
        ("Research Scientist - Human-AI Systems", RoleFamilyClass.CORE),
        ("Research Scientist - Frontier Benchmarks", RoleFamilyClass.ADJACENT),
        ("Research Scientist - Large Geometry Models", RoleFamilyClass.ADJACENT),
    ],
)
def test_context_policy_distinguishes_strong_and_weak_benchmark_title_evidence(
    title: str,
    classification: RoleFamilyClass,
) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.CONTEXT_QUALIFIED,
    )

    assert decision.classification == classification


@pytest.mark.parametrize(
    "title",
    [
        "SDE II - AI",
        "SDE IV (ML Engineering)",
        "Sr. Software Engineer, AI/ML Inference Platform",
        "Senior | Staff Software Engineer - AI/ML",
    ],
)
def test_research_policy_does_not_apply_to_sde_or_software_titles(title: str) -> None:
    decision = ResearchScientistPolicyEvaluator().evaluate(
        title,
        ResearchScientistPolicy.CONTEXT_QUALIFIED,
    )

    assert decision.applicable is False
    assert decision.classification is None
