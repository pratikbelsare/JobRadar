from uuid import uuid4

import pytest

from job_intelligence.config import Settings
from job_intelligence.filtering import (
    DeterministicJobFilter,
    ExperienceFilterConfig,
    ExperienceFitStatus,
    FilteringConfig,
)
from job_intelligence.models import CandidateProfile, Job
from job_intelligence.role_policy import (
    RoleFamily,
    RoleFamilyClass,
    RoleFamilyClassifier,
    RoleFamilyPolicy,
    UnknownRoleAction,
    default_ai_data_role_family_policy,
)


def make_job(title: str) -> Job:
    return Job(
        source_job_id="job-1",
        company_id=uuid4(),
        title=title,
        location="Pune, India",
        description="A role description.",
        source_url="https://example.test/jobs/1",
    )


def make_profile() -> CandidateProfile:
    return CandidateProfile(
        target_roles=["Machine Learning Engineer", "AI Engineer", "Data Scientist"],
        experience_years=1.2,
        preferred_locations=["Pune"],
    )


def policy_filter() -> DeterministicJobFilter:
    return DeterministicJobFilter(
        FilteringConfig(
            experience=ExperienceFilterConfig(
                acceptable_gap_years=1,
                max_stretch_gap_years=2,
            ),
            role_policy=default_ai_data_role_family_policy(),
        )
    )


@pytest.mark.parametrize(
    "title",
    [
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Scientist, Corporate",
    ],
)
def test_core_titles_pass_with_core_reason_code(title: str) -> None:
    result = policy_filter().evaluate(make_job(title), make_profile())

    assert result.passed is True
    assert result.rules[0].details["role_family_class"] == RoleFamilyClass.CORE.value
    assert result.rules[0].details["decision_code"] == "pass_core_role"


@pytest.mark.parametrize("title", ["ML Engineer II", "Applied AI Engineer", "LLM Engineer"])
def test_core_aliases_pass_without_substring_only_matching(title: str) -> None:
    result = policy_filter().evaluate(make_job(title), make_profile())

    assert result.passed is True
    assert result.rules[0].details["role_family_class"] == RoleFamilyClass.CORE.value


@pytest.mark.parametrize(
    "title",
    [
        "Data Analyst",
        "Data Product Analyst, Corporate",
        "Data QA Associate",
        "Investor Research Analyst",
        "Product Manager, AI",
        "Insights Associate - Retail",
    ],
)
def test_adjacent_roles_pass_for_deeper_analysis(title: str) -> None:
    result = policy_filter().evaluate(make_job(title), make_profile())

    assert result.passed is True
    assert result.rules[0].details["role_family_class"] == RoleFamilyClass.ADJACENT.value
    assert result.rules[0].details["decision_code"] == "pass_adjacent_role"


@pytest.mark.parametrize(
    "title",
    [
        "Business Development Associate",
        "Account Executive - Investor",
        "Client Account Management Associate",
        "Product Security Engineer",
        "Senior SDET",
        "IT Delivery Engineer II",
        "Senior Payroll & Benefits Administrator",
        "Associate Director of Product Development",
    ],
)
def test_explicit_exclude_families_reject_before_other_rules(title: str) -> None:
    result = policy_filter().evaluate(make_job(title), make_profile())

    assert result.passed is False
    assert result.rules[0].details["decision_code"] == "reject_role_family"
    assert len(result.rules) == 1


def test_ai_token_alone_does_not_make_generic_product_role_eligible() -> None:
    result = policy_filter().evaluate(make_job("Product Manager, AI Operations"), make_profile())

    assert result.passed is False
    assert result.rules[0].details["decision_code"] == "unknown_role_family"


def test_unknown_role_behavior_is_explicit_and_conservative() -> None:
    result = policy_filter().evaluate(make_job("Operations Manager"), make_profile())

    assert result.passed is False
    assert result.rules[0].details["role_family_class"] is None
    assert result.rules[0].details["decision_code"] == "unknown_role_family"


def test_policy_can_allow_unknown_roles_for_review() -> None:
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
        unknown_role_action=UnknownRoleAction.PASS,
    )
    classifier = RoleFamilyClassifier(policy)

    assert classifier.classify("Unclassified Role").classification is None
    result = DeterministicJobFilter(
        FilteringConfig(
            experience=ExperienceFilterConfig(
                acceptable_gap_years=1,
                max_stretch_gap_years=2,
            ),
            role_policy=policy,
        )
    ).evaluate(make_job("Unclassified Role"), CandidateProfile(target_roles=["Backend Engineer"]))

    assert result.passed is True
    assert result.rules[0].details["decision_code"] == "unknown_role_family"


def test_family_exclude_pattern_overrides_its_include_pattern() -> None:
    policy = RoleFamilyPolicy(
        families=[
            RoleFamily(
                id="engineering",
                canonical_name="Engineering",
                classification=RoleFamilyClass.CORE,
                include_patterns=[r"\bengineer\b"],
                exclude_patterns=[r"\bsecurity\b"],
            )
        ],
        target_family_ids=["engineering"],
    )

    match = RoleFamilyClassifier(policy).classify("Security Engineer")

    assert match.classification is None


def test_policy_is_generic_for_a_non_ai_profession() -> None:
    policy = RoleFamilyPolicy(
        families=[
            RoleFamily(
                id="java_backend",
                canonical_name="Java Backend Engineering",
                classification=RoleFamilyClass.CORE,
                exact_titles=["Java Developer"],
                aliases=["Java Engineer"],
            ),
            RoleFamily(
                id="java_support",
                canonical_name="Java Support",
                classification=RoleFamilyClass.ADJACENT,
                exact_titles=["Application Support Engineer"],
            ),
        ],
        target_family_ids=["java_backend"],
        adjacent_family_ids=["java_support"],
    )
    engine = DeterministicJobFilter(
        FilteringConfig(
            experience=ExperienceFilterConfig(
                acceptable_gap_years=1,
                max_stretch_gap_years=2,
            ),
            role_policy=policy,
        )
    )
    profile = CandidateProfile(target_roles=["Java Developer"], preferred_locations=["Pune"])

    assert engine.evaluate(make_job("Java Developer"), profile).passed is True
    assert engine.evaluate(make_job("Application Support Engineer"), profile).passed is True
    assert engine.evaluate(make_job("Python Developer"), profile).passed is False


def test_role_policy_can_be_loaded_from_environment_json() -> None:
    policy_json = (
        '{"families":[{"id":"backend","canonical_name":"Backend",'
        '"class":"core","exact_titles":["Backend Engineer"]}],'
        '"target_family_ids":["backend"]}'
    )
    settings = Settings.from_environment(
        {
            "JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS": "1",
            "JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS": "2",
            "JOB_INTELLIGENCE_ROLE_FAMILY_POLICY_JSON": policy_json,
        }
    )
    config = FilteringConfig.from_settings(settings)

    assert config.role_policy is not None
    assert config.role_policy.families[0].id == "backend"


def test_experience_status_is_separate_from_role_family() -> None:
    result = policy_filter().evaluate(
        make_job("Data Scientist, Corporate").model_copy(update={"min_experience_years": 2.5}),
        make_profile(),
    )

    assert result.passed is True
    assert result.outcome.value == "stretch"
    assert result.rules[0].details["role_family_class"] == RoleFamilyClass.CORE.value
    assert result.rules[2].details["experience_status"] == ExperienceFitStatus.STRETCH.value


def test_production_filter_exposes_normalized_title_evidence() -> None:
    result = policy_filter().evaluate(
        make_job("Senior Machine Learning Engineer, Voice AI"),
        make_profile(),
    )

    role = result.rules[0]
    assert result.passed is True
    assert role.details["original_title"] == "Senior Machine Learning Engineer, Voice AI"
    assert role.details["base_title"] == "Machine Learning Engineer"
    assert role.details["detected_seniority"] == ["senior"]
    assert role.details["specializations"] == ["Voice AI"]
    assert role.details["role_family_class"] == RoleFamilyClass.CORE.value
    assert role.details["title_used_for_match"] == "Machine Learning Engineer"


@pytest.mark.parametrize(
    "title",
    [
        "Senior Machine Learning Engineer",
        "Principal Machine Learning Engineer",
        "Staff Applied Scientist",
        "Machine Learning Engineer - Inference",
        "Sr. AI Engineer",
        "Senior / Staff AI Engineer",
        "Machine Learning Engineer (머신러닝 엔지니어)",
    ],
)
def test_production_classifier_uses_structural_normalization(title: str) -> None:
    match = RoleFamilyClassifier(default_ai_data_role_family_policy()).classify(title)

    assert match.classification == RoleFamilyClass.CORE


@pytest.mark.parametrize(
    "title",
    [
        "Staff Engineer - Salesforce AI",
        "Product Manager, AI Infrastructure",
        "Senior Data Engineer",
        "AI Transformation Architect",
        "AI Transformation Engineer",
        "Software Engineer",
    ],
)
def test_production_classifier_does_not_promote_keyword_only_titles(title: str) -> None:
    match = RoleFamilyClassifier(default_ai_data_role_family_policy()).classify(title)

    assert match.classification is None
