from uuid import uuid4

import pytest

from job_intelligence.config import Settings
from job_intelligence.filtering import (
    DeterministicJobFilter,
    ExperienceFilterConfig,
    FilteringConfig,
    FilterOutcome,
    FilterRule,
    RuleStatus,
)
from job_intelligence.models import (
    CandidatePreferences,
    CandidateProfile,
    EmploymentType,
    HardConstraints,
    Job,
    WorkMode,
)


def filter_config(
    *,
    acceptable_gap: float = 0.5,
    stretch_gap: float = 1.5,
) -> FilteringConfig:
    return FilteringConfig(
        experience=ExperienceFilterConfig(
            acceptable_gap_years=acceptable_gap,
            max_stretch_gap_years=stretch_gap,
        ),
        role_aliases={"software engineer": ["swe", "software developer"]},
        location_aliases={"bengaluru": ["bangalore"]},
    )


def make_job(**overrides) -> Job:
    values = {
        "source_job_id": "job-1",
        "company_id": uuid4(),
        "title": "Software Engineer",
        "location": "Bengaluru, India",
        "description": "Build software systems.",
        "source_url": "https://example.test/jobs/1",
        "min_experience_years": 2,
        "max_experience_years": 5,
    }
    values.update(overrides)
    return Job(**values)


def make_profile(**overrides) -> CandidateProfile:
    values = {
        "target_roles": ["Software Engineer"],
        "experience_years": 2,
        "preferred_locations": ["Bengaluru"],
    }
    values.update(overrides)
    return CandidateProfile(**values)


def evaluate(job: Job, profile: CandidateProfile, config: FilteringConfig | None = None):
    return DeterministicJobFilter(config or filter_config()).evaluate(job, profile)


def test_exact_role_match_passes() -> None:
    result = evaluate(make_job(), make_profile())

    assert result.passed is True
    assert result.outcome == FilterOutcome.PASS
    assert result.rules[0].rule == FilterRule.ROLE
    assert result.rules[0].status == RuleStatus.PASS


def test_role_alias_match_passes() -> None:
    job = make_job(title="SWE II")

    result = evaluate(job, make_profile())

    assert result.passed is True
    assert result.rules[0].details["matched_term"] == "swe"


def test_unrelated_role_is_rejected_early() -> None:
    result = evaluate(make_job(title="Product Marketing Manager"), make_profile())

    assert result.passed is False
    assert result.outcome == FilterOutcome.REJECT_ROLE
    assert len(result.rules) == 1
    assert "does not match" in result.reasons[-1]


def test_location_alias_match_passes() -> None:
    result = evaluate(make_job(location="Bangalore, India"), make_profile())

    assert result.passed is True
    assert result.rules[1].status == RuleStatus.PASS


def test_remote_preference_accepts_remote_job_and_rejects_onsite_job() -> None:
    profile = make_profile(
        preferred_locations=[],
        preferences=CandidatePreferences(preferred_work_modes=[WorkMode.REMOTE]),
    )
    remote = evaluate(make_job(location="Remote", work_mode=WorkMode.REMOTE), profile)
    onsite = evaluate(make_job(location="Bengaluru", work_mode=WorkMode.ONSITE), profile)

    assert remote.passed is True
    assert onsite.outcome == FilterOutcome.REJECT_LOCATION


def test_excluded_location_is_rejected() -> None:
    profile = make_profile(
        hard_constraints=HardConstraints(excluded_locations=["Bengaluru"])
    )

    result = evaluate(make_job(), profile)

    assert result.passed is False
    assert result.outcome == FilterOutcome.REJECT_LOCATION
    assert result.rules[1].status == RuleStatus.REJECT


def test_strong_experience_match_passes() -> None:
    result = evaluate(make_job(min_experience_years=1, max_experience_years=3), make_profile())

    assert result.outcome == FilterOutcome.PASS
    assert result.rules[2].status == RuleStatus.PASS


def test_small_experience_gap_is_acceptable() -> None:
    result = evaluate(
        make_job(min_experience_years=2),
        make_profile(experience_years=1.5),
    )

    assert result.passed is True
    assert result.outcome == FilterOutcome.PASS
    assert result.rules[2].status == RuleStatus.ACCEPTABLE
    assert result.rules[2].details["experience_gap_years"] == 0.5


def test_experience_stretch_passes_with_stretch_outcome() -> None:
    result = evaluate(
        make_job(min_experience_years=3),
        make_profile(experience_years=2),
    )

    assert result.passed is True
    assert result.outcome == FilterOutcome.STRETCH
    assert result.rules[2].status == RuleStatus.STRETCH


def test_extreme_experience_mismatch_is_rejected_early() -> None:
    result = evaluate(
        make_job(min_experience_years=5),
        make_profile(experience_years=1),
    )

    assert result.passed is False
    assert result.outcome == FilterOutcome.REJECT_EXPERIENCE
    assert len(result.rules) == 3


def test_multiple_rule_results_are_retained_for_a_pass() -> None:
    result = evaluate(make_job(), make_profile())

    assert [rule.rule for rule in result.rules] == [
        FilterRule.ROLE,
        FilterRule.LOCATION,
        FilterRule.EXPERIENCE,
        FilterRule.CONSTRAINT,
    ]
    assert len(result.reasons) == 4


def test_thresholds_are_configurable() -> None:
    job = make_job(min_experience_years=2)
    profile = make_profile(experience_years=1.5)

    permissive = evaluate(job, profile, filter_config(acceptable_gap=0.5, stretch_gap=1.0))
    strict = evaluate(job, profile, filter_config(acceptable_gap=0.1, stretch_gap=0.2))

    assert permissive.passed is True
    assert strict.outcome == FilterOutcome.REJECT_EXPERIENCE


def test_filtering_thresholds_can_be_built_from_settings() -> None:
    settings = Settings(
        filter_acceptable_experience_gap_years=0.25,
        filter_max_stretch_experience_gap_years=1.0,
    )

    config = FilteringConfig.from_settings(settings)

    assert config.experience.acceptable_gap_years == 0.25
    assert config.experience.max_stretch_gap_years == 1.0


def test_filtering_requires_configured_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        FilteringConfig.from_settings(Settings())


def test_unknown_job_fields_are_not_treated_as_semantic_rejections() -> None:
    job = make_job(
        location=None,
        work_mode=WorkMode.UNKNOWN,
        min_experience_years=None,
        max_experience_years=None,
    )

    result = evaluate(job, make_profile(preferred_locations=["Bengaluru"]))

    assert result.passed is True
    assert result.outcome == FilterOutcome.PASS
    assert result.rules[1].status == RuleStatus.ACCEPTABLE
    assert result.rules[2].status == RuleStatus.PASS


def test_explicit_constraints_are_hard_rejections() -> None:
    profile = make_profile(
        hard_constraints=HardConstraints(
            excluded_employment_types=[EmploymentType.CONTRACT],
            excluded_terms=["clearance"],
        )
    )
    contract = evaluate(make_job(employment_type=EmploymentType.CONTRACT), profile)
    excluded_text = evaluate(
        make_job(description="Requires active clearance."),
        make_profile(
            hard_constraints=HardConstraints(excluded_terms=["clearance"])
        ),
    )

    assert contract.outcome == FilterOutcome.REJECT_CONSTRAINT
    assert excluded_text.outcome == FilterOutcome.REJECT_CONSTRAINT


def test_remote_only_is_distinct_from_a_soft_remote_preference() -> None:
    profile = make_profile(
        preferred_locations=[],
        hard_constraints=HardConstraints(remote_only=True),
    )

    result = evaluate(make_job(location="Bengaluru", work_mode=WorkMode.ONSITE), profile)

    assert result.outcome == FilterOutcome.REJECT_CONSTRAINT


def test_non_ai_role_is_supported_without_special_cases() -> None:
    profile = make_profile(target_roles=["Java Developer"])
    job = make_job(title="Java Developer", description="Build JVM services.")

    result = evaluate(job, profile)

    assert result.passed is True
