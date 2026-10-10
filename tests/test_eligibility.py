from uuid import uuid4

import pytest

from job_intelligence.config import Settings
from job_intelligence.eligibility import (
    EligibilityConfig,
    EligibilityEvaluator,
    EligibilityExperienceStatus,
    EligibilityStatus,
    SeniorityStatus,
)
from job_intelligence.models import (
    CandidateProfile,
    EmploymentType,
    HardConstraints,
    Job,
    JobAnalysis,
    Seniority,
    WorkMode,
)
from job_intelligence.role_policy import RoleFamilyClass


def make_job(**overrides: object) -> Job:
    values: dict[str, object] = {
        "source_job_id": "job-1",
        "company_id": uuid4(),
        "title": "Data Scientist",
        "location": "Pune",
        "description": "Build data products.",
        "source_url": "https://example.test/jobs/1",
    }
    values.update(overrides)
    return Job(**values)


def make_analysis(**overrides: object) -> JobAnalysis:
    values: dict[str, object] = {
        "job_id": uuid4(),
        "job_version_id": uuid4(),
        "model_id": "test-model",
        "prompt_version": "test-prompt",
    }
    values.update(overrides)
    return JobAnalysis(**values)


def make_profile(**overrides: object) -> CandidateProfile:
    values: dict[str, object] = {
        "experience_years": 1.2,
        "preferred_locations": ["Pune"],
    }
    values.update(overrides)
    return CandidateProfile(**values)


def evaluator(**overrides: object) -> EligibilityEvaluator:
    values: dict[str, object] = {
        "acceptable_gap_years": 1,
        "max_stretch_gap_years": 2,
    }
    values.update(overrides)
    return EligibilityEvaluator(EligibilityConfig(**values))


def test_eligibility_config_uses_existing_filter_threshold_settings() -> None:
    config = EligibilityConfig.from_settings(
        Settings(
            filter_acceptable_experience_gap_years=1,
            filter_max_stretch_experience_gap_years=2,
        )
    )

    assert config.acceptable_gap_years == 1
    assert config.max_stretch_gap_years == 2


def test_aligned_experience_is_eligible() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(min_experience_years=1),
    )

    assert result.status == EligibilityStatus.ELIGIBLE
    assert result.experience_status == EligibilityExperienceStatus.ALIGNED
    assert result.experience_gap_years == 0
    assert result.visible is True


def test_acceptable_experience_gap_remains_aligned() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(min_experience_years=2),
    )

    assert result.status == EligibilityStatus.ELIGIBLE
    assert result.experience_gap_years == pytest.approx(0.8)


def test_stretch_gap_is_visible() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(min_experience_years=3),
    )

    assert result.status == EligibilityStatus.STRETCH
    assert result.experience_status == EligibilityExperienceStatus.STRETCH
    assert result.visible is True


def test_severe_gap_is_visible_and_not_ineligible() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(min_experience_years=4),
    )

    assert result.status == EligibilityStatus.SEVERE_STRETCH
    assert result.visible is True
    assert result.recommended_to_retain is True
    assert result.hard_block_reasons == []


def test_unknown_experience_is_not_rejected() -> None:
    result = evaluator().evaluate(make_profile(), make_job(), make_analysis())

    assert result.status == EligibilityStatus.UNKNOWN
    assert result.experience_status == EligibilityExperienceStatus.UNKNOWN
    assert result.visible is True


def test_missing_seniority_is_explicitly_unknown() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(min_experience_years=1, seniority=Seniority.UNKNOWN),
    )

    assert result.seniority_status == SeniorityStatus.UNKNOWN
    assert "seniority_unknown" in result.reason_codes


def test_seniority_mismatch_is_a_stretch_signal_not_a_hard_blocker() -> None:
    result = evaluator(seniority_min_experience_years={Seniority.SENIOR: 3}).evaluate(
        make_profile(),
        make_job(),
        make_analysis(seniority=Seniority.SENIOR),
    )

    assert result.seniority_status == SeniorityStatus.MISMATCH
    assert result.status == EligibilityStatus.STRETCH
    assert result.visible is True
    assert "seniority_mismatch" in result.reason_codes


def test_excluded_employment_type_is_a_hard_blocker() -> None:
    result = evaluator().evaluate(
        make_profile(
            hard_constraints=HardConstraints(
                excluded_employment_types=[EmploymentType.CONTRACT]
            )
        ),
        make_job(),
        make_analysis(employment_type=EmploymentType.CONTRACT),
    )

    assert result.status == EligibilityStatus.INELIGIBLE
    assert result.visible is False
    assert result.hard_block_reasons == ["employment_type_excluded"]


def test_unknown_employment_type_does_not_create_a_blocker() -> None:
    result = evaluator().evaluate(
        make_profile(
            hard_constraints=HardConstraints(
                excluded_employment_types=[EmploymentType.CONTRACT]
            )
        ),
        make_job(),
        make_analysis(employment_type=EmploymentType.UNKNOWN),
    )

    assert result.status == EligibilityStatus.UNKNOWN
    assert result.hard_block_reasons == []


def test_remote_only_mismatch_is_a_hard_blocker() -> None:
    result = evaluator().evaluate(
        make_profile(hard_constraints=HardConstraints(remote_only=True)),
        make_job(work_mode=WorkMode.ONSITE),
        make_analysis(work_modes=[WorkMode.ONSITE]),
    )

    assert result.status == EligibilityStatus.INELIGIBLE
    assert "remote_only_mismatch" in result.hard_block_reasons


def test_missing_skills_and_education_are_not_hard_rejections() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(),
        make_analysis(
            required_skills=["Kubernetes"],
            education=["Computer Science"],
        ),
    )

    assert result.status == EligibilityStatus.UNKNOWN
    assert result.visible is True
    assert "skills_are_not_hard_blockers" in result.reason_codes
    assert "education_not_verifiable_as_mandatory" in result.reason_codes


def test_core_role_class_is_preserved_when_experience_is_severe() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(title="Data Scientist"),
        make_analysis(min_experience_years=4),
        role_family_class=RoleFamilyClass.CORE,
    )

    assert result.role_family_class == RoleFamilyClass.CORE
    assert result.status == EligibilityStatus.SEVERE_STRETCH


def test_adjacent_role_with_aligned_experience_is_eligible() -> None:
    result = evaluator().evaluate(
        make_profile(),
        make_job(title="Data QA Associate"),
        make_analysis(min_experience_years=1),
        role_family_class=RoleFamilyClass.ADJACENT,
    )

    assert result.role_family_class == RoleFamilyClass.ADJACENT
    assert result.status == EligibilityStatus.ELIGIBLE
