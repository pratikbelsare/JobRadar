from typing import cast
from uuid import uuid4

import pytest

from job_intelligence.ai import AIResponseMetadata, AIResult, EmbeddingVector
from job_intelligence.assessment import JobAssessmentService
from job_intelligence.eligibility import (
    EligibilityConfig,
    EligibilityEvaluator,
    EligibilityStatus,
)
from job_intelligence.matching import HybridJobMatcher
from job_intelligence.models import (
    CandidateProfile,
    HardConstraints,
    Job,
    JobAnalysis,
    WorkExperience,
    WorkMode,
)
from job_intelligence.role_policy import (
    RoleFamilyClassifier,
    default_ai_data_role_family_policy,
)


class FlatEmbeddingProvider:
    def embed_text(self, _: str) -> AIResult[EmbeddingVector]:
        return AIResult(
            value=EmbeddingVector(values=[1.0, 0.0]),
            metadata=AIResponseMetadata(provider="test", model_id="flat"),
        )


def make_job(**overrides: object) -> Job:
    values: dict[str, object] = {
        "source_job_id": "job-1",
        "company_id": uuid4(),
        "title": "Data Scientist",
        "location": "Pune",
        "work_mode": WorkMode.ONSITE,
        "description": "Build data science systems.",
        "source_url": "https://example.test/jobs/1",
    }
    values.update(overrides)
    return Job(**values)


def make_profile(**overrides: object) -> CandidateProfile:
    values: dict[str, object] = {
        "target_roles": ["Data Scientist"],
        "experience_years": 1.0,
        "preferred_locations": ["Pune"],
        "skills": ["Python"],
        "domains": ["Payments"],
        "work_experience": [
            WorkExperience(
                employer="Example",
                role="Machine Learning Engineer",
                description="Build data science systems.",
                skills=["Python"],
            )
        ],
    }
    values.update(overrides)
    return CandidateProfile(**values)


def make_analysis(job: Job, **overrides: object) -> JobAnalysis:
    values: dict[str, object] = {
        "job_id": job.id,
        "job_version_id": uuid4(),
        "model_id": "test-model",
        "prompt_version": "test-prompt",
        "required_skills": ["Python"],
        "responsibilities": ["Build data science systems"],
        "domain": "Payments",
        "min_experience_years": 1.0,
    }
    values.update(overrides)
    return JobAnalysis(**values)


def assessment_service() -> JobAssessmentService:
    return JobAssessmentService(
        matcher=HybridJobMatcher(embedding_provider=FlatEmbeddingProvider()),
        eligibility=EligibilityEvaluator(
            EligibilityConfig(acceptable_gap_years=1, max_stretch_gap_years=2)
        ),
        role_families=RoleFamilyClassifier(default_ai_data_role_family_policy()),
    )


def test_relevance_excludes_experience_but_legacy_score_does_not() -> None:
    job = make_job()
    matcher = HybridJobMatcher(embedding_provider=FlatEmbeddingProvider())
    aligned = matcher.match(
        job,
        make_profile(),
        analysis=make_analysis(job, min_experience_years=1),
    )
    severe = matcher.match(
        job,
        make_profile(),
        analysis=make_analysis(job, min_experience_years=10),
    )

    assert aligned.relevance_score == pytest.approx(severe.relevance_score)
    assert aligned.final_score > severe.final_score
    assert aligned.legacy_final_score == aligned.final_score


def test_relevance_excludes_location_but_legacy_score_does_not() -> None:
    matcher = HybridJobMatcher(embedding_provider=FlatEmbeddingProvider())
    analysis = make_analysis(make_job())
    preferred = matcher.match(make_job(location="Pune"), make_profile(), analysis=analysis)
    elsewhere = matcher.match(make_job(location="Mumbai"), make_profile(), analysis=analysis)

    assert preferred.relevance_score == pytest.approx(elsewhere.relevance_score)
    assert preferred.final_score > elsewhere.final_score


def test_assessment_exposes_existing_eligibility_and_role_family_metadata() -> None:
    job = make_job()
    profile = make_profile(
        hard_constraints=HardConstraints(remote_only=True),
    )
    result = assessment_service().evaluate(
        job,
        profile,
        make_analysis(job, min_experience_years=4),
    )

    assert result.role_family.classification.value == "core"
    assert result.match.role_family_class == "core"
    assert result.eligibility.status == EligibilityStatus.INELIGIBLE
    assert result.match.eligibility is not None
    assert result.match.eligibility.experience_gap_years == 3
    assert result.match.eligibility.experience_status == "severe_stretch"
    assert "remote_only_mismatch" in result.match.eligibility.hard_block_reasons
    assert result.match.eligibility.visible is False


def test_role_family_is_metadata_without_a_numeric_bonus() -> None:
    service = assessment_service()
    core_job = make_job(title="Data Scientist")
    adjacent_job = make_job(title="Data Analyst")
    profile = make_profile()
    core = service.evaluate(core_job, profile, make_analysis(core_job))
    adjacent = service.evaluate(adjacent_job, profile, make_analysis(adjacent_job))

    assert core.match.role_family_class == "core"
    assert adjacent.match.role_family_class == "adjacent"
    assert core.match.relevance_score == pytest.approx(adjacent.match.relevance_score)


def test_missing_relevance_components_are_excluded() -> None:
    job = make_job()
    result = HybridJobMatcher().match(
        job,
        make_profile(),
        analysis=make_analysis(job, responsibilities=[], domain=None),
    )

    assert result.relevance_evidence is not None
    assert result.relevance_evidence.excluded_components == ["responsibility", "domain"]
    assert result.relevance_evidence.denominator == pytest.approx(0.5)


def test_genuine_relevance_mismatch_zero_remains_included() -> None:
    job = make_job()
    result = HybridJobMatcher().match(
        job,
        make_profile(),
        analysis=make_analysis(job, domain="Healthcare"),
    )

    assert result.relevance_evidence is not None
    assert result.relevance_evidence.components["domain"].raw_score == 0
    assert result.relevance_evidence.components["domain"].available is True
    assert "domain" in result.relevance_evidence.included_components
    assert result.relevance_evidence.denominator == pytest.approx(0.7)


def test_legacy_match_payload_remains_valid_without_new_fields() -> None:
    job = make_job()
    match = HybridJobMatcher().match(job, make_profile(), analysis=make_analysis(job))
    evidence = cast(dict[str, float], match.evidence["aggregation"])

    assert match.legacy_final_score == match.final_score
    assert match.final_score == pytest.approx(
        evidence["numerator"] / evidence["denominator"]
    )
    legacy_payload = match.model_dump(mode="python")
    for field in (
        "legacy_final_score",
        "relevance_score",
        "relevance_evidence",
        "eligibility",
        "role_family_class",
    ):
        legacy_payload.pop(field)
    restored = type(match).model_validate(legacy_payload)
    assert restored.final_score == match.final_score
    assert restored.relevance_score is None
    assert restored.eligibility is None
