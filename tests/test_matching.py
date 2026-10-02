from uuid import uuid4

import pytest

from job_intelligence.ai import AIResponseMetadata, AIResult, EmbeddingVector
from job_intelligence.filtering import FilterOutcome, FilterResult
from job_intelligence.matching import (
    HybridJobMatcher,
    MatchingConfig,
    MatchingWeights,
    RejectedJobError,
)
from job_intelligence.models import (
    CandidatePreferences,
    CandidateProfile,
    Job,
    WorkExperience,
    WorkMode,
)


class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        self.calls.append(text)
        return AIResult(
            value=EmbeddingVector(values=self.vectors.get(text, [0.0, 1.0])),
            metadata=AIResponseMetadata(
                provider="fake",
                model_id="fake-embedding",
            ),
        )


def make_job(**overrides) -> Job:
    values = {
        "source_job_id": "job-1",
        "company_id": uuid4(),
        "title": "Java Developer",
        "location": "Bengaluru",
        "description": "Build Java services.",
        "source_url": "https://example.test/jobs/1",
        "min_experience_years": 2,
        "max_experience_years": 5,
        "required_skills": ["Java"],
        "preferred_skills": ["AWS"],
        "responsibilities": ["Build reliable backend services"],
        "domain": "Payments",
    }
    values.update(overrides)
    return Job(**values)


def make_profile(**overrides) -> CandidateProfile:
    values = {
        "target_roles": ["Java Developer"],
        "experience_years": 3,
        "preferred_locations": ["Bengaluru"],
        "skills": ["Java", "AWS"],
        "domains": ["Payments"],
        "work_experience": [
            WorkExperience(
                employer="Example",
                role="Engineer",
                description="Built reliable backend services.",
                skills=["Java"],
            )
        ],
    }
    values.update(overrides)
    return CandidateProfile(**values)


def passing_filter() -> FilterResult:
    return FilterResult(passed=True, outcome=FilterOutcome.PASS)


def test_matching_weights_are_validated_and_configurable() -> None:
    weights = MatchingWeights(
        role=0.4,
        skill=0.2,
        experience=0.1,
        responsibility=0.1,
        domain=0.1,
        location=0.05,
        preference=0.05,
    )
    match = HybridJobMatcher(MatchingConfig(weights=weights)).match(
        make_job(),
        make_profile(),
    )

    assert match.final_score == pytest.approx(
        match.role_match * 0.4
        + match.skill_match * 0.2
        + match.experience_match * 0.1
        + match.responsibility_match * 0.1
        + match.domain_match * 0.1
        + match.location_match * 0.05
        + match.preference_match * 0.05
    )
    with pytest.raises(ValueError, match="sum to 1"):
        MatchingWeights(role=0.1)


def test_strong_role_and_skill_match_preserve_component_evidence() -> None:
    result = HybridJobMatcher().match(make_job(), make_profile(), filter_result=passing_filter())

    assert result.role_match == 1.0
    assert result.skill_match == pytest.approx(0.925)
    assert result.evidence["components"]["skill"]["matched_required_skills"] == ["Java"]
    assert result.evidence["components"]["skill"]["required_skills_without_evidence"] == []


def test_role_alias_and_semantic_role_match_are_supported() -> None:
    provider = FakeEmbeddingProvider(
        {
            "Backend Engineer": [1.0, 0.0],
            "Java Developer": [1.0, 0.0],
        }
    )
    config = MatchingConfig(role_aliases={"java developer": ["backend engineer"]})
    alias_job = make_job(title="Backend Engineer")
    alias_result = HybridJobMatcher(config, embedding_provider=provider).match(
        alias_job,
        make_profile(),
    )
    semantic_job = make_job(title="Distributed Systems Builder")
    semantic_result = HybridJobMatcher(
        MatchingConfig(),
        embedding_provider=FakeEmbeddingProvider(
            {
                "Distributed Systems Builder": [0.9, 0.1],
                "Java Developer": [1.0, 0.0],
            }
        ),
    ).match(semantic_job, make_profile())

    assert alias_result.role_match == 1.0
    assert semantic_result.role_match > 0.8


def test_required_skill_evidence_scores_more_than_bare_skill_mentions() -> None:
    bare = HybridJobMatcher().match(
        make_job(required_skills=["Java"], preferred_skills=[]),
        make_profile(skills=["Java"], work_experience=[]),
    )
    evidenced = HybridJobMatcher().match(
        make_job(required_skills=["Java"], preferred_skills=[]),
        make_profile(),
    )

    assert evidenced.skill_match > bare.skill_match
    assert "Java" in bare.evidence["components"]["skill"]["required_skills_without_evidence"]


def test_experience_scores_are_smooth_for_small_and_large_gaps() -> None:
    small = HybridJobMatcher().match(
        make_job(min_experience_years=3),
        make_profile(experience_years=2.5),
    )
    large = HybridJobMatcher().match(
        make_job(min_experience_years=8, max_experience_years=10),
        make_profile(experience_years=1),
    )

    assert 0.0 < small.experience_match < 1.0
    assert large.experience_match < small.experience_match


def test_responsibility_similarity_uses_injected_embeddings() -> None:
    provider = FakeEmbeddingProvider(
        {
            "Build reliable backend services": [1.0, 0.0],
            "Built reliable backend services.": [1.0, 0.0],
        }
    )

    result = HybridJobMatcher(embedding_provider=provider).match(make_job(), make_profile())

    assert result.responsibility_match == 1.0
    assert result.evidence["components"]["responsibility"]["method"] == "embedding"


def test_domain_location_and_preference_components_are_scored() -> None:
    profile = make_profile(
        preferences=CandidatePreferences(
            preferred_employment_types=["full_time"],
            preferred_domains=["Payments"],
        )
    )
    job = make_job(employment_type="full_time")

    result = HybridJobMatcher().match(job, profile)

    assert result.domain_match == 1.0
    assert result.location_match == 1.0
    assert result.preference_match == 1.0


def test_missing_fields_use_neutral_scores_and_rejected_jobs_do_not_rank() -> None:
    job = make_job(
        location=None,
        work_mode=WorkMode.UNKNOWN,
        min_experience_years=None,
        max_experience_years=None,
        required_skills=[],
        preferred_skills=[],
        responsibilities=[],
        domain=None,
    )
    result = HybridJobMatcher().match(job, CandidateProfile())

    assert result.experience_match == 0.5
    assert result.location_match == 0.5
    assert result.domain_match == 0.5
    with pytest.raises(RejectedJobError):
        HybridJobMatcher().match(
            job,
            CandidateProfile(),
            filter_result=FilterResult(
                passed=False,
                outcome=FilterOutcome.REJECT_ROLE,
            ),
        )


def test_non_ai_role_is_role_agnostic_and_results_are_reproducible() -> None:
    job = make_job(title="DevOps Engineer")
    profile = make_profile(target_roles=["DevOps Engineer"])
    matcher = HybridJobMatcher()

    first = matcher.match(job, profile)
    second = matcher.match(job, profile)

    assert first.role_match == 1.0
    assert first.final_score == second.final_score
