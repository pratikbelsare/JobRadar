import json
from uuid import uuid4

import pytest

from job_intelligence.ai import (
    AIProviderError,
    AIResponseError,
    AIResponseMetadata,
    AIResult,
    BedrockProvider,
    ExplanationReference,
    ExplanationSource,
    JobMatchExplanation,
)
from job_intelligence.config import Settings
from job_intelligence.explanations import (
    ExplanationGenerationError,
    ExplanationGroundingError,
    ExplanationService,
    build_explanation_context,
    validate_explanation_grounding,
)
from job_intelligence.feedback import (
    FeedbackNotFoundError,
    JsonUserFeedbackRepository,
    UserFeedbackService,
)
from job_intelligence.matching import HybridJobMatcher
from job_intelligence.models import (
    CandidateProfile,
    FeedbackLabel,
    FeedbackReason,
    Job,
)


class FakeExplanationProvider:
    def __init__(self, explanation: JobMatchExplanation | None = None, error=None) -> None:
        self.explanation = explanation
        self.error = error
        self.calls = 0
        self.contexts = []

    def generate_explanation(self, context):
        self.calls += 1
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return AIResult(
            value=self.explanation,
            metadata=AIResponseMetadata(
                provider="fake",
                model_id="fake-explanation-model",
                prompt_version="job-match-explanation-v1",
            ),
        )


class FakeConverseClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads

    def converse(self, **kwargs):
        return self.payloads.pop(0)

    def invoke_model(self, **kwargs):
        raise AssertionError("embedding transport should not be used")


def make_inputs() -> tuple[Job, CandidateProfile]:
    job = Job(
        source_job_id="job-1",
        company_id=uuid4(),
        title="Java Developer",
        description="Build Java services.",
        required_skills=["Java"],
        source_url="https://example.test/jobs/1",
    )
    profile = CandidateProfile(
        target_roles=["Java Developer"],
        skills=["Java"],
        experience_years=2,
    )
    return job, profile


def make_explanation() -> JobMatchExplanation:
    return JobMatchExplanation(
        why_fit=["The target role matches Java Developer."],
        strongest_matches=["Java is listed in the candidate skills."],
        potential_gaps=["No evidence found for Kubernetes."],
        experience_alignment="Candidate experience is available for comparison.",
        important_considerations=["Review the complete job description."],
        evidence_references=[
            ExplanationReference(
                source=ExplanationSource.CANDIDATE,
                claim="Java skill match",
                evidence="Java",
            ),
            ExplanationReference(
                source=ExplanationSource.JOB,
                claim="Role title",
                evidence="Java Developer",
            ),
        ],
    )


def test_valid_explanation_is_grounded_and_score_is_unchanged() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    original_score = match.final_score
    provider = FakeExplanationProvider(make_explanation())

    result = ExplanationService(provider).generate(job, profile, match)

    assert result.value.strongest_matches
    assert match.final_score == original_score
    assert provider.calls == 1
    assert provider.contexts[0].match_evidence["final_score"] == original_score


def test_explanation_generation_is_lazy_until_explicitly_requested() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    provider = FakeExplanationProvider(make_explanation())
    service = ExplanationService(provider)

    assert provider.calls == 0
    service.generate(job, profile, match)
    assert provider.calls == 1


def test_missing_evidence_language_is_allowed_but_unsupported_references_fail() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    context = build_explanation_context(job, profile, match)
    explanation = make_explanation()
    validate_explanation_grounding(explanation, context)

    bad_reference = explanation.model_copy(
        update={
            "evidence_references": [
                ExplanationReference(
                    source=ExplanationSource.CANDIDATE,
                    claim="unsupported",
                    evidence="Kubernetes experience",
                )
            ]
        }
    )
    with pytest.raises(ExplanationGroundingError):
        validate_explanation_grounding(bad_reference, context)


def test_explanations_reject_claims_that_assert_missing_skills() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    context = build_explanation_context(job, profile, match)
    explanation = make_explanation().model_copy(
        update={"potential_gaps": ["Candidate lacks Kubernetes."]}
    )

    with pytest.raises(ExplanationGroundingError, match="no evidence found"):
        validate_explanation_grounding(explanation, context)


def test_explanation_provider_failure_is_typed() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    provider = FakeExplanationProvider(error=AIProviderError("service unavailable"))

    with pytest.raises(ExplanationGenerationError):
        ExplanationService(provider).generate(job, profile, match)


def test_bedrock_explanation_output_is_schema_validated() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    context = build_explanation_context(job, profile, match)
    payload = {
        "output": {
            "message": {
                "content": [{"text": json.dumps(make_explanation().model_dump(mode="json"))}]
            }
        }
    }
    settings = Settings(
        aws_region="ap-south-1",
        job_analysis_model_id="provider/explanation-model",
        ai_max_attempts=1,
    )

    result = BedrockProvider(
        settings,
        client=FakeConverseClient([payload]),
    ).generate_explanation(context)

    assert result.value.experience_alignment
    assert result.metadata.prompt_version == "job-match-explanation-v1"


def test_bedrock_explanation_malformed_output_is_typed() -> None:
    job, profile = make_inputs()
    match = HybridJobMatcher().match(job, profile)
    context = build_explanation_context(job, profile, match)
    payload = {"output": {"message": {"content": [{"text": "not-json"}]}}}
    settings = Settings(
        aws_region="ap-south-1",
        job_analysis_model_id="provider/explanation-model",
        ai_max_attempts=1,
    )

    with pytest.raises(AIResponseError, match="JSON"):
        BedrockProvider(
            settings,
            client=FakeConverseClient([payload]),
        ).generate_explanation(context)


def test_feedback_create_update_load_and_export(tmp_path) -> None:
    job, profile = make_inputs()
    repository = JsonUserFeedbackRepository(tmp_path / "feedback.json")
    service = UserFeedbackService(repository)

    created = service.create(
        job_id=job.id,
        candidate_profile_id=profile.id,
        label=FeedbackLabel.WORTH_REVIEWING,
        reason=FeedbackReason.STRONG_FIT,
        notes="Review this role.",
    )
    updated = service.update(
        created.id,
        {"label": FeedbackLabel.STRONG_FIT, "notes": "Would apply."},
    )

    loaded = UserFeedbackService(JsonUserFeedbackRepository(tmp_path / "feedback.json")).load(
        created.id
    )
    exported = service.export_evaluation_examples()
    assert loaded == updated
    assert exported[0].job_id == job.id
    assert exported[0].label == FeedbackLabel.STRONG_FIT


def test_feedback_rejects_invalid_labels_and_context_changes(tmp_path) -> None:
    job, profile = make_inputs()
    service = UserFeedbackService(JsonUserFeedbackRepository(tmp_path / "feedback.json"))

    with pytest.raises(ValueError):
        service.create(job_id=job.id, candidate_profile_id=profile.id, label=4)
    created = service.create(
        job_id=job.id,
        candidate_profile_id=profile.id,
        label=FeedbackLabel.WEAK_FIT,
    )
    with pytest.raises(ValueError, match="job_id"):
        service.update(created.id, {"job_id": uuid4()})
    with pytest.raises(FeedbackNotFoundError):
        service.update(uuid4(), {"label": FeedbackLabel.IRRELEVANT})


def test_feedback_repository_reports_corrupt_storage(tmp_path) -> None:
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")

    with pytest.raises(Exception, match="Could not load feedback"):
        JsonUserFeedbackRepository(path).list()
