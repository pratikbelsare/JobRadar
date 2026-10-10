import json
from uuid import uuid4

import pytest

from job_intelligence.ai import (
    AIConfigurationError,
    AIResponseError,
    AIResult,
    AITimeoutError,
    AIValidationError,
    BedrockProvider,
    CandidateProfileProposal,
    EmbeddingVector,
    JobRequirements,
    merge_candidate_profile_proposal,
)
from job_intelligence.config import Settings
from job_intelligence.models import CandidateProfile


class FakeBody:
    def __init__(self, value: str) -> None:
        self.value = value

    def read(self) -> bytes:
        return self.value.encode("utf-8")


class FakeBedrockClient:
    def __init__(self, *, conversations=None, embeddings=None, error=None) -> None:
        self.conversations = list(conversations or [])
        self.embeddings = list(embeddings or [])
        self.error = error
        self.converse_calls: list[dict] = []
        self.embedding_calls: list[dict] = []

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.conversations.pop(0)

    def invoke_model(self, **kwargs):
        self.embedding_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"body": FakeBody(json.dumps(self.embeddings.pop(0)))}


def settings() -> Settings:
    return Settings(
        aws_region="ap-south-1",
        job_analysis_model_id="provider/job-model",
        embedding_model_id="provider/embedding-model",
        ai_max_attempts=2,
    )


def converse_payload(value: dict) -> dict:
    return {
        "output": {"message": {"content": [{"text": json.dumps(value)}]}},
        "usage": {"inputTokens": 12, "outputTokens": 8},
    }


def test_valid_job_extraction_is_structured_and_traced() -> None:
    client = FakeBedrockClient(
        conversations=[
            converse_payload(
                {
                    "min_experience_years": 2,
                    "required_skills": ["Python"],
                    "preferred_skills": ["AWS"],
                    "responsibilities": ["Build services"],
                }
            )
        ]
    )
    result = BedrockProvider(settings(), client=client).extract_job_requirements(
        "Build services with Python."
    )

    assert isinstance(result.value, JobRequirements)
    assert result.value.required_skills == ["Python"]
    assert result.value.preferred_skills == ["AWS"]
    assert result.metadata.model_id == "provider/job-model"
    assert result.metadata.prompt_version == "job-requirements-v1"
    assert result.metadata.input_tokens == 12


def test_missing_optional_fields_remain_unknown() -> None:
    client = FakeBedrockClient(conversations=[converse_payload({})])

    result = BedrockProvider(settings(), client=client).extract_job_requirements("A role")

    assert result.value.min_experience_years is None
    assert result.value.required_skills == []
    assert result.value.seniority.value == "unknown"


def test_invalid_json_retries_with_a_stricter_prompt() -> None:
    client = FakeBedrockClient(
        conversations=[
            {
                "output": {"message": {"content": [{"text": "not-json"}]}},
            },
            converse_payload({"required_skills": ["Java"]}),
        ]
    )

    result = BedrockProvider(settings(), client=client).extract_job_requirements("Java role")

    assert result.value.required_skills == ["Java"]
    assert len(client.converse_calls) == 2
    retry_text = client.converse_calls[1]["messages"][0]["content"][0]["text"]
    assert "Previous output was invalid" in retry_text


def test_schema_failure_after_retry_is_typed() -> None:
    client = FakeBedrockClient(
        conversations=[
            converse_payload({"min_experience_years": -1}),
            converse_payload({"min_experience_years": -2}),
        ]
    )

    with pytest.raises(AIValidationError):
        BedrockProvider(settings(), client=client).extract_job_requirements("Role")


def test_candidate_extraction_returns_a_proposal_and_manual_values_win() -> None:
    client = FakeBedrockClient(
        conversations=[
            converse_payload(
                {
                    "target_roles": ["Java Developer"],
                    "skills": ["Java", "SQL"],
                    "experience_years": 3,
                }
            )
        ]
    )
    provider = BedrockProvider(settings(), client=client)
    proposal = provider.extract_candidate_profile("Java and SQL resume").value
    profile = CandidateProfile(
        id=uuid4(),
        target_roles=["Platform Engineer"],
        skills=["Go"],
        experience_years=5,
    )

    merged = merge_candidate_profile_proposal(profile, proposal)

    assert isinstance(proposal, CandidateProfileProposal)
    assert merged.target_roles == ["Platform Engineer"]
    assert merged.skills == ["Go"]
    assert merged.experience_years == 5


def test_embedding_output_is_typed_and_model_configurable() -> None:
    client = FakeBedrockClient(embeddings=[{"embedding": [0.1, 0.2, 0.3]}])

    result = BedrockProvider(settings(), client=client).embed_text("job title")

    assert isinstance(result, AIResult)
    assert isinstance(result.value, EmbeddingVector)
    assert result.value.dimensions == 3
    assert client.embedding_calls[0]["modelId"] == "provider/embedding-model"


def test_initial_nova_and_titan_model_ids_are_sent_to_bedrock() -> None:
    client = FakeBedrockClient(
        conversations=[converse_payload({"required_skills": ["Python"]})],
        embeddings=[{"embedding": [0.1, 0.2]}],
    )
    configured = Settings(
        aws_region="ap-south-1",
        job_analysis_model_id="amazon.nova-lite-v1:0",
        embedding_model_id="amazon.titan-embed-text-v2:0",
    )
    provider = BedrockProvider(configured, client=client)

    provider.extract_job_requirements("Build Python services")
    provider.embed_text("Python services")

    assert client.converse_calls[0]["modelId"] == "amazon.nova-lite-v1:0"
    assert client.embedding_calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"


def test_embedding_validation_failure_is_retried_and_reported() -> None:
    client = FakeBedrockClient(embeddings=[{"wrong": []}, {"wrong": []}])

    with pytest.raises(AIResponseError, match="Embedding"):
        BedrockProvider(settings(), client=client).embed_text("text")
    assert len(client.embedding_calls) == 2


def test_timeout_is_exposed_as_a_typed_failure() -> None:
    client = FakeBedrockClient(error=TimeoutError("timed out"))

    with pytest.raises(AITimeoutError):
        BedrockProvider(settings(), client=client, max_attempts=1).embed_text("text")


def test_missing_model_configuration_is_rejected_before_transport() -> None:
    client = FakeBedrockClient()
    incomplete = Settings(aws_region="ap-south-1")

    with pytest.raises(AIConfigurationError):
        BedrockProvider(incomplete, client=client).extract_job_requirements("text")
    assert not client.converse_calls
