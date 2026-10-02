import json

import pytest

from job_intelligence.ai import (
    AIResponseError,
    BedrockMantleProvider,
    HuggingFaceEmbeddingProvider,
    build_ai_providers,
)
from job_intelligence.config import Settings


class FakeMantleClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat_completion(self, *, model_id: str, messages: list[dict[str, str]]) -> dict:
        self.calls.append({"model_id": model_id, "messages": messages})
        return self.responses.pop(0)


class FakeSentenceTransformer:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.normalize_embeddings: bool | None = None

    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.normalize_embeddings = normalize_embeddings
        return self.values


def mantle_response(value: object, *, input_tokens: int = 11) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(value)}}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": 7},
    }


def fallback_settings() -> Settings:
    return Settings(
        aws_region="ap-south-1",
        llm_provider="bedrock_mantle",
        embedding_provider="huggingface_local",
        mantle_model_id="openai.gpt-oss-20b",
        local_embedding_model_id="BAAI/bge-small-en-v1.5",
    )


def test_mantle_provider_reuses_structured_schema_and_metadata() -> None:
    client = FakeMantleClient(
        [
            mantle_response(
                {
                    "min_experience_years": 2,
                    "required_skills": ["Python"],
                    "responsibilities": ["Build services"],
                }
            )
        ]
    )

    result = BedrockMantleProvider(fallback_settings(), client=client).extract_job_requirements(
        "Build Python services"
    )

    assert result.value.required_skills == ["Python"]
    assert result.metadata.provider == "bedrock_mantle"
    assert result.metadata.model_id == "openai.gpt-oss-20b"
    assert result.metadata.input_tokens == 11
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_mantle_provider_retries_invalid_structured_output() -> None:
    client = FakeMantleClient(
        [
            mantle_response({"min_experience_years": -1}),
            mantle_response({"required_skills": ["SQL"]}),
        ]
    )

    result = BedrockMantleProvider(fallback_settings(), client=client).extract_job_requirements(
        "SQL role"
    )

    assert result.value.required_skills == ["SQL"]
    assert len(client.calls) == 2
    assert "Previous output was invalid" in client.calls[1]["messages"][1]["content"]


def test_mantle_provider_rejects_malformed_completion() -> None:
    client = FakeMantleClient([{"choices": []}, {"choices": []}])

    with pytest.raises(AIResponseError):
        BedrockMantleProvider(fallback_settings(), client=client).extract_job_requirements("role")


def test_local_embedding_provider_normalizes_and_preserves_dimensions() -> None:
    model = FakeSentenceTransformer([0.6, 0.8, 0.0])
    provider = HuggingFaceEmbeddingProvider.from_settings(fallback_settings(), model=model)

    result = provider.embed_text("Python services")

    assert result.metadata.provider == "huggingface_local"
    assert result.metadata.model_id == "BAAI/bge-small-en-v1.5"
    assert result.value.dimensions == 3
    assert result.value.values == [0.6, 0.8, 0.0]
    assert sum(value * value for value in result.value.values) == pytest.approx(1.0)
    assert model.normalize_embeddings is True


def test_provider_selection_is_configuration_driven() -> None:
    providers = build_ai_providers(
        fallback_settings(),
        mantle_client=FakeMantleClient([mantle_response({})]),
        local_model=FakeSentenceTransformer([1.0, 0.0]),
    )

    assert isinstance(providers.llm, BedrockMantleProvider)
    assert isinstance(providers.embeddings, HuggingFaceEmbeddingProvider)


def test_provider_selection_defaults_to_native_bedrock_path() -> None:
    settings = Settings(aws_region="ap-south-1", job_analysis_model_id="job-model")
    providers = build_ai_providers(
        settings,
        bedrock_client=object(),
    )

    assert type(providers.llm).__name__ == "BedrockProvider"
    assert type(providers.embeddings).__name__ == "BedrockProvider"


def test_provider_selection_settings_load_from_environment() -> None:
    settings = Settings.from_environment(
        {
            "JOB_INTELLIGENCE_AWS_REGION": "ap-south-1",
            "JOB_INTELLIGENCE_LLM_PROVIDER": "bedrock_mantle",
            "JOB_INTELLIGENCE_EMBEDDING_PROVIDER": "huggingface_local",
            "JOB_INTELLIGENCE_MANTLE_MODEL_ID": "openai.gpt-oss-20b",
            "JOB_INTELLIGENCE_MANTLE_MAX_TOKENS": "8192",
            "JOB_INTELLIGENCE_LOCAL_EMBEDDING_MODEL_ID": "BAAI/bge-small-en-v1.5",
            "JOB_INTELLIGENCE_LOCAL_EMBEDDING_DEVICE": "cpu",
        }
    )

    assert settings.llm_provider.value == "bedrock_mantle"
    assert settings.embedding_provider.value == "huggingface_local"
    assert settings.mantle_model_id == "openai.gpt-oss-20b"
    assert settings.mantle_max_tokens == 8192
    assert settings.local_embedding_model_id == "BAAI/bge-small-en-v1.5"
