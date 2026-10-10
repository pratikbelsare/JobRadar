from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from job_intelligence.ai import (
    AIConfigurationError,
    AIResponseError,
    AIValidationError,
    ExplanationContext,
    HttpOpenAICompatibleClient,
    HuggingFaceEmbeddingProvider,
    OpenAICompatibleChatProvider,
    build_ai_providers,
    configured_llm_model_id,
)
from job_intelligence.config import Settings


class FakeChatClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_completion(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        self.calls.append({"model_id": model_id, "messages": messages})
        return self.responses.pop(0)


class FakeSentenceTransformer:
    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        return [0.6, 0.8]


class FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def completion(value: object) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps(value)}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 8},
    }


def openai_settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "openai_compatible",
        "embedding_provider": "huggingface_local",
        "chat_model": "provider/model",
        "chat_base_url": "https://chat.example.test/v1",
        "chat_api_key": "top-secret-value",
        "local_embedding_model_id": "BAAI/bge-small-en-v1.5",
    }
    values.update(updates)
    return Settings.model_validate(values)


def explanation_context() -> ExplanationContext:
    return ExplanationContext(
        job_id="00000000-0000-0000-0000-000000000001",
        candidate_profile_id="00000000-0000-0000-0000-000000000002",
        job_title="Data Scientist",
        job_requirements={"required_skills": ["Python"]},
        candidate_evidence={"skills": ["Python"]},
        match_evidence={"relevance_score": 0.8},
    )


def grounded_explanation(evidence: str = "Python") -> dict[str, Any]:
    return {
        "why_fit": ["Python aligns."],
        "strongest_matches": ["Python"],
        "potential_gaps": [],
        "experience_alignment": "Review the supplied experience evidence.",
        "important_considerations": [],
        "evidence_references": [
            {
                "source": "candidate",
                "claim": "Python alignment",
                "evidence": evidence,
            }
        ],
    }


def test_chat_configuration_loads_from_environment_and_masks_secret() -> None:
    settings = Settings.from_environment(
        {
            "JOB_INTELLIGENCE_CHAT_PROVIDER": "openai_compatible",
            "JOB_INTELLIGENCE_CHAT_MODEL": "provider/model",
            "JOB_INTELLIGENCE_CHAT_BASE_URL": "https://chat.example.test/v1",
            "JOB_INTELLIGENCE_CHAT_API_KEY": "top-secret-value",
            "JOB_INTELLIGENCE_CHAT_MAX_TOKENS": "2048",
            "JOB_INTELLIGENCE_CHAT_TIMEOUT_SECONDS": "30",
        }
    )

    assert settings.llm_provider.value == "openai_compatible"
    assert settings.chat_model == "provider/model"
    assert settings.chat_base_url == "https://chat.example.test/v1"
    assert settings.chat_api_key is not None
    assert settings.chat_api_key.get_secret_value() == "top-secret-value"
    assert "top-secret-value" not in repr(settings)
    assert settings.chat_max_tokens == 2048
    assert settings.chat_timeout_seconds == 30


def test_unsupported_chat_provider_fails_during_configuration() -> None:
    with pytest.raises(ValueError, match="not a valid LLMProvider"):
        Settings.from_environment(
            {"JOB_INTELLIGENCE_CHAT_PROVIDER": "unsupported-provider"}
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"chat_model": None}, "chat model"),
        ({"chat_base_url": None}, "chat base URL"),
        ({"chat_api_key": None}, "chat API key"),
    ],
)
def test_missing_openai_compatible_configuration_fails_clearly(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AIConfigurationError, match=message):
        OpenAICompatibleChatProvider.from_settings(openai_settings(**updates))


def test_http_client_uses_configured_endpoint_model_and_bearer_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(bytes(request.data or b"").decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHttpResponse(completion({"required_skills": ["Python"]}))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = HttpOpenAICompatibleClient(
        base_url="https://chat.example.test/v1/",
        api_key="top-secret-value",
        timeout_seconds=12,
        max_tokens=1024,
    )

    client.chat_completion(
        model_id="provider/model",
        messages=[{"role": "user", "content": "Return JSON"}],
    )

    assert captured["url"] == "https://chat.example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer top-secret-value"
    assert captured["payload"]["model"] == "provider/model"
    assert captured["payload"]["messages"][0]["content"] == "Return JSON"
    assert captured["payload"]["max_tokens"] == 1024
    assert captured["timeout"] == 12


def test_valid_structured_response_uses_existing_job_analysis_schema() -> None:
    client = FakeChatClient(
        [
            completion(
                {
                    "min_experience_years": 2,
                    "required_skills": ["Python"],
                    "responsibilities": ["Build production models"],
                }
            )
        ]
    )

    result = OpenAICompatibleChatProvider(
        openai_settings(), client=client
    ).extract_job_requirements("Build production models with Python")

    assert result.value.min_experience_years == 2
    assert result.value.required_skills == ["Python"]
    assert result.metadata.provider == "openai_compatible"
    assert result.metadata.model_id == "provider/model"
    assert result.metadata.input_tokens == 13
    assert client.calls[0]["model_id"] == "provider/model"
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_invalid_structured_output_is_corrected_then_strictly_validated() -> None:
    client = FakeChatClient(
        [
            completion({"min_experience_years": -2}),
            completion({"required_skills": ["SQL"]}),
        ]
    )
    provider = OpenAICompatibleChatProvider(openai_settings(), client=client)

    result = provider.extract_job_requirements("SQL role")

    assert result.value.required_skills == ["SQL"]
    assert len(client.calls) == 2
    assert "Previous output was invalid" in client.calls[1]["messages"][1]["content"]


def test_malformed_output_never_bypasses_validation() -> None:
    client = FakeChatClient([completion([]), completion([])])

    with pytest.raises(AIResponseError):
        OpenAICompatibleChatProvider(openai_settings(), client=client).extract_job_requirements(
            "role"
        )


def test_grounded_explanation_retries_unsupported_evidence() -> None:
    client = FakeChatClient(
        [
            completion(grounded_explanation("invented evidence")),
            completion(grounded_explanation()),
        ]
    )
    provider = OpenAICompatibleChatProvider(openai_settings(), client=client)

    result = provider.generate_explanation(explanation_context())

    assert result.value.evidence_references[0].evidence == "Python"
    assert len(client.calls) == 2
    assert "Explanation grounding failed" in client.calls[1]["messages"][1]["content"]


def test_explanation_failure_remains_bounded_and_typed() -> None:
    client = FakeChatClient(
        [
            completion(grounded_explanation("invented evidence")),
            completion(grounded_explanation("still invented")),
        ]
    )
    provider = OpenAICompatibleChatProvider(openai_settings(), client=client)

    with pytest.raises(AIValidationError, match="Explanation grounding failed"):
        provider.generate_explanation(explanation_context())


def test_provider_factory_selects_chat_and_local_embedding_independently() -> None:
    providers = build_ai_providers(
        openai_settings(),
        openai_compatible_client=FakeChatClient([completion({})]),
        local_model=FakeSentenceTransformer(),
    )

    assert isinstance(providers.llm, OpenAICompatibleChatProvider)
    assert isinstance(providers.embeddings, HuggingFaceEmbeddingProvider)
    assert configured_llm_model_id(openai_settings()) == "provider/model"


def test_invalid_base_url_is_rejected_before_any_request() -> None:
    with pytest.raises(AIConfigurationError, match="absolute HTTP"):
        OpenAICompatibleChatProvider.from_settings(
            openai_settings(chat_base_url="not-a-url")
        )
