"""Amazon Bedrock provider kept behind the provider-independent AI contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from ..config import Settings
from .errors import (
    AIConfigurationError,
    AIResponseError,
    AIServiceError,
    AIThrottledError,
    AITimeoutError,
    AIValidationError,
)
from .models import (
    AIResponseMetadata,
    AIResult,
    CandidateProfileProposal,
    EmbeddingVector,
    ExplanationContext,
    JobMatchExplanation,
    JobRequirements,
)
from .prompts import (
    CANDIDATE_PROFILE_PROMPT_VERSION,
    CANDIDATE_PROFILE_SYSTEM_PROMPT,
    EXPLANATION_PROMPT_VERSION,
    EXPLANATION_SYSTEM_PROMPT,
    JOB_REQUIREMENTS_PROMPT_VERSION,
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    candidate_profile_prompt,
    explanation_prompt,
    job_requirements_prompt,
)

T = TypeVar("T")


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]:
        """Call the Bedrock Converse API."""

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        """Call a model-specific Bedrock inference API."""


class BedrockProvider:
    """Bedrock implementation of structured extraction and embeddings."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: BedrockRuntimeClient,
        max_attempts: int | None = None,
        provider_name: str = "bedrock",
    ) -> None:
        self._settings = settings
        self._client = client
        self._max_attempts = (
            max_attempts
            if max_attempts is not None
            else settings.ai_max_attempts or 2
        )
        if self._max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._provider_name = provider_name

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: BedrockRuntimeClient | None = None,
        max_attempts: int | None = None,
    ) -> BedrockProvider:
        """Build a provider, lazily creating the AWS client only when needed."""

        runtime_client = client if client is not None else _create_bedrock_client(settings)
        return cls(settings, client=runtime_client, max_attempts=max_attempts)

    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        _validate_text_input(job_text)
        model_id = _required_setting(self._settings.job_analysis_model_id, "job model ID")
        return self._extract_json(
            model_id=model_id,
            system_prompt=JOB_REQUIREMENTS_SYSTEM_PROMPT,
            user_prompt=job_requirements_prompt(job_text),
            prompt_version=JOB_REQUIREMENTS_PROMPT_VERSION,
            parser=JobRequirements.model_validate,
        )

    def extract_candidate_profile(
        self,
        resume_text: str,
    ) -> AIResult[CandidateProfileProposal]:
        _validate_text_input(resume_text)
        model_id = _required_setting(self._settings.job_analysis_model_id, "job model ID")
        return self._extract_json(
            model_id=model_id,
            system_prompt=CANDIDATE_PROFILE_SYSTEM_PROMPT,
            user_prompt=candidate_profile_prompt(resume_text),
            prompt_version=CANDIDATE_PROFILE_PROMPT_VERSION,
            parser=CandidateProfileProposal.model_validate,
        )

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        _validate_text_input(text)
        model_id = _required_setting(self._settings.embedding_model_id, "embedding model ID")
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({"inputText": text.strip()}),
                    contentType="application/json",
                    accept="application/json",
                )
                payload = _decode_response_body(response)
                raw_vector = payload.get("embedding")
                if not isinstance(raw_vector, list):
                    raise AIResponseError("Embedding response did not contain an embedding list")
                vector = EmbeddingVector(values=raw_vector)
                return AIResult(
                    value=vector,
                    metadata=self._metadata(model_id=model_id),
                )
            except AIResponseError as exc:
                last_error = exc
            except ValidationError:
                last_error = AIValidationError("Embedding output failed schema validation")
            except AIServiceError as exc:
                last_error = exc
            except Exception as exc:
                last_error = _classify_service_error(exc)
            if attempt + 1 < self._max_attempts:
                continue
        assert last_error is not None
        raise last_error

    def generate_explanation(
        self,
        context: ExplanationContext,
    ) -> AIResult[JobMatchExplanation]:
        model_id = _required_setting(self._settings.job_analysis_model_id, "job model ID")
        return self._extract_json(
            model_id=model_id,
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
            user_prompt=explanation_prompt(context),
            prompt_version=EXPLANATION_PROMPT_VERSION,
            parser=JobMatchExplanation.model_validate,
        )

    def _extract_json(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        parser: Callable[[Any], T],
    ) -> AIResult[T]:
        last_error: Exception | None = None
        retry_prompt = user_prompt
        for attempt in range(self._max_attempts):
            try:
                response = self._client.converse(
                    modelId=model_id,
                    system=[{"text": system_prompt}],
                    messages=[
                        {"role": "user", "content": [{"text": retry_prompt}]}
                    ],
                )
                text, usage = _extract_converse_text(response)
                payload = _decode_json(text)
                try:
                    value = parser(payload)
                except ValidationError as exc:
                    raise AIValidationError(
                        "Structured model output failed schema validation"
                    ) from exc
                return AIResult(
                    value=value,
                    metadata=self._metadata(
                        model_id=model_id,
                        prompt_version=prompt_version,
                        usage=usage,
                    ),
                )
            except (AIResponseError, AIValidationError) as exc:
                last_error = exc
                retry_prompt = (
                    user_prompt
                    + "\nPrevious output was invalid. Return only one valid JSON object "
                    "with no markdown or additional commentary."
                )
            except AIServiceError as exc:
                last_error = exc
            except Exception as exc:
                last_error = _classify_service_error(exc)
            if attempt + 1 < self._max_attempts:
                continue
        assert last_error is not None
        raise last_error

    def _metadata(
        self,
        *,
        model_id: str,
        prompt_version: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> AIResponseMetadata:
        usage = usage or {}
        return AIResponseMetadata(
            provider=self._provider_name,
            model_id=model_id,
            prompt_version=prompt_version,
            input_tokens=_optional_int(usage.get("inputTokens")),
            output_tokens=_optional_int(usage.get("outputTokens")),
            requested_at=datetime.now(timezone.utc),
        )


def _create_bedrock_client(settings: Settings) -> BedrockRuntimeClient:
    if settings.aws_region is None:
        raise AIConfigurationError("AWS region must be configured for Bedrock")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise AIConfigurationError(
            "Bedrock integration requires boto3; install the project dependencies"
        ) from exc

    config = None
    if settings.bedrock_timeout_seconds is not None:
        timeout = settings.bedrock_timeout_seconds
        config = Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"max_attempts": 0},
        )
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if config is not None:
        kwargs["config"] = config
    return boto3.client("bedrock-runtime", **kwargs)


def _required_setting(value: str | None, name: str) -> str:
    if value is None:
        raise AIConfigurationError(f"{name} must be configured for Bedrock")
    return value


def _validate_text_input(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI text input must not be blank")


def _extract_converse_text(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        content = response["output"]["message"]["content"]
        texts = [item["text"] for item in content if isinstance(item.get("text"), str)]
    except (KeyError, TypeError) as exc:
        raise AIResponseError("Bedrock Converse response did not contain text output") from exc
    if not texts:
        raise AIResponseError("Bedrock Converse response contained no text output")
    usage = response.get("usage")
    return "\n".join(texts), usage if isinstance(usage, dict) else {}


def _decode_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AIResponseError("Bedrock response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AIResponseError("Bedrock response JSON must be an object")
    return payload


def _decode_response_body(response: dict[str, Any]) -> dict[str, Any]:
    body = response.get("body")
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if not isinstance(body, str):
        raise AIResponseError("Bedrock embedding response body was unreadable")
    return _decode_json(body)


def _classify_service_error(exc: Exception) -> AIServiceError:
    message = str(exc)
    lowered = message.casefold()
    if "timeout" in lowered or "timed out" in lowered:
        return AITimeoutError(message or "Bedrock request timed out")
    if "throttl" in lowered or "rate exceeded" in lowered:
        return AIThrottledError(message or "Bedrock request was throttled")
    if "credential" in lowered or "accessdenied" in lowered or "unauthorized" in lowered:
        return AIConfigurationError(message or "Bedrock credentials were rejected")
    return AIServiceError(message or "Bedrock request failed")


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


__all__ = ["BedrockProvider", "BedrockRuntimeClient"]
