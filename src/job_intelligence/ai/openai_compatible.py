"""Structured chat provider for OpenAI-compatible Chat Completions APIs."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
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


class OpenAICompatibleClient(Protocol):
    def chat_completion(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Return one OpenAI-compatible Chat Completions response."""


class HttpOpenAICompatibleClient:
    """Small bearer-authenticated client with no vendor SDK dependency."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60,
        max_tokens: int = 4096,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._api_key = _required(api_key, "chat API key")
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens

    def chat_completion(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = _http_error_message(exc)
            if exc.code == 429:
                raise AIThrottledError(message) from exc
            if exc.code in {408, 504}:
                raise AITimeoutError(message) from exc
            if exc.code in {401, 403}:
                raise AIConfigurationError(message) from exc
            raise AIServiceError(message) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.URLError) and not isinstance(
                exc.reason, TimeoutError
            ):
                raise AIServiceError(str(exc) or "Chat Completions request failed") from exc
            raise AITimeoutError("Chat Completions request timed out") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AIResponseError("Chat Completions response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise AIResponseError("Chat Completions response JSON must be an object")
        return decoded


class OpenAICompatibleChatProvider:
    """Strict structured-output provider for compatible chat APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAICompatibleClient,
        max_attempts: int | None = None,
    ) -> None:
        self._model_id = _required(settings.chat_model, "chat model")
        self._client = client
        self._max_attempts = (
            max_attempts if max_attempts is not None else settings.ai_max_attempts or 2
        )
        if self._max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: OpenAICompatibleClient | None = None,
        max_attempts: int | None = None,
    ) -> OpenAICompatibleChatProvider:
        if client is not None:
            return cls(settings, client=client, max_attempts=max_attempts)
        base_url = _required(settings.chat_base_url, "chat base URL")
        api_key = settings.chat_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise AIConfigurationError(
                "chat API key must be configured for the OpenAI-compatible provider"
            )
        return cls(
            settings,
            client=HttpOpenAICompatibleClient(
                base_url=base_url,
                api_key=api_key.get_secret_value(),
                timeout_seconds=settings.chat_timeout_seconds,
                max_tokens=settings.chat_max_tokens,
            ),
            max_attempts=max_attempts,
        )

    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        _validate_text_input(job_text)
        return self._extract_json(
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
        return self._extract_json(
            system_prompt=CANDIDATE_PROFILE_SYSTEM_PROMPT,
            user_prompt=candidate_profile_prompt(resume_text),
            prompt_version=CANDIDATE_PROFILE_PROMPT_VERSION,
            parser=CandidateProfileProposal.model_validate,
        )

    def generate_explanation(
        self,
        context: ExplanationContext,
    ) -> AIResult[JobMatchExplanation]:
        def parse_grounded_explanation(payload: Any) -> JobMatchExplanation:
            explanation = JobMatchExplanation.model_validate(payload)
            from ..explanations import ExplanationGroundingError, validate_explanation_grounding

            try:
                validate_explanation_grounding(explanation, context)
            except ExplanationGroundingError as exc:
                raise AIValidationError(f"Explanation grounding failed: {exc}") from exc
            return explanation

        return self._extract_json(
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
            user_prompt=explanation_prompt(context),
            prompt_version=EXPLANATION_PROMPT_VERSION,
            parser=parse_grounded_explanation,
        )

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        raise AIConfigurationError(
            "The OpenAI-compatible chat provider does not provide embeddings; "
            "select a separate embedding provider"
        )

    def _extract_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        parser: Callable[[Any], T],
    ) -> AIResult[T]:
        last_error: Exception | None = None
        retry_prompt = user_prompt
        for attempt in range(self._max_attempts):
            try:
                response = self._client.chat_completion(
                    model_id=self._model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": retry_prompt},
                    ],
                )
                payload_text, usage = _extract_completion_text(response)
                payload = _decode_json(payload_text)
                try:
                    value = parser(payload)
                except ValidationError as exc:
                    raise AIValidationError(
                        "Structured model output failed schema validation: "
                        + _validation_hint(exc)
                    ) from exc
                return AIResult(
                    value=value,
                    metadata=AIResponseMetadata(
                        provider="openai_compatible",
                        model_id=self._model_id,
                        prompt_version=prompt_version,
                        requested_at=datetime.now(timezone.utc),
                        input_tokens=_optional_int(usage.get("prompt_tokens")),
                        output_tokens=_optional_int(usage.get("completion_tokens")),
                    ),
                )
            except (AIResponseError, AIValidationError) as exc:
                last_error = exc
                retry_prompt = (
                    user_prompt
                    + "\nPrevious output was invalid. Return only one valid JSON object "
                    + "with no markdown or additional commentary. Fix these field-level "
                    + f"validation errors: {last_error}"
                )
            except AIServiceError as exc:
                last_error = exc
            except Exception as exc:
                last_error = AIServiceError(
                    str(exc) or "OpenAI-compatible request failed"
                )
            if attempt + 1 < self._max_attempts:
                continue
        assert last_error is not None
        raise last_error


def _extract_completion_text(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIResponseError("Chat Completions response did not contain a chat message") from exc
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    else:
        text = ""
    if not text.strip():
        raise AIResponseError("Chat Completions response contained no text output")
    usage = response.get("usage")
    return text, usage if isinstance(usage, dict) else {}


def _decode_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AIResponseError("Chat Completions response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AIResponseError("Chat Completions response JSON must be an object")
    return payload


def _validate_text_input(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI text input must not be blank")


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise AIConfigurationError(
            f"{name} must be configured for the OpenAI-compatible provider"
        )
    return value


def _validate_base_url(value: str) -> str:
    url = _required(value, "chat base URL").rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AIConfigurationError("chat base URL must be an absolute HTTP(S) URL")
    return url


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _validation_hint(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )


def _http_error_message(error: urllib.error.HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        detail = ""
    return f"Chat Completions HTTP {error.code}: {detail or error.reason}"


__all__ = [
    "HttpOpenAICompatibleClient",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleClient",
]
