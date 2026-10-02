"""Bedrock Mantle provider using its AWS-authenticated OpenAI-compatible API."""

from __future__ import annotations

import json
import urllib.error
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


class MantleClient(Protocol):
    def chat_completion(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Return one OpenAI-compatible chat completion response."""


class SignedMantleClient:
    """Small SigV4 transport for the Bedrock Mantle Chat Completions API."""

    def __init__(
        self,
        *,
        base_url: str,
        region: str,
        timeout_seconds: float = 60,
        max_tokens: int = 4096,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._region = region
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
        url = f"{self._base_url}/chat/completions"
        request = _signed_request(url, payload, self._region)
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
            if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
                raise AIServiceError(str(exc) or "Mantle request failed") from exc
            raise AITimeoutError("Mantle request timed out") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AIResponseError("Mantle response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise AIResponseError("Mantle response JSON must be an object")
        return decoded


class BedrockMantleProvider:
    """Structured AI provider backed by Bedrock Mantle, not Bedrock Runtime."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: MantleClient,
        max_attempts: int | None = None,
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

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: MantleClient | None = None,
        max_attempts: int | None = None,
    ) -> BedrockMantleProvider:
        if client is not None:
            return cls(settings, client=client, max_attempts=max_attempts)
        region = _required(settings.aws_region, "AWS region")
        base_url = settings.mantle_base_url or (
            f"https://bedrock-mantle.{region}.api.aws/v1"
        )
        timeout = settings.bedrock_timeout_seconds or 60
        return cls(
            settings,
            client=SignedMantleClient(
                base_url=base_url,
                region=region,
                timeout_seconds=timeout,
                max_tokens=settings.mantle_max_tokens,
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
        return self._extract_json(
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
            user_prompt=explanation_prompt(context),
            prompt_version=EXPLANATION_PROMPT_VERSION,
            parser=JobMatchExplanation.model_validate,
        )

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        raise AIConfigurationError(
            "Bedrock Mantle does not provide a configured embedding model; "
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
        model_id = self._settings.mantle_model_id
        last_error: Exception | None = None
        retry_prompt = user_prompt
        for attempt in range(self._max_attempts):
            try:
                response = self._client.chat_completion(
                    model_id=model_id,
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
                        provider="bedrock_mantle",
                        model_id=model_id,
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
                last_error = AIServiceError(str(exc) or "Mantle request failed")
            if attempt + 1 < self._max_attempts:
                continue
        assert last_error is not None
        raise last_error


def _signed_request(url: str, payload: bytes, region: str) -> urllib.request.Request:
    try:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ImportError as exc:
        raise AIConfigurationError(
            "Bedrock Mantle integration requires boto3; install the project dependencies"
        ) from exc
    credentials = boto3.Session(region_name=region).get_credentials()
    if credentials is None:
        raise AIConfigurationError("AWS credentials are required for Bedrock Mantle")
    frozen = credentials.get_frozen_credentials()
    aws_request = AWSRequest(
        method="POST",
        url=url,
        data=payload,
        headers={"content-type": "application/json", "accept": "application/json"},
    )
    SigV4Auth(frozen, "bedrock-mantle", region).add_auth(aws_request)
    prepared = aws_request.prepare()
    headers = {str(key): str(value) for key, value in prepared.headers.items()}
    return urllib.request.Request(url, data=payload, headers=headers, method="POST")


def _extract_completion_text(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        choices = response["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIResponseError("Mantle response did not contain a chat message") from exc
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
        raise AIResponseError("Mantle response contained no text output")
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
        raise AIResponseError("Mantle response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AIResponseError("Mantle response JSON must be an object")
    return payload


def _validate_text_input(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI text input must not be blank")


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise AIConfigurationError(f"{name} must be configured for Bedrock Mantle")
    return value


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
    return f"Mantle HTTP {error.code}: {detail or error.reason}"


__all__ = ["BedrockMantleProvider", "MantleClient", "SignedMantleClient"]
