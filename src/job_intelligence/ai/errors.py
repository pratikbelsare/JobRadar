"""Typed failures raised by AI providers."""

from __future__ import annotations


class AIProviderError(RuntimeError):
    """Base class for expected AI provider failures."""


class AIConfigurationError(AIProviderError):
    """Raised when provider settings or credentials are not usable."""


class AIServiceError(AIProviderError):
    """Raised for an upstream service failure."""


class AIThrottledError(AIServiceError):
    """Raised when the upstream service throttles a request."""


class AITimeoutError(AIServiceError):
    """Raised when an upstream call times out."""


class AIResponseError(AIProviderError):
    """Raised when a provider response cannot be read as model output."""


class AIValidationError(AIProviderError):
    """Raised when model output fails the requested structured schema."""


__all__ = [
    "AIConfigurationError",
    "AIProviderError",
    "AIResponseError",
    "AIServiceError",
    "AIThrottledError",
    "AITimeoutError",
    "AIValidationError",
]
