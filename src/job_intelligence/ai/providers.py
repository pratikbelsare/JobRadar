"""Configuration-driven construction of the interchangeable AI providers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import EmbeddingProvider, LLMProvider, Settings
from .bedrock import BedrockProvider, BedrockRuntimeClient
from .errors import AIConfigurationError
from .local_embeddings import HuggingFaceEmbeddingProvider, SentenceTransformerModel
from .mantle import BedrockMantleProvider, MantleClient
from .openai_compatible import OpenAICompatibleChatProvider, OpenAICompatibleClient
from .protocols import ChatModelProvider, TextEmbeddingProvider


@dataclass(frozen=True)
class AIProviders:
    llm: ChatModelProvider
    embeddings: TextEmbeddingProvider


def build_llm_provider(
    settings: Settings,
    *,
    bedrock_client: BedrockRuntimeClient | None = None,
    mantle_client: MantleClient | None = None,
    openai_compatible_client: OpenAICompatibleClient | None = None,
) -> ChatModelProvider:
    if settings.llm_provider == LLMProvider.BEDROCK:
        return BedrockProvider.from_settings(settings, client=bedrock_client)
    if settings.llm_provider == LLMProvider.BEDROCK_MANTLE:
        return BedrockMantleProvider.from_settings(settings, client=mantle_client)
    if settings.llm_provider == LLMProvider.OPENAI_COMPATIBLE:
        return OpenAICompatibleChatProvider.from_settings(
            settings,
            client=openai_compatible_client,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_embedding_provider(
    settings: Settings,
    *,
    bedrock_client: BedrockRuntimeClient | None = None,
    local_model: SentenceTransformerModel | None = None,
) -> TextEmbeddingProvider:
    if settings.embedding_provider == EmbeddingProvider.BEDROCK:
        return BedrockProvider.from_settings(settings, client=bedrock_client)
    if settings.embedding_provider == EmbeddingProvider.HUGGINGFACE_LOCAL:
        return HuggingFaceEmbeddingProvider.from_settings(settings, model=local_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def build_ai_providers(
    settings: Settings,
    *,
    bedrock_client: BedrockRuntimeClient | None = None,
    mantle_client: MantleClient | None = None,
    openai_compatible_client: OpenAICompatibleClient | None = None,
    local_model: SentenceTransformerModel | None = None,
) -> AIProviders:
    llm = build_llm_provider(
        settings,
        bedrock_client=bedrock_client,
        mantle_client=mantle_client,
        openai_compatible_client=openai_compatible_client,
    )
    embeddings = build_embedding_provider(
        settings,
        bedrock_client=bedrock_client,
        local_model=local_model,
    )
    return AIProviders(llm=llm, embeddings=embeddings)


def configured_llm_model_id(settings: Settings) -> str:
    """Return the cache identity for the selected structured chat model."""

    values = {
        LLMProvider.BEDROCK: settings.job_analysis_model_id,
        LLMProvider.BEDROCK_MANTLE: settings.mantle_model_id,
        LLMProvider.OPENAI_COMPATIBLE: settings.chat_model,
    }
    value = values.get(settings.llm_provider)
    if value is None or not value.strip():
        raise AIConfigurationError(
            f"Model ID must be configured for {settings.llm_provider.value}"
        )
    return value


__all__ = [
    "AIProviders",
    "build_ai_providers",
    "build_embedding_provider",
    "build_llm_provider",
    "configured_llm_model_id",
]
