"""Configuration-driven construction of the interchangeable AI providers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import EmbeddingProvider, LLMProvider, Settings
from .bedrock import BedrockProvider, BedrockRuntimeClient
from .local_embeddings import HuggingFaceEmbeddingProvider, SentenceTransformerModel
from .mantle import BedrockMantleProvider, MantleClient
from .protocols import AIProvider, TextEmbeddingProvider


@dataclass(frozen=True)
class AIProviders:
    llm: AIProvider
    embeddings: TextEmbeddingProvider


def build_llm_provider(
    settings: Settings,
    *,
    bedrock_client: BedrockRuntimeClient | None = None,
    mantle_client: MantleClient | None = None,
) -> AIProvider:
    if settings.llm_provider == LLMProvider.BEDROCK:
        return BedrockProvider.from_settings(settings, client=bedrock_client)
    if settings.llm_provider == LLMProvider.BEDROCK_MANTLE:
        return BedrockMantleProvider.from_settings(settings, client=mantle_client)
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
    local_model: SentenceTransformerModel | None = None,
) -> AIProviders:
    llm = build_llm_provider(
        settings,
        bedrock_client=bedrock_client,
        mantle_client=mantle_client,
    )
    embeddings = build_embedding_provider(
        settings,
        bedrock_client=bedrock_client,
        local_model=local_model,
    )
    return AIProviders(llm=llm, embeddings=embeddings)


__all__ = [
    "AIProviders",
    "build_ai_providers",
    "build_embedding_provider",
    "build_llm_provider",
]
