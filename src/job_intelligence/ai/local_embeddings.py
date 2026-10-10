"""Local sentence-transformers embeddings for development and offline matching."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import ValidationError

from ..config import Settings
from .errors import AIConfigurationError, AIResponseError, AIValidationError
from .models import AIResponseMetadata, AIResult, EmbeddingVector


class SentenceTransformerModel(Protocol):
    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Encode one text locally."""


class HuggingFaceEmbeddingProvider:
    """CPU-oriented local provider backed by a cached sentence-transformers model."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: SentenceTransformerModel | None = None,
    ) -> None:
        self._settings = settings
        self._model = model

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: SentenceTransformerModel | None = None,
    ) -> HuggingFaceEmbeddingProvider:
        provider = cls(settings, model=model)
        if model is None:
            provider._model = provider._load_model()
        return provider

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("AI text input must not be blank")
        model = self._model or self._load_model()
        self._model = model
        try:
            encoded = model.encode(
                text.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            values = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
            vector = EmbeddingVector(values=[float(value) for value in values])
        except ValidationError as exc:
            raise AIValidationError("Local embedding output failed schema validation") from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise AIResponseError("Local embedding output was not a finite vector") from exc
        return AIResult(
            value=vector,
            metadata=AIResponseMetadata(
                provider="huggingface_local",
                model_id=self._settings.local_embedding_model_id,
                requested_at=datetime.now(timezone.utc),
            ),
        )

    def _load_model(self) -> SentenceTransformerModel:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AIConfigurationError(
                "Local embeddings require sentence-transformers; install the local-ai extra"
            ) from exc
        try:
            return SentenceTransformer(
                self._settings.local_embedding_model_id,
                device=self._settings.local_embedding_device,
            )
        except Exception as exc:
            raise AIConfigurationError(
                f"Could not load local embedding model {self._settings.local_embedding_model_id}"
            ) from exc


__all__ = ["HuggingFaceEmbeddingProvider", "SentenceTransformerModel"]
