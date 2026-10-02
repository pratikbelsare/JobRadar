"""Small interfaces consumed by business logic and matching."""

from __future__ import annotations

from typing import Protocol

from .models import (
    AIResult,
    CandidateProfileProposal,
    EmbeddingVector,
    ExplanationContext,
    JobMatchExplanation,
    JobRequirements,
)


class AIProvider(Protocol):
    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        """Extract validated facts from normalized job text."""

    def extract_candidate_profile(
        self,
        resume_text: str,
    ) -> AIResult[CandidateProfileProposal]:
        """Return a resume-derived proposal without applying it to a profile."""

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        """Return a validated vector for one text input."""

    def generate_explanation(
        self,
        context: ExplanationContext,
    ) -> AIResult[JobMatchExplanation]:
        """Generate a structured explanation from supplied evidence only."""


class TextEmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        """Return a validated embedding for matching."""


__all__ = ["AIProvider", "TextEmbeddingProvider"]
