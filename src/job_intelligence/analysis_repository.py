"""Replaceable persistence for cached structured job analysis."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .models import JobAnalysis


class JobAnalysisRepository(Protocol):
    def get(
        self,
        job_id: UUID,
        job_version_id: UUID,
        model_id: str,
        prompt_version: str,
    ) -> JobAnalysis | None:
        """Return cached analysis for an exact job/version/model/prompt combination."""

    def save(self, analysis: JobAnalysis) -> JobAnalysis:
        """Create or replace one cached analysis record."""


__all__ = ["JobAnalysisRepository"]
