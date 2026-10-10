"""Replaceable persistence for cached structured job analysis."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from .models import JobAnalysis


class JobAnalysisRepositoryError(RuntimeError):
    """Raised when cached analysis cannot be read or written."""


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


class JsonJobAnalysisRepository:
    """Store immutable cache entries in a local directory.

    Filenames use a digest because provider model IDs and prompt versions are not
    necessarily safe filesystem names. The validated payload remains the source of
    truth and is checked against the requested cache key when loaded.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def get(
        self,
        job_id: UUID,
        job_version_id: UUID,
        model_id: str,
        prompt_version: str,
    ) -> JobAnalysis | None:
        path = self._path(job_id, job_version_id, model_id, prompt_version)
        if not path.exists():
            return None
        try:
            analysis = JobAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise JobAnalysisRepositoryError(f"Could not load analysis cache: {path}") from exc
        expected = (job_id, job_version_id, model_id, prompt_version)
        actual = (
            analysis.job_id,
            analysis.job_version_id,
            analysis.model_id,
            analysis.prompt_version,
        )
        if actual != expected:
            raise JobAnalysisRepositoryError(f"Analysis cache key mismatch: {path}")
        return analysis

    def save(self, analysis: JobAnalysis) -> JobAnalysis:
        path = self._path(
            analysis.job_id,
            analysis.job_version_id,
            analysis.model_id,
            analysis.prompt_version,
        )
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(analysis.model_dump_json(indent=2))
            temporary_path.replace(path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise JobAnalysisRepositoryError(f"Could not save analysis cache: {path}") from exc
        return analysis

    def _path(
        self,
        job_id: UUID,
        job_version_id: UUID,
        model_id: str,
        prompt_version: str,
    ) -> Path:
        raw_key = f"{job_id}:{job_version_id}:{model_id}:{prompt_version}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return self._directory / str(job_id) / f"{digest}.json"


__all__ = [
    "JobAnalysisRepository",
    "JobAnalysisRepositoryError",
    "JsonJobAnalysisRepository",
]
