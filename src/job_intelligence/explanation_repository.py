"""Replaceable local storage for generated job explanations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from .ai.models import JobMatchExplanation


class ExplanationRepositoryError(RuntimeError):
    """Raised when generated explanations cannot be read or written."""


class ExplanationRepository(Protocol):
    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatchExplanation | None:
        """Return one stored explanation."""

    def save(
        self,
        job_id: UUID,
        candidate_profile_id: UUID,
        explanation: JobMatchExplanation,
    ) -> None:
        """Persist or replace one explanation."""


class JsonExplanationRepository:
    """Store explanations keyed by job and candidate profile."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatchExplanation | None:
        records = self._read()
        value = records.get(_key(job_id, candidate_profile_id))
        return JobMatchExplanation.model_validate(value) if value is not None else None

    def save(
        self,
        job_id: UUID,
        candidate_profile_id: UUID,
        explanation: JobMatchExplanation,
    ) -> None:
        records = self._read()
        records[_key(job_id, candidate_profile_id)] = explanation.model_dump(mode="json")
        self._write(records)

    def _read(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("explanation file must contain a JSON object")
            return raw
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            raise ExplanationRepositoryError(f"Could not load explanations: {self._path}") from exc

    def _write(self, records: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(records, temporary, indent=2)
            temporary_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ExplanationRepositoryError(f"Could not save explanations: {self._path}") from exc


def _key(job_id: UUID, candidate_profile_id: UUID) -> str:
    return f"{job_id}:{candidate_profile_id}"


__all__ = ["ExplanationRepository", "ExplanationRepositoryError", "JsonExplanationRepository"]
