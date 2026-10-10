"""Replaceable storage for computed job matches used by the API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Protocol
from uuid import UUID

from pydantic import ValidationError

from .models import JobMatch


class JobMatchRepositoryError(RuntimeError):
    """Raised when computed match data cannot be read or written."""


class JobMatchRepository(Protocol):
    def save(self, match: JobMatch) -> JobMatch:
        """Create or replace the current match for a job/profile pair."""

    def list(self, candidate_profile_id: UUID | None = None) -> list[JobMatch]:
        """Return persisted matches, optionally scoped to one profile."""

    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatch | None:
        """Return the current match for a job/profile pair."""


class JsonJobMatchRepository:
    """Store match records in one local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def list(self, candidate_profile_id: UUID | None = None) -> list[JobMatch]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("match file must contain a JSON list")
            matches = [JobMatch.model_validate(item) for item in raw]
            if candidate_profile_id is not None:
                matches = [
                    item for item in matches if item.candidate_profile_id == candidate_profile_id
                ]
            return matches
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            raise JobMatchRepositoryError(f"Could not load matches: {self._path}") from exc

    def save(self, match: JobMatch) -> JobMatch:
        matches = [
            item
            for item in self.list()
            if not (
                item.job_id == match.job_id
                and item.candidate_profile_id == match.candidate_profile_id
            )
        ]
        matches.append(match)
        self.save_all(matches)
        return match

    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatch | None:
        return next(
            (item for item in self.list(candidate_profile_id) if item.job_id == job_id),
            None,
        )

    def save_all(self, matches: List[JobMatch]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump([item.model_dump(mode="json") for item in matches], temporary, indent=2)
            temporary_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise JobMatchRepositoryError(f"Could not save matches: {self._path}") from exc


__all__ = ["JobMatchRepository", "JobMatchRepositoryError", "JsonJobMatchRepository"]
