"""Persistence interfaces and the local JSON candidate-profile repository."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from .models import CandidateProfile


class ProfileRepositoryError(RuntimeError):
    """Raised when a candidate profile cannot be persisted or decoded."""


class CandidateProfileRepository(Protocol):
    """Persistence boundary that a future PostgreSQL adapter can implement."""

    def save(self, profile: CandidateProfile) -> CandidateProfile:
        """Create or replace a profile."""

    def load(self, profile_id: UUID) -> CandidateProfile | None:
        """Load a profile, returning ``None`` when it does not exist."""


class JsonCandidateProfileRepository:
    """Store one JSON document per candidate profile in a local directory."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, profile: CandidateProfile) -> CandidateProfile:
        target = self._profile_path(profile.id)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=f".{profile.id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(profile.model_dump_json(indent=2))
            temporary_path.replace(target)
        except OSError as exc:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            raise ProfileRepositoryError(f"Could not save profile {profile.id}") from exc
        return profile

    def load(self, profile_id: UUID) -> CandidateProfile | None:
        path = self._profile_path(profile_id)
        if not path.exists():
            return None
        try:
            return CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise ProfileRepositoryError(f"Could not load profile {profile_id}") from exc

    def _profile_path(self, profile_id: UUID) -> Path:
        return self._directory / f"{profile_id}.json"


__all__ = [
    "CandidateProfileRepository",
    "JsonCandidateProfileRepository",
    "ProfileRepositoryError",
]
