"""Candidate-profile management independent of UI, storage, and resume parsing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from .models import CandidateProfile
from .profile_repository import CandidateProfileRepository


class ProfileAlreadyExistsError(RuntimeError):
    """Raised when creating a profile with an existing identifier."""


class ProfileNotFoundError(LookupError):
    """Raised when updating a profile that is not persisted."""


class CandidateProfileService:
    """Create, update, and load profiles through an injected repository."""

    def __init__(self, repository: CandidateProfileRepository) -> None:
        self._repository = repository

    def create(self, profile: CandidateProfile) -> CandidateProfile:
        if self._repository.load(profile.id) is not None:
            raise ProfileAlreadyExistsError(f"Profile already exists: {profile.id}")
        return self._repository.save(profile)

    def load(self, profile_id: UUID) -> CandidateProfile | None:
        return self._repository.load(profile_id)

    def update(
        self,
        profile_id: UUID,
        changes: Mapping[str, Any] | CandidateProfile,
    ) -> CandidateProfile:
        current = self._repository.load(profile_id)
        if current is None:
            raise ProfileNotFoundError(f"Profile not found: {profile_id}")

        update_values = _update_values(changes)
        requested_id = update_values.pop("id", profile_id)
        if requested_id != profile_id:
            raise ValueError("A profile update cannot change the profile id")

        merged = current.model_dump(mode="python")
        merged.update(update_values)
        merged["id"] = profile_id
        try:
            updated = CandidateProfile.model_validate(merged)
        except ValidationError:
            raise
        return self._repository.save(updated)


def _update_values(changes: Mapping[str, Any] | CandidateProfile) -> dict[str, Any]:
    if isinstance(changes, CandidateProfile):
        return changes.model_dump(exclude_unset=True, mode="python")
    return dict(changes)


__all__ = [
    "CandidateProfileService",
    "ProfileAlreadyExistsError",
    "ProfileNotFoundError",
]
