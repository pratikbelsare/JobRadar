"""Feedback service and replaceable local JSON persistence."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import List, Protocol
from uuid import UUID

from pydantic import ValidationError

from .evaluation import EvaluationExample
from .models import FeedbackLabel, FeedbackReason, UserFeedback


class FeedbackRepositoryError(RuntimeError):
    """Raised when feedback cannot be persisted or decoded."""


class FeedbackAlreadyExistsError(RuntimeError):
    """Raised when creating feedback with an existing identifier."""


class FeedbackNotFoundError(LookupError):
    """Raised when updating or loading an unknown feedback record."""


class UserFeedbackRepository(Protocol):
    def save(self, feedback: UserFeedback) -> UserFeedback:
        """Create or replace one feedback record."""

    def load(self, feedback_id: UUID) -> UserFeedback | None:
        """Load one feedback record."""

    def list(self) -> List[UserFeedback]:
        """Return all feedback records."""


class JsonUserFeedbackRepository:
    """Store feedback records in one atomically replaced local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, feedback: UserFeedback) -> UserFeedback:
        records = self._read()
        records[str(feedback.id)] = feedback
        self._write(records)
        return feedback

    def load(self, feedback_id: UUID) -> UserFeedback | None:
        return self._read().get(str(feedback_id))

    def list(self) -> List[UserFeedback]:
        return sorted(self._read().values(), key=lambda item: item.created_at)

    def _read(self) -> dict[str, UserFeedback]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("feedback file must contain a JSON list")
            records = [UserFeedback.model_validate(item) for item in raw]
            return {str(item.id): item for item in records}
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            raise FeedbackRepositoryError(f"Could not load feedback: {self._path}") from exc

    def _write(self, records: dict[str, UserFeedback]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                payload = [
                    item.model_dump(mode="json")
                    for item in sorted(records.values(), key=lambda value: value.created_at)
                ]
                json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise FeedbackRepositoryError(f"Could not save feedback: {self._path}") from exc


class UserFeedbackService:
    """Manage immutable context plus editable feedback label/reason/notes."""

    def __init__(self, repository: UserFeedbackRepository) -> None:
        self._repository = repository

    def create(
        self,
        *,
        job_id: UUID,
        candidate_profile_id: UUID,
        label: FeedbackLabel,
        reason: FeedbackReason | None = None,
        notes: str | None = None,
    ) -> UserFeedback:
        feedback = UserFeedback(
            job_id=job_id,
            candidate_profile_id=candidate_profile_id,
            label=label,
            reason=reason,
            notes=notes,
        )
        if self._repository.load(feedback.id) is not None:
            raise FeedbackAlreadyExistsError(f"Feedback already exists: {feedback.id}")
        return self._repository.save(feedback)

    def load(self, feedback_id: UUID) -> UserFeedback | None:
        return self._repository.load(feedback_id)

    def update(
        self,
        feedback_id: UUID,
        changes: Mapping[str, object] | UserFeedback,
    ) -> UserFeedback:
        current = self._repository.load(feedback_id)
        if current is None:
            raise FeedbackNotFoundError(f"Feedback not found: {feedback_id}")
        update_values = _feedback_update_values(changes)
        if "id" in update_values and update_values.pop("id") != feedback_id:
            raise ValueError("A feedback update cannot change the feedback id")
        for immutable in ("job_id", "candidate_profile_id"):
            if (
                immutable in update_values
                and update_values[immutable] != getattr(current, immutable)
            ):
                raise ValueError(f"A feedback update cannot change {immutable}")
        merged = current.model_dump(mode="python")
        merged.update(update_values)
        updated = UserFeedback.model_validate(merged)
        return self._repository.save(updated)

    def list(self) -> list[UserFeedback]:
        return self._repository.list()

    def export_evaluation_examples(self) -> List[EvaluationExample]:
        return [
            EvaluationExample(
                candidate_profile_id=item.candidate_profile_id,
                job_id=item.job_id,
                label=item.label,
                reason=item.reason,
                notes=item.notes,
            )
            for item in self._repository.list()
        ]


def _feedback_update_values(
    changes: Mapping[str, object] | UserFeedback,
) -> dict[str, object]:
    if isinstance(changes, UserFeedback):
        return changes.model_dump(exclude_unset=True, mode="python")
    return dict(changes)


__all__ = [
    "FeedbackAlreadyExistsError",
    "FeedbackNotFoundError",
    "FeedbackRepositoryError",
    "JsonUserFeedbackRepository",
    "UserFeedbackRepository",
    "UserFeedbackService",
]
