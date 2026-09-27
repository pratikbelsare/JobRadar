"""Raw job snapshot storage with a filesystem implementation."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field, ValidationError

from .connectors.base import RawJobDetails
from .models import Model, NonEmptyText


class SnapshotStoreError(RuntimeError):
    """Raised when a raw snapshot cannot be stored or loaded."""


class SnapshotNotFoundError(SnapshotStoreError):
    """Raised when a requested snapshot does not exist."""


class RawSnapshotMetadata(Model):
    snapshot_id: UUID = Field(default_factory=uuid4)
    company_id: UUID
    source_job_id: NonEmptyText
    captured_at: datetime
    path: Path


class StoredRawSnapshot(Model):
    metadata: RawSnapshotMetadata
    job: RawJobDetails


class RawSnapshotStore(Protocol):
    """Storage boundary suitable for a later S3 adapter."""

    def save(
        self,
        raw_job: RawJobDetails,
        *,
        captured_at: datetime | None = None,
    ) -> RawSnapshotMetadata:
        """Persist one raw connector observation."""

    def load(self, snapshot_id: UUID) -> RawJobDetails:
        """Load a raw connector observation."""


class FileRawSnapshotStore:
    """Write one immutable JSON snapshot per connector observation."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(
        self,
        raw_job: RawJobDetails,
        *,
        captured_at: datetime | None = None,
    ) -> RawSnapshotMetadata:
        snapshot_id = uuid4()
        timestamp = _utc(captured_at or datetime.now(timezone.utc))
        path = self._directory / str(raw_job.company_id) / f"{snapshot_id}.json"
        metadata = RawSnapshotMetadata(
            snapshot_id=snapshot_id,
            company_id=raw_job.company_id,
            source_job_id=raw_job.source_job_id,
            captured_at=timestamp,
            path=path,
        )
        stored = StoredRawSnapshot(metadata=metadata, job=raw_job)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, stored.model_dump_json(indent=2))
        except SnapshotStoreError:
            raise
        except (OSError, ValidationError, ValueError) as exc:
            raise SnapshotStoreError(f"Could not save raw snapshot {snapshot_id}") from exc
        return metadata

    def load(self, snapshot_id: UUID) -> RawJobDetails:
        matches = list(self._directory.rglob(f"{snapshot_id}.json"))
        if not matches:
            raise SnapshotNotFoundError(f"Raw snapshot not found: {snapshot_id}")
        try:
            stored = StoredRawSnapshot.model_validate_json(
                matches[0].read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise SnapshotStoreError(f"Could not load raw snapshot {snapshot_id}") from exc
        return stored.job


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise SnapshotStoreError(f"Could not write raw snapshot file {path}") from exc


__all__ = [
    "FileRawSnapshotStore",
    "RawSnapshotMetadata",
    "RawSnapshotStore",
    "SnapshotNotFoundError",
    "SnapshotStoreError",
    "StoredRawSnapshot",
]
