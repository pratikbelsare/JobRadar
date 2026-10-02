"""S3 implementation of the existing raw snapshot storage boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from ..connectors.base import RawJobDetails
from ..models import Model
from ..snapshots import (
    RawSnapshotMetadata,
    RawSnapshotStore,
    SnapshotNotFoundError,
    SnapshotStoreError,
)


class S3Client(Protocol):
    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        """Store one object."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        """Read one object."""


class S3SnapshotIndex(Model):
    key: str


class S3RawSnapshotStore(RawSnapshotStore):
    """Store immutable snapshots under ``raw/<company>/<date>/`` in S3.

    A small index object makes lookup by snapshot UUID deterministic without listing
    an entire bucket. Both the snapshot and index are private bucket objects.
    """

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str = "",
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def save(
        self,
        raw_job: RawJobDetails,
        *,
        captured_at: datetime | None = None,
    ) -> RawSnapshotMetadata:
        snapshot_id = uuid4()
        timestamp = _utc(captured_at or datetime.now(timezone.utc))
        date_path = timestamp.strftime("%Y/%m/%d")
        key = self._key(
            "raw",
            str(raw_job.company_id),
            date_path,
            f"{snapshot_id}.json",
        )
        metadata = RawSnapshotMetadata(
            snapshot_id=snapshot_id,
            company_id=raw_job.company_id,
            source_job_id=raw_job.source_job_id,
            captured_at=timestamp,
            path=Path(f"s3://{self._bucket}/{key}"),
        )
        payload = _StoredSnapshot(metadata=metadata, job=raw_job).model_dump_json().encode("utf-8")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
            index = S3SnapshotIndex(key=key).model_dump_json().encode("utf-8")
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._index_key(snapshot_id),
                Body=index,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        except Exception as exc:
            raise SnapshotStoreError(f"Could not save S3 snapshot {snapshot_id}") from exc
        return metadata

    def load(self, snapshot_id: UUID) -> RawJobDetails:
        try:
            index_response = self._client.get_object(
                Bucket=self._bucket,
                Key=self._index_key(snapshot_id),
            )
        except Exception as exc:
            raise SnapshotNotFoundError(f"Raw snapshot not found: {snapshot_id}") from exc

        try:
            index = S3SnapshotIndex.model_validate_json(_read_body(index_response))
            response = self._client.get_object(Bucket=self._bucket, Key=index.key)
            return _StoredSnapshot.model_validate_json(_read_body(response)).job
        except Exception as exc:
            if isinstance(exc, SnapshotStoreError):
                raise
            raise SnapshotStoreError(f"Could not load S3 snapshot {snapshot_id}") from exc

    def _key(self, *parts: str) -> str:
        values = [part.strip("/") for part in parts if part.strip("/")]
        return "/".join(([self._prefix] if self._prefix else []) + values)

    def _index_key(self, snapshot_id: UUID) -> str:
        return self._key("raw", "index", f"{snapshot_id}.json")


class _StoredSnapshot(Model):
    metadata: RawSnapshotMetadata
    job: RawJobDetails


def _read_body(response: dict[str, Any]) -> str:
    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise SnapshotStoreError("S3 response did not contain a readable body")
    value = body.read()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise SnapshotStoreError("S3 response body was not text")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["S3Client", "S3RawSnapshotStore", "S3SnapshotIndex"]
