from uuid import uuid4

import pytest

from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.snapshots import (
    FileRawSnapshotStore,
    SnapshotNotFoundError,
    SnapshotStoreError,
)


def make_raw() -> RawJobDetails:
    return RawJobDetails(
        company_id=uuid4(),
        source_job_id="job-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.test/jobs/1",
        details_url="https://api.example.test/jobs/1",
        description="Raw description",
        raw_payload={"content": "raw"},
    )


def test_file_snapshot_store_round_trips_raw_job(tmp_path) -> None:
    store = FileRawSnapshotStore(tmp_path / "snapshots")
    raw = make_raw()

    metadata = store.save(raw)
    loaded = store.load(metadata.snapshot_id)

    assert loaded == raw
    assert metadata.path.exists()


def test_file_snapshot_store_reports_missing_and_corrupt_files(tmp_path) -> None:
    store = FileRawSnapshotStore(tmp_path / "snapshots")
    with pytest.raises(SnapshotNotFoundError):
        store.load(uuid4())

    raw = make_raw()
    metadata = store.save(raw)
    metadata.path.write_text("not json", encoding="utf-8")
    with pytest.raises(SnapshotStoreError):
        store.load(metadata.snapshot_id)
