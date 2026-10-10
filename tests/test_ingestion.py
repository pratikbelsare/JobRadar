from datetime import datetime, timezone
from uuid import uuid4

from job_intelligence.change_detection import JobChangeType
from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.ingestion import JobIngestionService
from job_intelligence.job_repository import JsonJobRepository
from job_intelligence.snapshots import FileRawSnapshotStore


def test_ingestion_flow_is_new_then_unchanged_then_changed(tmp_path) -> None:
    company_id = uuid4()
    raw = RawJobDetails(
        company_id=company_id,
        source_job_id="job-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.test/jobs/1",
        details_url="https://api.example.test/jobs/1",
        description="Original description",
    )
    repository = JsonJobRepository(tmp_path / "jobs")
    snapshots = FileRawSnapshotStore(tmp_path / "snapshots")
    service = JobIngestionService(repository, snapshots)

    first = service.ingest(raw, observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    second = service.ingest(raw, observed_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    changed = service.ingest(
        raw.model_copy(update={"description": "Changed description"}),
        observed_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
    )

    assert first.classification == JobChangeType.NEW
    assert second.classification == JobChangeType.UNCHANGED
    assert changed.classification == JobChangeType.CHANGED
    state = repository.find_by_identity(changed.identity)
    assert state is not None
    assert [version.version_number for version in state.versions] == [1, 2]
    assert len(list((tmp_path / "snapshots").rglob("*.json"))) == 3
