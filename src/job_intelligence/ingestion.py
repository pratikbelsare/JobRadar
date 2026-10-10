"""Local orchestration from raw connector output to stored job observations."""

from __future__ import annotations

from datetime import datetime

from .change_detection import JobChangeDetector, JobChangeResult, StoredJobState
from .connectors.base import RawJobDetails
from .job_repository import JobRepository
from .normalization import JobNormalizer, job_identity
from .snapshots import RawSnapshotStore


class JobIngestionService:
    """Normalize, compare, persist, and snapshot one raw job observation."""

    def __init__(
        self,
        job_repository: JobRepository,
        snapshot_store: RawSnapshotStore,
        *,
        normalizer: JobNormalizer | None = None,
        change_detector: JobChangeDetector | None = None,
    ) -> None:
        self._job_repository = job_repository
        self._snapshot_store = snapshot_store
        self._normalizer = normalizer or JobNormalizer()
        self._change_detector = change_detector or JobChangeDetector()

    def ingest(
        self,
        raw_job: RawJobDetails,
        *,
        observed_at: datetime | None = None,
    ) -> JobChangeResult:
        normalized_job = self._normalizer.normalize(raw_job, observed_at=observed_at)
        previous = self._job_repository.find_by_identity(job_identity(normalized_job))
        result = self._change_detector.detect(previous, normalized_job)
        versions = list(previous.versions) if previous is not None else []
        if result.new_version is not None:
            versions.append(result.new_version)
        self._job_repository.save(StoredJobState(job=result.job, versions=versions))
        self._snapshot_store.save(raw_job, captured_at=result.job.last_seen_at)
        return result


__all__ = ["JobIngestionService"]
