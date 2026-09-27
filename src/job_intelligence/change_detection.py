"""Pure new/unchanged/changed classification and version creation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from .models import Job, JobVersion, Model
from .normalization import JobIdentity, ensure_content_hash, job_identity


class JobChangeType(str, Enum):
    NEW = "new"
    UNCHANGED = "unchanged"
    CHANGED = "changed"


class StoredJobState(Model):
    """Current canonical job plus immutable historical versions."""

    job: Job
    versions: list[JobVersion] = Field(default_factory=list)


class JobChangeResult(Model):
    classification: JobChangeType
    identity: JobIdentity
    job: Job
    new_version: JobVersion | None = None


class JobChangeDetector:
    """Classify one normalized observation against stored state."""

    def detect(
        self,
        previous: StoredJobState | None,
        current: Job,
    ) -> JobChangeResult:
        current = ensure_content_hash(current)
        identity = job_identity(current)
        if previous is None:
            version = _create_version(current, version_number=1)
            return JobChangeResult(
                classification=JobChangeType.NEW,
                identity=identity,
                job=current,
                new_version=version,
            )

        previous_identity = job_identity(previous.job)
        if not previous_identity.matches(identity):
            raise ValueError("previous job state does not match the current job identity")

        observed_job = _preserve_observation_state(previous.job, current)
        if previous.job.content_hash == current.content_hash:
            return JobChangeResult(
                classification=JobChangeType.UNCHANGED,
                identity=job_identity(observed_job),
                job=observed_job,
            )

        next_version = max((version.version_number for version in previous.versions), default=0) + 1
        version = _create_version(observed_job, version_number=next_version)
        return JobChangeResult(
            classification=JobChangeType.CHANGED,
            identity=job_identity(observed_job),
            job=observed_job,
            new_version=version,
        )


def _create_version(job: Job, *, version_number: int) -> JobVersion:
    return JobVersion(
        job_id=job.id,
        version_number=version_number,
        content_hash=job.content_hash or "",
        description=job.description,
        captured_at=job.last_seen_at,
        job_snapshot=job,
    )


def _preserve_observation_state(previous: Job, current: Job) -> Job:
    values = current.model_dump(mode="python")
    values.update(
        {
            "id": previous.id,
            "first_seen_at": previous.first_seen_at,
            "status": previous.status,
        }
    )
    return Job.model_validate(values)


__all__ = [
    "JobChangeDetector",
    "JobChangeResult",
    "JobChangeType",
    "StoredJobState",
]
