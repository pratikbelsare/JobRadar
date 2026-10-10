"""Deterministic exact duplicate detection for canonical jobs."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Job
from .normalization import job_identity


class ExactDuplicateDetector:
    """Find duplicates by stable identity or same-company content hash."""

    def find_duplicate(self, candidate: Job, existing: Iterable[Job]) -> Job | None:
        candidate_identity = job_identity(candidate)
        for job in existing:
            if job_identity(job).matches(candidate_identity):
                return job
            if (
                job.company_id == candidate.company_id
                and job.content_hash is not None
                and job.content_hash == candidate.content_hash
            ):
                return job
        return None


__all__ = ["ExactDuplicateDetector"]
