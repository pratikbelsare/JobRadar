from datetime import datetime, timezone
from uuid import uuid4

import pytest

from job_intelligence.change_detection import (
    JobChangeDetector,
    JobChangeType,
    StoredJobState,
)
from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.normalization import JobNormalizer


def make_raw(description: str) -> RawJobDetails:
    return RawJobDetails(
        company_id=uuid4(),
        source_job_id="job-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.test/jobs/1",
        details_url="https://api.example.test/jobs/1",
        description=description,
    )


def test_new_observation_creates_initial_version() -> None:
    raw = make_raw("Original description")
    job = JobNormalizer().normalize(raw, observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))

    result = JobChangeDetector().detect(None, job)

    assert result.classification == JobChangeType.NEW
    assert result.new_version is not None
    assert result.new_version.version_number == 1
    assert result.new_version.job_snapshot == result.job


def test_unchanged_observation_preserves_first_seen_and_history() -> None:
    raw = make_raw("Original description")
    normalizer = JobNormalizer()
    first = normalizer.normalize(raw, observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    detector = JobChangeDetector()
    created = detector.detect(None, first)
    state = StoredJobState(job=created.job, versions=[created.new_version])

    second = normalizer.normalize(raw, observed_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    result = detector.detect(state, second)

    assert result.classification == JobChangeType.UNCHANGED
    assert result.new_version is None
    assert result.job.id == first.id
    assert result.job.first_seen_at == first.first_seen_at
    assert result.job.last_seen_at == second.last_seen_at


def test_changed_observation_creates_next_version_without_mutating_history() -> None:
    company_id = uuid4()
    normalizer = JobNormalizer()
    first_raw = make_raw("Original description").model_copy(update={"company_id": company_id})
    first = normalizer.normalize(first_raw, observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    detector = JobChangeDetector()
    created = detector.detect(None, first)
    first_version = created.new_version
    assert first_version is not None
    state = StoredJobState(job=created.job, versions=[first_version])

    changed_raw = make_raw("Meaningfully changed description").model_copy(
        update={"company_id": company_id}
    )
    changed = normalizer.normalize(
        changed_raw,
        observed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    result = detector.detect(state, changed)

    assert result.classification == JobChangeType.CHANGED
    assert result.new_version is not None
    assert result.new_version.version_number == 2
    assert result.new_version.description == "Meaningfully changed description"
    assert first_version.description == "Original description"


def test_detector_rejects_a_different_identity() -> None:
    normalizer = JobNormalizer()
    first = normalizer.normalize(make_raw("Original"))
    created = JobChangeDetector().detect(None, first)
    state = StoredJobState(job=created.job, versions=[created.new_version])
    different = normalizer.normalize(
        make_raw("Different").model_copy(update={"source_job_id": "other-job"})
    )

    with pytest.raises(ValueError, match="identity"):
        JobChangeDetector().detect(state, different)
