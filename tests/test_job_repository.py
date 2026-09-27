from uuid import uuid4

import pytest

from job_intelligence.change_detection import JobChangeDetector, StoredJobState
from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.job_repository import JobRepositoryError, JsonJobRepository
from job_intelligence.normalization import JobNormalizer, job_identity


def test_json_job_repository_preserves_versions(tmp_path) -> None:
    company_id = uuid4()
    normalizer = JobNormalizer()
    detector = JobChangeDetector()
    repository = JsonJobRepository(tmp_path / "jobs")
    first_raw = RawJobDetails(
        company_id=company_id,
        source_job_id="job-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.test/jobs/1",
        details_url="https://api.example.test/jobs/1",
        description="Original",
    )
    first_job = normalizer.normalize(first_raw)
    first_result = detector.detect(None, first_job)
    repository.save(
        StoredJobState(job=first_result.job, versions=[first_result.new_version])
    )

    second_job = normalizer.normalize(first_raw.model_copy(update={"description": "Changed"}))
    second_result = detector.detect(
        repository.find_by_identity(job_identity(second_job)),
        second_job,
    )
    state = repository.find_by_identity(job_identity(second_job))
    assert state is not None
    repository.save(
        StoredJobState(
            job=second_result.job,
            versions=[*state.versions, second_result.new_version],
        )
    )

    loaded = repository.find_by_identity(job_identity(second_result.job))
    assert loaded is not None
    assert [version.version_number for version in loaded.versions] == [1, 2]
    assert loaded.versions[0].description == "Original"
    assert loaded.versions[1].description == "Changed"


def test_json_job_repository_does_not_overwrite_historical_versions(tmp_path) -> None:
    repository = JsonJobRepository(tmp_path / "jobs")
    raw = RawJobDetails(
        company_id=uuid4(),
        source_job_id="job-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.test/jobs/1",
        details_url="https://api.example.test/jobs/1",
        description="Original",
    )
    job = JobNormalizer().normalize(raw)
    result = JobChangeDetector().detect(None, job)
    repository.save(StoredJobState(job=result.job, versions=[result.new_version]))

    altered = result.new_version.model_copy(update={"description": "Altered history"})
    with pytest.raises(JobRepositoryError, match="cannot be overwritten"):
        repository.save(StoredJobState(job=result.job, versions=[altered]))
