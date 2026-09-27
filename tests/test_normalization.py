from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.normalization import (
    JobNormalizer,
    canonicalize_url,
    compute_content_hash,
    job_identity,
)


def raw_job(
    *,
    description: str = "Build reliable systems.",
    source_job_id: str = "  job-123 ",
    source_url: str = "https://Example.test/jobs/123/?b=2&a=1#details",
) -> RawJobDetails:
    return RawJobDetails(
        company_id=uuid4(),
        source_job_id=source_job_id,
        title="  Senior   Engineer\n",
        location=" Remote  /  Bengaluru ",
        source_url=source_url,
        details_url="https://api.example.test/jobs/123",
        description=description,
        raw_payload={"source": "fixture"},
    )


def test_normalizer_cleans_text_urls_and_timestamps() -> None:
    company_id = uuid4()
    source = raw_job()
    source = source.model_copy(update={"company_id": company_id})
    observed_at = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)

    normalized = JobNormalizer().normalize(source, observed_at=observed_at)

    assert normalized.company_id == company_id
    assert normalized.source_job_id == "job-123"
    assert normalized.title == "Senior Engineer"
    assert normalized.location == "Remote / Bengaluru"
    assert normalized.description == "Build reliable systems."
    assert str(normalized.source_url) == "https://example.test/jobs/123?a=1&b=2"
    assert normalized.first_seen_at == observed_at
    assert normalized.last_seen_at == observed_at
    assert normalized.content_hash == compute_content_hash(normalized)


def test_formatting_only_changes_keep_the_same_hash() -> None:
    company_id = uuid4()
    first = raw_job().model_copy(update={"company_id": company_id})
    second = raw_job(
        source_job_id="job-123",
        source_url="https://example.test/jobs/123?a=1&b=2",
        description="  Build reliable\n systems. ",
    ).model_copy(update={"company_id": company_id})

    normalizer = JobNormalizer()
    first_job = normalizer.normalize(first)
    second_job = normalizer.normalize(second)

    assert first_job.content_hash == second_job.content_hash


def test_meaningful_description_change_changes_hash() -> None:
    company_id = uuid4()
    first = raw_job().model_copy(update={"company_id": company_id})
    second = raw_job(description="Build distributed data platforms.").model_copy(
        update={"company_id": company_id}
    )

    normalizer = JobNormalizer()
    assert normalizer.normalize(first).content_hash != normalizer.normalize(second).content_hash


def test_identity_uses_company_source_id_and_canonical_url() -> None:
    company_id = uuid4()
    first = JobNormalizer().normalize(
        raw_job().model_copy(update={"company_id": company_id})
    )
    same_url = JobNormalizer().normalize(
        raw_job(source_job_id="different-id").model_copy(update={"company_id": company_id})
    )
    different_company = JobNormalizer().normalize(
        raw_job().model_copy(update={"company_id": uuid4()})
    )

    assert job_identity(first).matches(job_identity(same_url))
    assert not job_identity(first).matches(job_identity(different_company))


def test_malformed_connector_data_is_rejected_before_normalization() -> None:
    with pytest.raises(ValidationError):
        RawJobDetails(
            company_id=uuid4(),
            source_job_id="job-123",
            title="Engineer",
            source_url="not-a-url",
            details_url="https://api.example.test/jobs/123",
            description="Description",
        )

    with pytest.raises(ValidationError):
        canonicalize_url("not-a-url")
