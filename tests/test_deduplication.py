from uuid import uuid4

from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.deduplication import ExactDuplicateDetector
from job_intelligence.normalization import JobNormalizer


def make_job(source_id: str, description: str = "Same description"):
    raw = RawJobDetails(
        company_id=uuid4(),
        source_job_id=source_id,
        title="Engineer",
        location="Remote",
        source_url=f"https://example.test/jobs/{source_id}",
        details_url=f"https://api.example.test/jobs/{source_id}",
        description=description,
    )
    return JobNormalizer().normalize(raw)


def test_exact_duplicate_matches_stable_identity() -> None:
    original = make_job("job-1")
    same_identity = original.model_copy(update={"title": "Renamed title"})

    assert ExactDuplicateDetector().find_duplicate(same_identity, [original]) == original


def test_exact_duplicate_matches_same_company_content_hash() -> None:
    original = make_job("job-1")
    duplicate = make_job("job-2", description="Same description")
    duplicate = duplicate.model_copy(update={"company_id": original.company_id})

    assert ExactDuplicateDetector().find_duplicate(duplicate, [original]) == original


def test_different_company_is_not_deduplicated_by_hash_alone() -> None:
    original = make_job("job-1")
    different_company = make_job("job-1")

    assert ExactDuplicateDetector().find_duplicate(different_company, [original]) is None
