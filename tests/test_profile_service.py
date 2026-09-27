from uuid import uuid4

import pytest
from pydantic import ValidationError

from job_intelligence.models import CandidateProfile, CandidatePreferences, WorkMode
from job_intelligence.profile_repository import (
    JsonCandidateProfileRepository,
    ProfileRepositoryError,
)
from job_intelligence.profile_service import (
    CandidateProfileService,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
)
from job_intelligence.resume import PdfTextExtractor, ResumeIngestor


def make_profile() -> CandidateProfile:
    return CandidateProfile(
        target_roles=["Data Scientist"],
        preferred_locations=["Remote"],
        skills=["Python"],
        preferences=CandidatePreferences(preferred_work_modes=[WorkMode.REMOTE]),
    )


def test_json_repository_round_trips_profile(tmp_path) -> None:
    repository = JsonCandidateProfileRepository(tmp_path / "profiles")
    profile = make_profile()

    repository.save(profile)
    loaded = repository.load(profile.id)

    assert loaded == profile
    assert (tmp_path / "profiles" / f"{profile.id}.json").exists()


def test_service_creates_loads_and_updates_without_losing_existing_fields(tmp_path) -> None:
    service = CandidateProfileService(JsonCandidateProfileRepository(tmp_path / "profiles"))
    profile = make_profile()
    service.create(profile)

    updated = service.update(
        profile.id,
        {
            "skills": ["Python", "SQL"],
            "target_roles": ["Analytics Engineer"],
        },
    )

    assert updated.id == profile.id
    assert updated.skills == ["Python", "SQL"]
    assert updated.target_roles == ["Analytics Engineer"]
    assert updated.preferred_locations == ["Remote"]
    assert service.load(profile.id) == updated


def test_manual_profile_values_remain_intact_after_resume_ingestion(tmp_path) -> None:
    repository = JsonCandidateProfileRepository(tmp_path / "profiles")
    service = CandidateProfileService(repository)
    profile = make_profile()
    service.create(profile)

    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"fixture PDF bytes")

    class Page:
        def extract_text(self) -> str:
            return "Kubernetes"

    class Reader:
        pages = [Page()]

    ingestor = ResumeIngestor(
        max_size_bytes=1024,
        extractor=PdfTextExtractor(reader_factory=lambda _: Reader()),
    )
    document = ingestor.ingest(resume_path)

    assert "Kubernetes" in document.raw_text
    assert service.load(profile.id).skills == ["Python"]
    assert service.load(profile.id).target_roles == ["Data Scientist"]


def test_profile_service_rejects_duplicate_missing_and_invalid_updates(tmp_path) -> None:
    service = CandidateProfileService(JsonCandidateProfileRepository(tmp_path / "profiles"))
    profile = make_profile()
    service.create(profile)

    with pytest.raises(ProfileAlreadyExistsError):
        service.create(profile)
    with pytest.raises(ProfileNotFoundError):
        service.update(uuid4(), {"skills": ["Python"]})
    with pytest.raises(ValidationError):
        service.update(profile.id, {"experience_years": -1})
    assert service.load(profile.id) == profile


def test_repository_reports_corrupt_json(tmp_path) -> None:
    repository = JsonCandidateProfileRepository(tmp_path / "profiles")
    profile_id = uuid4()
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / f"{profile_id}.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ProfileRepositoryError):
        repository.load(profile_id)
