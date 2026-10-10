from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import AnyHttpUrl

from job_intelligence.ai.models import AIResponseMetadata, AIResult, EmbeddingVector
from job_intelligence.benchmark import (
    BenchmarkIntegrityError,
    BenchmarkLabel,
    BenchmarkManifest,
    BenchmarkRow,
    FrozenAnalysisSet,
    FrozenJob,
    FrozenJobSet,
    benchmark_fingerprint,
    calculate_benchmark_metrics,
    evaluate_frozen_benchmark,
    load_benchmark_labels,
    load_frozen_benchmark,
    sha256_file,
    write_json,
)
from job_intelligence.eligibility import EligibilityConfig, EligibilityEvaluator
from job_intelligence.filtering import (
    DeterministicJobFilter,
    ExperienceFilterConfig,
    FilteringConfig,
)
from job_intelligence.matching import MatchingConfig
from job_intelligence.models import CandidateProfile, Job, JobAnalysis
from job_intelligence.role_policy import (
    RoleFamilyClassifier,
    default_ai_data_role_family_policy,
)

JOB_ID = UUID("11111111-1111-1111-1111-111111111111")
VERSION_ID = UUID("22222222-2222-2222-2222-222222222222")
PROFILE_ID = UUID("33333333-3333-3333-3333-333333333333")
COMPANY_ID = UUID("44444444-4444-4444-4444-444444444444")


class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        normalized = text.casefold()
        values = [
            1.0 if any(term in normalized for term in ("data", "python", "model")) else 0.0,
            1.0 if "scient" in normalized else 0.0,
            0.5,
        ]
        return AIResult(
            value=EmbeddingVector(values=values),
            metadata=AIResponseMetadata(provider="fake", model_id="fake-v1"),
        )


def test_frozen_benchmark_is_repeatable_and_needs_no_llm(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)

    first = evaluate_frozen_benchmark(benchmark, FakeEmbeddingProvider())
    second = evaluate_frozen_benchmark(benchmark, FakeEmbeddingProvider())

    assert first == second
    assert benchmark_fingerprint(first) == benchmark_fingerprint(second)
    assert first[0].role_family == "core"
    assert first[0].relevance_score is not None
    assert first[0].eligibility_status == "eligible"


def test_loading_detects_modified_immutable_input(tmp_path: Path) -> None:
    _benchmark(tmp_path)
    profile_path = tmp_path / "candidate_profile.json"
    profile_path.write_text(profile_path.read_text() + " ", encoding="utf-8")

    with pytest.raises(BenchmarkIntegrityError, match="hash mismatch"):
        load_frozen_benchmark(tmp_path)


def test_labels_keep_provisional_and_human_values_separate(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("job_id", "provisional_label", "human_label", "human_notes"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_id": str(JOB_ID),
                "provisional_label": "1",
                "human_label": "3",
                "human_notes": "reviewed",
            }
        )

    labels = load_benchmark_labels(path)

    assert labels == [
        BenchmarkLabel(job_id=JOB_ID, provisional_label=1, human_label=3)
    ]


def test_labels_allow_blank_provisional_and_human_values(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "job_id,provisional_label,human_label\n"
        f"{JOB_ID},,\n",
        encoding="utf-8",
    )

    assert load_benchmark_labels(path) == [BenchmarkLabel(job_id=JOB_ID)]


def test_metrics_support_provisional_human_and_subsets() -> None:
    rows = [
        _row(JOB_ID, 0.9, "core", "eligible"),
        _row(UUID("55555555-5555-5555-5555-555555555555"), 0.8, "adjacent", "stretch"),
        _row(UUID("66666666-6666-6666-6666-666666666666"), 0.2, "exclude", "eligible"),
    ]
    labels = [
        BenchmarkLabel(job_id=rows[0].job_id, provisional_label=2, human_label=3),
        BenchmarkLabel(job_id=rows[1].job_id, provisional_label=2),
        BenchmarkLabel(job_id=rows[2].job_id, provisional_label=0, human_label=2),
    ]

    provisional = calculate_benchmark_metrics(
        rows, labels, label_source="provisional", subset="all"
    )
    human = calculate_benchmark_metrics(rows, labels, label_source="human", subset="all")
    core_adjacent = calculate_benchmark_metrics(
        rows, labels, label_source="auto", subset="core_adjacent"
    )

    assert provisional.evaluated_count == 3
    assert provisional.relevant_count == 2
    assert provisional.recall_at_10 == 1.0
    assert human.evaluated_count == 2
    assert human.relevant_count == 2
    assert core_adjacent.evaluated_count == 2


def _benchmark(directory: Path) -> Any:
    profile = CandidateProfile(
        id=PROFILE_ID,
        profile_key="benchmark",
        target_roles=["Data Scientist"],
        experience_years=2,
        preferred_locations=["Remote"],
        skills=["Python", "machine learning"],
        domains=["data science"],
    )
    job = Job(
        id=JOB_ID,
        source_job_id="source-1",
        company_id=COMPANY_ID,
        title="Data Scientist",
        location="Remote",
        description="Build data models with Python.",
        source_url=AnyHttpUrl("https://example.com/jobs/1"),
        first_seen_at=datetime(2026, 1, 1),
        last_seen_at=datetime(2026, 1, 1),
    )
    analysis = JobAnalysis(
        job_id=JOB_ID,
        job_version_id=VERSION_ID,
        model_id="fixed-model",
        prompt_version="fixed-prompt",
        required_skills=["Python"],
        min_experience_years=1,
        responsibilities=["Build data models"],
        domain="data science",
        provider="fixture",
    )
    filtering = FilteringConfig(
        experience=ExperienceFilterConfig(
            acceptable_gap_years=1,
            max_stretch_gap_years=2,
        ),
        role_policy=default_ai_data_role_family_policy(),
    )
    eligibility_config = EligibilityConfig(
        acceptable_gap_years=1,
        max_stretch_gap_years=2,
    )
    family = RoleFamilyClassifier(filtering.role_policy).classify(job.title)  # type: ignore[arg-type]
    jobs = FrozenJobSet(
        jobs=[
            FrozenJob(
                job=job,
                job_version_id=VERSION_ID,
                category="core_ai_data",
                role_family=family,
                stage1_filter=DeterministicJobFilter(filtering).evaluate(job, profile),
                eligibility=EligibilityEvaluator(eligibility_config).evaluate(
                    profile,
                    job,
                    analysis,
                    role_family_class=family.classification,
                ),
            )
        ]
    )
    analyses = FrozenAnalysisSet(analyses=[analysis])
    write_json(directory / "jobs.json", jobs.model_dump(mode="json"))
    write_json(directory / "job_analyses.json", analyses.model_dump(mode="json"))
    write_json(directory / "candidate_profile.json", profile.model_dump(mode="json"))
    hashes = {
        name: sha256_file(directory / name)
        for name in ("jobs.json", "job_analyses.json", "candidate_profile.json")
    }
    manifest = BenchmarkManifest(
        benchmark_version="test-v1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        company_name="Example",
        job_count=1,
        profile_version="profile-v1",
        profile_sha256=hashes["candidate_profile.json"],
        file_sha256=hashes,
        analysis_provider="fixture",
        analysis_model_id="fixed-model",
        analysis_prompt_version="fixed-prompt",
        analyses_reused_local=1,
        embedding_provider="fixture",
        embedding_model_id="fixture",
        embedding_device="cpu",
        matching_config=MatchingConfig(),
        filtering_config=filtering,
        eligibility_config=eligibility_config,
    )
    write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
    return load_frozen_benchmark(directory)


def _row(
    job_id: UUID,
    relevance: float,
    family: str,
    eligibility: str,
) -> BenchmarkRow:
    return BenchmarkRow(
        job_id=job_id,
        title=f"Job {job_id}",
        category="test",
        role_family=family,
        stage1_passed=family != "exclude",
        stage1_outcome="pass" if family != "exclude" else "reject_role",
        eligibility_status=eligibility,
        experience_status="aligned",
        seniority_status="aligned",
        relevance_score=relevance,
        relevance_rank=1,
        legacy_blended_score=relevance,
        legacy_rank=1,
        role_score=relevance,
        skill_score=relevance,
        responsibility_score=relevance,
        domain_score=relevance,
        location_score=1,
    )
