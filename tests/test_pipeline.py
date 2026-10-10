from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from job_intelligence.ai.models import (
    AIResponseMetadata,
    AIResult,
    EmbeddingVector,
    ExplanationContext,
    JobMatchExplanation,
    JobRequirements,
)
from job_intelligence.analysis_repository import JsonJobAnalysisRepository
from job_intelligence.assessment import JobAssessmentService
from job_intelligence.aws.workers import JobAnalysisWorker
from job_intelligence.company_repository import JsonCompanyRepository
from job_intelligence.config import Settings
from job_intelligence.connectors.base import (
    ConnectorStatus,
    JobDetailsResult,
    JobListingResult,
    JobReference,
    RawJobDetails,
)
from job_intelligence.eligibility import EligibilityConfig, EligibilityEvaluator
from job_intelligence.explanation_repository import JsonExplanationRepository
from job_intelligence.explanations import ExplanationService
from job_intelligence.filtering import DeterministicJobFilter, FilteringConfig
from job_intelligence.ingestion import JobIngestionService
from job_intelligence.job_repository import JsonJobRepository
from job_intelligence.match_repository import JsonJobMatchRepository
from job_intelligence.matching import HybridJobMatcher, MatchingConfig
from job_intelligence.models import CandidateProfile, Company, ConnectorType
from job_intelligence.multi_company import MultiCompanyPipelineRunner
from job_intelligence.pipeline import ExplanationStatus, JobIntelligencePipeline, PipelineStage
from job_intelligence.profile_repository import JsonCandidateProfileRepository
from job_intelligence.role_policy import RoleFamilyClassifier, default_ai_data_role_family_policy
from job_intelligence.run_repository import JsonScanRunRepository
from job_intelligence.snapshots import FileRawSnapshotStore


class FakeConnector:
    connector_type = ConnectorType.GENERIC

    def __init__(self, company: Company, jobs: dict[str, tuple[str, str]]) -> None:
        self.company = company
        self.jobs = jobs
        self.detail_failures: set[str] = set()

    def list_jobs(self, company: Company) -> JobListingResult:
        return JobListingResult(
            status=ConnectorStatus.SUCCESS,
            jobs=[
                JobReference(
                    company_id=company.id,
                    source_job_id=source_id,
                    title=title,
                    source_url=f"https://example.test/jobs/{source_id}",
                    details_url=f"https://example.test/api/jobs/{source_id}",
                )
                for source_id, (title, _) in self.jobs.items()
            ],
        )

    def get_job(self, reference: JobReference) -> JobDetailsResult:
        if reference.source_job_id in self.detail_failures:
            return JobDetailsResult(status=ConnectorStatus.PARSE_ERROR, error="bad detail")
        title, description = self.jobs[reference.source_job_id]
        return JobDetailsResult(
            status=ConnectorStatus.SUCCESS,
            job=RawJobDetails(
                company_id=reference.company_id,
                source_job_id=reference.source_job_id,
                title=title,
                source_url=reference.source_url,
                details_url=reference.details_url,
                description=description,
            ),
        )


class FakeRegistry:
    def __init__(self, connector: FakeConnector) -> None:
        self.connector = connector

    def for_company(self, company: Company) -> FakeConnector:
        return self.connector


class FakeAIProvider:
    def __init__(self) -> None:
        self.extraction_calls = 0
        self.embedding_calls = 0
        self.explanation_calls = 0

    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        self.extraction_calls += 1
        if "FAIL_ANALYSIS" in job_text:
            raise RuntimeError("synthetic analysis failure")
        return AIResult(
            value=JobRequirements(
                min_experience_years=2,
                required_skills=["Python"],
                responsibilities=[job_text],
                domain="AI/ML",
            ),
            metadata=AIResponseMetadata(
                provider="fake",
                model_id="test-model",
                prompt_version="job-requirements-v1",
            ),
        )

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        self.embedding_calls += 1
        if "FAIL_EMBEDDING" in text:
            raise RuntimeError("synthetic embedding failure")
        return AIResult(
            value=EmbeddingVector(values=[1.0, 0.0, 0.0]),
            metadata=AIResponseMetadata(provider="fake", model_id="test-embedding"),
        )

    def generate_explanation(
        self,
        context: ExplanationContext,
    ) -> AIResult[JobMatchExplanation]:
        self.explanation_calls += 1
        if "FAIL_EXPLANATION" in context.job_title:
            raise RuntimeError("synthetic explanation failure")
        return AIResult(
            value=JobMatchExplanation(
                why_fit=["The supplied match evidence supports review."],
                experience_alignment="The structured eligibility result is available.",
            ),
            metadata=AIResponseMetadata(provider="fake", model_id="test-model"),
        )

    def extract_candidate_profile(self, resume_text: str) -> Any:
        raise NotImplementedError


def _pipeline(
    tmp_path: Path,
    jobs_data: dict[str, tuple[str, str]],
) -> tuple[
    JobIntelligencePipeline,
    FakeConnector,
    FakeAIProvider,
    JsonJobRepository,
    JsonJobMatchRepository,
    Company,
    CandidateProfile,
]:
    settings = Settings(
        filter_acceptable_experience_gap_years=1,
        filter_max_stretch_experience_gap_years=2,
    )
    company = Company(
        name="Example",
        career_url="https://example.test/careers",
        connector_type=ConnectorType.GENERIC,
    )
    profile = CandidateProfile(
        target_roles=["Data Scientist", "Machine Learning Engineer"],
        experience_years=1.2,
        skills=["Python"],
        projects=[
            {
                "name": "ML project",
                "description": "Built and evaluated machine learning models.",
                "skills": ["Python"],
            }
        ],
    )
    profiles = JsonCandidateProfileRepository(tmp_path / "profiles")
    profiles.save(profile)
    companies = JsonCompanyRepository(tmp_path / "companies.json")
    companies.save(company)
    jobs = JsonJobRepository(tmp_path / "jobs")
    analyses = JsonJobAnalysisRepository(tmp_path / "analyses")
    matches = JsonJobMatchRepository(tmp_path / "matches.json")
    explanations = JsonExplanationRepository(tmp_path / "explanations.json")
    provider = FakeAIProvider()
    matcher = HybridJobMatcher(MatchingConfig(), embedding_provider=provider)
    assessment = JobAssessmentService(
        matcher=matcher,
        eligibility=EligibilityEvaluator(EligibilityConfig.from_settings(settings)),
        role_families=RoleFamilyClassifier(default_ai_data_role_family_policy()),
    )
    worker = JobAnalysisWorker(
        jobs=jobs,
        profiles=profiles,
        analyses=analyses,
        matches=matches,
        provider=provider,
        embedding_provider=provider,
        matcher=matcher,
        assessment_service=assessment,
        model_id="test-model",
    )
    connector = FakeConnector(company, jobs_data)
    pipeline = JobIntelligencePipeline(
        connectors=FakeRegistry(connector),
        ingestion=JobIngestionService(
            jobs,
            FileRawSnapshotStore(tmp_path / "snapshots"),
        ),
        filtering=DeterministicJobFilter(FilteringConfig.from_settings(settings)),
        analysis_worker=worker,
        jobs=jobs,
        analyses=analyses,
        matches=matches,
        explanations=explanations,
        explanation_service=ExplanationService(provider),
        runs=JsonScanRunRepository(tmp_path / "runs.json"),
        analysis_model_id="test-model",
    )
    return pipeline, connector, provider, jobs, matches, company, profile


def test_successful_company_run_composes_existing_services(tmp_path: Path) -> None:
    pipeline, _, provider, jobs, matches, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "Build models with Python."),
            "2": ("Sales Representative", "Sell software."),
        },
    )

    result = pipeline.run_company(company, profile)

    assert result.summary.jobs_discovered == 2
    assert result.summary.new_jobs == 2
    assert result.summary.stage1_retained == 1
    assert result.summary.stage1_rejected == 1
    assert result.summary.analysis_successes == 1
    assert result.summary.matches_created == 1
    assert result.summary.explanations_created == 1
    assert len(jobs.list()) == 2
    assert len(matches.list(profile.id)) == 1
    assert provider.extraction_calls == 1


def test_mixed_run_isolates_one_analysis_failure(tmp_path: Path) -> None:
    pipeline, _, _, _, matches, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "FAIL_ANALYSIS"),
            "2": ("Machine Learning Engineer", "Build ML systems with Python."),
        },
    )

    result = pipeline.run_company(company, profile)

    assert result.summary.analysis_failures == 1
    assert result.summary.analysis_successes == 1
    assert result.summary.failures == 1
    assert result.failures[0].stage == PipelineStage.ANALYSIS_MATCHING
    assert len(matches.list(profile.id)) == 1


def test_idempotent_rerun_reuses_analysis_match_and_explanation(tmp_path: Path) -> None:
    pipeline, _, provider, jobs, matches, company, profile = _pipeline(
        tmp_path,
        {"1": ("Data Scientist", "Build models with Python.")},
    )

    first = pipeline.run_company(company, profile)
    first_match = matches.list(profile.id)[0]
    second = pipeline.run_company(company, profile)

    assert first.summary.new_jobs == 1
    assert second.summary.unchanged_jobs == 1
    assert second.summary.duplicate_jobs_created == 0
    assert second.summary.extraction_calls == 0
    assert second.summary.extraction_cache_hits == 1
    assert second.summary.matches_reused == 1
    assert second.summary.embeddings_reused == 1
    assert second.summary.explanations_reused == 1
    assert second.summary.explanation_calls == 0
    assert len(jobs.list()) == 1
    assert len(matches.list(profile.id)) == 1
    assert matches.list(profile.id)[0].id == first_match.id
    assert provider.extraction_calls == 1
    assert provider.explanation_calls == 1
    assert second.jobs[0].explanation_status == ExplanationStatus.REUSED


def test_changed_job_creates_version_and_invalidates_analysis(tmp_path: Path) -> None:
    pipeline, connector, provider, jobs, matches, company, profile = _pipeline(
        tmp_path,
        {"1": ("Data Scientist", "Build models with Python.")},
    )
    first = pipeline.run_company(company, profile)
    first_match_id = matches.list(profile.id)[0].id
    connector.jobs["1"] = ("Data Scientist", "Build production models with Python and SQL.")

    second = pipeline.run_company(company, profile)

    assert first.summary.new_jobs == 1
    assert second.summary.changed_jobs == 1
    assert second.summary.extraction_calls == 1
    assert second.summary.matches_updated == 1
    assert second.summary.explanations_created == 1
    assert len(jobs.list()) == 1
    assert len(jobs.list()[0].versions) == 2
    assert matches.list(profile.id)[0].id == first_match_id
    assert provider.extraction_calls == 2
    assert provider.explanation_calls == 2


def test_detail_and_explanation_failures_do_not_destroy_healthy_matches(
    tmp_path: Path,
) -> None:
    pipeline, connector, provider, _, matches, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "Build models."),
            "2": ("Machine Learning Engineer", "Build services."),
            "3": ("Data Scientist - FAIL_EXPLANATION", "Build models."),
        },
    )
    connector.detail_failures.add("1")

    result = pipeline.run_company(company, profile)

    assert result.summary.analysis_successes == 2
    assert result.summary.explanation_failures == 1
    assert len(matches.list(profile.id)) == 2
    assert provider.extraction_calls == 2
    assert {failure.stage for failure in result.failures} == {
        PipelineStage.DETAIL_FETCH,
        PipelineStage.EXPLANATION,
    }


def test_embedding_failure_is_isolated_to_affected_job(tmp_path: Path) -> None:
    pipeline, _, _, _, matches, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "FAIL_EMBEDDING"),
            "2": ("Machine Learning Engineer", "Build reliable ML systems."),
        },
    )

    result = pipeline.run_company(company, profile)

    assert result.summary.analysis_failures == 1
    assert result.summary.analysis_successes == 1
    assert len(matches.list(profile.id)) == 1
    assert result.failures[0].stage == PipelineStage.ANALYSIS_MATCHING
    assert result.summary.persistent_records_created["analyses"] == 2


def test_repository_write_failure_is_isolated_to_affected_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _, _, jobs, matches, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "Build models with Python."),
            "2": ("Machine Learning Engineer", "Build reliable ML systems."),
        },
    )
    original_save = jobs.save

    def fail_one_job(state: Any) -> Any:
        if state.job.source_job_id == "1":
            raise RuntimeError("synthetic repository write failure")
        return original_save(state)

    monkeypatch.setattr(jobs, "save", fail_one_job)

    result = pipeline.run_company(company, profile)

    assert result.summary.analysis_successes == 1
    assert len(matches.list(profile.id)) == 1
    assert result.failures[0].stage == PipelineStage.INGESTION
    assert result.failures[0].source_job_id == "1"


def test_exact_same_content_with_different_source_identity_is_deduplicated(
    tmp_path: Path,
) -> None:
    pipeline, _, _, jobs, _, company, profile = _pipeline(
        tmp_path,
        {
            "1": ("Data Scientist", "Build the same models."),
            "2": ("Data Scientist", "Build the same models."),
        },
    )

    result = pipeline.run_company(company, profile)

    assert result.summary.jobs_discovered == 2
    assert result.summary.jobs_normalized == 2
    assert result.summary.duplicates == 1
    assert result.summary.new_jobs == 1
    assert result.summary.duplicate_jobs_created == 0
    assert len(jobs.list()) == 1


def test_multi_company_run_is_idempotent_and_keeps_company_identity_separate(
    tmp_path: Path,
) -> None:
    pipeline, _, provider, jobs, matches, first_company, profile = _pipeline(
        tmp_path,
        {"shared-source-id": ("Data Scientist", "Build models with Python.")},
    )
    second_company = Company(
        name="Second Company",
        career_url="https://second.example.test/careers",
        connector_type=ConnectorType.GENERIC,
    )
    runner = MultiCompanyPipelineRunner(pipeline)

    first = runner.run([first_company, second_company], profile)
    second = runner.run([first_company, second_company], profile)

    assert first.summary.companies_attempted == 2
    assert first.summary.companies_succeeded == 2
    assert first.summary.new_jobs == 2
    assert first.summary.matches_generated == 2
    assert second.summary.new_jobs == 0
    assert second.summary.unchanged_jobs == 2
    assert second.summary.job_versions_created == 0
    assert second.summary.extraction_calls == 0
    assert second.summary.analyses_reused == 2
    assert second.summary.embeddings_reused == 2
    assert second.summary.matches_reused == 2
    assert second.summary.explanations_reused == 2
    assert second.summary.duplicate_jobs_created == 0
    assert len(jobs.list()) == 2
    assert sum(len(state.versions) for state in jobs.list()) == 2
    assert {state.job.company_id for state in jobs.list()} == {
        first_company.id,
        second_company.id,
    }
    assert len(matches.list(profile.id)) == 2
    assert len(list((tmp_path / "analyses").rglob("*.json"))) == 2
    assert len(list((tmp_path / "snapshots").rglob("*.json"))) == 4
    assert provider.extraction_calls == 2


def test_multi_company_runner_isolates_unexpected_company_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _, _, jobs, _, first_company, profile = _pipeline(
        tmp_path,
        {"1": ("Data Scientist", "Build models with Python.")},
    )
    failed_company = Company(
        name="Failed Company",
        career_url="https://failed.example.test/careers",
        connector_type=ConnectorType.GENERIC,
    )
    final_company = Company(
        name="Final Company",
        career_url="https://final.example.test/careers",
        connector_type=ConnectorType.GENERIC,
    )
    original_run = pipeline.run_company

    def fail_middle_company(company: Company, candidate: CandidateProfile) -> Any:
        if company.id == failed_company.id:
            raise RuntimeError("synthetic company failure")
        return original_run(company, candidate)

    monkeypatch.setattr(pipeline, "run_company", fail_middle_company)

    result = MultiCompanyPipelineRunner(pipeline).run(
        [first_company, failed_company, final_company],
        profile,
    )

    assert result.summary.companies_attempted == 3
    assert result.summary.companies_succeeded == 2
    assert result.summary.companies_failed == 1
    assert result.summary.failed_company_names == ["Failed Company"]
    assert result.summary.new_jobs == 2
    assert len(jobs.list()) == 2
    assert result.company_failures[0].error_type == "RuntimeError"
