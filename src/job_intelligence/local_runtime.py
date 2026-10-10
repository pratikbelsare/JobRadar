"""Shared local JSON/filesystem wiring for the pipeline, API, and smoke tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ai import ChatModelProvider, TextEmbeddingProvider, configured_llm_model_id
from .analysis_repository import JsonJobAnalysisRepository
from .assessment import JobAssessmentService
from .aws.workers import JobAnalysisWorker
from .company_repository import JsonCompanyRepository
from .config import Settings
from .connectors import ConnectorRegistry
from .eligibility import EligibilityConfig, EligibilityEvaluator
from .explanation_repository import JsonExplanationRepository
from .explanations import ExplanationService
from .feedback import JsonUserFeedbackRepository
from .filtering import DeterministicJobFilter, FilteringConfig
from .http import HttpClient
from .ingestion import JobIngestionService
from .job_repository import JsonJobRepository
from .match_repository import JsonJobMatchRepository
from .matching import HybridJobMatcher, MatchingConfig
from .pipeline import JobIntelligencePipeline
from .profile_repository import JsonCandidateProfileRepository
from .role_policy import RoleFamilyClassifier, default_ai_data_role_family_policy
from .run_repository import JsonScanRunRepository
from .snapshots import FileRawSnapshotStore


@dataclass(frozen=True)
class LocalRepositories:
    data_directory: Path
    profiles: JsonCandidateProfileRepository
    companies: JsonCompanyRepository
    runs: JsonScanRunRepository
    jobs: JsonJobRepository
    analyses: JsonJobAnalysisRepository
    matches: JsonJobMatchRepository
    explanations: JsonExplanationRepository
    feedback: JsonUserFeedbackRepository
    snapshots: FileRawSnapshotStore


def build_local_repositories(settings: Settings) -> LocalRepositories:
    """Build every local adapter from one stable directory layout."""

    data_directory = settings.application_data_directory or Path("data")
    profile_directory = settings.candidate_profile_directory or data_directory / "profiles"
    return LocalRepositories(
        data_directory=data_directory,
        profiles=JsonCandidateProfileRepository(profile_directory),
        companies=JsonCompanyRepository(data_directory / "companies.json"),
        runs=JsonScanRunRepository(data_directory / "runs.json"),
        jobs=JsonJobRepository(data_directory / "jobs"),
        analyses=JsonJobAnalysisRepository(data_directory / "analyses"),
        matches=JsonJobMatchRepository(data_directory / "matches.json"),
        explanations=JsonExplanationRepository(data_directory / "explanations.json"),
        feedback=JsonUserFeedbackRepository(data_directory / "feedback.json"),
        snapshots=FileRawSnapshotStore(data_directory / "job-snapshots"),
    )


def build_local_pipeline(
    settings: Settings,
    repositories: LocalRepositories,
    llm: ChatModelProvider,
    embeddings: TextEmbeddingProvider,
) -> JobIntelligencePipeline:
    """Compose the production domain services with local persistence adapters."""

    if settings.http_timeout_seconds is None or settings.http_user_agent is None:
        raise ValueError("HTTP timeout and user agent are required for local connectors")
    filtering_config = FilteringConfig.from_settings(settings)
    matcher = HybridJobMatcher(
        MatchingConfig.from_settings(settings),
        embedding_provider=embeddings,
    )
    assessment = JobAssessmentService(
        matcher=matcher,
        eligibility=EligibilityEvaluator(EligibilityConfig.from_settings(settings)),
        role_families=RoleFamilyClassifier(
            filtering_config.role_policy or default_ai_data_role_family_policy()
        ),
    )
    model_id = configured_llm_model_id(settings)
    worker = JobAnalysisWorker(
        jobs=repositories.jobs,
        profiles=repositories.profiles,
        analyses=repositories.analyses,
        matches=repositories.matches,
        provider=llm,
        embedding_provider=embeddings,
        matcher=matcher,
        assessment_service=assessment,
        model_id=model_id,
    )
    http = HttpClient(
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.http_user_agent,
        max_retries=settings.http_max_retries or 0,
    )
    return JobIntelligencePipeline(
        connectors=ConnectorRegistry(http),
        ingestion=JobIngestionService(repositories.jobs, repositories.snapshots),
        filtering=DeterministicJobFilter(filtering_config),
        analysis_worker=worker,
        jobs=repositories.jobs,
        analyses=repositories.analyses,
        matches=repositories.matches,
        explanations=repositories.explanations,
        explanation_service=ExplanationService(llm),
        runs=repositories.runs,
        analysis_model_id=model_id,
    )


__all__ = ["LocalRepositories", "build_local_pipeline", "build_local_repositories"]
