from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import boto3

from job_intelligence.ai import (
    BedrockMantleProvider,
    HuggingFaceEmbeddingProvider,
)
from job_intelligence.analysis_repository import JobAnalysisRepository
from job_intelligence.aws.dynamodb import (
    DynamoCandidateProfileRepository,
    DynamoJobRepository,
)
from job_intelligence.aws.sqs import JobAnalysisMessage
from job_intelligence.aws.workers import JobAnalysisWorker
from job_intelligence.config import Settings
from job_intelligence.explanation_repository import JsonExplanationRepository
from job_intelligence.explanations import ExplanationService
from job_intelligence.job_repository import JobRepository
from job_intelligence.match_repository import JsonJobMatchRepository
from job_intelligence.models import CandidateProfile, JobAnalysis
from job_intelligence.profile_repository import CandidateProfileRepository


class MemoryJobRepository(JobRepository):
    def __init__(self, state) -> None:
        self.state = state

    def find_by_identity(self, identity):
        return None

    def find_by_id(self, job_id):
        return self.state if self.state.job.id == job_id else None

    def list(self):
        return [self.state]

    def save(self, state):
        self.state = state
        return state


class MemoryProfileRepository(CandidateProfileRepository):
    def __init__(self, profile: CandidateProfile) -> None:
        self.profile = profile

    def save(self, profile):
        self.profile = profile
        return profile

    def load(self, profile_id):
        return self.profile if self.profile.id == profile_id else None

    def find_by_key(self, profile_key):
        return self.profile if self.profile.profile_key == profile_key else None

    def list(self):
        return [self.profile]


class MemoryAnalysisRepository(JobAnalysisRepository):
    def __init__(self) -> None:
        self.records: dict[tuple[UUID, UUID, str, str], JobAnalysis] = {}

    def get(self, job_id, job_version_id, model_id, prompt_version):
        return self.records.get((job_id, job_version_id, model_id, prompt_version))

    def save(self, analysis):
        key = (analysis.job_id, analysis.job_version_id, analysis.model_id, analysis.prompt_version)
        self.records[key] = analysis
        return analysis


def main() -> None:
    settings = Settings.from_environment()
    if settings.aws_region is None:
        raise RuntimeError("JOB_INTELLIGENCE_AWS_REGION must be configured")
    if settings.dynamodb_table_name is None:
        raise RuntimeError("JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME must be configured")
    if settings.candidate_profile_key is None:
        raise RuntimeError("JOB_INTELLIGENCE_CANDIDATE_PROFILE_KEY must be configured")
    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.dynamodb_table_name
    )
    target_title = os.environ.get("JOB_INTELLIGENCE_SMOKE_JOB_TITLE", "").strip()
    if not target_title:
        raise RuntimeError("JOB_INTELLIGENCE_SMOKE_JOB_TITLE must be configured")
    state = next(
        item
        for item in DynamoJobRepository(table).list()
        if item.job.title == target_title
    )
    profile = DynamoCandidateProfileRepository(table).find_by_key(settings.candidate_profile_key)
    if profile is None:
        raise RuntimeError(f"Candidate profile was not found: {settings.candidate_profile_key}")
    if settings.llm_provider.value != "bedrock_mantle":
        raise RuntimeError("Set JOB_INTELLIGENCE_LLM_PROVIDER=bedrock_mantle")
    if settings.embedding_provider.value != "huggingface_local":
        raise RuntimeError("Set JOB_INTELLIGENCE_EMBEDDING_PROVIDER=huggingface_local")

    llm = BedrockMantleProvider.from_settings(settings)
    embeddings = HuggingFaceEmbeddingProvider.from_settings(settings)
    analyses = MemoryAnalysisRepository()
    with tempfile.TemporaryDirectory() as temporary_directory:
        matches = JsonJobMatchRepository(Path(temporary_directory) / "matches.json")
        worker = JobAnalysisWorker(
            jobs=MemoryJobRepository(state),
            profiles=MemoryProfileRepository(profile),
            analyses=analyses,
            matches=matches,
            provider=llm,
            embedding_provider=embeddings,
            model_id=settings.mantle_model_id,
        )
        result = worker.process(
            JobAnalysisMessage(
                run_id=uuid4(),
                job_id=state.job.id,
                candidate_profile_id=profile.id,
            )
        )
        match = matches.get(state.job.id, profile.id)
        if match is None:
            raise RuntimeError("Match was not persisted")

        match_persisted = matches.get(state.job.id, profile.id) is not None
        explanation_persisted = False
        if os.environ.get("JOB_INTELLIGENCE_SMOKE_SKIP_EXPLANATION") != "1":
            explanation_repo = JsonExplanationRepository(
                Path(temporary_directory) / "explanations.json"
            )
            explanation = ExplanationService(llm).generate(
                state.job,
                profile,
                match,
                analysis=result.analysis,
            )
            explanation_repo.save(state.job.id, profile.id, explanation.value)
            explanation_persisted = explanation_repo.get(state.job.id, profile.id) is not None

    print(f"job_id={state.job.id}")
    print(f"job_title={state.job.title}")
    print(f"analysis_provider={result.analysis.provider}")
    print(f"analysis_model={result.analysis.model_id}")
    print(f"analysis_required_skills={len(result.analysis.required_skills)}")
    print(f"embedding_provider={embeddings.embed_text('smoke check').metadata.provider}")
    print(f"embedding_dimensions={embeddings.embed_text('smoke check').value.dimensions}")
    role_evidence = match.evidence["components"]["role"]
    print(f"semantic_role_method={role_evidence.get('method')}")
    print(f"semantic_role_score={match.role_match:.6f}")
    print(f"final_match_score={match.final_score:.6f}")
    print(f"match_persisted={match_persisted}")
    print(f"explanation_persisted={explanation_persisted}")


if __name__ == "__main__":
    main()
