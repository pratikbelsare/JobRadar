from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from job_intelligence.ai.models import AIResponseMetadata, AIResult, JobRequirements
from job_intelligence.aws.handlers import analysis_handler, coordinator_handler, scanner_handler
from job_intelligence.aws.sqs import CompanyScanMessage, JobAnalysisMessage
from job_intelligence.aws.workers import CompanyScanner, JobAnalysisWorker, RunCoordinator
from job_intelligence.change_detection import StoredJobState
from job_intelligence.config import Settings
from job_intelligence.connectors.base import (
    ConnectorStatus,
    JobDetailsResult,
    JobListingResult,
    JobReference,
    RawJobDetails,
)
from job_intelligence.filtering import DeterministicJobFilter, FilteringConfig
from job_intelligence.ingestion import JobIngestionService
from job_intelligence.job_repository import JsonJobRepository
from job_intelligence.match_repository import JsonJobMatchRepository
from job_intelligence.models import (
    CandidateProfile,
    Company,
    ConnectorType,
    Job,
    JobAnalysis,
    JobVersion,
    ScanRun,
)
from job_intelligence.profile_repository import JsonCandidateProfileRepository
from job_intelligence.snapshots import FileRawSnapshotStore


class RecordingQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> str:
        self.messages.append(message)
        return str(len(self.messages))


class FakeCompanies:
    def __init__(self, companies: list[Company]) -> None:
        self.items = {company.id: company for company in companies}

    def list(self) -> list[Company]:
        return list(self.items.values())

    def get(self, company_id: object) -> Company | None:
        return self.items.get(company_id)

    def save(self, company: Company) -> Company:
        self.items[company.id] = company
        return company


class FakeRuns:
    def __init__(self) -> None:
        self.items: dict[object, object] = {}

    def save(self, run: object) -> object:
        self.items[run.id] = run
        return run

    def get(self, run_id: object) -> object | None:
        return self.items.get(run_id)

    def list(self) -> list[object]:
        return list(self.items.values())


class FakeConnector:
    def __init__(self, company: Company) -> None:
        self.company = company

    def list_jobs(self, company: Company) -> JobListingResult:
        reference = JobReference(
            company_id=company.id,
            source_job_id="source-1",
            title="Software Engineer",
            source_url="https://example.com/jobs/1",
            details_url="https://example.com/jobs/1",
        )
        return JobListingResult(status=ConnectorStatus.SUCCESS, jobs=[reference])

    def get_job(self, reference: JobReference) -> JobDetailsResult:
        return JobDetailsResult(
            status=ConnectorStatus.SUCCESS,
            job=RawJobDetails(
                company_id=reference.company_id,
                source_job_id=reference.source_job_id,
                title=reference.title,
                source_url=reference.source_url,
                details_url=reference.details_url,
                description="Build software systems.",
            ),
        )


class FakeRegistry:
    def __init__(self, connector: FakeConnector) -> None:
        self.connector = connector

    def for_company(self, company: Company) -> FakeConnector:
        return self.connector


class FakeAnalysisRepository:
    def __init__(self) -> None:
        self.items: dict[tuple[object, object, str, str], JobAnalysis] = {}

    def get(
        self,
        job_id: object,
        version_id: object,
        model_id: str,
        prompt_version: str,
    ) -> JobAnalysis | None:
        return self.items.get((job_id, version_id, model_id, prompt_version))

    def save(self, analysis: JobAnalysis) -> JobAnalysis:
        self.items[
            (analysis.job_id, analysis.job_version_id, analysis.model_id, analysis.prompt_version)
        ] = analysis
        return analysis


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        self.calls += 1
        return AIResult(
            value=JobRequirements(),
            metadata=AIResponseMetadata(
                provider="test",
                model_id="test-model",
                prompt_version="job-requirements-v1",
            ),
        )


class FailingProvider(FakeProvider):
    def extract_job_requirements(self, job_text: str) -> AIResult[JobRequirements]:
        raise RuntimeError("Bedrock access denied")


def test_coordinator_batches_enabled_companies_and_handler_output() -> None:
    companies = [
        Company(name="First", career_url="https://first.example"),
        Company(name="Second", career_url="https://second.example"),
        Company(name="Disabled", career_url="https://disabled.example", enabled=False),
    ]
    queue = RecordingQueue()
    runs = FakeRuns()
    coordinator = RunCoordinator(FakeCompanies(companies), runs, queue, batch_size=1)
    result = coordinator.start()
    assert result.selected_company_ids == [companies[0].id]
    output = coordinator_handler({}, None, coordinator=coordinator)
    assert "run_id" in output
    assert len(queue.messages) == 2


def test_scanner_reuses_existing_ingestion_and_enqueues_filtered_job() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        company = Company(
            name="Example",
            career_url="https://example.com",
            connector_type=ConnectorType.GENERIC,
        )
        profile = CandidateProfile(target_roles=["Software Engineer"])
        companies = FakeCompanies([company])
        profiles = JsonCandidateProfileRepository(root / "profiles")
        profiles.save(profile)
        runs = FakeRuns()
        queue = RecordingQueue()
        jobs = JsonJobRepository(root / "jobs")
        ingestion = JobIngestionService(jobs, FileRawSnapshotStore(root / "snapshots"))
        filtering = DeterministicJobFilter(
            FilteringConfig.from_settings(
                Settings(
                    filter_acceptable_experience_gap_years=1,
                    filter_max_stretch_experience_gap_years=3,
                )
            )
        )
        scanner = CompanyScanner(
            companies=companies,
            profiles=profiles,
            runs=runs,
            connectors=FakeRegistry(FakeConnector(company)),
            ingestion=ingestion,
            filtering=filtering,
            analysis_queue=queue,
        )
        run = ScanRun(
            companies_requested=1,
            status="running",
        )
        runs.save(run)
        result = scanner.process(CompanyScanMessage(run_id=run.id, company_id=company.id))
        assert result.jobs_discovered == 1
        assert result.jobs_queued == 1
        assert len(queue.messages) == 1


def test_analysis_worker_caches_bedrock_and_preserves_match_identity() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        profile = CandidateProfile()
        profiles = JsonCandidateProfileRepository(root / "profiles")
        profiles.save(profile)
        job = Job(
            source_job_id="source-1",
            company_id=uuid4(),
            title="Engineer",
            description="Build systems.",
            source_url="https://example.com/jobs/1",
        )
        jobs = JsonJobRepository(root / "jobs")
        version = JobVersion(
            job_id=job.id,
            version_number=1,
            content_hash="hash-1",
            description=job.description,
            job_snapshot=job,
        )
        jobs.save(StoredJobState(job=job, versions=[version]))
        state = jobs.find_by_id(job.id)
        assert state is not None
        provider = FakeProvider()
        matches = JsonJobMatchRepository(root / "matches.json")
        worker = JobAnalysisWorker(
            jobs=jobs,
            profiles=profiles,
            analyses=FakeAnalysisRepository(),
            matches=matches,
            provider=provider,
            model_id="test-model",
        )
        message = JobAnalysisMessage(run_id=uuid4(), job_id=job.id, candidate_profile_id=profile.id)
        first = worker.process(message)
        second = worker.process(message)
        assert first.match_id == second.match_id
        assert provider.calls == 1


def test_analysis_worker_logs_stage_and_safe_identifiers_on_failure(caplog) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        profile = CandidateProfile(profile_key="pratik")
        profiles = JsonCandidateProfileRepository(root / "profiles")
        profiles.save(profile)
        job = Job(
            source_job_id="source-1",
            company_id=uuid4(),
            title="Engineer",
            description="Build systems.",
            source_url="https://example.com/jobs/1",
        )
        jobs = JsonJobRepository(root / "jobs")
        version = JobVersion(
            job_id=job.id,
            version_number=1,
            content_hash="hash-1",
            description=job.description,
            job_snapshot=job,
        )
        jobs.save(StoredJobState(job=job, versions=[version]))
        worker = JobAnalysisWorker(
            jobs=jobs,
            profiles=profiles,
            analyses=FakeAnalysisRepository(),
            matches=JsonJobMatchRepository(root / "matches.json"),
            provider=FailingProvider(),
            model_id="amazon.nova-lite-v1:0",
        )

        with caplog.at_level(logging.ERROR, logger="job_intelligence.aws"):
            with pytest.raises(RuntimeError, match="access denied"):
                worker.process(
                    JobAnalysisMessage(
                        run_id=uuid4(),
                        job_id=job.id,
                        candidate_profile_id=profile.id,
                    )
                )

        events = [json.loads(record.getMessage()) for record in caplog.records]
        event = next(item for item in events if item["event"] == "analysis_stage_failed")
        assert event["stage"] == "bedrock_structured_extraction"
        assert event["job_id"] == str(job.id)
        assert event["job_version_id"] == str(version.id)
        assert event["candidate_profile_key"] == "pratik"
        assert event["exception_type"] == "RuntimeError"


def test_analysis_handler_logs_message_context_and_keeps_retry_failure(caplog) -> None:
    class FailingWorker:
        def process(self, message: JobAnalysisMessage) -> None:
            raise RuntimeError("analysis persistence failed")

    message = JobAnalysisMessage(
        run_id=uuid4(),
        job_id=uuid4(),
        candidate_profile_id=uuid4(),
    )
    with caplog.at_level(logging.ERROR, logger="job_intelligence.aws"):
        output = analysis_handler(
            {"Records": [{"messageId": "analysis-1", "body": message.model_dump_json()}]},
            None,
            worker=FailingWorker(),
        )
    assert output == {"batchItemFailures": [{"itemIdentifier": "analysis-1"}]}
    events = [json.loads(record.getMessage()) for record in caplog.records]
    event = next(item for item in events if item["event"] == "sqs_record_failed")
    assert event["message_id"] == "analysis-1"
    assert event["job_id"] == str(message.job_id)
    assert event["candidate_profile_id"] == str(message.candidate_profile_id)
    assert event["stage"] == "worker"


def test_sqs_handler_returns_failed_message_ids_for_malformed_work() -> None:
    output = scanner_handler(
        {"Records": [{"messageId": "bad-1", "body": "not-json"}]},
        None,
        scanner=object(),  # parsing fails before the fake worker is used
    )
    assert output == {"batchItemFailures": [{"itemIdentifier": "bad-1"}]}
