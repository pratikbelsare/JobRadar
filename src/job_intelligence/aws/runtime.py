"""Concrete AWS wiring kept out of core services and Lambda handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ai import build_ai_providers
from ..company_repository import CompanyRepository
from ..config import Settings
from ..connectors import ConnectorRegistry
from ..filtering import DeterministicJobFilter, FilteringConfig
from ..http import HttpClient
from ..ingestion import JobIngestionService
from ..job_repository import JobRepository
from ..profile_repository import CandidateProfileRepository
from ..snapshots import RawSnapshotStore
from .dynamodb import (
    DynamoCandidateProfileRepository,
    DynamoCompanyRepository,
    DynamoJobAnalysisRepository,
    DynamoJobMatchRepository,
    DynamoJobRepository,
    DynamoScanRunRepository,
)
from .s3 import S3RawSnapshotStore
from .sqs import SQSQueuePublisher
from .workers import CompanyScanner, JobAnalysisWorker, RunCoordinator


class AWSConfigurationError(RuntimeError):
    """Raised when a Lambda cannot be wired from environment configuration."""


@dataclass(frozen=True)
class AWSClients:
    table: Any
    s3: Any
    sqs: Any


def create_aws_clients(settings: Settings) -> AWSClients:
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(settings.dynamodb_table_name, "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME")
    try:
        import boto3
    except ImportError as exc:
        raise AWSConfigurationError("AWS runtime wiring requires boto3") from exc
    return AWSClients(
        table=boto3.resource("dynamodb", region_name=region).Table(table_name),
        s3=boto3.client("s3", region_name=region),
        sqs=boto3.client("sqs", region_name=region),
    )


def build_coordinator(settings: Settings | None = None) -> RunCoordinator:
    configured = settings or Settings.from_environment()
    clients = create_aws_clients(configured)
    return RunCoordinator(
        _company_repository(clients.table),
        _run_repository(clients.table),
        SQSQueuePublisher(
            clients.sqs,
            queue_url=_required(
                configured.company_scan_queue_url,
                "JOB_INTELLIGENCE_COMPANY_SCAN_QUEUE_URL",
            ),
        ),
        batch_size=configured.company_batch_size or 1,
    )


def build_scanner(settings: Settings | None = None) -> CompanyScanner:
    configured = settings or Settings.from_environment()
    clients = create_aws_clients(configured)
    http = HttpClient(
        timeout_seconds=_required(
            configured.http_timeout_seconds,
            "JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS",
        ),
        user_agent=_required(
            configured.http_user_agent,
            "JOB_INTELLIGENCE_HTTP_USER_AGENT",
        ),
        max_retries=configured.http_max_retries or 1,
    )
    job_repository = _job_repository(clients.table)
    snapshot_store = _snapshot_store(clients.s3, configured)
    filtering = FilteringConfig.from_settings(configured)
    return CompanyScanner(
        companies=_company_repository(clients.table),
        profiles=_profile_repository(clients.table),
        runs=_run_repository(clients.table),
        connectors=ConnectorRegistry(http),
        ingestion=JobIngestionService(job_repository, snapshot_store),
        filtering=DeterministicJobFilter(filtering),
        analysis_queue=SQSQueuePublisher(
            clients.sqs,
            queue_url=_required(
                configured.job_analysis_queue_url,
                "JOB_INTELLIGENCE_JOB_ANALYSIS_QUEUE_URL",
            ),
        ),
        candidate_profile_id=configured.candidate_profile_id,
        candidate_profile_key=configured.candidate_profile_key,
    )


def build_analysis_worker(settings: Settings | None = None) -> JobAnalysisWorker:
    configured = settings or Settings.from_environment()
    clients = create_aws_clients(configured)
    providers = build_ai_providers(configured)
    return JobAnalysisWorker(
        jobs=_job_repository(clients.table),
        profiles=_profile_repository(clients.table),
        analyses=DynamoJobAnalysisRepository(clients.table),
        matches=DynamoJobMatchRepository(clients.table),
        provider=providers.llm,
        embedding_provider=providers.embeddings,
        model_id=_required(
            configured.job_analysis_model_id,
            "JOB_INTELLIGENCE_JOB_ANALYSIS_MODEL_ID",
        ),
    )


def _company_repository(table: Any) -> CompanyRepository:
    return DynamoCompanyRepository(table)


def _profile_repository(table: Any) -> CandidateProfileRepository:
    return DynamoCandidateProfileRepository(table)


def _run_repository(table: Any) -> Any:
    return DynamoScanRunRepository(table)


def _job_repository(table: Any) -> JobRepository:
    return DynamoJobRepository(table)


def _snapshot_store(client: Any, settings: Settings) -> RawSnapshotStore:
    return S3RawSnapshotStore(
        client,
        bucket=_required(settings.s3_bucket_name, "JOB_INTELLIGENCE_S3_BUCKET_NAME"),
        prefix=settings.s3_raw_prefix or "",
    )


def _required(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise AWSConfigurationError(f"{name} must be configured")
    return value


__all__ = [
    "AWSClients",
    "AWSConfigurationError",
    "build_analysis_worker",
    "build_coordinator",
    "build_scanner",
    "create_aws_clients",
]
