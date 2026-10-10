"""Shared connector contracts, typed outputs, and failure statuses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import AnyHttpUrl, Field

from ..models import Company, ConnectorType, Model, NonEmptyText


class ConnectorStatus(str, Enum):
    SUCCESS = "success"
    NO_JOBS = "no_jobs"
    UNSUPPORTED = "unsupported"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    PARSE_ERROR = "parse_error"
    FAILED = "failed"


class JobReference(Model):
    """A stable listing reference returned by a source connector."""

    company_id: UUID
    source_job_id: NonEmptyText
    title: NonEmptyText
    location: NonEmptyText | None = None
    source_url: AnyHttpUrl
    details_url: AnyHttpUrl


class RawJobDetails(Model):
    """Raw source details before normalization or AI analysis."""

    company_id: UUID
    source_job_id: NonEmptyText
    title: NonEmptyText
    location: NonEmptyText | None = None
    source_url: AnyHttpUrl
    details_url: AnyHttpUrl
    description: NonEmptyText
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class JobListingResult(Model):
    status: ConnectorStatus
    jobs: list[JobReference] = Field(default_factory=list)
    error: NonEmptyText | None = None


class JobDetailsResult(Model):
    status: ConnectorStatus
    job: RawJobDetails | None = None
    error: NonEmptyText | None = None


class ConnectorError(RuntimeError):
    """Base error carrying a representable connector status."""

    def __init__(self, status: ConnectorStatus, message: str) -> None:
        self.status = status
        super().__init__(message)


class UnsupportedConnectorError(ConnectorError):
    def __init__(self, connector_type: ConnectorType) -> None:
        super().__init__(
            ConnectorStatus.UNSUPPORTED,
            f"No connector registered for type {connector_type.value}",
        )


class ConnectorNetworkError(ConnectorError):
    def __init__(self, message: str) -> None:
        super().__init__(ConnectorStatus.NETWORK_ERROR, message)


class ConnectorTimeoutError(ConnectorError):
    def __init__(self, message: str) -> None:
        super().__init__(ConnectorStatus.TIMEOUT, message)


class ConnectorRateLimitedError(ConnectorError):
    def __init__(self, message: str) -> None:
        super().__init__(ConnectorStatus.RATE_LIMITED, message)


class ConnectorInvalidResponseError(ConnectorError):
    def __init__(self, message: str) -> None:
        super().__init__(ConnectorStatus.INVALID_RESPONSE, message)


class ConnectorParseError(ConnectorError):
    def __init__(self, message: str) -> None:
        super().__init__(ConnectorStatus.PARSE_ERROR, message)


@runtime_checkable
class JobSourceConnector(Protocol):
    """Interface implemented by each career-site connector."""

    connector_type: ConnectorType

    def list_jobs(self, company: Company) -> JobListingResult:
        """Discover currently listed jobs for a configured company."""

    def get_job(self, job_reference: JobReference) -> JobDetailsResult:
        """Fetch raw details for one discovered job."""


__all__ = [
    "ConnectorError",
    "ConnectorInvalidResponseError",
    "ConnectorNetworkError",
    "ConnectorParseError",
    "ConnectorRateLimitedError",
    "ConnectorStatus",
    "ConnectorTimeoutError",
    "JobDetailsResult",
    "JobListingResult",
    "JobReference",
    "JobSourceConnector",
    "RawJobDetails",
    "UnsupportedConnectorError",
]
