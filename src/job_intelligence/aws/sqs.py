"""Typed SQS messages and a small publish adapter."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from ..models import Model


class SQSClient(Protocol):
    def send_message(self, **kwargs: Any) -> dict[str, Any]:
        """Publish one message."""


class QueuePublisher(Protocol):
    def publish(self, message: Model) -> str:
        """Publish a typed message and return its provider message id."""


class CompanyScanMessage(Model):
    run_id: UUID
    company_id: UUID


class JobAnalysisMessage(Model):
    run_id: UUID
    job_id: UUID
    candidate_profile_id: UUID


class SQSQueuePublisher:
    """Publish JSON messages to one standard SQS queue."""

    def __init__(self, client: SQSClient, *, queue_url: str) -> None:
        self._client = client
        self._queue_url = queue_url

    def publish(self, message: Model) -> str:
        response = self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=message.model_dump_json(),
        )
        message_id = response.get("MessageId")
        if not isinstance(message_id, str) or not message_id:
            raise RuntimeError("SQS did not return a message id")
        return message_id


def parse_company_scan_message(body: str) -> CompanyScanMessage:
    return CompanyScanMessage.model_validate_json(body)


def parse_job_analysis_message(body: str) -> JobAnalysisMessage:
    return JobAnalysisMessage.model_validate_json(body)


__all__ = [
    "CompanyScanMessage",
    "JobAnalysisMessage",
    "QueuePublisher",
    "SQSClient",
    "SQSQueuePublisher",
    "parse_company_scan_message",
    "parse_job_analysis_message",
]
