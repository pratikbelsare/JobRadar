from __future__ import annotations

import io
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from job_intelligence.aws.dynamodb import (
    DynamoCandidateProfileRepository,
    DynamoDBError,
    DynamoJobRepository,
)
from job_intelligence.aws.s3 import S3RawSnapshotStore
from job_intelligence.aws.sqs import (
    CompanyScanMessage,
    JobAnalysisMessage,
    SQSQueuePublisher,
    parse_company_scan_message,
)
from job_intelligence.change_detection import StoredJobState
from job_intelligence.connectors.base import RawJobDetails
from job_intelligence.models import CandidateProfile, Job, JobVersion
from job_intelligence.normalization import job_identity


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **kwargs: object) -> dict[str, object]:
        body = kwargs["Body"]
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = (
            body if isinstance(body, bytes) else str(body).encode()
        )
        return {}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        body = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))]
        return {"Body": io.BytesIO(body)}


class FakeSQS:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send_message(self, **kwargs: object) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"MessageId": "message-1"}


class FakeDynamoTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = kwargs["Key"]
        assert isinstance(key, dict)
        item = self.items.get((str(key["PK"]), str(key["SK"])))
        return {"Item": item} if item is not None else {}

    def put_item(self, **kwargs: object) -> dict[str, object]:
        item = kwargs["Item"]
        assert isinstance(item, dict)
        key = (str(item["PK"]), str(item["SK"]))
        if kwargs.get("ConditionExpression") and key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = item
        return {}

    def query(self, **kwargs: object) -> dict[str, object]:
        values = kwargs["ExpressionAttributeValues"]
        assert isinstance(values, dict)
        partition_key = str(values[":pk"])
        expression = str(kwargs["KeyConditionExpression"])
        if kwargs.get("IndexName") == "GSI1":
            candidates = [
                item for item in self.items.values() if item.get("GSI1PK") == partition_key
            ]
        elif kwargs.get("IndexName") == "GSI2":
            candidates = [
                item for item in self.items.values() if item.get("GSI2PK") == partition_key
            ]
        else:
            candidates = [item for item in self.items.values() if item.get("PK") == partition_key]
            if "begins_with" in expression:
                prefix = str(values[":prefix"])
                candidates = [item for item in candidates if str(item["SK"]).startswith(prefix)]
        return {"Items": candidates}


def _raw_job(company_id: object) -> RawJobDetails:
    return RawJobDetails(
        company_id=company_id,
        source_job_id="source-1",
        title="Engineer",
        location="Remote",
        source_url="https://example.com/jobs/1",
        details_url="https://example.com/jobs/1",
        description="Build reliable systems.",
    )


def test_s3_snapshot_round_trip_and_missing_snapshot() -> None:
    client = FakeS3()
    store = S3RawSnapshotStore(client, bucket="private-bucket", prefix="jobradar")
    metadata = store.save(_raw_job(uuid4()), captured_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    loaded = store.load(metadata.snapshot_id)
    assert loaded.description == "Build reliable systems."
    from job_intelligence.snapshots import SnapshotNotFoundError

    with pytest.raises(SnapshotNotFoundError):
        store.load(uuid4())


def test_sqs_messages_are_typed_and_serialized() -> None:
    client = FakeSQS()
    publisher = SQSQueuePublisher(client, queue_url="https://sqs.example/queue")
    message = CompanyScanMessage(run_id=uuid4(), company_id=uuid4())
    assert publisher.publish(message) == "message-1"
    parsed = parse_company_scan_message(str(client.messages[0]["MessageBody"]))
    assert parsed == message
    analysis = JobAnalysisMessage(
        run_id=message.run_id,
        job_id=uuid4(),
        candidate_profile_id=uuid4(),
    )
    assert analysis.job_id != message.company_id


def test_dynamodb_profile_and_job_identity_version_storage() -> None:
    table = FakeDynamoTable()
    profiles = DynamoCandidateProfileRepository(table)
    profile = CandidateProfile()
    profiles.save(profile)
    assert profiles.load(profile.id) == profile
    assert profiles.list() == [profile]

    company_id = uuid4()
    job = Job(
        source_job_id="source-1",
        company_id=company_id,
        title="Engineer",
        description="Build systems.",
        source_url="https://example.com/jobs/1",
    )
    version = JobVersion(
        job_id=job.id,
        version_number=1,
        content_hash="hash-1",
        description=job.description,
        job_snapshot=job,
    )
    jobs = DynamoJobRepository(table)
    jobs.save(StoredJobState(job=job, versions=[version]))
    loaded = jobs.find_by_identity(job_identity(job))
    assert loaded is not None
    assert loaded.versions[0].version_number == 1
    with pytest.raises(DynamoDBError):
        jobs.save(
            StoredJobState(
                job=job,
                versions=[version.model_copy(update={"description": "different"})],
            )
        )
