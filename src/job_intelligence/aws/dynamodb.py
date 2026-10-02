"""DynamoDB adapters for JobRadar's replaceable repository interfaces.

The adapters use one deliberately small table with explicit entity keys. ``PK`` and
``SK`` keep related job versions together, while two GSIs support the actual list and
identity lookups used by the application.
"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from pydantic import ValidationError

from ..ai.models import JobMatchExplanation
from ..analysis_repository import JobAnalysisRepository
from ..change_detection import StoredJobState
from ..company_repository import CompanyRepository
from ..explanation_repository import ExplanationRepository
from ..feedback import UserFeedbackRepository
from ..job_repository import JobRepository
from ..match_repository import JobMatchRepository
from ..models import (
    CandidateProfile,
    Company,
    Job,
    JobAnalysis,
    JobMatch,
    JobVersion,
    Model,
    ScanRun,
    UserFeedback,
)
from ..normalization import JobIdentity, job_identity
from ..profile_repository import CandidateProfileRepository
from ..run_repository import ScanRunRepository


class DynamoDBError(RuntimeError):
    """Raised when a DynamoDB operation fails or contains invalid data."""


class DynamoTable(Protocol):
    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        """Read one item."""

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        """Write one item."""

    def query(self, **kwargs: Any) -> dict[str, Any]:
        """Query one partition or index."""


class DynamoEntityStore:
    """Serialize validated Pydantic models into a shared DynamoDB table."""

    def __init__(self, table: DynamoTable) -> None:
        self.table = table

    def put(
        self,
        *,
        pk: str,
        sk: str,
        model: Model,
        entity: str,
        gsi1: tuple[str, str] | None = None,
        gsi2: tuple[str, str] | None = None,
        condition: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "Entity": entity,
            "Payload": model.model_dump_json(),
        }
        if gsi1 is not None:
            item["GSI1PK"], item["GSI1SK"] = gsi1
        if gsi2 is not None:
            item["GSI2PK"], item["GSI2SK"] = gsi2
        try:
            kwargs: dict[str, Any] = {"Item": item}
            if condition is not None:
                kwargs["ConditionExpression"] = condition
            self.table.put_item(**kwargs)
        except Exception as exc:
            raise DynamoDBError(f"Could not write DynamoDB item {pk}/{sk}") from exc

    def get(self, pk: str, sk: str) -> dict[str, Any] | None:
        try:
            return self.table.get_item(Key={"PK": pk, "SK": sk}).get("Item")
        except Exception as exc:
            raise DynamoDBError(f"Could not read DynamoDB item {pk}/{sk}") from exc

    def query(
        self,
        *,
        key_expression: str,
        values: dict[str, str],
        index: str | None = None,
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": key_expression,
            "ExpressionAttributeValues": values,
        }
        if index is not None:
            kwargs["IndexName"] = index
        if filter_expression is not None:
            kwargs["FilterExpression"] = filter_expression
        items: list[dict[str, Any]] = []
        try:
            while True:
                response = self.table.query(**kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    return items
                kwargs["ExclusiveStartKey"] = last_key
        except Exception as exc:
            raise DynamoDBError("Could not query DynamoDB") from exc


class DynamoCandidateProfileRepository(CandidateProfileRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def save(self, profile: CandidateProfile) -> CandidateProfile:
        self._store.put(
            pk=f"PROFILE#{profile.id}",
            sk="RECORD",
            model=profile,
            entity="PROFILE",
            gsi2=("PROFILE", str(profile.id)),
        )
        return profile

    def load(self, profile_id: UUID) -> CandidateProfile | None:
        item = self._store.get(f"PROFILE#{profile_id}", "RECORD")
        return _model_from_item(item, CandidateProfile) if item else None

    def find_by_key(self, profile_key: str) -> CandidateProfile | None:
        return next(
            (profile for profile in self.list() if profile.profile_key == profile_key),
            None,
        )

    def list(self) -> list[CandidateProfile]:
        return [
            _model_from_item(item, CandidateProfile)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": "PROFILE"},
                index="GSI2",
            )
        ]


class DynamoCompanyRepository(CompanyRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def save(self, company: Company) -> Company:
        self._store.put(
            pk=f"COMPANY#{company.id}",
            sk="RECORD",
            model=company,
            entity="COMPANY",
            gsi2=("COMPANY", str(company.id)),
        )
        return company

    def get(self, company_id: UUID) -> Company | None:
        item = self._store.get(f"COMPANY#{company_id}", "RECORD")
        return _model_from_item(item, Company) if item else None

    def list(self) -> list[Company]:
        return [
            _model_from_item(item, Company)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": "COMPANY"},
                index="GSI2",
            )
        ]


class DynamoScanRunRepository(ScanRunRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def save(self, run: ScanRun) -> ScanRun:
        self._store.put(
            pk=f"RUN#{run.id}",
            sk="RECORD",
            model=run,
            entity="RUN",
            gsi2=("RUN", str(run.id)),
        )
        return run

    def get(self, run_id: UUID) -> ScanRun | None:
        item = self._store.get(f"RUN#{run_id}", "RECORD")
        return _model_from_item(item, ScanRun) if item else None

    def list(self) -> list[ScanRun]:
        return [
            _model_from_item(item, ScanRun)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": "RUN"},
                index="GSI2",
            )
        ]


class DynamoJobRepository(JobRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def find_by_identity(self, identity: JobIdentity) -> StoredJobState | None:
        for key in (identity.source_key, identity.url_key):
            items = self._store.query(
                key_expression="GSI1PK = :pk",
                values={":pk": key},
                index="GSI1",
            )
            if items:
                return self._state_from_current(items[0])
        return None

    def find_by_id(self, job_id: UUID) -> StoredJobState | None:
        item = self._store.get(f"JOB#{job_id}", "CURRENT")
        return self._state_from_current(item) if item else None

    def list(self) -> list[StoredJobState]:
        return [
            self._state_from_current(item)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": "JOB"},
                index="GSI2",
            )
        ]

    def save(self, state: StoredJobState) -> StoredJobState:
        identity = job_identity(state.job)
        for version in state.versions:
            self._save_version(version)
        self._store.put(
            pk=f"JOB#{state.job.id}",
            sk="CURRENT",
            model=state.job,
            entity="JOB",
            gsi1=(identity.source_key, identity.url_key),
            gsi2=("JOB", state.job.last_seen_at.isoformat()),
        )
        return state

    def _save_version(self, version: JobVersion) -> None:
        pk = f"JOB#{version.job_id}"
        sk = f"VERSION#{version.version_number:08d}"
        try:
            self._store.put(
                pk=pk,
                sk=sk,
                model=version,
                entity="JOB_VERSION",
                condition="attribute_not_exists(PK)",
            )
        except DynamoDBError:
            existing = self._store.get(pk, sk)
            if existing is None or existing.get("Payload") != version.model_dump_json():
                raise DynamoDBError(
                    f"Historical job version cannot be overwritten: {version.job_id}"
                )

    def _state_from_current(self, item: dict[str, Any]) -> StoredJobState:
        job = _model_from_item(item, Job)
        versions = [
            _model_from_item(version, JobVersion)
            for version in self._store.query(
                key_expression="PK = :pk AND begins_with(SK, :prefix)",
                values={":pk": f"JOB#{job.id}", ":prefix": "VERSION#"},
            )
        ]
        return StoredJobState(
            job=job,
            versions=sorted(versions, key=lambda value: value.version_number),
        )


class DynamoJobAnalysisRepository(JobAnalysisRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def get(
        self,
        job_id: UUID,
        job_version_id: UUID,
        model_id: str,
        prompt_version: str,
    ) -> JobAnalysis | None:
        key = _analysis_key(job_id, job_version_id, model_id, prompt_version)
        item = self._store.get(key, "RECORD")
        return _model_from_item(item, JobAnalysis) if item else None

    def save(self, analysis: JobAnalysis) -> JobAnalysis:
        key = _analysis_key(
            analysis.job_id,
            analysis.job_version_id,
            analysis.model_id,
            analysis.prompt_version,
        )
        self._store.put(pk=key, sk="RECORD", model=analysis, entity="JOB_ANALYSIS")
        return analysis


class DynamoJobMatchRepository(JobMatchRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def save(self, match: JobMatch) -> JobMatch:
        self._store.put(
            pk=f"MATCH#{match.candidate_profile_id}#{match.job_id}",
            sk="CURRENT",
            model=match,
            entity="JOB_MATCH",
            gsi2=(f"MATCH#{match.candidate_profile_id}", str(match.job_id)),
        )
        return match

    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatch | None:
        item = self._store.get(f"MATCH#{candidate_profile_id}#{job_id}", "CURRENT")
        return _model_from_item(item, JobMatch) if item else None

    def list(self, candidate_profile_id: UUID | None = None) -> list[JobMatch]:
        if candidate_profile_id is None:
            return []
        return [
            _model_from_item(item, JobMatch)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": f"MATCH#{candidate_profile_id}"},
                index="GSI2",
            )
        ]


class DynamoUserFeedbackRepository(UserFeedbackRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def save(self, feedback: UserFeedback) -> UserFeedback:
        self._store.put(
            pk=f"FEEDBACK#{feedback.id}",
            sk="RECORD",
            model=feedback,
            entity="FEEDBACK",
            gsi2=("FEEDBACK", f"{feedback.candidate_profile_id}#{feedback.created_at.isoformat()}"),
        )
        return feedback

    def load(self, feedback_id: UUID) -> UserFeedback | None:
        item = self._store.get(f"FEEDBACK#{feedback_id}", "RECORD")
        return _model_from_item(item, UserFeedback) if item else None

    def list(self) -> list[UserFeedback]:
        return [
            _model_from_item(item, UserFeedback)
            for item in self._store.query(
                key_expression="GSI2PK = :pk",
                values={":pk": "FEEDBACK"},
                index="GSI2",
            )
        ]


class DynamoExplanationRepository(ExplanationRepository):
    def __init__(self, table: DynamoTable) -> None:
        self._store = DynamoEntityStore(table)

    def get(self, job_id: UUID, candidate_profile_id: UUID) -> JobMatchExplanation | None:
        item = self._store.get(f"EXPLANATION#{candidate_profile_id}#{job_id}", "CURRENT")
        return _model_from_item(item, JobMatchExplanation) if item else None

    def save(
        self,
        job_id: UUID,
        candidate_profile_id: UUID,
        explanation: JobMatchExplanation,
    ) -> None:
        self._store.put(
            pk=f"EXPLANATION#{candidate_profile_id}#{job_id}",
            sk="CURRENT",
            model=explanation,
            entity="EXPLANATION",
        )


def _analysis_key(job_id: UUID, version_id: UUID, model_id: str, prompt_version: str) -> str:
    return f"ANALYSIS#{job_id}#{version_id}#{model_id}#{prompt_version}"


def _model_from_item(item: dict[str, Any], model_type: type[Model]) -> Any:
    try:
        return model_type.model_validate_json(item["Payload"])
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise DynamoDBError("DynamoDB item contained invalid model data") from exc


__all__ = [
    "DynamoCandidateProfileRepository",
    "DynamoCompanyRepository",
    "DynamoDBError",
    "DynamoEntityStore",
    "DynamoExplanationRepository",
    "DynamoJobAnalysisRepository",
    "DynamoJobMatchRepository",
    "DynamoJobRepository",
    "DynamoScanRunRepository",
    "DynamoTable",
    "DynamoUserFeedbackRepository",
]
