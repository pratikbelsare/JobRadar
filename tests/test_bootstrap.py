from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pytest
from pydantic import ValidationError

from job_intelligence.aws.dynamodb import DynamoCandidateProfileRepository
from job_intelligence.bootstrap import (
    BootstrapError,
    CandidateProfileBootstrap,
    CompanyBootstrap,
    main,
)
from job_intelligence.company_repository import JsonCompanyRepository
from job_intelligence.models import CandidateProfile, Company, ConnectorType
from job_intelligence.profile_repository import JsonCandidateProfileRepository


class BootstrapDynamoTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = kwargs["Key"]
        assert isinstance(key, dict)
        item = self.items.get((str(key["PK"]), str(key["SK"])))
        return {"Item": item} if item else {}

    def put_item(self, **kwargs: object) -> dict[str, object]:
        item = kwargs["Item"]
        assert isinstance(item, dict)
        self.items[(str(item["PK"]), str(item["SK"]))] = item
        return {}

    def query(self, **kwargs: object) -> dict[str, object]:
        values = kwargs["ExpressionAttributeValues"]
        assert isinstance(values, dict)
        key = str(values[":pk"])
        return {
            "Items": [item for item in self.items.values() if item.get("GSI2PK") == key]
        }


def _company() -> Company:
    return Company(
        name="Example Greenhouse Company",
        career_url="https://boards.greenhouse.io/example-board",
        connector_type=ConnectorType.GREENHOUSE,
        enabled=False,
    )


def test_profile_bootstrap_is_idempotent_and_replaces_by_key() -> None:
    with TemporaryDirectory() as directory:
        repository = JsonCandidateProfileRepository(directory)
        bootstrap = CandidateProfileBootstrap(repository)
        first = bootstrap.upsert("pratik", CandidateProfile(target_roles=["Configured role"]))
        second = bootstrap.upsert("pratik", CandidateProfile(target_roles=["Updated role"]))

        assert second.id == first.id
        assert repository.list() == [second]
        assert repository.find_by_key("pratik") == second


def test_profile_bootstrap_rejects_invalid_profile_data() -> None:
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate({"experience_years": -1})


def test_company_bootstrap_is_idempotent_and_rejects_unsupported_connectors() -> None:
    with TemporaryDirectory() as directory:
        repository = JsonCompanyRepository(Path(directory) / "companies.json")
        bootstrap = CompanyBootstrap(repository)
        first = bootstrap.upsert([_company()])[0]
        second = bootstrap.upsert([first.model_copy(update={"enabled": True})])[0]

        assert second.id == first.id
        assert repository.list() == [second]

        unsupported = first.model_copy(
            update={"connector_type": ConnectorType.LEVER}
        )
        with pytest.raises(BootstrapError, match="not supported"):
            bootstrap.upsert([unsupported])


def test_company_bootstrap_rejects_duplicate_urls_in_one_input() -> None:
    with TemporaryDirectory() as directory:
        repository = JsonCompanyRepository(Path(directory) / "companies.json")
        company = _company()
        with pytest.raises(BootstrapError, match="duplicate company URL"):
            duplicate = company.model_copy(
                update={"id": UUID("00000000-0000-4000-8000-000000000002")}
            )
            CompanyBootstrap(repository).upsert(
                [company, duplicate]
            )


def test_profile_cli_uses_the_repository_abstraction() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "profile.json"
        input_path.write_text(json.dumps({"target_roles": ["Configured role"]}), encoding="utf-8")
        result = main(
            [
                "profile",
                "--id",
                "pratik",
                "--input",
                str(input_path),
                "--directory",
                str(root / "profiles"),
            ]
        )
        assert result == 0
        assert len(list((root / "profiles").glob("*.json"))) == 1


def test_profile_bootstrap_uses_the_dynamodb_repository_adapter() -> None:
    repository = DynamoCandidateProfileRepository(BootstrapDynamoTable())
    bootstrap = CandidateProfileBootstrap(repository)
    first = bootstrap.upsert("pratik", CandidateProfile())
    second = bootstrap.upsert("pratik", CandidateProfile(target_roles=["Updated role"]))

    assert first.id == second.id
    assert repository.find_by_key("pratik") == second
