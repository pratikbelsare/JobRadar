"""Validated, idempotent bootstrap services and CLI for AWS seed data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from pydantic import ValidationError

from .company_repository import CompanyRepository, JsonCompanyRepository
from .config import Settings
from .models import CandidateProfile, Company, ConnectorType
from .normalization import canonicalize_url
from .profile_repository import CandidateProfileRepository, JsonCandidateProfileRepository


class BootstrapError(ValueError):
    """Raised when bootstrap input or deployment configuration is invalid."""


class CandidateProfileBootstrap:
    """Create or intentionally replace one profile selected by a stable key."""

    def __init__(self, repository: CandidateProfileRepository) -> None:
        self._repository = repository

    def upsert(self, profile_key: str, profile: CandidateProfile) -> CandidateProfile:
        if not profile_key.strip():
            raise BootstrapError("profile key must not be blank")
        existing = self._repository.find_by_key(profile_key)
        values = profile.model_dump(mode="python")
        values["profile_key"] = profile_key.strip()
        if existing is not None:
            values["id"] = existing.id
        return self._repository.save(CandidateProfile.model_validate(values))


class CompanyBootstrap:
    """Validate and upsert a small configured company catalog without duplicates."""

    def __init__(self, repository: CompanyRepository) -> None:
        self._repository = repository

    def upsert(self, companies: Sequence[Company]) -> list[Company]:
        if not companies:
            raise BootstrapError("at least one company is required")
        existing = self._repository.list()
        by_id = {company.id: company for company in existing}
        by_url = {_company_url(company): company for company in existing}
        seen_urls: set[str] = set()
        saved: list[Company] = []
        for company in companies:
            _validate_connector(company)
            url = _company_url(company)
            if url in seen_urls:
                raise BootstrapError(f"duplicate company URL in input: {company.career_url}")
            seen_urls.add(url)
            prior = by_id.get(company.id) or by_url.get(url)
            if prior is not None and prior.id != company.id:
                company = company.model_copy(update={"id": prior.id})
            saved.append(self._repository.save(company))
        return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result: dict[str, Any] | list[dict[str, Any]]
        if args.command == "profile":
            result = _bootstrap_profile(args)
        else:
            result = _bootstrap_companies(args)
    except (BootstrapError, OSError, TypeError, ValueError, ValidationError) as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _bootstrap_profile(args: argparse.Namespace) -> dict[str, Any]:
    repository = _profile_repository(args)
    profile = CandidateProfile.model_validate(_read_json_object(args.input))
    saved = CandidateProfileBootstrap(repository).upsert(args.profile_key, profile)
    return {"profile_key": saved.profile_key, "profile_id": str(saved.id)}


def _bootstrap_companies(args: argparse.Namespace) -> list[dict[str, Any]]:
    repository = _company_repository(args)
    raw: Any = _read_json(args.input)
    if not isinstance(raw, list):
        raise BootstrapError("companies input must contain a JSON list")
    companies = [Company.model_validate(item) for item in raw]
    saved = CompanyBootstrap(repository).upsert(companies)
    return [
        {
            "company_id": str(company.id),
            "name": company.name,
            "enabled": company.enabled,
            "connector_type": company.connector_type.value,
        }
        for company in saved
    ]


def _profile_repository(args: argparse.Namespace) -> CandidateProfileRepository:
    if args.backend == "json":
        return JsonCandidateProfileRepository(args.directory or Path("data/profiles"))
    return _dynamo_profile_repository(args)


def _company_repository(args: argparse.Namespace) -> CompanyRepository:
    if args.backend == "json":
        return JsonCompanyRepository(args.directory or Path("data/companies.json"))
    return _dynamo_company_repository(args)


def _dynamo_profile_repository(args: argparse.Namespace) -> CandidateProfileRepository:
    from .aws.dynamodb import DynamoCandidateProfileRepository

    return DynamoCandidateProfileRepository(_dynamo_table(args))


def _dynamo_company_repository(args: argparse.Namespace) -> CompanyRepository:
    from .aws.dynamodb import DynamoCompanyRepository

    return DynamoCompanyRepository(_dynamo_table(args))


def _dynamo_table(args: argparse.Namespace) -> Any:
    settings = Settings.from_environment()
    region = args.region or settings.aws_region
    table_name = args.table or settings.dynamodb_table_name
    if not region or not table_name:
        raise BootstrapError(
            "DynamoDB bootstrap requires --region/--table or the corresponding "
            "JOB_INTELLIGENCE_AWS_REGION/JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME settings"
        )
    try:
        import boto3
    except ImportError as exc:
        raise BootstrapError("DynamoDB bootstrap requires boto3") from exc
    return boto3.resource("dynamodb", region_name=region).Table(table_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m job_intelligence.bootstrap")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="upsert one candidate profile")
    profile.add_argument(
        "--profile-key",
        "--id",
        dest="profile_key",
        required=True,
        help="stable external profile key (the input JSON may carry the UUID)",
    )
    profile.add_argument("--input", required=True, type=Path)
    _add_repository_options(profile)

    companies = commands.add_parser("companies", help="upsert configured companies")
    companies.add_argument("input", type=Path)
    _add_repository_options(companies)
    return parser


def _add_repository_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("json", "dynamodb"), default="json")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--region")
    parser.add_argument("--table")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise BootstrapError("profile input must contain a JSON object")
    return raw


def _company_url(company: Company) -> str:
    return str(canonicalize_url(company.career_url))


def _validate_connector(company: Company) -> None:
    host = (urlsplit(str(company.career_url)).hostname or "").casefold()
    if company.connector_type == ConnectorType.GREENHOUSE:
        if "greenhouse.io" not in host:
            raise BootstrapError(
                "Greenhouse companies must use a greenhouse.io career URL"
            )
        return
    raise BootstrapError(
        f"connector type {company.connector_type.value!r} is not supported by the "
        "current connector registry"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BootstrapError", "CandidateProfileBootstrap", "CompanyBootstrap", "main"]
