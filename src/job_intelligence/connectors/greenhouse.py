"""Greenhouse public Job Board API connector."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlparse

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from ..http import (
    HttpClient,
    HttpNetworkError,
    HttpStatusError,
    HttpTimeoutError,
    RateLimitedHttpError,
)
from ..models import Company, ConnectorType
from .base import (
    ConnectorError,
    ConnectorInvalidResponseError,
    ConnectorNetworkError,
    ConnectorParseError,
    ConnectorRateLimitedError,
    ConnectorStatus,
    ConnectorTimeoutError,
    JobDetailsResult,
    JobListingResult,
    JobReference,
    JobSourceConnector,
    RawJobDetails,
)


class GreenhouseConnector(JobSourceConnector):
    """Read jobs from Greenhouse's public board API."""

    connector_type = ConnectorType.GREENHOUSE
    _DEFAULT_API_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"
    _PAGE_SIZE = 100

    def __init__(
        self,
        http_client: HttpClient,
        *,
        api_base_url: str = _DEFAULT_API_BASE_URL,
    ) -> None:
        self._http_client = http_client
        self._api_base_url = api_base_url.rstrip("/")

    def list_jobs(self, company: Company) -> JobListingResult:
        try:
            board_token = _board_token(company)
            references: list[JobReference] = []
            page = 1
            while True:
                payload = self._get_json(
                    f"{self._api_base_url}/{quote(board_token, safe='')}/jobs",
                    {"content": "true", "page": page, "per_page": self._PAGE_SIZE},
                )
                entries = _jobs_array(payload)
                references.extend(
                    _job_reference(company, board_token, self._api_base_url, entry)
                    for entry in entries
                )
                if len(entries) < self._PAGE_SIZE:
                    break
                page += 1

            if not references:
                return JobListingResult(status=ConnectorStatus.NO_JOBS)
            return JobListingResult(status=ConnectorStatus.SUCCESS, jobs=references)
        except ConnectorError as exc:
            return JobListingResult(status=exc.status, error=str(exc))
        except (TypeError, ValidationError, ValueError) as exc:
            return JobListingResult(
                status=ConnectorStatus.PARSE_ERROR,
                error=f"Could not parse Greenhouse listing response: {exc}",
            )

    def get_job(self, job_reference: JobReference) -> JobDetailsResult:
        try:
            payload = self._get_json(
                str(job_reference.details_url),
                {"content": "true"},
            )
            job = _job_details(job_reference, payload)
            return JobDetailsResult(status=ConnectorStatus.SUCCESS, job=job)
        except ConnectorError as exc:
            return JobDetailsResult(status=exc.status, error=str(exc))
        except (TypeError, ValidationError, ValueError) as exc:
            return JobDetailsResult(
                status=ConnectorStatus.PARSE_ERROR,
                error=f"Could not parse Greenhouse job response: {exc}",
            )

    def _get_json(
        self,
        url: str,
        params: Mapping[str, str | int] | None,
    ) -> dict[str, Any]:
        try:
            response = self._http_client.get(url, params=params)
        except RateLimitedHttpError as exc:
            raise ConnectorRateLimitedError(str(exc)) from exc
        except HttpTimeoutError as exc:
            raise ConnectorTimeoutError(str(exc)) from exc
        except HttpNetworkError as exc:
            raise ConnectorNetworkError(str(exc)) from exc
        except HttpStatusError as exc:
            raise ConnectorInvalidResponseError(str(exc)) from exc

        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConnectorParseError("Greenhouse response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ConnectorInvalidResponseError("Greenhouse response must be a JSON object")
        return payload


def _board_token(company: Company) -> str:
    configured_token = company.connector_config.get("board_token")
    if configured_token:
        return configured_token

    parsed = urlparse(str(company.career_url))
    if parsed.hostname not in {"boards.greenhouse.io", "job-boards.greenhouse.io"}:
        raise ConnectorParseError(
            "Greenhouse company career_url must use a supported Greenhouse board host"
        )
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        raise ConnectorParseError("Greenhouse career_url does not contain a board token")
    return segments[0]


def _jobs_array(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = payload.get("jobs")
    if not isinstance(entries, list):
        raise ConnectorInvalidResponseError("Greenhouse response is missing a jobs array")
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise ConnectorParseError("Greenhouse jobs array contains a non-object item")
    return entries


def _job_reference(
    company: Company,
    board_token: str,
    api_base_url: str,
    entry: Mapping[str, Any],
) -> JobReference:
    source_job_id = entry.get("id")
    title = entry.get("title")
    source_url = entry.get("absolute_url")
    if source_job_id is None or not isinstance(title, str) or not title.strip():
        raise ConnectorParseError("Greenhouse job listing is missing id or title")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ConnectorParseError("Greenhouse job listing is missing absolute_url")

    details_url = _details_url(api_base_url, board_token, source_job_id)
    return JobReference(
        company_id=company.id,
        source_job_id=str(source_job_id),
        title=title,
        location=_location(entry.get("location")),
        source_url=source_url,
        details_url=details_url,
    )


def _job_details(reference: JobReference, payload: Mapping[str, Any]) -> RawJobDetails:
    source_job_id = payload.get("id", reference.source_job_id)
    if str(source_job_id) != reference.source_job_id:
        raise ConnectorParseError("Greenhouse job detail id does not match its reference")

    title = payload.get("title", reference.title)
    if not isinstance(title, str) or not title.strip():
        raise ConnectorParseError("Greenhouse job detail is missing title")
    description = next(
        (
            payload.get(field)
            for field in ("content", "description", "job_description")
            if isinstance(payload.get(field), str) and payload.get(field).strip()
        ),
        None,
    )
    if not isinstance(description, str):
        raise ConnectorParseError("Greenhouse job detail is missing description content")

    source_url = payload.get("absolute_url", str(reference.source_url))
    if not isinstance(source_url, str):
        raise ConnectorParseError("Greenhouse job detail has an invalid absolute_url")
    return RawJobDetails(
        company_id=reference.company_id,
        source_job_id=reference.source_job_id,
        title=title,
        location=_location(payload.get("location")) or reference.location,
        source_url=source_url,
        details_url=reference.details_url,
        description=description,
        raw_payload=dict(payload),
    )


def _location(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("name")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _details_url(api_base_url: str, board_token: str, source_job_id: Any) -> AnyHttpUrl:
    url = (
        f"{api_base_url.rstrip('/')}/{quote(board_token, safe='')}"
        f"/jobs/{quote(str(source_job_id), safe='')}"
    )
    return TypeAdapter(AnyHttpUrl).validate_python(url)


__all__ = ["GreenhouseConnector"]
