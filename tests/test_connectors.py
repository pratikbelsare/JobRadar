import json
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest

from job_intelligence.connectors import (
    ConnectorRegistry,
    ConnectorStatus,
    GreenhouseConnector,
    JobSourceConnector,
    UnsupportedConnectorError,
)
from job_intelligence.http import HttpClient, HttpResponse
from job_intelligence.models import Company, ConnectorType


class FakeTransport:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[str] = []

    def request(self, url, *, headers, timeout_seconds):
        self.calls.append(url)
        return self.handler(url)


def make_client(handler) -> tuple[HttpClient, FakeTransport]:
    transport = FakeTransport(handler)
    client = HttpClient(
        timeout_seconds=1,
        user_agent="connector-test/1.0",
        max_retries=0,
        transport=transport,
    )
    return client, transport


def make_company() -> Company:
    return Company(
        id=uuid4(),
        name="Example Greenhouse",
        career_url="https://boards.greenhouse.io/example",
        connector_type=ConnectorType.GREENHOUSE,
    )


def json_response(payload, status_code=200) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        url="https://boards-api.greenhouse.io",
        body=json.dumps(payload).encode("utf-8"),
    )


def listing(job_id=123, title="ML Engineer", location="Remote") -> dict:
    return {
        "id": job_id,
        "title": title,
        "location": {"name": location},
        "absolute_url": f"https://boards.greenhouse.io/example/jobs/{job_id}",
    }


def detail(job_id=123) -> dict:
    return {
        **listing(job_id),
        "content": "<p>Build and operate machine learning systems.</p>",
    }


def test_greenhouse_connector_lists_and_retrieves_typed_jobs() -> None:
    company = make_company()

    def handler(url: str) -> HttpResponse:
        if "/jobs/123" in url:
            return json_response(detail())
        return json_response({"jobs": [listing()]})

    client, transport = make_client(handler)
    connector = GreenhouseConnector(
        client,
        api_base_url="https://boards-api.greenhouse.test/v1/boards",
    )

    listing_result = connector.list_jobs(company)
    reference = listing_result.jobs[0]
    detail_result = connector.get_job(reference)

    assert isinstance(connector, JobSourceConnector)
    assert listing_result.status == ConnectorStatus.SUCCESS
    assert reference.source_job_id == "123"
    assert reference.title == "ML Engineer"
    assert reference.location == "Remote"
    assert str(reference.details_url) == (
        "https://boards-api.greenhouse.test/v1/boards/example/jobs/123"
    )
    assert detail_result.status == ConnectorStatus.SUCCESS
    assert detail_result.job is not None
    assert detail_result.job.description.startswith("<p>Build")
    assert len(transport.calls) == 2


def test_greenhouse_connector_handles_pagination() -> None:
    company = make_company()
    page_one = [listing(job_id=index) for index in range(2)]
    page_two = [listing(job_id=3)]

    def handler(url: str) -> HttpResponse:
        page = parse_qs(urlparse(url).query).get("page", ["1"])[0]
        return json_response({"jobs": page_one if page == "1" else page_two})

    client, transport = make_client(handler)
    connector = GreenhouseConnector(client, api_base_url="https://api.test")
    connector._PAGE_SIZE = 2

    result = connector.list_jobs(company)

    assert result.status == ConnectorStatus.SUCCESS
    assert [job.source_job_id for job in result.jobs] == ["0", "1", "3"]
    assert len(transport.calls) == 2


def test_greenhouse_connector_reports_empty_and_malformed_responses() -> None:
    company = make_company()
    client, _ = make_client(lambda _: json_response({"jobs": []}))
    connector = GreenhouseConnector(client, api_base_url="https://api.test")
    assert connector.list_jobs(company).status == ConnectorStatus.NO_JOBS

    malformed_client, _ = make_client(
        lambda _: HttpResponse(status_code=200, url="https://api.test", body=b"not-json")
    )
    malformed = GreenhouseConnector(malformed_client, api_base_url="https://api.test")
    assert malformed.list_jobs(company).status == ConnectorStatus.PARSE_ERROR

    missing_jobs_client, _ = make_client(lambda _: json_response({"results": []}))
    missing_jobs = GreenhouseConnector(missing_jobs_client, api_base_url="https://api.test")
    assert missing_jobs.list_jobs(company).status == ConnectorStatus.INVALID_RESPONSE


def test_greenhouse_connector_reports_http_timeout_and_parsing_failures() -> None:
    company = make_company()
    timeout_client, _ = make_client(lambda _: (_ for _ in ()).throw(TimeoutError()))
    timeout_connector = GreenhouseConnector(timeout_client, api_base_url="https://api.test")
    assert timeout_connector.list_jobs(company).status == ConnectorStatus.TIMEOUT

    network_client, _ = make_client(lambda _: (_ for _ in ()).throw(URLError("offline")))
    network_connector = GreenhouseConnector(network_client, api_base_url="https://api.test")
    assert network_connector.list_jobs(company).status == ConnectorStatus.NETWORK_ERROR

    malformed_listing_client, _ = make_client(
        lambda _: json_response({"jobs": [{"title": "No ID"}]})
    )
    malformed_listing = GreenhouseConnector(
        malformed_listing_client,
        api_base_url="https://api.test",
    )
    assert malformed_listing.list_jobs(company).status == ConnectorStatus.PARSE_ERROR

    detail_client, _ = make_client(lambda _: json_response({"id": 123, "title": "Missing content"}))
    detail_connector = GreenhouseConnector(detail_client, api_base_url="https://api.test")
    reference = connector_reference(company)
    assert detail_connector.get_job(reference).status == ConnectorStatus.PARSE_ERROR


def connector_reference(company: Company):
    client, _ = make_client(lambda _: json_response({"jobs": [listing()]}))
    result = GreenhouseConnector(client, api_base_url="https://api.test").list_jobs(company)
    return result.jobs[0]


def test_registry_selects_supported_connector_and_rejects_unsupported() -> None:
    client, _ = make_client(lambda _: json_response({"jobs": []}))
    registry = ConnectorRegistry(client)
    assert isinstance(registry.for_company(make_company()), GreenhouseConnector)

    unsupported_company = Company(
        name="Generic",
        career_url="https://example.test/careers",
        connector_type=ConnectorType.GENERIC,
    )
    with pytest.raises(UnsupportedConnectorError) as error:
        registry.for_company(unsupported_company)
    assert error.value.status == ConnectorStatus.UNSUPPORTED
