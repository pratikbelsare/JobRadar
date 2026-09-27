import socket
from collections.abc import Mapping
from urllib.error import URLError

import pytest

from job_intelligence.http import (
    HttpClient,
    HttpNetworkError,
    HttpResponse,
    HttpStatusError,
    HttpTimeoutError,
    RateLimitedHttpError,
)


class FakeTransport:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def request(self, url, *, headers, timeout_seconds):
        self.calls.append((url, headers, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response(status_code: int, body: bytes = b"{}") -> HttpResponse:
    return HttpResponse(status_code=status_code, url="https://example.test", body=body)


def test_http_client_adds_configured_headers_and_query_parameters() -> None:
    transport = FakeTransport([response(200, b"ok")])
    client = HttpClient(
        timeout_seconds=7.5,
        user_agent="test-agent/1.0",
        max_retries=0,
        transport=transport,
    )

    result = client.get("https://example.test/jobs", params={"page": 2})

    assert result.body == b"ok"
    url, headers, timeout = transport.calls[0]
    assert url == "https://example.test/jobs?page=2"
    assert headers["User-Agent"] == "test-agent/1.0"
    assert timeout == 7.5


def test_http_client_retries_transient_status_once() -> None:
    transport = FakeTransport([response(503), response(200, b"ok")])
    client = HttpClient(
        timeout_seconds=1,
        user_agent="test-agent/1.0",
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )

    assert client.get("https://example.test").body == b"ok"
    assert len(transport.calls) == 2


def test_http_client_reports_status_rate_limit_and_timeout_errors() -> None:
    with pytest.raises(HttpStatusError):
        HttpClient(
            timeout_seconds=1,
            user_agent="test-agent/1.0",
            max_retries=0,
            transport=FakeTransport([response(500)]),
        ).get("https://example.test")

    with pytest.raises(RateLimitedHttpError):
        HttpClient(
            timeout_seconds=1,
            user_agent="test-agent/1.0",
            max_retries=0,
            transport=FakeTransport([response(429)]),
        ).get("https://example.test")

    with pytest.raises(HttpTimeoutError):
        HttpClient(
            timeout_seconds=1,
            user_agent="test-agent/1.0",
            max_retries=0,
            transport=FakeTransport([socket.timeout()]),
        ).get("https://example.test")

    with pytest.raises(HttpNetworkError):
        HttpClient(
            timeout_seconds=1,
            user_agent="test-agent/1.0",
            max_retries=0,
            transport=FakeTransport([URLError("offline")]),
        ).get("https://example.test")
