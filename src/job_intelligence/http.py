"""Small dependency-free HTTP client for local connector use."""

from __future__ import annotations

import socket
import time
from collections.abc import Mapping
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class HttpResponse(BaseModel):
    """The small response surface connectors need."""

    model_config = ConfigDict(extra="forbid")

    status_code: PositiveInt = Field(le=599)
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        """Perform one HTTP request."""


class HttpError(RuntimeError):
    """Base class for expected HTTP failures."""


class HttpTimeoutError(HttpError):
    """The request exceeded its configured timeout."""


class HttpNetworkError(HttpError):
    """The request could not reach the remote server."""


class HttpStatusError(HttpError):
    """The server returned a non-success status code."""

    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} returned for {url}")


class RateLimitedHttpError(HttpStatusError):
    """The server rejected the request due to rate limiting."""


class UrllibTransport:
    """Default transport backed by Python's standard library."""

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        with urlopen(request, timeout=timeout_seconds) as response:
            response_headers = {
                key: value for key, value in response.headers.items()
            }
            return HttpResponse(
                status_code=response.status,
                url=response.geturl(),
                headers=response_headers,
                body=response.read(),
            )


class HttpClient:
    """GET-only HTTP client with bounded retries for transient failures."""

    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        transport: HttpTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not user_agent.strip():
            raise ValueError("user_agent must not be blank")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")

        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._transport = transport or UrllibTransport()

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> HttpResponse:
        request_url = _with_query(url, params)
        headers = {"Accept": "application/json", "User-Agent": self._user_agent}

        for attempt in range(self._max_retries + 1):
            try:
                response = self._transport.request(
                    request_url,
                    headers=headers,
                    timeout_seconds=self._timeout_seconds,
                )
                if 200 <= response.status_code < 300:
                    return response
                error = _status_error(response.status_code, request_url)
            except HTTPError as exc:
                error = _status_error(exc.code, request_url)
            except (socket.timeout, TimeoutError) as exc:
                error = HttpTimeoutError(f"HTTP request timed out for {request_url}")
                error.__cause__ = exc
            except URLError as exc:
                error = HttpNetworkError(f"HTTP request failed for {request_url}")
                error.__cause__ = exc
            except HttpError:
                raise

            if not _should_retry(error, attempt, self._max_retries):
                raise error
            if self._retry_backoff_seconds:
                time.sleep(self._retry_backoff_seconds * (attempt + 1))

        raise AssertionError("HTTP retry loop ended unexpectedly")


def _with_query(url: str, params: Mapping[str, str | int] | None) -> str:
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def _status_error(status_code: int, url: str) -> HttpStatusError:
    if status_code == 429:
        return RateLimitedHttpError(status_code, url)
    return HttpStatusError(status_code, url)


def _should_retry(error: HttpError, attempt: int, max_retries: int) -> bool:
    if attempt >= max_retries:
        return False
    if isinstance(error, RateLimitedHttpError):
        return True
    if isinstance(error, HttpStatusError):
        return error.status_code in HttpClient._RETRYABLE_STATUS_CODES
    return isinstance(error, (HttpTimeoutError, HttpNetworkError))


__all__ = [
    "HttpClient",
    "HttpError",
    "HttpNetworkError",
    "HttpResponse",
    "HttpStatusError",
    "HttpTimeoutError",
    "HttpTransport",
    "RateLimitedHttpError",
]
