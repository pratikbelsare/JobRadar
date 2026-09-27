"""Pure normalization, identity, and content-hashing logic for jobs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from pydantic import AnyHttpUrl, TypeAdapter

from .connectors.base import RawJobDetails
from .models import Job, Model, NonEmptyText

_HTTP_URL = TypeAdapter(AnyHttpUrl)
_WHITESPACE = re.compile(r"\s+")


class JobIdentity(Model):
    """Stable identifiers used to associate observations with one job."""

    company_id: UUID
    source_job_id: NonEmptyText
    canonical_source_url: NonEmptyText

    def matches(self, other: JobIdentity) -> bool:
        """Match on company plus source ID or canonical source URL."""

        if self.company_id != other.company_id:
            return False
        return (
            self.source_job_id == other.source_job_id
            or self.canonical_source_url == other.canonical_source_url
        )

    @property
    def source_key(self) -> str:
        return f"company:{self.company_id}|source:{self.source_job_id}"

    @property
    def url_key(self) -> str:
        return f"company:{self.company_id}|url:{self.canonical_source_url}"


class JobNormalizer:
    """Convert connector details into the canonical ``Job`` model."""

    def normalize(
        self,
        raw_job: RawJobDetails,
        *,
        observed_at: datetime | None = None,
    ) -> Job:
        observed = _normalize_datetime(observed_at or datetime.now(timezone.utc))
        job = Job(
            source_job_id=normalize_text(raw_job.source_job_id),
            company_id=raw_job.company_id,
            title=normalize_text(raw_job.title),
            location=normalize_optional_text(raw_job.location),
            description=normalize_text(raw_job.description),
            source_url=canonicalize_url(raw_job.source_url),
            first_seen_at=observed,
            last_seen_at=observed,
        )
        return job.model_copy(update={"content_hash": compute_content_hash(job)})


def normalize_text(value: str) -> str:
    """Trim and collapse whitespace without interpreting the text."""

    return _WHITESPACE.sub(" ", value).strip()


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    return normalized or None


def canonicalize_url(value: str | AnyHttpUrl) -> AnyHttpUrl:
    """Canonicalize URL casing, fragments, query ordering, and trailing slashes."""

    parsed = urlsplit(str(value).strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_pairs = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    query = urlencode(query_pairs)
    return _HTTP_URL.validate_python(urlunsplit((scheme, netloc, path, query, "")))


def job_identity(job: Job) -> JobIdentity:
    return JobIdentity(
        company_id=job.company_id,
        source_job_id=normalize_text(job.source_job_id),
        canonical_source_url=str(canonicalize_url(job.source_url)),
    )


def canonical_job_content(job: Job) -> dict[str, Any]:
    """Return the stable, meaningful subset used by the content hash."""

    content = job.model_dump(
        mode="json",
        include={
            "title",
            "location",
            "description",
            "work_mode",
            "min_experience_years",
            "max_experience_years",
            "seniority",
            "required_skills",
            "preferred_skills",
            "responsibilities",
            "education",
            "domain",
            "employment_type",
            "posted_at",
        },
    )
    for field in ("title", "location", "description", "domain"):
        value = content.get(field)
        if isinstance(value, str):
            content[field] = normalize_text(value)
    for field in ("required_skills", "preferred_skills", "responsibilities", "education"):
        values = content.get(field)
        if isinstance(values, list):
            content[field] = sorted(normalize_text(value) for value in values)
    return content


def compute_content_hash(job: Job) -> str:
    """Hash canonical meaningful content, excluding identity and observation metadata."""

    serialized = json.dumps(
        canonical_job_content(job),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def ensure_content_hash(job: Job) -> Job:
    expected_hash = compute_content_hash(job)
    if job.content_hash == expected_hash:
        return job
    return _revalidate_job(job, content_hash=expected_hash)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _revalidate_job(job: Job, **updates: Any) -> Job:
    values = job.model_dump(mode="python")
    values.update(updates)
    return Job.model_validate(values)


__all__ = [
    "JobIdentity",
    "JobNormalizer",
    "canonical_job_content",
    "canonicalize_url",
    "compute_content_hash",
    "ensure_content_hash",
    "job_identity",
    "normalize_optional_text",
    "normalize_text",
]
