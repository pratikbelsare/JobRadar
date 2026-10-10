"""Local resume validation and PDF text extraction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from .models import NonEmptyText


class ResumeIngestionError(RuntimeError):
    """Raised when a resume cannot be validated or read."""


class ResumeFileType(str, Enum):
    PDF = "pdf"


class ResumeMetadata(BaseModel):
    """Validated metadata describing one locally ingested resume file."""

    model_config = ConfigDict(extra="forbid")

    file_name: NonEmptyText
    file_path: Path
    file_type: ResumeFileType
    size_bytes: PositiveInt
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: NonNegativeInt
    extracted_at: datetime


class ResumeDocument(BaseModel):
    """Resume metadata and extracted text, kept separate from candidate fields."""

    model_config = ConfigDict(extra="forbid")

    metadata: ResumeMetadata
    raw_text: str


class PdfPage(Protocol):
    def extract_text(self) -> str | None:
        """Return text extracted from a page, if any."""


class PdfReader(Protocol):
    pages: Sequence[PdfPage]


PdfReaderFactory = Callable[[str], PdfReader]


class PdfTextExtractor:
    """Extract page text using an injectable PDF reader implementation."""

    def __init__(self, reader_factory: PdfReaderFactory | None = None) -> None:
        self._reader_factory = reader_factory or _create_pypdf_reader

    def extract(self, path: Path) -> tuple[str, int]:
        try:
            reader = self._reader_factory(str(path))
            pages = list(reader.pages)
            text = "\n".join(
                page_text
                for page in pages
                if (page_text := page.extract_text())
            )
        except ResumeIngestionError:
            raise
        except Exception as exc:
            raise ResumeIngestionError(f"Could not extract PDF text from {path}") from exc

        return text, len(pages)


class ResumeIngestor:
    """Validate a local PDF and return its metadata plus raw extracted text."""

    def __init__(
        self,
        max_size_bytes: int,
        extractor: PdfTextExtractor | None = None,
    ) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        self._max_size_bytes = max_size_bytes
        self._extractor = extractor or PdfTextExtractor()

    def ingest(self, path: str | Path) -> ResumeDocument:
        resume_path = Path(path)
        size_bytes = self._validate_file(resume_path)
        raw_text, page_count = self._extractor.extract(resume_path)

        metadata = ResumeMetadata(
            file_name=resume_path.name,
            file_path=resume_path,
            file_type=ResumeFileType.PDF,
            size_bytes=size_bytes,
            sha256=_sha256(resume_path),
            page_count=page_count,
            extracted_at=datetime.now(timezone.utc),
        )
        return ResumeDocument(metadata=metadata, raw_text=raw_text)

    def _validate_file(self, path: Path) -> int:
        if not path.exists():
            raise ResumeIngestionError(f"Resume file does not exist: {path}")
        if not path.is_file():
            raise ResumeIngestionError(f"Resume path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ResumeIngestionError("Only PDF resume files are supported")

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise ResumeIngestionError(f"Could not inspect resume file: {path}") from exc
        if size_bytes <= 0:
            raise ResumeIngestionError("Resume file must not be empty")
        if size_bytes > self._max_size_bytes:
            raise ResumeIngestionError(
                f"Resume file exceeds the configured size limit of {self._max_size_bytes} bytes"
            )
        return size_bytes


def _create_pypdf_reader(path: str) -> PdfReader:
    try:
        from pypdf import PdfReader as PypdfReader
    except ImportError as exc:
        raise ResumeIngestionError(
            "PDF extraction requires the pypdf dependency; install the project dependencies"
        ) from exc
    return PypdfReader(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResumeIngestionError(f"Could not read resume file: {path}") from exc
    return digest.hexdigest()


__all__ = [
    "PdfReader",
    "PdfTextExtractor",
    "ResumeDocument",
    "ResumeFileType",
    "ResumeIngestionError",
    "ResumeIngestor",
    "ResumeMetadata",
]
