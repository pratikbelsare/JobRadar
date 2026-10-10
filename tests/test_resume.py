from hashlib import sha256

import pytest

from job_intelligence.resume import (
    PdfTextExtractor,
    ResumeIngestionError,
    ResumeIngestor,
)


class FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakeReader:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages


def test_pdf_text_extraction_returns_metadata_and_raw_text(tmp_path) -> None:
    resume_path = tmp_path / "resume.PDF"
    contents = b"local PDF fixture bytes"
    resume_path.write_bytes(contents)

    def reader_factory(path: str) -> FakeReader:
        assert path == str(resume_path)
        return FakeReader([FakePage("First page"), FakePage(None), FakePage("Second page")])

    ingestor = ResumeIngestor(
        max_size_bytes=1024,
        extractor=PdfTextExtractor(reader_factory=reader_factory),
    )
    document = ingestor.ingest(resume_path)

    assert document.raw_text == "First page\nSecond page"
    assert document.metadata.file_name == "resume.PDF"
    assert document.metadata.page_count == 3
    assert document.metadata.size_bytes == len(contents)
    assert document.metadata.sha256 == sha256(contents).hexdigest()


@pytest.mark.parametrize(
    ("path_kind", "max_size_bytes", "message"),
    [
        ("missing", 1024, "does not exist"),
        ("wrong_type", 1024, "Only PDF"),
        ("empty", 1024, "must not be empty"),
        ("too_large", 1, "exceeds"),
    ],
)
def test_resume_validation_failures(tmp_path, path_kind, max_size_bytes, message) -> None:
    if path_kind == "missing":
        path = tmp_path / "missing.pdf"
    elif path_kind == "wrong_type":
        path = tmp_path / "resume.txt"
        path.write_text("text", encoding="utf-8")
    elif path_kind == "empty":
        path = tmp_path / "resume.pdf"
        path.write_bytes(b"")
    else:
        path = tmp_path / "resume.pdf"
        path.write_bytes(b"too large")

    ingestor = ResumeIngestor(
        max_size_bytes=max_size_bytes,
        extractor=PdfTextExtractor(reader_factory=lambda _: FakeReader([])),
    )
    with pytest.raises(ResumeIngestionError, match=message):
        ingestor.ingest(path)


def test_resume_extraction_failure_is_wrapped(tmp_path) -> None:
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"fixture")

    def failing_reader(_: str) -> FakeReader:
        raise ValueError("invalid PDF")

    ingestor = ResumeIngestor(
        max_size_bytes=1024,
        extractor=PdfTextExtractor(reader_factory=failing_reader),
    )
    with pytest.raises(ResumeIngestionError, match="Could not extract PDF text"):
        ingestor.ingest(path)
