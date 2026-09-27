"""Replaceable job-state persistence with a local JSON implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .change_detection import StoredJobState
from .models import Job, JobVersion
from .normalization import JobIdentity, job_identity


class JobRepositoryError(RuntimeError):
    """Raised when local job state cannot be read or written."""


class JobRepository(Protocol):
    """Persistence boundary for a future PostgreSQL implementation."""

    def find_by_identity(self, identity: JobIdentity) -> StoredJobState | None:
        """Return the current state matching a stable job identity."""

    def save(self, state: StoredJobState) -> StoredJobState:
        """Save current state and append any new immutable versions."""


class JsonJobRepository:
    """Store current jobs and immutable versions as separate JSON files."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._jobs_directory = self._directory / "jobs"
        self._versions_directory = self._directory / "versions"

    def find_by_identity(self, identity: JobIdentity) -> StoredJobState | None:
        if not self._jobs_directory.exists():
            return None
        try:
            for path in self._jobs_directory.glob("*.json"):
                job = Job.model_validate_json(path.read_text(encoding="utf-8"))
                if job_identity(job).matches(identity):
                    return StoredJobState(job=job, versions=self._load_versions(job))
        except (OSError, ValidationError, ValueError) as exc:
            raise JobRepositoryError("Could not inspect local job state") from exc
        return None

    def save(self, state: StoredJobState) -> StoredJobState:
        try:
            self._jobs_directory.mkdir(parents=True, exist_ok=True)
            self._versions_directory.mkdir(parents=True, exist_ok=True)
            for version in state.versions:
                self._save_version(version)
            self._write_atomic(
                self._jobs_directory / f"{state.job.id}.json",
                state.job.model_dump_json(indent=2),
            )
        except JobRepositoryError:
            raise
        except (OSError, ValidationError, ValueError) as exc:
            raise JobRepositoryError(f"Could not save job {state.job.id}") from exc
        return state

    def _load_versions(self, job: Job) -> list[JobVersion]:
        directory = self._versions_directory / str(job.id)
        if not directory.exists():
            return []
        versions: list[JobVersion] = []
        for path in directory.glob("*.json"):
            versions.append(JobVersion.model_validate_json(path.read_text(encoding="utf-8")))
        return sorted(versions, key=lambda version: version.version_number)

    def _save_version(self, version: JobVersion) -> None:
        directory = self._versions_directory / str(version.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{version.version_number}.json"
        if path.exists():
            existing = JobVersion.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != version:
                raise JobRepositoryError(
                    f"Historical job version cannot be overwritten: {version.job_id}"
                )
            return
        self._write_atomic(path, version.model_dump_json(indent=2))

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
            temporary_path.replace(path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise JobRepositoryError(f"Could not write local job file {path}") from exc


__all__ = ["JobRepository", "JobRepositoryError", "JsonJobRepository"]
