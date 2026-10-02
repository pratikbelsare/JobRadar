"""Replaceable scan-run persistence for local API read access."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Protocol
from uuid import UUID

from pydantic import ValidationError

from .models import ScanRun


class ScanRunRepositoryError(RuntimeError):
    """Raised when local scan-run data cannot be read or written."""


class ScanRunRepository(Protocol):
    def save(self, run: ScanRun) -> ScanRun:
        """Create or replace one scan run."""

    def list(self) -> list[ScanRun]:
        """Return scan runs ordered by their persisted order."""

    def get(self, run_id: UUID) -> ScanRun | None:
        """Return one scan run by id."""


class JsonScanRunRepository:
    """Store scan-run summaries in one local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def list(self) -> list[ScanRun]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("run file must contain a JSON list")
            return [ScanRun.model_validate(item) for item in raw]
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            raise ScanRunRepositoryError(f"Could not load scan runs: {self._path}") from exc

    def save(self, run: ScanRun) -> ScanRun:
        runs = [item for item in self.list() if item.id != run.id]
        runs.append(run)
        self.save_all(runs)
        return run

    def get(self, run_id: UUID) -> ScanRun | None:
        return next((run for run in self.list() if run.id == run_id), None)

    def save_all(self, runs: List[ScanRun]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump([run.model_dump(mode="json") for run in runs], temporary, indent=2)
            temporary_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ScanRunRepositoryError(f"Could not save scan runs: {self._path}") from exc


__all__ = ["JsonScanRunRepository", "ScanRunRepository", "ScanRunRepositoryError"]
