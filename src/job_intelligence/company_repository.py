"""Replaceable company catalog persistence for the local API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Protocol
from uuid import UUID

from pydantic import ValidationError

from .models import Company


class CompanyRepositoryError(RuntimeError):
    """Raised when the local company catalog cannot be read or written."""


class CompanyRepository(Protocol):
    def save(self, company: Company) -> Company:
        """Create or replace one company."""

    def list(self) -> list[Company]:
        """Return configured companies."""

    def get(self, company_id: UUID) -> Company | None:
        """Return one company by id."""


class JsonCompanyRepository:
    """Store a small company catalog in one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def list(self) -> list[Company]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("company file must contain a JSON list")
            return [Company.model_validate(item) for item in raw]
        except (OSError, TypeError, ValueError, ValidationError) as exc:
            raise CompanyRepositoryError(f"Could not load companies: {self._path}") from exc

    def save(self, company: Company) -> Company:
        companies = [item for item in self.list() if item.id != company.id]
        companies.append(company)
        self.save_all(companies)
        return company

    def get(self, company_id: UUID) -> Company | None:
        return next((company for company in self.list() if company.id == company_id), None)

    def save_all(self, companies: List[Company]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    [company.model_dump(mode="json") for company in companies],
                    temporary,
                    indent=2,
                )
            temporary_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise CompanyRepositoryError(f"Could not save companies: {self._path}") from exc


__all__ = ["CompanyRepository", "CompanyRepositoryError", "JsonCompanyRepository"]
