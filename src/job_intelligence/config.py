"""Explicit, environment-backed application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from .models import NonEmptyText


class Settings(BaseModel):
    """Runtime settings with no user- or provider-specific defaults.

    Values can be passed directly in tests or loaded from environment variables with
    :meth:`from_environment`. Keeping loading explicit avoids hidden global state.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    target_roles: list[NonEmptyText] = Field(default_factory=list)
    target_locations: list[NonEmptyText] = Field(default_factory=list)
    min_experience_years: NonNegativeFloat | None = None
    max_experience_years: NonNegativeFloat | None = None
    company_batch_size: PositiveInt | None = None
    aws_region: NonEmptyText | None = None
    job_analysis_model_id: NonEmptyText | None = None
    embedding_model_id: NonEmptyText | None = None
    resume_max_size_bytes: PositiveInt | None = None
    candidate_profile_directory: Path | None = None

    @model_validator(mode="after")
    def validate_experience_range(self) -> Settings:
        if (
            self.min_experience_years is not None
            and self.max_experience_years is not None
            and self.max_experience_years < self.min_experience_years
        ):
            raise ValueError("max_experience_years must not be below min_experience_years")
        return self

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Build settings from ``JOB_INTELLIGENCE_*`` environment variables.

        Lists are comma-separated. Missing variables remain ``None`` or an empty list,
        so deployment-specific values are never silently hardcoded here.
        """

        values = environ if environ is not None else os.environ
        prefix = "JOB_INTELLIGENCE_"

        return cls(
            target_roles=_read_list(values, prefix + "TARGET_ROLES"),
            target_locations=_read_list(values, prefix + "TARGET_LOCATIONS"),
            min_experience_years=_read_float(values, prefix + "MIN_EXPERIENCE_YEARS"),
            max_experience_years=_read_float(values, prefix + "MAX_EXPERIENCE_YEARS"),
            company_batch_size=_read_int(values, prefix + "COMPANY_BATCH_SIZE"),
            aws_region=_read_optional(values, prefix + "AWS_REGION"),
            job_analysis_model_id=_read_optional(values, prefix + "JOB_ANALYSIS_MODEL_ID"),
            embedding_model_id=_read_optional(values, prefix + "EMBEDDING_MODEL_ID"),
            resume_max_size_bytes=_read_int(values, prefix + "RESUME_MAX_SIZE_BYTES"),
            candidate_profile_directory=_read_optional(values, prefix + "PROFILE_DIRECTORY"),
        )


def _read_optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def _read_list(environ: Mapping[str, str], name: str) -> list[str]:
    value = environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_float(environ: Mapping[str, str], name: str) -> float | None:
    value = _read_optional(environ, name)
    return float(value) if value is not None else None


def _read_int(environ: Mapping[str, str], name: str) -> int | None:
    value = _read_optional(environ, name)
    return int(value) if value is not None else None
