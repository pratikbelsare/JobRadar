"""Explicit, environment-backed application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from .models import NonEmptyText


class LLMProvider(str, Enum):
    BEDROCK = "bedrock"
    BEDROCK_MANTLE = "bedrock_mantle"


class EmbeddingProvider(str, Enum):
    BEDROCK = "bedrock"
    HUGGINGFACE_LOCAL = "huggingface_local"


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
    llm_provider: LLMProvider = LLMProvider.BEDROCK
    embedding_provider: EmbeddingProvider = EmbeddingProvider.BEDROCK
    job_analysis_model_id: NonEmptyText | None = None
    embedding_model_id: NonEmptyText | None = None
    mantle_base_url: NonEmptyText | None = None
    mantle_model_id: NonEmptyText = "openai.gpt-oss-20b"
    mantle_max_tokens: PositiveInt = 4096
    local_embedding_model_id: NonEmptyText = "BAAI/bge-small-en-v1.5"
    local_embedding_device: NonEmptyText = "cpu"
    resume_max_size_bytes: PositiveInt | None = None
    candidate_profile_directory: Path | None = None
    application_data_directory: Path | None = None
    candidate_profile_id: UUID | None = None
    candidate_profile_key: NonEmptyText | None = None
    s3_bucket_name: NonEmptyText | None = None
    s3_raw_prefix: NonEmptyText | None = None
    dynamodb_table_name: NonEmptyText | None = None
    company_scan_queue_url: NonEmptyText | None = None
    job_analysis_queue_url: NonEmptyText | None = None
    scheduler_expression: NonEmptyText | None = None
    http_timeout_seconds: PositiveFloat | None = None
    http_user_agent: NonEmptyText | None = None
    http_max_retries: NonNegativeInt | None = None
    filter_acceptable_experience_gap_years: NonNegativeFloat | None = None
    filter_max_stretch_experience_gap_years: NonNegativeFloat | None = None
    bedrock_timeout_seconds: PositiveFloat | None = None
    ai_max_attempts: PositiveInt | None = None
    matching_role_weight: NonNegativeFloat | None = None
    matching_skill_weight: NonNegativeFloat | None = None
    matching_experience_weight: NonNegativeFloat | None = None
    matching_responsibility_weight: NonNegativeFloat | None = None
    matching_domain_weight: NonNegativeFloat | None = None
    matching_location_weight: NonNegativeFloat | None = None
    matching_preference_weight: NonNegativeFloat | None = None
    matching_experience_gap_scale_years: PositiveFloat | None = None
    matching_required_skill_weight: NonNegativeFloat | None = None
    matching_preferred_skill_weight: NonNegativeFloat | None = None

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
            llm_provider=LLMProvider(
                _read_optional(values, prefix + "LLM_PROVIDER") or LLMProvider.BEDROCK
            ),
            embedding_provider=EmbeddingProvider(
                _read_optional(values, prefix + "EMBEDDING_PROVIDER")
                or EmbeddingProvider.BEDROCK
            ),
            job_analysis_model_id=_read_optional(values, prefix + "JOB_ANALYSIS_MODEL_ID"),
            embedding_model_id=_read_optional(values, prefix + "EMBEDDING_MODEL_ID"),
            mantle_base_url=_read_optional(values, prefix + "MANTLE_BASE_URL"),
            mantle_model_id=(
                _read_optional(values, prefix + "MANTLE_MODEL_ID")
                or "openai.gpt-oss-20b"
            ),
            mantle_max_tokens=(
                _read_int(values, prefix + "MANTLE_MAX_TOKENS") or 4096
            ),
            local_embedding_model_id=(
                _read_optional(values, prefix + "LOCAL_EMBEDDING_MODEL_ID")
                or "BAAI/bge-small-en-v1.5"
            ),
            local_embedding_device=(
                _read_optional(values, prefix + "LOCAL_EMBEDDING_DEVICE") or "cpu"
            ),
            resume_max_size_bytes=_read_int(values, prefix + "RESUME_MAX_SIZE_BYTES"),
            candidate_profile_directory=_read_path(values, prefix + "PROFILE_DIRECTORY"),
            application_data_directory=_read_path(values, prefix + "DATA_DIRECTORY"),
            candidate_profile_id=_read_uuid(values, prefix + "CANDIDATE_PROFILE_ID"),
            candidate_profile_key=_read_optional(values, prefix + "CANDIDATE_PROFILE_KEY"),
            s3_bucket_name=_read_optional(values, prefix + "S3_BUCKET_NAME"),
            s3_raw_prefix=_read_optional(values, prefix + "S3_RAW_PREFIX"),
            dynamodb_table_name=_read_optional(values, prefix + "DYNAMODB_TABLE_NAME"),
            company_scan_queue_url=_read_optional(values, prefix + "COMPANY_SCAN_QUEUE_URL"),
            job_analysis_queue_url=_read_optional(values, prefix + "JOB_ANALYSIS_QUEUE_URL"),
            scheduler_expression=_read_optional(values, prefix + "SCHEDULER_EXPRESSION"),
            http_timeout_seconds=_read_float(values, prefix + "HTTP_TIMEOUT_SECONDS"),
            http_user_agent=_read_optional(values, prefix + "HTTP_USER_AGENT"),
            http_max_retries=_read_int(values, prefix + "HTTP_MAX_RETRIES"),
            filter_acceptable_experience_gap_years=_read_float(
                values,
                prefix + "FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS",
            ),
            filter_max_stretch_experience_gap_years=_read_float(
                values,
                prefix + "FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS",
            ),
            bedrock_timeout_seconds=_read_float(
                values,
                prefix + "BEDROCK_TIMEOUT_SECONDS",
            ),
            ai_max_attempts=_read_int(values, prefix + "AI_MAX_ATTEMPTS"),
            matching_role_weight=_read_float(values, prefix + "MATCHING_ROLE_WEIGHT"),
            matching_skill_weight=_read_float(values, prefix + "MATCHING_SKILL_WEIGHT"),
            matching_experience_weight=_read_float(
                values,
                prefix + "MATCHING_EXPERIENCE_WEIGHT",
            ),
            matching_responsibility_weight=_read_float(
                values,
                prefix + "MATCHING_RESPONSIBILITY_WEIGHT",
            ),
            matching_domain_weight=_read_float(values, prefix + "MATCHING_DOMAIN_WEIGHT"),
            matching_location_weight=_read_float(
                values,
                prefix + "MATCHING_LOCATION_WEIGHT",
            ),
            matching_preference_weight=_read_float(
                values,
                prefix + "MATCHING_PREFERENCE_WEIGHT",
            ),
            matching_experience_gap_scale_years=_read_float(
                values,
                prefix + "MATCHING_EXPERIENCE_GAP_SCALE_YEARS",
            ),
            matching_required_skill_weight=_read_float(
                values,
                prefix + "MATCHING_REQUIRED_SKILL_WEIGHT",
            ),
            matching_preferred_skill_weight=_read_float(
                values,
                prefix + "MATCHING_PREFERRED_SKILL_WEIGHT",
            ),
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


def _read_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = _read_optional(environ, name)
    return Path(value) if value is not None else None


def _read_uuid(environ: Mapping[str, str], name: str) -> UUID | None:
    value = _read_optional(environ, name)
    return UUID(value) if value is not None else None
