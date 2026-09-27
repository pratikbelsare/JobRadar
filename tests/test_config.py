from pathlib import Path

import pytest
from pydantic import ValidationError

from job_intelligence.config import Settings


def test_settings_load_configurable_values_from_environment_mapping() -> None:
    settings = Settings.from_environment(
        {
            "JOB_INTELLIGENCE_TARGET_ROLES": "ML Engineer, Data Scientist",
            "JOB_INTELLIGENCE_TARGET_LOCATIONS": "Bengaluru, Remote",
            "JOB_INTELLIGENCE_MIN_EXPERIENCE_YEARS": "1.5",
            "JOB_INTELLIGENCE_MAX_EXPERIENCE_YEARS": "5",
            "JOB_INTELLIGENCE_COMPANY_BATCH_SIZE": "12",
            "JOB_INTELLIGENCE_AWS_REGION": "ap-south-1",
            "JOB_INTELLIGENCE_JOB_ANALYSIS_MODEL_ID": "provider/job-model",
            "JOB_INTELLIGENCE_EMBEDDING_MODEL_ID": "provider/embedding-model",
            "JOB_INTELLIGENCE_RESUME_MAX_SIZE_BYTES": "5000000",
            "JOB_INTELLIGENCE_PROFILE_DIRECTORY": "data/profiles",
            "JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS": "15",
            "JOB_INTELLIGENCE_HTTP_USER_AGENT": "job-intelligence-test/1.0",
            "JOB_INTELLIGENCE_HTTP_MAX_RETRIES": "2",
        }
    )

    assert settings.target_roles == ["ML Engineer", "Data Scientist"]
    assert settings.target_locations == ["Bengaluru", "Remote"]
    assert settings.min_experience_years == 1.5
    assert settings.max_experience_years == 5
    assert settings.company_batch_size == 12
    assert settings.aws_region == "ap-south-1"
    assert settings.job_analysis_model_id == "provider/job-model"
    assert settings.embedding_model_id == "provider/embedding-model"
    assert settings.resume_max_size_bytes == 5000000
    assert settings.candidate_profile_directory == Path("data/profiles")
    assert settings.http_timeout_seconds == 15
    assert settings.http_user_agent == "job-intelligence-test/1.0"
    assert settings.http_max_retries == 2


def test_settings_have_no_user_or_provider_specific_defaults() -> None:
    settings = Settings.from_environment({})

    assert settings.target_roles == []
    assert settings.target_locations == []
    assert settings.aws_region is None
    assert settings.job_analysis_model_id is None
    assert settings.resume_max_size_bytes is None
    assert settings.candidate_profile_directory is None
    assert settings.http_timeout_seconds is None
    assert settings.http_user_agent is None


def test_settings_reject_inverted_experience_range() -> None:
    with pytest.raises(ValidationError):
        Settings(min_experience_years=5, max_experience_years=2)
