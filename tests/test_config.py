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


def test_settings_have_no_user_or_provider_specific_defaults() -> None:
    settings = Settings.from_environment({})

    assert settings.target_roles == []
    assert settings.target_locations == []
    assert settings.aws_region is None
    assert settings.job_analysis_model_id is None


def test_settings_reject_inverted_experience_range() -> None:
    with pytest.raises(ValidationError):
        Settings(min_experience_years=5, max_experience_years=2)
