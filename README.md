# Job Intelligence

Milestones 1 and 2 contain the repository foundation, typed Pydantic domain models, local candidate-profile management, PDF resume text ingestion, and environment-backed configuration for the Job Intelligence product.

## Local setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Configuration is loaded explicitly with `Settings.from_environment()`. Supported variables use the `JOB_INTELLIGENCE_` prefix, including:

- `JOB_INTELLIGENCE_TARGET_ROLES` — comma-separated roles
- `JOB_INTELLIGENCE_TARGET_LOCATIONS` — comma-separated locations
- `JOB_INTELLIGENCE_MIN_EXPERIENCE_YEARS`
- `JOB_INTELLIGENCE_MAX_EXPERIENCE_YEARS`
- `JOB_INTELLIGENCE_COMPANY_BATCH_SIZE`
- `JOB_INTELLIGENCE_AWS_REGION`
- `JOB_INTELLIGENCE_JOB_ANALYSIS_MODEL_ID`
- `JOB_INTELLIGENCE_EMBEDDING_MODEL_ID`
- `JOB_INTELLIGENCE_RESUME_MAX_SIZE_BYTES`
- `JOB_INTELLIGENCE_PROFILE_DIRECTORY`

Resume ingestion is intentionally limited to local PDF validation and text extraction. It does not infer profile fields or call an LLM. Profile JSON files are stored under the configured profile directory; the directory is created when a profile is saved.

Example usage from Python:

```python
from pathlib import Path

from job_intelligence.config import Settings
from job_intelligence.profile_service import CandidateProfileService
from job_intelligence.profile_repository import JsonCandidateProfileRepository
from job_intelligence.resume import ResumeIngestor

settings = Settings.from_environment()
profile_directory = settings.candidate_profile_directory
max_resume_size = settings.resume_max_size_bytes
if profile_directory is None or max_resume_size is None:
    raise RuntimeError("Configure JOB_INTELLIGENCE_PROFILE_DIRECTORY and JOB_INTELLIGENCE_RESUME_MAX_SIZE_BYTES")
repository = JsonCandidateProfileRepository(profile_directory)
profiles = CandidateProfileService(repository)
resume = ResumeIngestor(max_size_bytes=max_resume_size)
document = resume.ingest(Path("resume.pdf"))
```

The resume document is returned separately from `CandidateProfile`, so ingesting a resume never overwrites manual profile values.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```
