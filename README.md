# Job Intelligence

Milestones 1–5 contain the repository foundation, typed Pydantic domain models, local candidate-profile management, PDF resume text ingestion, career-site connectors, job normalization, change detection, local development storage, and deterministic job filtering for the Job Intelligence product.

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
- `JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS`
- `JOB_INTELLIGENCE_HTTP_USER_AGENT`
- `JOB_INTELLIGENCE_HTTP_MAX_RETRIES`
- `JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS`
- `JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS`

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

For a configured Greenhouse company, the local connector flow is:

```python
from job_intelligence.connectors import ConnectorRegistry
from job_intelligence.http import HttpClient
from job_intelligence.models import Company, ConnectorType

timeout = settings.http_timeout_seconds
user_agent = settings.http_user_agent
max_retries = settings.http_max_retries
if timeout is None or user_agent is None or max_retries is None:
    raise RuntimeError("Configure the HTTP settings before creating a connector")

company = Company(
    name="Configured company",
    career_url="https://boards.greenhouse.io/board-token",
    connector_type=ConnectorType.GREENHOUSE,
)
http = HttpClient(
    timeout_seconds=timeout,
    user_agent=user_agent,
    max_retries=max_retries,
)
connector = ConnectorRegistry(http).for_company(company)
listing = connector.list_jobs(company)
if listing.jobs:
    details = connector.get_job(listing.jobs[0])
```

The connector returns typed `JobReference`, `RawJobDetails`, and status-bearing results. Tests use mocked HTTP transports and do not require internet access.

Normalized job ingestion uses the connector's `RawJobDetails` output and keeps volatile observation timestamps out of the content hash:

```python
from job_intelligence.ingestion import JobIngestionService
from job_intelligence.job_repository import JsonJobRepository
from job_intelligence.snapshots import FileRawSnapshotStore

service = JobIngestionService(
    JsonJobRepository("data/jobs"),
    FileRawSnapshotStore("data/job-snapshots"),
)
result = service.ingest(details.job)
print(result.classification, result.job.content_hash)
```

The first observation is `NEW`, a repeated observation with the same normalized content is `UNCHANGED`, and a meaningful content change is `CHANGED` with a new immutable `JobVersion`. No matching, ranking, or AI processing is performed.

Deterministic filtering is configured separately from the filter implementation. Role and location aliases are supplied by the application, while experience thresholds can come from `Settings`:

```python
from job_intelligence.filtering import DeterministicJobFilter, FilteringConfig

filter_config = FilteringConfig.from_settings(
    settings,
    role_aliases={"software engineer": ["swe"]},
    location_aliases={"bengaluru": ["bangalore"]},
)
result = DeterministicJobFilter(filter_config).evaluate(job, profile)
print(result.passed, result.outcome, result.reasons)
```

Filtering runs role → location → experience → constraints and stops on the first hard rejection. It performs no semantic or AI analysis.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```
