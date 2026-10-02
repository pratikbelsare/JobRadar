# Job Intelligence

Milestones 1–11 include the repository foundation, local ingestion and development storage,
deterministic filtering, isolated AI/matching/evaluation services, a local API/UI, and the
AWS processing adapters. Milestone 12 adds deployment-readiness configuration and bootstrap
tools without deploying AWS resources.

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
- `JOB_INTELLIGENCE_DATA_DIRECTORY`
- `JOB_INTELLIGENCE_CANDIDATE_PROFILE_ID`
- `JOB_INTELLIGENCE_CANDIDATE_PROFILE_KEY`
- `JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS`
- `JOB_INTELLIGENCE_HTTP_USER_AGENT`
- `JOB_INTELLIGENCE_HTTP_MAX_RETRIES`
- `JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS`
- `JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS`
- `JOB_INTELLIGENCE_BEDROCK_TIMEOUT_SECONDS`
- `JOB_INTELLIGENCE_AI_MAX_ATTEMPTS`
- `JOB_INTELLIGENCE_MATCHING_*_WEIGHT`
- `JOB_INTELLIGENCE_MATCHING_EXPERIENCE_GAP_SCALE_YEARS`

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

## Local API and UI

Install the API dependencies with the development extras, then start the local server:

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn job_intelligence.api:create_app --factory --reload
```

The API uses local JSON data under `data/` by default. Set
`JOB_INTELLIGENCE_DATA_DIRECTORY` to change that location. A profile is created on first
startup when no profile exists; set `JOB_INTELLIGENCE_CANDIDATE_PROFILE_ID` to select a
specific existing profile. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

To run the optional Streamlit interface in a second terminal:

```powershell
python -m pip install -e ".[dev,ui]"
streamlit run streamlit_app.py
```

Set `JOB_INTELLIGENCE_API_URL` if the API is not at `http://127.0.0.1:8000`. The UI only
consumes the API; filtering, matching, ranking, feedback, and explanation behavior remain
in the existing application services.

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

Bedrock extraction and embeddings are isolated behind `AIProvider`/`TextEmbeddingProvider`. Configure the AWS region and model IDs before constructing `BedrockProvider`; inject a mock client in unit tests. Matching uses configurable component weights and preserves component evidence in `JobMatch` rather than asking an LLM to create the final score.

For local development while Bedrock Runtime access is unavailable, install the optional
CPU embedding dependency and select the providers explicitly:

```powershell
python -m pip install -e ".[dev,local-ai]"
$env:JOB_INTELLIGENCE_AWS_REGION = "ap-south-1"
$env:JOB_INTELLIGENCE_LLM_PROVIDER = "bedrock_mantle"
$env:JOB_INTELLIGENCE_EMBEDDING_PROVIDER = "huggingface_local"
$env:JOB_INTELLIGENCE_MANTLE_MODEL_ID = "openai.gpt-oss-20b"
$env:JOB_INTELLIGENCE_MANTLE_MAX_TOKENS = "4096"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_DEVICE = "cpu"
```

`bedrock_mantle` uses the AWS-authenticated OpenAI-compatible Mantle endpoint derived from
the configured region (or `JOB_INTELLIGENCE_MANTLE_BASE_URL` when supplied). The local
embedding provider downloads and caches `sentence-transformers` model files locally and
does not call hosted inference. The default providers remain Bedrock Runtime and Titan.

To run the checked-in local smoke harness against one already-persisted job, configure
the DynamoDB table/profile/job title read-only and run:

```powershell
$env:AWS_PROFILE = "jobradar-dev"
$env:JOB_INTELLIGENCE_AWS_REGION = "ap-south-1"
$env:JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME = "<deployed-data-table>"
$env:JOB_INTELLIGENCE_CANDIDATE_PROFILE_KEY = "pratik"
$env:JOB_INTELLIGENCE_SMOKE_JOB_TITLE = "Product Manager, AI"
$env:JOB_INTELLIGENCE_LLM_PROVIDER = "bedrock_mantle"
$env:JOB_INTELLIGENCE_EMBEDDING_PROVIDER = "huggingface_local"
$env:JOB_INTELLIGENCE_MANTLE_MODEL_ID = "openai.gpt-oss-20b"
$env:JOB_INTELLIGENCE_MANTLE_MAX_TOKENS = "8192"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_DEVICE = "cpu"
$env:NO_PROXY = "bedrock-mantle.ap-south-1.api.aws"
$env:PYTHONPATH = "src"
python scripts/local_fallback_smoke.py
```

The harness reads DynamoDB only, does not read or redrive SQS, and writes match and
explanation artifacts only to a temporary local directory.

To evaluate a deterministic sample of persisted YipitData jobs with the same providers:

```powershell
python scripts/evaluate_persisted_jobs.py --sample-size 18 --output-dir evaluation/results
```

This writes ignored JSON, CSV, and Markdown reports. Use `--skip-explanations` when
ranking-only evaluation is desired; explanation failures are otherwise recorded per job
without discarding its ranking result.

To compare the same persisted 18-job sample before and after a local candidate-profile
revision, use the read-only profile-impact harness:

```powershell
python scripts/evaluate_profile_impact.py `
  --profile evaluation/profile/pratik-enriched.json `
  --baseline-profile examples/candidate_profile.example.json `
  --reference evaluation/results/persisted-job-ranking.json `
  --output-dir evaluation/results/profile-impact
```

It reads existing jobs from DynamoDB, performs no AWS writes, and writes ignored
before/after ranking and editable provisional-label artifacts locally. The enriched
profile directory is ignored because it contains local candidate data.

To evaluate the existing deterministic filter before ranking the retained jobs:

```powershell
$env:JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS = "1"
$env:JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS = "2"
python scripts/evaluate_filter_impact.py `
  --profile evaluation/profile/pratik-enriched.json `
  --reference evaluation/results/persisted-job-ranking.json `
  --output-dir evaluation/results/filter-impact
```

This also reads existing DynamoDB jobs only and writes filter outcomes, retained
ranking, and provisional metric comparisons to local ignored artifacts.

Evaluation labels use JSONL and can be compared with the keyword, embedding, and hybrid approaches through `evaluate_approaches`. Precision@K treats labels `>= 2` as relevant by default; the threshold is configurable. Reports are plain JSON and require no database or dashboard.

Explanations are generated on demand through `ExplanationService` from the structured `JobMatch` evidence. `JsonUserFeedbackRepository` stores editable relevance feedback locally, and `UserFeedbackService.export_evaluation_examples()` converts it back into the evaluation format without changing ranking weights.

## AWS deployment preparation

Milestone 11 adds AWS adapters and SAM infrastructure without deploying resources. The
pipeline is EventBridge Scheduler → coordinator Lambda → company-scan SQS → scanner Lambda
→ private S3/DynamoDB → deterministic filtering → job-analysis SQS → analysis Lambda →
Bedrock and persisted matching. Local JSON repositories, the local filesystem snapshot store,
mocked AI providers, and the FastAPI/Streamlit workflow remain available.

Prerequisites for later validation/deployment are AWS SAM CLI, an AWS region, Bedrock model
IDs, and credentials supplied through the standard AWS credential chain. Validate/build
locally with:

```powershell
sam validate --template-file infrastructure/template.yaml
sam build --template-file infrastructure/template.yaml
```

Review `infrastructure/README.md` and `infrastructure/samconfig.toml.example` for the
DynamoDB key/index design, cost controls, configurable schedule, IAM boundaries, and later
deployment commands. `sam deploy` was intentionally not run.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```
