# Job Intelligence

Milestones 1–11 include the repository foundation, local ingestion and development storage,
deterministic filtering, isolated AI/matching/evaluation services, a local API/UI, and the
AWS processing adapters. Milestone 12 adds deployment-readiness configuration and bootstrap
tools without deploying AWS resources.

The current frozen filtering and ranking behavior is identified as **JobRadar matching
v0.1**. Known taxonomy and ranking improvements are tracked in `BACKLOG.md`; integrated
product validation does not change those rules.

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
- `JOB_INTELLIGENCE_CHAT_PROVIDER`
- `JOB_INTELLIGENCE_CHAT_MODEL`
- `JOB_INTELLIGENCE_CHAT_BASE_URL`
- `JOB_INTELLIGENCE_CHAT_API_KEY`
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
- `JOB_INTELLIGENCE_ROLE_FAMILY_POLICY_JSON` or `JOB_INTELLIGENCE_ROLE_FAMILY_POLICY_FILE`
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

## Local integrated Moloco validation

Milestone 13 composes the production connector/domain services in a local company-level
pipeline. It writes canonical jobs, immutable versions, analysis cache entries, matches,
explanations, workflow state, and run summaries under one configured local data directory.
It does not use DynamoDB, S3, SQS, or deployed Lambda functions.

Install the local AI and UI extras, configure the temporary providers, and run exactly two
Moloco scans:

```powershell
python -m pip install -e ".[dev,local-ai,ui]"
$env:AWS_PROFILE = "jobradar-dev"
$env:JOB_INTELLIGENCE_AWS_REGION = "ap-south-1"
$env:JOB_INTELLIGENCE_LLM_PROVIDER = "bedrock_mantle"
$env:JOB_INTELLIGENCE_MANTLE_MODEL_ID = "openai.gpt-oss-20b"
$env:JOB_INTELLIGENCE_AI_MAX_ATTEMPTS = "3"
$env:JOB_INTELLIGENCE_EMBEDDING_PROVIDER = "huggingface_local"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_DEVICE = "cpu"
$env:JOB_INTELLIGENCE_HTTP_TIMEOUT_SECONDS = "20"
$env:JOB_INTELLIGENCE_HTTP_USER_AGENT = "JobRadar/0.1 local-validation"
$env:JOB_INTELLIGENCE_HTTP_MAX_RETRIES = "1"
$env:JOB_INTELLIGENCE_FILTER_ACCEPTABLE_EXPERIENCE_GAP_YEARS = "1"
$env:JOB_INTELLIGENCE_FILTER_MAX_STRETCH_EXPERIENCE_GAP_YEARS = "2"
$env:JOB_INTELLIGENCE_DATA_DIRECTORY = "data/moloco-e2e-v1"
python scripts/run_moloco_e2e.py
```

The runner reads the frozen public `candidate_001` profile without changing it and writes the
validation report to `evaluation/results/moloco-e2e-v1/`. Run 2 should report unchanged
jobs, analysis/match/explanation reuse, and zero duplicate canonical jobs.

To inspect the same local state through the product:

```powershell
python -m uvicorn job_intelligence.api:create_app --factory --reload
streamlit run streamlit_app.py
```

The API and UI present career relevance separately from eligibility. For example, a role
can be highly relevant while still showing a severe experience stretch.

### Multi-company local product validation

Milestone 14 reuses the same local pipeline and repositories for a configurable company
catalog. The default catalog contains Moloco, Snorkel AI, PhysicsX, Garner Health, and
Dialpad, all through the existing Greenhouse connector. With the provider environment from
the previous section still configured, run exactly two passes with:

```powershell
$env:JOB_INTELLIGENCE_DATA_DIRECTORY = "data/multi-company-e2e-v1"
python scripts/run_multi_company_e2e.py
```

The generated product artifacts are written under
`evaluation/results/multi-company-e2e-v1/`, including the aggregate summaries and an
unlabeled `product-review.csv`. To inspect that same combined state:

```powershell
$env:JOB_INTELLIGENCE_DATA_DIRECTORY = "data/multi-company-e2e-v1"
python -m uvicorn job_intelligence.api:create_app --factory --reload
streamlit run streamlit_app.py
```

The Streamlit dashboard separates career relevance from eligibility, summarizes workflow
counts, and provides Recommended, All, Shortlisted, Applied, and Dismissed views. The job
list supports company, role-family, eligibility, workflow-status, location, and text
filters plus relevance/date/eligibility/company/title sorting. Job details keep evidence,
gaps, source links, feedback, and workflow actions visible while placing diagnostic
component scores under **Advanced match details**. Matching and filtering remain v0.1;
this workflow is product validation rather than ranking evaluation.

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

Deterministic filtering is configured separately from the filter implementation. The default
structured role-family policy contains configurable CORE, ADJACENT, and EXCLUDE families for
the initial AI/ML/Data profile. Replace it with a JSON policy through
`JOB_INTELLIGENCE_ROLE_FAMILY_POLICY_JSON` or `JOB_INTELLIGENCE_ROLE_FAMILY_POLICY_FILE` for
another profession. The policy supports canonical names, exact titles, aliases, include and
exclude patterns, eligible target/adjacent family IDs, and an explicit unknown-title action.
Role-family classification is independent of experience fit; the filter exposes aligned,
stretch, severe-gap, or unknown experience status in the rule details.
Classification first honors an original-title match, then deterministically removes
structural seniority, level, annotation, and specialization text and retries the same
configured taxonomy against the resulting base title. It does not promote titles merely
because they contain words such as `AI`, `ML`, or `Data`.

```json
{
  "families": [
    {
      "id": "backend_engineering",
      "canonical_name": "Backend Engineering",
      "class": "core",
      "exact_titles": ["Backend Engineer"],
      "aliases": ["API Engineer"]
    }
  ],
  "target_family_ids": ["backend_engineering"],
  "unknown_role_action": "reject"
}
```

Role and location aliases can still be supplied by the application, while experience
thresholds can come from `Settings`:

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

Filtering runs role → location → experience → constraints and stops on the first hard
rejection. Preferred locations and work modes are soft: a mismatch is retained as an
explainable preference mismatch. Explicit excluded locations, remote-only requirements,
excluded employment types, and excluded terms remain hard constraints. Filtering performs
no semantic or AI analysis.

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

For normal local use without Bedrock or Mantle, select any service exposing an
OpenAI-compatible Chat Completions endpoint while keeping the existing local BGE
embeddings:

```powershell
$env:JOB_INTELLIGENCE_CHAT_PROVIDER = "openai_compatible"
$env:JOB_INTELLIGENCE_CHAT_MODEL = "<provider-model-name>"
$env:JOB_INTELLIGENCE_CHAT_BASE_URL = "https://<provider-host>/v1"
$env:JOB_INTELLIGENCE_CHAT_API_KEY = "<secret>"
$env:JOB_INTELLIGENCE_CHAT_MAX_TOKENS = "4096"
$env:JOB_INTELLIGENCE_CHAT_TIMEOUT_SECONDS = "60"
$env:JOB_INTELLIGENCE_EMBEDDING_PROVIDER = "huggingface_local"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
$env:JOB_INTELLIGENCE_LOCAL_EMBEDDING_DEVICE = "cpu"
```

`JOB_INTELLIGENCE_CHAT_PROVIDER` is the preferred local setting;
`JOB_INTELLIGENCE_LLM_PROVIDER` remains supported for existing Bedrock and Mantle
configuration. API keys are read only from configuration and are masked by the settings
model. Copy `.env.example` only as a reference; the application intentionally does not
silently load `.env` files.

With the connector/filter settings from the integrated-validation section configured, a
single-company smoke run can use a fresh local data directory:

```powershell
python scripts/run_moloco_e2e.py --runs 1 `
  --data-dir data/openai-compatible-smoke `
  --output-dir evaluation/results/openai-compatible-smoke
```

To run the checked-in local smoke harness against one already-persisted job, configure
the DynamoDB table/profile/job title read-only and run:

```powershell
$env:AWS_PROFILE = "jobradar-dev"
$env:JOB_INTELLIGENCE_AWS_REGION = "ap-south-1"
$env:JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME = "<deployed-data-table>"
$env:JOB_INTELLIGENCE_CANDIDATE_PROFILE_KEY = "candidate_001"
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
  --profile evaluation/benchmark/v2/candidate_profile.json `
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
  --profile evaluation/benchmark/v2/candidate_profile.json `
  --reference evaluation/results/persisted-job-ranking.json `
  --output-dir evaluation/results/filter-impact
```

This also reads existing DynamoDB jobs only and writes filter outcomes, retained
ranking, and provisional metric comparisons to local ignored artifacts.

The reproducible ranking benchmark is frozen under `evaluation/benchmark/v1`. Its
manifest hashes the canonical jobs, fixed structured analyses, and candidate profile,
and snapshots the exact matching, filtering, and eligibility configuration. Run it
without DynamoDB, queues, or an LLM (the local BGE model must already be cached):

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts/run_frozen_benchmark.py `
  --benchmark-dir evaluation/benchmark/v1 `
  --output-dir evaluation/results/benchmark-v1-run `
  --label-source provisional
```

For human evaluation, edit only `human_label` and `human_notes` in
`evaluation/benchmark/v1/labels.csv`, then rerun with `--label-source human` or `auto`.
Provisional labels are explicitly separate and are not ground truth.

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
