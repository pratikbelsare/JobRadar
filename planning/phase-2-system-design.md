# Phase 2 — System Design and Architecture

## 1. Goal

Design a **portfolio-balanced, low-cost AWS architecture** for the personal Job Intelligence product defined in Phase 1.

The system should periodically check official company career sites, discover new or changed jobs, filter and rank them against a configurable candidate profile, and generate evidence-backed explanations.

---

## 2. Locked Constraints

- Cloud: **AWS**
- AI platform: **Amazon Bedrock**
- Product mode: **single-user / personal**
- Job sources: **official company career sites only**
- Company list: one flat configurable list initially
- Roles: configurable; not hardcoded to AI/ML
- Locations: configurable
- Experience rules: configurable
- Candidate profile: **resume + editable structured profile**
- Architecture complexity: **portfolio-balanced**
- Cost approach: prefer low-cost, usage-based AWS services
- Frontend: undecided
- Exact budget, run frequency, batch size, and company count: configurable / deferred

Initial evaluation will focus on AI/ML/Data roles even though the architecture is role-agnostic.

---

## 3. High-Level Architecture

```text
EventBridge Scheduler
        ↓
Run Coordinator
        ↓
Select next company batch
        ↓
SQS: Company Scan Queue
        ↓
Career-Site Connectors
        ↓
Raw Data → S3
        ↓
Normalize + Detect New/Changed Jobs
        ↓
Deterministic Filters
(role / location / experience)
        ↓
SQS: Job Analysis Queue
        ↓
Bedrock Extraction / Embeddings
        ↓
Hybrid Matching + Ranking
        ↓
Grounded Explanation / Skill Gaps
        ↓
PostgreSQL
        ↓
API / Dashboard
```

---

## 4. AWS Services

| Service | Purpose |
|---|---|
| EventBridge Scheduler | Trigger periodic scans |
| Lambda | Coordinators, scanners, processing workers |
| SQS | Decouple company scanning and AI analysis |
| S3 | Raw HTML/JSON, snapshots, exports, evaluation data |
| Amazon Bedrock | LLM extraction, semantic analysis, embeddings, explanations |
| Aurora PostgreSQL Serverless v2 | Structured application data |
| CloudWatch | Logs, metrics, alarms |
| IAM | Least-privilege access control |
| API Gateway | Backend API if required by the final frontend |

### Avoid initially

- ECS / EKS
- OpenSearch
- SageMaker
- Step Functions
- Kafka / Kinesis
- ElastiCache
- Separate vector database
- LangChain / LangGraph unless later justified

---

## 5. Company Scheduling

The system should not hardcode a fixed number of companies.

Each run:

1. Load enabled companies.
2. Sort/select companies based on `last_checked_at`.
3. Pick a configurable batch.
4. Queue each company independently.
5. Update scan status and timestamps.

Initial strategy: **simple rotation**.

Later, priority-based schedules can be added if useful.

---

## 6. Career-Site Ingestion

Use a generic connector abstraction.

```text
JobSourceConnector
├── list_jobs(company)
└── get_job(job_reference)
```

Different companies may require different implementations, but the rest of the system should not care whether data came from:

- normal HTML
- embedded JSON
- public career endpoints
- Workday
- Greenhouse
- Lever
- Oracle
- another supported career system

Do not create business logic tied to individual companies unless necessary.

Connector outcomes should include:

- SUCCESS
- NO_JOBS
- UNSUPPORTED
- RATE_LIMITED
- FAILED

---

## 7. Core Data Model

Main entities:

```text
Company
CandidateProfile
Job
JobVersion
JobAnalysis
JobMatch
UserFeedback
ScanRun
```

### Company

Key fields:

- id
- name
- career_url
- connector_type
- enabled
- last_checked_at
- last_success_at
- failure_count

### Job

Key fields:

- source_job_id
- company_id
- title
- location
- description
- source_url
- posted_at
- first_seen_at
- last_seen_at
- content_hash
- status

### CandidateProfile

Contains:

- target roles
- experience
- locations
- skills
- skill evidence
- projects
- domains
- education
- preferences
- hard constraints

The resume is an input source, not the candidate database itself.

---

## 8. New / Changed Job Detection

Before AI processing:

```text
Fetch job
   ↓
Normalize content
   ↓
Generate content hash
   ↓
Compare with stored version
```

If unchanged:

> reuse previous analysis

If new or changed:

> continue processing

This is a major cost-control mechanism.

---

## 9. Filtering Pipeline

Use cheap deterministic logic before Bedrock.

```text
New/Changed Job
      ↓
Role Filter
      ↓
Location Filter
      ↓
Experience Filter
      ↓
Other Hard Constraints
      ↓
Semantic / AI Analysis
```

Experience rules should support:

- strong match
- acceptable
- stretch
- reject

rather than a single strict cutoff.

All filters must be configurable.

---

## 10. AI / Bedrock Responsibilities

Use Bedrock only where semantic understanding adds value.

### AI tasks

1. Extract structured job requirements.
2. Distinguish required vs preferred skills.
3. Extract / update candidate profile from resume.
4. Generate embeddings where useful.
5. Perform semantic candidate-job comparison.
6. Generate grounded explanations.
7. Produce skill-gap summaries.

### Do not use LLMs for

- scheduling
- database filtering
- exact duplicate detection
- simple numeric constraints
- job-status tracking
- basic configuration rules

All LLM outputs must use structured schemas and validation.

---

## 11. Ranking Design

Do not ask an LLM to directly invent a 0–100 score.

Use interpretable components such as:

```text
role_match
skill_match
experience_match
location_match
domain_match
preference_match
```

Initial ranking:

```text
final_score =
    weighted combination of component scores
```

Weights will later be evaluated and tuned using user labels.

Expected progression:

```text
Keyword baseline
→ Embedding baseline
→ Hybrid structured ranking
→ Feedback-tuned ranking
```

---

## 12. Explainability

Recommendations should be grounded in evidence from:

- job description
- candidate profile

Example:

```text
Strength:
Forecasting — supported by candidate project experience.

Gap:
Kubernetes — no evidence found in candidate profile.
```

Use **“no evidence found”** instead of claiming the candidate does not know a skill.

Long explanations should be generated lazily for top jobs or on demand to reduce cost.

---

## 13. Storage Strategy

### S3

Store:

- raw career-site responses
- HTML / JSON snapshots
- evaluation datasets
- exports
- processing artifacts

### PostgreSQL

Store:

- companies
- normalized jobs
- job versions
- candidate profile
- analysis results
- rankings
- user feedback
- run metadata

A separate vector database is not required initially.

If vector persistence becomes useful, PostgreSQL + `pgvector` can be evaluated before adding another service.

---

## 14. Reliability

The system should be:

- idempotent
- retryable
- resilient to individual company failures
- tolerant of malformed pages
- tolerant of Bedrock failures

Use:

- SQS retries
- Dead Letter Queues
- structured error codes
- schema validation
- safe retry policies
- cached previous results

One failed company must not fail the whole scan.

---

## 15. Observability

Every run should have a `run_id`.

Track:

- companies scanned
- companies failed
- jobs discovered
- new jobs
- changed jobs
- duplicates
- jobs filtered
- jobs analyzed
- Bedrock calls
- Bedrock failures
- processing latency
- cache hits
- relevant jobs surfaced

Use CloudWatch logs and metrics initially.

---

## 16. API Boundary

Keep the backend independent from the frontend.

Possible endpoints:

```text
GET  /jobs
GET  /jobs/{id}
GET  /companies
GET  /runs

PUT  /profile
PUT  /preferences

POST /jobs/{id}/shortlist
POST /jobs/{id}/dismiss
POST /jobs/{id}/applied
```

This allows Streamlit, React/Next.js, or another frontend later without changing the core system.

---

## 17. Repository Structure

```text
job-intelligence/
├── docs/
├── src/
│   ├── ingestion/
│   ├── connectors/
│   ├── filtering/
│   ├── ai/
│   ├── matching/
│   ├── ranking/
│   ├── storage/
│   ├── api/
│   └── common/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── evaluation/
├── infrastructure/
└── README.md
```

---

## 18. Phase 2 Decisions

### Locked

- AWS-based architecture
- Serverless/event-driven design
- Amazon Bedrock for AI
- S3 for raw data
- Lambda for processing
- SQS for queues
- EventBridge Scheduler for periodic runs
- CloudWatch for observability
- PostgreSQL for structured data
- Resume + structured editable candidate profile
- Deterministic filtering before AI
- Hybrid interpretable ranking
- Evidence-grounded explanations
- No separate vector DB initially
- No agents / LangGraph / Step Functions initially
- Generic connector interface for career sites

### Deferred

- Exact AWS region
- Exact budget
- Exact company count
- Batch size
- Run frequency
- Bedrock models
- Embedding model
- Ranking weights
- Frontend framework
- Final deployment method
- Exact connector implementations

---

## 19. Architecture Principle

> Use simple deterministic engineering wherever possible and use AI only where semantic understanding provides measurable value.

The project should demonstrate **good AI Engineering decisions**, not maximum architectural complexity.

---

## 20. Next Phase

**Phase 3 — Data, AI, Ranking and Evaluation Design**

Phase 3 should define:

- detailed job schema
- candidate profile schema
- structured LLM outputs
- skill normalization
- embedding strategy
- matching features
- ranking formula
- baselines
- evaluation dataset
- Precision@K / NDCG methodology
- explanation quality evaluation
