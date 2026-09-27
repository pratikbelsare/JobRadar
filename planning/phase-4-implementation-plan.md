# Phase 4 — Implementation Plan

## 1. Goal

Define the build order for the Job Intelligence product so implementation can be done incrementally, tested at each step, and handed to Codex in small tasks.

---

## 2. Development Approach

- Build locally first where possible.
- Deploy to AWS only after each core component works.
- Keep tasks small and independently testable.
- Avoid building the full system in one Codex prompt.
- Add complexity only when the previous stage is stable.

---

## 3. Milestones

### Milestone 1 — Repository and Core Models

Build:

- repository structure
- configuration system
- typed domain models
- shared utilities
- test setup

Core models:

- Company
- CandidateProfile
- Job
- JobVersion
- JobAnalysis
- JobMatch
- UserFeedback
- ScanRun

---

### Milestone 2 — Candidate Profile

Build:

- resume ingestion
- structured candidate profile
- editable preferences
- target roles
- locations
- experience rules
- skill evidence

Test profile extraction and manual overrides.

---

### Milestone 3 — Career-Site Ingestion

Build:

- `JobSourceConnector` interface
- first working connector
- company configuration
- job listing discovery
- job description retrieval

Add connectors incrementally.

---

### Milestone 4 — Normalization and Change Detection

Build:

- normalized job schema
- content hashing
- new/changed job detection
- exact deduplication
- raw snapshot storage

---

### Milestone 5 — Deterministic Filtering

Implement configurable:

- role filtering
- location filtering
- experience filtering
- hard constraints

Verify that irrelevant jobs are removed before AI processing.

---

### Milestone 6 — Bedrock Integration

Build:

- Bedrock client abstraction
- structured job extraction
- candidate extraction
- schema validation
- retries and failure handling

Do not add agents.

---

### Milestone 7 — Matching and Ranking

Implement:

- role match
- skill match
- experience match
- responsibility similarity
- domain match
- location match
- preference match
- weighted final score

Keep ranking interpretable.

---

### Milestone 8 — Evaluation

Build:

- labeled evaluation dataset
- keyword baseline
- embedding baseline
- hybrid ranking evaluation

Track:

- Precision@5
- Precision@10
- NDCG@10

---

### Milestone 9 — Explanation and Feedback

Build:

- grounded explanations
- skill-gap summaries
- relevance labels
- shortlist / dismiss / applied tracking

---

### Milestone 10 — API and UI

Build backend API first.

Frontend choice can be finalized here.

Possible options:

- Streamlit
- React / Next.js

---

### Milestone 11 — AWS Deployment

Deploy:

- S3
- Lambda
- SQS
- EventBridge Scheduler
- Aurora PostgreSQL
- Bedrock integration
- CloudWatch

Add infrastructure-as-code.

---

### Milestone 12 — Hardening

Add:

- retries
- dead-letter queues
- structured errors
- logging
- metrics
- cost tracking
- caching
- integration tests
- CI/CD

---

## 4. Testing Strategy

Use:

- unit tests for business logic
- fixture-based tests for connectors
- integration tests for Bedrock and database access
- mocked external services where possible
- evaluation tests for ranking quality

Avoid requiring live websites or paid APIs for every test run.

---

## 5. AWS Strategy

Develop locally first.

Move components to AWS only when stable.

Prefer usage-based services and keep infrastructure minimal.

Exact budget, region, batch size, and schedule remain configurable.

---

## 6. Codex Workflow

For each milestone:

1. Give Codex the relevant phase documents.
2. Ask it to implement one small milestone only.
3. Require tests.
4. Review the output.
5. Fix issues before moving forward.
6. Commit the milestone.
7. Continue to the next milestone.

Do not ask Codex to build the entire project in one prompt.

---

## 7. Completion Criteria

Implementation planning is complete when:

- build order is clear
- component boundaries are defined
- tests are included in every milestone
- AWS deployment happens after local validation
- AI evaluation is part of implementation, not an afterthought

After this phase, coding can begin with **Milestone 1 — Repository and Core Models**.
