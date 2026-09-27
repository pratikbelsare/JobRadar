# Phase 1 — Product Scope

## 1. Product

**AI-Powered Job Intelligence Platform**

A personal system that monitors selected company career sites, discovers relevant jobs, filters unsuitable roles, ranks opportunities against a candidate profile, explains the match, identifies skill gaps, and supports shortlist / dismiss / applied tracking.

The system is a **decision-support product**, not an autonomous job-application bot.

---

## 2. Problem Statement

Job seekers spend significant time:

- repeatedly checking company career sites
- reading irrelevant job descriptions
- comparing role, location, experience, and skills manually
- dealing with duplicate or reposted jobs
- deciding which opportunities are actually worth reviewing

The product should reduce this effort by producing a:

> **deduplicated, filtered, personalized, ranked, and explainable shortlist of jobs.**

---

## 3. Target User

Initial product:

- single-user / personal
- early-career technical professional
- initially validated on:
  - ML Engineer
  - AI Engineer
  - Data Scientist
  - related AI/ML roles

The architecture should remain configurable so other roles such as Java Developer or DevOps Engineer can be supported later.

---

## 4. Core Product Value

The system should help answer:

1. **Which jobs are worth reviewing?**
2. **Why is this job relevant or irrelevant?**
3. **What are the main strengths and gaps?**
4. **Should I shortlist, dismiss, or apply?**

The user remains the final decision-maker.

---

## 5. Core Workflow

```text
Candidate Profile
        +
Configured Company List
        ↓
Career-Site Job Discovery
        ↓
Normalization
        ↓
New / Changed Job Detection
        ↓
Deduplication
        ↓
Hard Filtering
        ↓
AI / Semantic Analysis
        ↓
Candidate-Job Matching
        ↓
Ranking
        ↓
Grounded Explanation
        ↓
Skill Gap Analysis
        ↓
User Decision
        ↓
Feedback
```

---

## 6. Functional Requirements

### FR-1 — Candidate Profile

Maintain:

- target roles
- experience
- preferred locations
- work mode
- skills
- skill evidence
- projects
- education
- domains
- preferences
- hard constraints

Input should be:

> **resume + editable structured profile**

---

### FR-2 — Company List

Maintain a configurable list of target companies.

Each company should support:

- career URL
- enabled/disabled status
- connector/source type
- last checked timestamp
- scan status

Exact company count is configurable.

---

### FR-3 — Job Discovery

Periodically check official company career sites.

The system should support batch-based scanning rather than requiring all companies to be checked every run.

---

### FR-4 — Job Normalization

Convert all discovered jobs into a common internal schema.

Important fields:

- company
- title
- location
- description
- experience
- skills
- responsibilities
- posted date
- source URL

---

### FR-5 — New / Changed Detection

Avoid reprocessing unchanged jobs.

Use stable identifiers and/or content hashes.

---

### FR-6 — Deduplication

Detect:

- exact duplicates
- reposted jobs
- near-duplicate versions where practical

---

### FR-7 — Hard Filtering

Apply cheap deterministic checks before AI.

Initial filters:

- role
- location
- experience
- employment type
- hard user constraints

---

### FR-8 — Requirement Extraction

Extract:

- required skills
- preferred skills
- experience requirements
- seniority
- responsibilities
- education
- location constraints
- domain requirements

---

### FR-9 — Matching and Ranking

Calculate interpretable components such as:

- role match
- skill match
- experience match
- responsibility match
- domain match
- location match
- preference match

Do not use an opaque LLM-generated score as the primary ranking method.

---

### FR-10 — Explainability

Explain why a job ranked where it did.

Important claims should be supported by:

- job-description evidence
- candidate-profile evidence

---

### FR-11 — Skill Gaps

Distinguish:

- strong evidence
- partial evidence
- no evidence found

Do not claim the candidate lacks a skill simply because it is absent from the resume.

---

### FR-12 — Job Tracking

Support statuses such as:

- New
- Reviewed
- Shortlisted
- Maybe
- Applied
- Dismissed

---

### FR-13 — Feedback

Allow relevance labels:

```text
3 = Strong fit / would apply
2 = Worth reviewing
1 = Weak fit
0 = Irrelevant
```

Feedback will support evaluation and future ranking improvement.

---

## 7. Non-Functional Requirements

The system should be:

- reliable
- idempotent
- retryable
- testable
- observable
- cost-aware
- privacy-conscious
- easy to extend

Important engineering requirements:

- one failed company must not fail an entire run
- unchanged jobs should reuse cached analysis
- LLM outputs must use structured validation
- model / prompt / ranking versions should be traceable
- external failures should be logged clearly

---

## 8. AI vs Deterministic Engineering

### Use AI / embeddings for

- structured requirement extraction
- required vs preferred skill interpretation
- resume/profile extraction
- semantic role similarity
- semantic responsibility matching
- grounded explanations
- skill-gap summaries

### Do not use AI for

- scheduling
- database filtering
- exact duplicate detection
- basic numeric experience checks
- job status tracking
- simple configuration rules

### Principle

> Use AI only where semantic understanding provides measurable value.

---

## 9. MVP Scope

MVP includes:

- one personal candidate profile
- configurable company list
- official career-site monitoring
- batch-based scans
- job normalization
- change detection
- deduplication
- role/location/experience filtering
- structured job extraction
- semantic matching
- ranking
- grounded explanations
- skill-gap analysis
- job tracking
- user feedback

---

## 10. Explicitly Out of MVP

Do not include initially:

- automatic application submission
- LinkedIn automation
- recruiter messaging
- cover-letter generation
- resume rewriting per job
- interview preparation
- salary prediction
- hiring-probability prediction
- networking automation
- multi-user SaaS
- billing
- generic chatbot
- unnecessary autonomous agents

---

## 11. Success Metrics

### Ranking

Track:

- Precision@5
- Precision@10
- NDCG@10

Primary product metric:

> Percentage of top-ranked jobs that are genuinely worth reviewing.

Initial target:

> **≥ 70% of top-10 recommendations rated 2 or 3**

This target may change after a real baseline is measured.

### Extraction

Evaluate:

- required skill extraction
- preferred skill extraction
- experience extraction
- seniority extraction
- location extraction

### Explanation

Track:

> percentage of important explanation claims supported by source evidence

Target high grounding quality.

### System

Track:

- companies scanned
- jobs discovered
- new jobs
- jobs changed
- jobs filtered
- relevant jobs
- failures
- cache hits
- AI calls
- latency
- cost

---

## 12. Major Risks

### Career-Site Diversity

Different companies use different site technologies.

**Mitigation:** connector-based ingestion architecture.

### Scope Creep

The product could expand into a complete career platform.

**Mitigation:** MVP ends at ranked job intelligence and tracking.

### Arbitrary Ranking

Weighted scores may look meaningful without being useful.

**Mitigation:** compare against baselines and evaluate with labeled jobs.

### LLM Errors

Extraction or explanations may be wrong.

**Mitigation:** structured output, validation, evidence grounding, evaluation.

### Cost Growth

Repeated AI processing may become expensive.

**Mitigation:** deterministic filtering, caching, change detection, selective AI calls.

---

## 13. Product Boundary

The MVP begins with:

> **candidate profile + configured company list**

The MVP ends with:

> **ranked, explainable jobs that the user can review, shortlist, dismiss, or mark as applied**

It does **not** automatically submit applications.

---

## 14. Guiding Principles

1. Solve a real personal problem first.
2. Keep the architecture configurable rather than hardcoded to AI/ML roles.
3. Use deterministic engineering before AI.
4. Filter cheaply before expensive model calls.
5. Prefer interpretable ranking over opaque LLM scoring.
6. Ground important AI claims in evidence.
7. Measure usefulness using real user labels.
8. Add complexity only when evaluation justifies it.

---

## 15. Next Phase

**Phase 2 — System Design and Architecture**

Phase 2 defines the AWS architecture, storage, queues, scheduling, connectors, Bedrock integration, reliability, observability, and system boundaries.
