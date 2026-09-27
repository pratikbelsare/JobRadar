# Phase 3 — Data, AI, Ranking and Evaluation Design

## 1. Goal

Define how the system represents candidates and jobs, uses Bedrock, calculates job relevance, explains results, and evaluates whether ranking quality is actually useful.

---

## 2. Candidate Profile

Use a structured editable profile initialized from the resume.

```text
CandidateProfile
├── target_roles
├── experience_years
├── preferred_locations
├── work_mode
├── skills
├── skill_evidence
├── work_experience
├── projects
├── education
├── domains
├── preferences
└── hard_constraints
```

The resume is an input source, not the main candidate database.

---

## 3. Normalized Job Schema

Every career-site connector should produce a common job structure.

```text
Job
├── title
├── company
├── location
├── work_mode
├── description
├── min_experience
├── max_experience
├── seniority
├── required_skills
├── preferred_skills
├── responsibilities
├── education
├── domain
├── employment_type
├── posted_date
└── source_url
```

---

## 4. Bedrock Usage

### Bedrock should handle

- structured job requirement extraction
- required vs preferred skill classification
- resume-to-profile extraction
- semantic embeddings
- semantic candidate-job comparison
- grounded explanations
- skill-gap summaries

### Bedrock should not handle

- scheduling
- hard filters
- exact deduplication
- numeric experience rules
- job status tracking
- final ranking score directly

All LLM outputs must use structured schemas and validation.

---

## 5. Skill Normalization

Normalize equivalent terms into canonical skills.

Examples:

```text
Amazon Web Services → AWS
Postgres            → PostgreSQL
Pytorch             → PyTorch
```

Start with a configurable taxonomy and use semantic fallback where needed.

---

## 6. Hard Filters vs Ranking

### Hard Filters

Reject only clearly unsuitable jobs.

Examples:

- unsupported country / location
- mandatory work authorization unavailable
- very large experience mismatch
- completely unrelated role
- excluded employment type

### Ranking Signals

Use softer signals for borderline cases:

- role relevance
- skill fit
- experience fit
- responsibility similarity
- domain fit
- location preference
- other user preferences

Avoid rejecting reasonable stretch roles too early.

---

## 7. Matching Features

Primary features:

```text
Role Match
Skill Match
Experience Match
Responsibility Match
Domain Match
Location Match
Preference Match
```

### Role Match

Combine:

- normalized title matching
- semantic title / role similarity

### Skill Match

Required skills should have more weight than preferred skills.

Candidate evidence should matter more than a skill merely appearing in a skill list.

### Experience Match

Use a gradual score instead of a strict cutoff.

Example:

```text
Strong alignment  → high score
Small gap         → slight penalty
Stretch role      → larger penalty
Extreme mismatch  → hard reject
```

### Responsibility Match

Use embeddings to compare candidate experience/projects with job responsibilities.

This captures semantic similarity even when wording differs.

---

## 8. Initial Ranking Model

Start with an interpretable weighted score.

```text
Final Score =
    25% Role Match
  + 25% Skill Match
  + 20% Experience Match
  + 15% Responsibility Match
  +  5% Domain Match
  +  5% Location Match
  +  5% Preference Match
```

These weights are **initial placeholders**, not final truth.

They will be tuned using evaluation data.

The LLM must not directly invent the final 0–100 job score.

---

## 9. Ranking Evolution

Evaluate increasingly capable approaches:

```text
Baseline 1: Keyword Matching
Baseline 2: Embedding Similarity
Model 3: Hybrid Structured Ranking
Model 4: Feedback-Tuned Ranking
```

Add complexity only if it improves measured results.

---

## 10. Explanation Design

Generate explanations only after ranking.

Input to Bedrock:

- candidate evidence
- job requirements
- component scores

Output:

```text
Why this role fits
Strong matches
Potential gaps
Experience alignment
Important considerations
```

Every important claim should be traceable to source evidence.

Use:

> No evidence found for Kubernetes.

Not:

> Candidate does not know Kubernetes.

---

## 11. User Feedback

Allow explicit relevance labels:

```text
3 = Strong fit / would apply
2 = Worth reviewing
1 = Weak fit
0 = Irrelevant
```

Optional reasons:

- wrong role
- wrong location
- too senior
- skill mismatch
- strong fit
- other

Feedback becomes evaluation data and can later improve ranking.

---

## 12. Ranking Evaluation

Create a labeled dataset of real jobs.

Primary metrics:

- Precision@5
- Precision@10
- NDCG@10

Main product question:

> Of the top-ranked jobs, how many are genuinely worth reviewing?

Compare all ranking approaches against the same labeled dataset.

---

## 13. Extraction Evaluation

Create a smaller manually annotated set of job descriptions.

Evaluate:

- required skill extraction
- preferred skill extraction
- experience extraction
- seniority extraction
- location extraction

Use precision, recall, F1, or accuracy depending on the field.

---

## 14. Explanation Evaluation

Track an evidence-support metric:

> Percentage of important explanation claims supported by the candidate profile or job description.

Target high factual grounding rather than persuasive wording.

---

## 15. End-to-End Intelligence Flow

```text
Resume
   ↓
Structured Candidate Profile

Job Description
   ↓
Structured Job Requirements

Candidate + Job
   ↓
Hard Filters
   ↓
Feature Generation
   ↓
Hybrid Ranking
   ↓
Ranked Jobs
   ↓
Grounded Explanation
   ↓
User Feedback
   ↓
Evaluation
   ↓
Ranking Improvement
```

---

## 16. Phase 3 Decisions

### Locked

- structured editable candidate profile
- normalized job schema
- Bedrock for structured extraction
- skill normalization
- hard filters separate from ranking
- embeddings for semantic responsibility matching
- interpretable hybrid ranking
- LLM explains scores but does not create them
- explicit relevance feedback
- Precision@K and NDCG evaluation
- baseline comparisons before adding complexity

### Deferred

- exact Bedrock model
- exact embedding model
- final skill taxonomy
- final ranking weights
- exact hard-filter thresholds
- exact evaluation dataset size
- whether a learned ranking model becomes worthwhile

---

## 17. Next Phase

**Phase 4 — Implementation Planning**

Phase 4 should define:

- development milestones
- implementation order
- local vs AWS development workflow
- infrastructure setup
- testing strategy
- CI/CD
- deployment sequence
- evaluation milestones
- what Codex should build in each step
