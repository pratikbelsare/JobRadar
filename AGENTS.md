# AGENTS.md

## Project

This repository contains a personal AI-powered Job Intelligence product.

The canonical planning documents are in `planning/`:

- `planning/phase-1-product-scope.md`
- `planning/phase-2-system-design.md`
- `planning/phase-3-data-ai-ranking-evaluation.md`
- `planning/phase-4-implementation-plan.md`

Read the relevant planning files before making architectural or implementation decisions.

## Working Principles

1. Implement one milestone at a time.
2. Do not expand scope beyond the planning documents unless explicitly requested.
3. Prefer simple, maintainable engineering over unnecessary frameworks.
4. Do not introduce LangChain, LangGraph, agents, Step Functions, OpenSearch, a separate vector database, ECS, EKS, SageMaker, or similar infrastructure unless explicitly requested or clearly justified.
5. Keep the core architecture role-agnostic and configurable.
6. Keep business logic independent from AWS-specific code where practical.
7. Use deterministic logic where AI is unnecessary.
8. Keep AI-provider access behind abstractions so models/providers can change later.
9. Use typed models and explicit interfaces.
10. Validate external and LLM-generated data.
11. Design components to be independently testable.
12. Avoid hidden global state and hardcoded user-specific values.
13. Prefer clear names and small focused modules over large utility files.
14. Do not create premature abstractions for hypothetical future requirements.
15. Preserve explainability and reproducibility in ranking-related code.

## Python Standards

- Use modern Python.
- Use type hints throughout application code.
- Use Pydantic for domain/config validation where appropriate.
- Use pytest for tests.
- Keep formatting and linting configuration simple.
- Prefer dependency injection or explicit constructor parameters for external dependencies.
- Keep pure business logic separate from I/O.

## Testing

Every milestone should include tests for its important behavior.

Tests should:

- run locally
- avoid live AWS calls unless explicitly marked as integration tests
- avoid live career-site requests in normal unit tests
- use fixtures/mocks for external systems
- cover validation and failure cases, not only happy paths

## Configuration

Do not hardcode:

- target roles
- locations
- experience thresholds
- company count
- batch size
- scan frequency
- model IDs
- AWS region
- credentials
- API keys

Use configuration/environment variables where appropriate.

Never commit secrets.

## AWS

The planned AWS direction is:

- Amazon Bedrock
- S3
- Lambda
- SQS
- EventBridge Scheduler
- Aurora PostgreSQL Serverless v2
- CloudWatch
- IAM
- API Gateway if needed

Do not provision or connect AWS resources until the relevant implementation milestone requires them.

Local development should come first where possible.

## AI / Ranking

The LLM must not directly invent the final job-match score.

The planned ranking approach is:

- deterministic hard filters
- structured features
- semantic signals where useful
- interpretable hybrid scoring
- grounded explanations after ranking

Important explanation claims must ultimately be traceable to candidate or job evidence.

## Repository Hygiene

- Keep generated files and local secrets out of Git.
- Update README/docs when behavior or setup materially changes.
- Do not modify planning documents unless explicitly asked.
- If implementation requires deviating from a planning decision, explain the conflict before changing the architecture.

## Current Implementation Order

Follow `planning/phase-4-implementation-plan.md`.

The first implementation milestone is:

**Milestone 1 — Repository and Core Models**

Do not implement later milestones unless explicitly requested.
