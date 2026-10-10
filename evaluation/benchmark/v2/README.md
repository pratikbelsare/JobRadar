# JobRadar frozen benchmark v2

This benchmark contains 84 current jobs frozen from official company
career sites. The selection strata deliberately cover target, stretch, adjacent, and
hard-negative roles; they are not human relevance labels.

The immutable files are hashed in `manifest.json`. The semantic fingerprint is
`87a5cec16c269c59225531301713fac72b9d10877eac1dec502e9f3c2ae13fb4`. Raw count: 84; independent count:
84; duplicate groups: 0.

`job_analyses.json` was extracted once with `bedrock_mantle` / 
`openai.gpt-oss-20b` and strict schema validation. Normal evaluation never
uses Mantle, DynamoDB, SQS, or any AWS API; it uses only the frozen files and local
`BAAI/bge-small-en-v1.5` embeddings.

Run twice offline and compare fingerprints:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.venv\Scripts\python.exe scripts/run_frozen_benchmark.py `
  --benchmark-dir evaluation/benchmark/v2 `
  --output-dir evaluation/results/benchmark-v2-run-1
.venv\Scripts\python.exe scripts/run_frozen_benchmark.py `
  --benchmark-dir evaluation/benchmark/v2 `
  --output-dir evaluation/results/benchmark-v2-run-2
```

To label, edit only `human_label` and `human_notes` in `labels.csv`. Use
`3=strong_target`, `2=relevant_stretch`, `1=adjacent`, `0=irrelevant`. Do not infer a
label from `selection_bucket`, current rank, role family, or eligibility.

The full pre-label quality audit is in `audit.json`.
