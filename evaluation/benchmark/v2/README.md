# JobRadar frozen benchmark v2

This benchmark contains 84 current jobs frozen from official company
career sites. The selection strata deliberately cover target, stretch, adjacent, and
hard-negative roles; they are not human relevance labels.
The candidate input is a synthetic public benchmark profile (`candidate_001`); it is not
an export of a real person's profile.

The immutable files are hashed in `manifest.json`. The semantic fingerprint is
`1659939a818926901de2b8ee42a6b797017c9018a410514898ac14f0daca75c3`. Raw count: 84; independent count:
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

`labels_v2_filled.csv` contains the reviewed public human-label ground truth. `labels.csv`
remains the blank labeling template, and `labels.md` is its readable review form.

The full pre-label quality audit is in `audit.json`.
