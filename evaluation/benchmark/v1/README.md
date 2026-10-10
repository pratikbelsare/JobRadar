# JobRadar frozen benchmark v1

This directory contains 30 immutable job/profile/analysis inputs and
editable labels. `manifest.json` records SHA-256 hashes and the exact configurations.
The candidate input is a synthetic public benchmark profile (`candidate_001`); it is not
an export of a real person's profile.

Run offline evaluation:

```powershell
python scripts/run_frozen_benchmark.py --benchmark-dir evaluation/benchmark/v1
```

Edit only `human_label` and `human_notes` in `labels.csv`. Provisional labels are not
ground truth. The runner uses human labels when present and otherwise reports the
explicit provisional baseline.

`labels_filled.csv` contains the reviewed public human-label ground truth. `labels.csv`
remains the blank labeling template, and `labels.md` is its readable review form.
