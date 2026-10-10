# JobRadar frozen benchmark v1

This directory contains 30 immutable job/profile/analysis inputs and
editable labels. `manifest.json` records SHA-256 hashes and the exact configurations.

Run offline evaluation:

```powershell
python scripts/run_frozen_benchmark.py --benchmark-dir evaluation/benchmark/v1
```

Edit only `human_label` and `human_notes` in `labels.csv`. Provisional labels are not
ground truth. The runner uses human labels when present and otherwise reports the
explicit provisional baseline.
