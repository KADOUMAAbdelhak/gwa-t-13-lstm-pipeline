# Task 08 Baselines 100VM

## What was run
I loaded the exact Task 5b sample from `dynapm/data/eval_sample_summary.parquet` by taking unique `(trace, vm_id, volatility_tier)` tuples, confirmed `100` unique VMs, and evaluated four methods (`static_v2`, `periodic_6h_v2`, `reactive_v2`, `full_rewrite_v2`) using `dynapm/eval/runner_v2.py` with `joblib.Parallel(n_jobs=8)`. This produced `400` runs total, persisted per-run JSON outputs, a summary parquet, aggregate diagnostics, and two Task 8 figures.

## Per-method aggregate statistics
(Mean/median across 100 VMs; sorted by `cpu_class_accuracy` mean descending.)

| method | cpu_class_accuracy mean | cpu_class_accuracy median | memory_class_accuracy mean | memory_class_accuracy median | cpu_p95_mae mean | cpu_p95_mae median | mem_p95_mae mean | mem_p95_mae median | total_updates mean | total_updates median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_rewrite_v2 | 0.7329 | 0.7566 | 0.7497 | 0.7670 | 1.3715 | 0.2917 | 1.6247 | 0.8340 | 1293.46 | 1330.50 |
| periodic_6h_v2 | 0.7085 | 0.7304 | 0.7275 | 0.7376 | 1.5225 | 0.3051 | 1.8248 | 1.0549 | 141.22 | 148.00 |
| reactive_v2 | 0.6158 | 0.6626 | 0.5814 | 0.5584 | 2.5012 | 0.4337 | 2.7079 | 1.3932 | 571.96 | 575.50 |
| static_v2 | 0.4207 | 0.3537 | 0.5408 | 0.6146 | 3.7864 | 0.4068 | 3.4472 | 0.9789 | 0.00 | 0.00 |

## Per-tier stratification
Per-method mean `cpu_class_accuracy` by volatility tier.

| method | low | medium | high |
|---|---:|---:|---:|
| full_rewrite_v2 | 0.7407 | 0.7542 | 0.7046 |
| periodic_6h_v2 | 0.7149 | 0.7296 | 0.6819 |
| reactive_v2 | 0.6876 | 0.5741 | 0.5865 |
| static_v2 | 0.3296 | 0.4112 | 0.5183 |

## Snapshot-drop diagnostic
- Mean `snapshots_dropped_unverified`: **0.00**
- Max `snapshots_dropped_unverified`: **0**
- Mean `n_eval_points_verified`: **40.60**
- Max `n_eval_points_verified`: **48**
- Drop-to-verified mean ratio: **0.00%** (`0.00 / 40.60`)

Interpretation against task thresholds:
- Mean dropped is not >5% of mean verified.
- Mean dropped is not >20% of mean verified.

## Checkpoint questions
- **Q1** PASS: `static_v2` vs `full_rewrite_v2` CPU accuracy gap = **0.3122** (`0.7329 - 0.4207`), which is >= `0.20`.
- **Q2** PASS: `periodic_6h_v2` mean `total_updates` = **141.22**, within `[50, 300]`.
- **Q3** PASS: `full_rewrite_v2` mean `total_updates` = **1293.46**, which is > `1000`.
- **Q4** PASS: `|periodic_6h_v2 - full_rewrite_v2|` CPU accuracy = **0.0243**, within `0.10`.
- **Q5** FAIL: high-tier spread (**0.1863**) is not greater than low-tier spread (**0.4111**); medium-tier spread is **0.3430**.

## Figures
- `dynapm/reports/task_08/baselines_v2_accuracy_by_method.png`
- `dynapm/reports/task_08/baselines_v2_accuracy_by_tier.png`

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/*.json` (created; 400 per-run outputs)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_summary.parquet` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_diagnostics.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_08/baselines_v2_accuracy_by_method.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_08/baselines_v2_accuracy_by_tier.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_08_baselines_100vm.md` (created)

## Test results
No new unit tests were required by this task. Baseline pytest still passes:

```text
.......                                                                  [100%]
7 passed in 2.05s
```

## Assumptions
- The exact Task 5b sample is represented by unique `(trace, vm_id, volatility_tier)` tuples in `dynapm/data/eval_sample_summary.parquet`.
- `volatility_tier` values in the Task 5b summary are valid for reuse without recomputation.
- `runner_v2` output fields are already in final form and are consumed directly for aggregation.
- Snapshot-drop ratio check uses `mean_dropped / mean_verified` as requested.

## Surprises
- Snapshot drops were zero across all 400 runs, indicating no verification-tail loss on this sample under current `runner_v2` timelines.
- Tier spread pattern for Q5 is opposite the expected heuristic: low-tier method spread is much larger than high-tier spread.

## Commands

```bash
dynapm/.venv/bin/python - << 'PY'
# Loaded Task5b sample from eval_sample_summary.parquet (100 unique VMs)
# Ran 4 baselines x 100 VMs with joblib.Parallel(n_jobs=8)
# Wrote eval_sample_v2 summary, diagnostics JSON, and two figures
PY
# output snippet:
# {
#   "n_unique_vms": 100,
#   "n_runs": 400,
#   "wall_seconds": 65.16880825010594,
#   "summary_parquet": ".../dynapm/data/eval_sample_v2_summary.parquet",
#   "diag_json": ".../dynapm/data/eval_sample_v2_diagnostics.json",
#   "fig1": ".../dynapm/reports/task_08/baselines_v2_accuracy_by_method.png",
#   "fig2": ".../dynapm/reports/task_08/baselines_v2_accuracy_by_tier.png"
# }
```

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_forecasting.py dynapm/tests/test_runner_v2.py -q
# output:
# .......                                                                  [100%]
# 7 passed in 2.05s
```

## Next-step suggestion
Task 9 should integrate DynAPM into `runner_v2` and evaluate it on this same locked 100-VM sample, using `periodic_6h_v2` and `full_rewrite_v2` as immediate reference baselines, since Q4 shows periodic is already close to full rewrite on CPU accuracy at far lower churn and therefore sets a strong practical target for DynAPM’s contribution.
