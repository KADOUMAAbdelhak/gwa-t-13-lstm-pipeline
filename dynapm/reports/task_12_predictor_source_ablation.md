# Task 12 Predictor Source Ablation

## What was tested
This task isolates the **forecast source** under the same **6-hour refresh cadence** as `PeriodicPolicyV2`: one baseline refreshes using the **mean of observed values over the lookback window** at each tick (`periodic_6h_v2`, Task 8), the other (`periodic_predictor_6h_v2`) refreshes using the **mean class / p95 of the next 288 predictor steps** (`y_pred`) at each refresh. Both use `runner_v2` and the locked **100-VM** sample from `dynapm/data/eval_sample_v2_summary.parquet`. Reference metrics for `periodic_6h_v2` are taken from that same parquet (Task 8 aggregates).

## Implementation
- Added **`PeriodicPredictorPolicyV2`** to `dynapm/methods/baselines_v2.py` (new class only; **`PeriodicPolicyV2` unchanged**).
- Initialization matches **`PeriodicPolicyV2`**: initial forecast from the last **`period_hours`** of bootstrap at bootstrap end, via `_window_last_hours` + `_set_initial_forecast`.
- **`runner_v2`** already passes **`cpu_pred_full`** and **`mem_pred_full`** into `method_factory` (Task 9); no `runner_v2` edit was required.

## Head-to-head comparison
Mean and median across **100** VMs. **`periodic_6h_v2`** from `eval_sample_v2_summary.parquet`; **`periodic_predictor_6h_v2`** from this task’s run.

| method | cpu_class_accuracy mean | cpu_class_accuracy median | memory_class_accuracy mean | memory_class_accuracy median | cpu_p95_mae mean | cpu_p95_mae median | mem_p95_mae mean | mem_p95_mae median | total_updates mean | total_updates median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic_6h_v2 | 0.7085 | 0.7304 | 0.7275 | 0.7376 | 1.5225 | 0.3051 | 1.8248 | 1.0549 | 141.22 | 148.0 |
| periodic_predictor_6h_v2 | 0.8878 | 0.9322 | 0.8730 | 0.8947 | 1.4981 | 0.2658 | 1.3157 | 0.7275 | 189.72 | 188.0 |

## Per-tier comparison
Mean `cpu_class_accuracy` by **volatility_tier**.

| tier | periodic_6h_v2 | periodic_predictor_6h_v2 |
|---|---:|---:|
| low | 0.7149 | 0.9043 |
| medium | 0.7296 | 0.8874 |
| high | 0.6819 | 0.8720 |

## Per-VM pairwise delta
Definitions: **`delta_cpu_acc`** = periodic_predictor CPU acc − periodic_6h CPU acc; **`delta_mem_acc`** = same for memory.

| Statistic | delta_cpu_acc | delta_mem_acc |
|---|---:|---:|
| Mean | 0.1792 | 0.1455 |
| Median | 0.1795 | 0.1316 |

| Rule | Count (of 100) |
|---|---:|
| `delta_cpu_acc > 0` | 85 |
| `delta_cpu_acc < -0.05` | 3 |
| `delta_mem_acc > 0` | 83 |
| `delta_mem_acc < -0.05` | 4 |

## Decisive checkpoint
Periodic reference means: CPU **0.708545**, memory **0.727501** (from Task 8 summary). Predictor-periodic means: CPU **0.887774**, memory **0.873024**.

- **Q1** (predictor mean CPU ≥ periodic mean CPU + **0.02**): **PASS** — gap **+0.1792** (0.8878 − 0.7085).
- **Q2** (predictor mean memory ≥ periodic mean memory + **0.02**): **PASS** — gap **+0.1455**.
- **Q3** (≥ **60** VMs with `delta_cpu_acc ≥ 0`): **PASS** — **85** VMs.
- **Q4** (high tier: predictor mean CPU &gt; periodic high tier **0.6819**): **PASS** — **0.8720** &gt; **0.6819**.
- **Q5** (“irrelevant” if **|mean delta_cpu_acc| &lt; 0.02**): **FAIL** — **|0.1792| ≥ 0.02**, so the gap is **not** negligible.

## Interpretation
**Q1 passes** with a large mean CPU lift (**~+18pp**) and **Q4 passes** on the high-volatility tier, so under this protocol the **RandomForest lookahead** is not substitutable by the **recent observation window** at the same calendar refresh rate: the predictor-sourced aggregate tracks the evaluation target materially better on this sample. **Q5 fails** precisely because that difference is **large**, so the “predictor is irrelevant” hypothesis is **not** supported by these numbers. This does **not** prescribe the next implementation (e.g. hybrid design); it only states that **predictor-grounded forecast content** is associated with substantially higher agreement with the Task 7/8 ground-truth definition here than observation-window aggregation at 6h refreshes.

## Figures
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_12/predictor_vs_window_per_vm.png` — per-VM CPU accuracy scatter (periodic *x*, predictor-periodic *y*) with **y = x** line.
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_12/predictor_vs_window_per_tier.png` — grouped bars of mean CPU accuracy by tier for both methods.

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v2.py` (modified — added `PeriodicPredictorPolicyV2` only)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_dynapm_v2.py` (modified — added `test_periodic_predictor_v2_runs`)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task12_periodic_predictor.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet` (created; 100 rows)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_periodic_predictor_diagnostics.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/periodic_predictor_6h_v2__<trace>__<vm_id>.json` (created; 100 files)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_12/predictor_vs_window_per_vm.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_12/predictor_vs_window_per_tier.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_12_predictor_source_ablation.md` (created)

`dynapm_v2.py` and `runner_v2.py` were **not** modified for this task.

## Test results
```text
============================== 11 passed in 3.18s ==============================
```
(collected: `test_dynapm_v2` × 4, `test_runner_v2` × 3, `test_forecasting` × 4)

## Assumptions
- Locked sample IDs and **`periodic_6h_v2`** metrics come from **`dynapm/data/eval_sample_v2_summary.parquet`** (Task 8).
- **Q1/Q2** thresholds use **+0.02** on **mean** accuracy vs the reference **mean** (not vs per-VM periodic).
- **Q5** uses **|**mean of per-VM `delta_cpu_acc`**|** &lt; 0.02** as “predictor irrelevant.”
- **`total_updates`** compares **logged events**; `PeriodicPredictorPolicyV2` appends **p95** rows every refresh while `PeriodicPolicyV2` uses `_commit_forecast`, so **absolute update counts are not strictly comparable** even when refresh **timestamps** align.

## Surprises
- **Mean `total_updates`** is higher for the predictor policy (**~190** vs **~141**) despite the same **6h** refresh intent, driven largely by **logging four field commits per refresh** (including both p95 fields each time), whereas **`_commit_forecast`** only logs **changed** fields.

## Commands

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_dynapm_v2.py dynapm/tests/test_runner_v2.py dynapm/tests/test_forecasting.py -v
# ... 11 passed in 3.18s
```

```bash
dynapm/.venv/bin/python dynapm/scripts/task12_periodic_predictor.py
# output snippet:
# "wall_seconds": 6.33,
# "mean_cpu_gap": 0.17923,
# "checkpoint": { "Q1": true, "Q2": true, "Q3": true, "Q4": true, "Q5": false }
```

## Next-step suggestion
Treat **`total_updates`** as **not directly comparable** across these two policies until logging is normalized (e.g. “refresh events” vs raw log rows); for any hybrid or DynAPM paper table, report both **refresh-aligned cost** and **metric deltas** so readers are not misled by differing `update_log` semantics while the schedule is the same.
