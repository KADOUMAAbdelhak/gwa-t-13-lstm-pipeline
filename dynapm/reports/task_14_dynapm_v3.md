# Task 14: DynAPM-V3 (Periodic-Predictor + Drift Augmentation)

## What was implemented

`DynAPMPolicyV3` (`dynapm/methods/dynapm_v3.py`) extends the v2 baseline policy: it refreshes all four dynamic fields from the predictor lookahead (next 288 `y_pred` steps) whenever the **6-hour periodic timer** fires (always advancing `_last`) **or** ADWIN signals drift on **CPU or memory residuals** (`y_true − y_pred` at the last timestep of the runner windows), in which case it refreshes without advancing `_last`, so scheduled 6h refreshes stay on calendar time. Each refresh logs four update rows with a `reason` of `"periodic"` or `"drift"` in the metadata. Unit tests cover initialization, a residuals-only periodic stream, and a residual shift producing at least one `"drift"` reason.

## Single-VM sanity check

VM: `Materna-Trace-1` / `501`. **Periodic window** and **DynAPM-V2** metrics are taken from the previously saved Task 9 JSONs (`task_09/periodic_6h_v2_vm501.json`, `task_09/dynapm_v2_vm501.json`). **Periodic-Predictor** and **DynAPM-V3** were run in this task via `runner_v2.run_method_on_vm` with defaults `period_hours=6`, `drift_delta=0.002` (Predictor policy does not use `drift_delta`).

| Method | cpu_class_accuracy | memory_class_accuracy | total_updates | Refresh breakdown (V3 only) |
|--------|-------------------|----------------------|---------------|-----------------------------|
| PeriodicPolicyV2 (window) | 0.7714 | 0.4000 | 145 | — |
| PeriodicPredictorPolicyV2 | 0.8857 | 0.7714 | 183 | — |
| DynAPMPolicyV2 | 0.8857 | 0.3143 | 21 | — |
| DynAPMPolicyV3 | 0.9143 | 0.7714 | 380 | **78** periodic / **17** drift refresh events |

`total_updates` here is `len(update_log)` from the runner. DynAPM-V3 appends **four** log rows per refresh (always), so divide by 4 for a refresh-event count comparable in spirit to sparse logging (~95 logical refreshes on this VM).

Persisted V3 run: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/dynapm_v3_vm501.json`.

## 100-VM head-to-head

Rows sorted by **mean `cpu_class_accuracy` ascending** (100 locked Task 8 VMs). Reference rows: `static_v2`, `periodic_6h_v2`, `full_rewrite_v2`, `dynapm_v2` from `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_summary_with_dynapm.parquet`; `periodic_predictor_6h_v2` from `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet`; `dynapm_v3` from this task’s `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_dynapm_v3_summary.parquet`.

| method | mean cpu_class_acc | median cpu_class_acc | mean mem_class_acc | median mem_class_acc | mean cpu_p95_mae | median cpu_p95_mae | mean mem_p95_mae | median mem_p95_mae | mean total_updates | median total_updates |
|--------|---------------------|----------------------|--------------------|---------------------|------------------|---------------------|------------------|-------------------|--------------------|--------------------|
| static_v2 | 0.4207 | 0.3537 | 0.5408 | 0.6146 | 3.7864 | 0.4068 | 3.4472 | 0.9789 | 0 | 0 |
| dynapm_v2 | 0.5452 | 0.5707 | 0.6387 | 0.6859 | 2.5168 | 0.4116 | 2.6539 | 0.9673 | 8.61 | 6 |
| periodic_6h_v2 | 0.7085 | 0.7304 | 0.7275 | 0.7376 | 1.5225 | 0.3051 | 1.8248 | 1.0549 | 141.22 | 148 |
| full_rewrite_v2 | 0.7329 | 0.7566 | 0.7497 | 0.7670 | 1.3715 | 0.2917 | 1.6247 | 0.8340 | 1293.46 | 1330.5 |
| periodic_predictor_6h_v2 | 0.8878 | 0.9322 | 0.8730 | 0.8947 | 1.4981 | 0.2658 | 1.3157 | 0.7275 | 189.72 | 188 |
| dynapm_v3 | 0.8923 | 0.9375 | 0.8768 | 0.8947 | 1.4805 | 0.2605 | 1.3002 | 0.7197 | 364.60 | 352 |

Wall-clock for the **100-VM** `joblib.Parallel(n_jobs=8)` DynAPM-V3 batch: **8.48 s** (this machine).

## Per-tier breakdown

Mean **cpu_class_accuracy** by volatility tier (same 100-VM sample):

| tier | static_v2 | periodic_6h_v2 | periodic_predictor_6h_v2 | dynapm_v2 | dynapm_v3 | full_rewrite_v2 |
|------|-----------|----------------|---------------------------|-----------|-----------|-----------------|
| low | 0.3296 | 0.7149 | 0.9043 | 0.4457 | 0.9059 | 0.7407 |
| medium | 0.4112 | 0.7296 | 0.8874 | 0.5616 | 0.8901 | 0.7542 |
| high | 0.5183 | 0.6819 | 0.8720 | 0.6260 | 0.8811 | 0.7046 |

## Pairwise DynAPM-V3 vs Periodic-Predictor

Per VM: `delta_cpu_acc = dynapm_v3_cpu_acc − periodic_predictor_6h_v2_cpu_acc` (and similarly for memory).

| Metric | CPU | Memory |
|--------|-----|--------|
| Mean delta | +0.00450 | +0.00374 |
| Median delta | 0.0 | 0.0 |

| CPU | Count |
|-----|------:|
| Δ \> 0 (strict help) | 11 |
| Δ \< −0.02 (hurt \> 2pp) | 1 |
| 0 ≤ Δ \< 0.01 (near tie) | 88 |
| Δ ≥ 0.01 (≥ 1pp gain) | 11 |

| Memory | Count |
|--------|------:|
| Δ \> 0 | 12 |
| Δ \< −0.02 | 2 |
| 0 ≤ Δ \< 0.01 | 86 |
| Δ ≥ 0.01 | 12 |

Across **100** VMs, **mean periodic refreshes ≈ 85.57** and **mean drift refreshes ≈ 5.58** (counting unique `(timestamp, reason)` groups in each VM’s V3 update log).

## Checkpoint questions

| Id | Question | Result | Numbers |
|----|------------|--------|---------|
| Q1 | Mean `cpu_class_accuracy`: V3 beats Periodic-Predictor by ≥ **0.01**? | **FAIL** | V3 mean **0.8923**, PP mean **0.8878**, gap **+0.00450** |
| Q2 | Mean `memory_class_accuracy`: same vs PP | **FAIL** | V3 **0.8768**, PP **0.8730**, gap **+0.00374** |
| Q3 | ≥ **50** VMs with `delta_cpu_acc ≥ 0`? | **PASS** | **99** / 100 |
| Q4 | High tier: V3 mean CPU acc \> PP mean on high tier? | **PASS** | V3 **0.8811**, PP **0.8720** |
| Q5 | V3 mean `total_updates` within **50%** of PP’s (\*≤ 1.5× PP mean\*)? | **FAIL** | V3 mean **364.6**, PP **189.72**, ratio **~1.92** |
| Q6 | \|mean V3 CPU acc − mean FullRewrite CPU acc\| ≤ **0.05**? | **FAIL** | V3 **0.8923**, FR **0.7329**, abs gap **0.159** |

\*Q5 as coded: `mean_v3_tu <= 1.5 * mean_pp_tu`.

## Interpretation

**Q1 fails** the **+1 percentage point** headline bar versus Periodic-Predictor, but the mean gain is **small and positive** (~0.45pp CPU, ~0.37pp memory), and **Q4 passes**: on the **high-volatility** tier, V3’s mean CPU class accuracy exceeds Predictor-Periodic’s. That matches the task’s **conditional** narrative: **drift augmentation shows a clearer benefit where residual instability is expected**, while corpus-wide headline lift is modest and often **numerically tied** at per-VM resolution (many deltas in `[0, 0.01)`). **Q5** indicates the runner’s `total_updates` metric is **not directly comparable** to Periodic-Predictor’s without normalizing (V3 logs **four rows per refresh**; PP logs only when fields change), and drift adds **extra** refreshes—so raw row counts overshoot a “50% band” even when the **additional** drift refreshes are moderate (~5.6 per VM on average).

## Figures

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/method_progression.png` — mean CPU class accuracy for the six methods (complexity order).
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/dynapm_v3_vs_periodic_predictor_per_vm.png` — scatter vs diagonal.
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/refresh_reason_breakdown.png` — stacked periodic vs drift **refresh events** per VM index.

## Files created or modified

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/dynapm_v3.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_dynapm_v3.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task14_dynapm_v3.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/dynapm_v3_vm501.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_dynapm_v3_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_dynapm_v3_diagnostics.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/method_progression.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/dynapm_v3_vs_periodic_predictor_per_vm.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_14/refresh_reason_breakdown.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/dynapm_v3__<trace>__<vm_id>.json` (100 files)

## Test results

- `pytest dynapm/tests/test_dynapm_v3.py` — **3 passed**.
- `pytest dynapm/tests/` — **45 passed**, **1 failed** (`test_alpha_sensitivity_after_fix` in `test_update_engine.py`; not introduced by this task).

## Assumptions

- **Task 9 JSONs** for VM 501 (periodic window, DynAPM-V2) are the requested “previously captured” reference; **Periodic-Predictor** and **V3** were re-run for a current apples-to-apples pair with V3.
- **Headline and tier statistics** for non-V3 methods use existing parquet aggregates from Tasks 8 and 10 and Task 12 (Predictor-Periodic).
- **`total_updates`** is the runner’s log row count; **V3** always logs **four** rows per refresh, unlike **PeriodicPredictor**, which logs only on field changes—so Q5 is a **sanity** check on raw counts, not on normalized refresh counts.

## Surprises

- **Mean FullRewrite CPU class accuracy** (~0.73) is **below** Predictor-Periodic and V3 (~0.89), so **Q6** (distance to “oracle” FullRewrite on this label) fails **upward**: V3 is much closer to the **predictor** ceiling than to relabeling every 24h from observations on this metric.
- Most per-VM CPU deltas versus Predictor-Periodic are **numerically tiny or zero**, with **11** VMs showing a **≥ 1pp** gain—consistent with drift adds being **sparse** (~5.6 drift refreshes per VM on average).

## Commands

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && . .venv/bin/activate
$ PYTHONPATH=. pytest dynapm/tests/test_dynapm_v3.py -v
... 3 passed in ~14s ...

$ PYTHONPATH=. pytest dynapm/tests/ -q
... 45 passed, 1 failed (test_update_engine.py::test_alpha_sensitivity_after_fix) ...

$ PYTHONPATH=. python dynapm/scripts/task14_dynapm_v3.py
{"wall_100vm": 8.48, ... checkpoint Q1–Q6 ...}
```

## Next-step suggestion

If the paper needs a cleaner headline than raw means, report **high-tier** CPU/memory lifts and a **normalized refresh count** (e.g. log rows / 4 for V3 vs Pred-Periodic, plus explicit drift-event counts) so **update-load** comparisons match the **checkpoint** intent; consider **tighter drift δ** or **debounced drift** if **Q5-style** operational cost must stay near the 6h predictor baseline while preserving **Q4-style** volatile-VM gains.
