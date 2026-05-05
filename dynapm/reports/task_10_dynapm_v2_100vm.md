# Task 10 DynAPM v2 100VM

## What was run
I loaded the locked Task 8 sample as unique `(trace, vm_id, volatility_tier)` tuples from `dynapm/data/eval_sample_v2_summary.parquet`, confirmed **100** unique VMs, and evaluated **DynAPM-V2** (`DynAPMPolicyV2`, `drift_delta=0.002`, `alpha=0.5`) via `runner_v2.run_method_on_vm` for each VM using `joblib.Parallel(n_jobs=8)`. Per-run JSON files were written under `dynapm/results/eval_sample_v2/`, and results were concatenated with the existing four baseline rows into `dynapm/data/eval_sample_v2_summary_with_dynapm.parquet` (**500** rows total). Aggregates, pairwise deltas vs Periodic-6h, checkpoints, memory diagnostics, and three figures were derived from that merged table.

## Per-method aggregate table
Mean and median over **100** VMs per method, sorted by **cpu_class_accuracy** mean (descending).

| method | cpu_acc mean | cpu_acc median | mem_acc mean | mem_acc median | cpu_p95_mae mean | cpu_p95_mae median | mem_p95_mae mean | mem_p95_mae median | total_updates mean | total_updates median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_rewrite_v2 | 0.7329 | 0.7566 | 0.7497 | 0.7670 | 1.3715 | 0.2917 | 1.6247 | 0.8340 | 1293.46 | 1330.50 |
| periodic_6h_v2 | 0.7085 | 0.7304 | 0.7275 | 0.7376 | 1.5225 | 0.3051 | 1.8248 | 1.0549 | 141.22 | 148.00 |
| reactive_v2 | 0.6158 | 0.6626 | 0.5814 | 0.5584 | 2.5012 | 0.4337 | 2.7079 | 1.3932 | 571.96 | 575.50 |
| dynapm_v2 | 0.5452 | 0.5707 | 0.6387 | 0.6859 | 2.5168 | 0.4116 | 2.6539 | 0.9673 | 8.61 | 6.00 |
| static_v2 | 0.4207 | 0.3537 | 0.5408 | 0.6146 | 3.7864 | 0.4068 | 3.4472 | 0.9789 | 0.00 | 0.00 |

## Per-tier breakdown
Mean **cpu_class_accuracy** and mean **total_updates** by volatility tier and method.

| tier | method | mean cpu_class_accuracy | mean total_updates |
|---|---|---:|---:|
| low | static_v2 | 0.3296 | 0.00 |
| low | dynapm_v2 | 0.4457 | 3.70 |
| low | reactive_v2 | 0.6876 | 558.12 |
| low | periodic_6h_v2 | 0.7149 | 115.27 |
| low | full_rewrite_v2 | 0.7407 | 892.52 |
| medium | static_v2 | 0.4112 | 0.00 |
| medium | dynapm_v2 | 0.5616 | 9.61 |
| medium | reactive_v2 | 0.5741 | 611.88 |
| medium | periodic_6h_v2 | 0.7296 | 152.21 |
| medium | full_rewrite_v2 | 0.7542 | 1444.15 |
| high | static_v2 | 0.5183 | 0.00 |
| high | reactive_v2 | 0.5865 | 546.65 |
| high | dynapm_v2 | 0.6260 | 12.41 |
| high | periodic_6h_v2 | 0.6819 | 155.74 |
| high | full_rewrite_v2 | 0.7046 | 1536.35 |

## Pairwise DynAPM vs Periodic-6h
Per-VM deltas: **DynAPM − Periodic** (negative MAE delta implies DynAPM is better on error).

| Statistic | delta_cpu_acc | delta_mem_acc | delta_cpu_p95_mae | delta_mem_p95_mae | delta_updates |
|---|---:|---:|---:|---:|---:|
| Mean | −0.1633 | −0.0888 | +0.9942 | +0.8290 | −132.61 |
| Median | −0.0417 | +0.0104 | +0.0112 | −0.0197 | −138.50 |

Counts (out of **100** VMs):

| Rule | Count |
|---|---:|
| `delta_cpu_acc > 0` (DynAPM wins CPU) | 36 |
| `delta_cpu_acc < −0.05` (DynAPM loses CPU by >5pp) | 47 |
| `delta_mem_acc > 0` (DynAPM wins memory) | 50 |
| `delta_mem_acc < −0.05` | 36 |

## Checkpoint questions
- **Q1** (mean CPU acc: DynAPM ≥ Periodic-6h): **FAIL** — mean gap (DynAPM − Periodic) = **−0.1633** (0.5452 vs 0.7085).
- **Q2** (DynAPM mean updates &lt; 50% of Periodic): **PASS** — 8.61 vs 141.22 → ratio **0.061** (&lt; 0.5).
- **Q3** (≥70 VMs with `dyn_cpu_acc ≥ periodic_cpu_acc − 0.05`): **FAIL** — **53** / 100 satisfy the bound.
- **Q4** (high tier: DynAPM mean CPU &gt; Periodic): **FAIL** — high-tier means **0.6260** (DynAPM) vs **0.6819** (Periodic); gap **−0.0559**.
- **Q5** (memory: DynAPM loses by ≤0.10 on average): **PASS** — Periodic mean − DynAPM mean = **0.0888** (0.7275 − 0.6387).

## Memory-class diagnostic (DynAPM-V2 only, *n* = 100)

**`n_memory_class_updates`:** min **0**, p25 **0.0**, median **0.5**, p75 **2.0**, max **7**.

**`n_cpu_class_updates`:** min **0**, p25 **0.0**, median **1.0**, p75 **1.25**, max **9**.

- VMs with `n_memory_class_updates == 0`: **50**
- VMs with `n_memory_class_updates < 2`: **71**

## Figures
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/method_comparison_5way.png` — mean CPU class accuracy, five methods, ±1 SE across VMs.
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/dynapm_vs_periodic_per_vm.png` — scatter Periodic vs DynAPM CPU accuracy with *y* = *x*.
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/dynapm_updates_vs_accuracy.png` — `total_updates` vs `cpu_class_accuracy`, color by method (all methods × all VMs).

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/dynapm_v2__<trace>__<vm_id>.json` (created; 100 files)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_summary_with_dynapm.parquet` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_with_dynapm_diagnostics.json` (created; auxiliary aggregates for this run)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/method_comparison_5way.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/dynapm_vs_periodic_per_vm.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10/dynapm_updates_vs_accuracy.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_10_dynapm_v2_100vm.md` (created)

No changes were made to `dynapm_v2.py`, `runner_v2.py`, or `baselines_v2.py`.

## Test results
Baseline regression tests (forecasting, `runner_v2`, `dynapm_v2`) still pass:

```text
..........                                                               [100%]
10 passed in 2.75s
```

## Assumptions
- The Task 8 sample is completely represented by unique VM keys in `eval_sample_v2_summary.parquet` (100 rows after `drop_duplicates` on trace/vm/tier).
- `method` label **`dynapm_v2`** matches the Task 10 JSON filename prefix and keeps the parquet schema aligned with baseline columns.
- Pairwise deltas join on `(trace, vm_id)` with unique rows per method.
- Standard error bars in the 5-way figure use the sample standard deviation of per-VM CPU accuracy divided by √100.

## Surprises
- On the **100-VM** sample, mean CPU class accuracy for DynAPM-V2 (**0.545**) is far below Periodic-6h (**0.709**), unlike the Trace-1/501 single-VM story—consistent with **very sparse** drift events (mean **8.6** updates per VM vs **141** for periodic).
- Half the VMs (**50**) recorded **zero** `n_memory_class_updates`, explaining why memory strategy at class level often never moves off the bootstrap forecast unless mem ADWIN fires.

## Commands

```bash
dynapm/.venv/bin/python - << 'PY'
# Loaded 100 VMs from eval_sample_v2_summary.parquet; ran DynAPM-V2 in parallel;
# merged to eval_sample_v2_summary_with_dynapm.parquet; wrote figures + diagnostics JSON
PY
# output snippet:
# { "wall_seconds": 8.045435791951604, "merged_rows": 500, "diag": ".../eval_sample_v2_with_dynapm_diagnostics.json" }
```

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_dynapm_v2.py dynapm/tests/test_runner_v2.py dynapm/tests/test_forecasting.py -q
# output:
# ..........                                                               [100%]
# 10 passed in 2.75s
```

## Next-step suggestion
Before adding a decision head, run a sensitivity pass on **`drift_delta`** (and optionally a light periodic safety refresh) strictly under `runner_v2`, because the current ADWIN settings appear to starve updates on many VMs—once mean update count moves into a plausible band vs periodic, re-check the same Q1/Q3/Q4 gates; if CPU remains low, pivot the mem/CPU drift logic (e.g., separate calibration or predictor-window definition) rather than layering learning on top of a near-static profile.
