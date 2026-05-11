# Task 11 Drift Delta Sweep

## What was run
On the same **100** unique VMs as Task 8 (from `dynapm/data/eval_sample_v2_summary.parquet`), I re-ran `DynAPMPolicyV2` under `runner_v2` for **five** new `drift_delta` values (`0.0001`, `0.0005`, `0.001`, `0.005`, `0.01`), excluding the Task 10 default `0.002`. Each configuration produced **100** runs (**500** runs total) with `joblib.Parallel(n_jobs=8)`. Outputs were written as `dynapm_v2_delta_<delta>__<trace>__<vm_id>.json`, aggregated to `dynapm/data/eval_sample_v2_dynapm_delta_sweep.parquet`, and cross-checked against reference rows from `dynapm/data/eval_sample_v2_summary_with_dynapm.parquet`. Aggregates, checkpoints, and figures were saved via `dynapm/scripts/task11_delta_sweep.py` to `dynapm/data/eval_sample_v2_delta_sweep_diagnostics.json`. Wall-clock for the 500 runs was **35.63 s**.

## Per-delta aggregate table
Mean and median across **100** VMs. Rows include **five sweep deltas**, **Task 10 DynAPM** (`dynapm_v2`, **δ=0.002**), and **three baselines** ( **`periodic_6h_v2`**, **`full_rewrite_v2`**, **`static_v2`** ). Sorted by **mean `total_updates`** ascending (per spec).

| method | drift_delta | cpu_acc mean | cpu_acc med | mem_acc mean | mem_acc med | cpu_p95_mae mean | mem_p95_mae mean | upd mean | upd med | n_cpu_cls mean | n_mem_cls mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_v2 | — | 0.4207 | 0.3537 | 0.5408 | 0.6146 | 3.7864 | 3.4472 | 0.00 | 0.0 | 0.00 | 0.00 |
| dynapm_v2_delta_0.0001 | 0.0001 | 0.5322 | 0.5521 | 0.6093 | 0.6842 | 2.5570 | 2.8178 | 6.35 | 4.0 | 0.88 | 0.86 |
| dynapm_v2_delta_0.0005 | 0.0005 | 0.5415 | 0.5576 | 0.6182 | 0.6842 | 2.5510 | 2.7657 | 7.55 | 4.5 | 1.00 | 1.01 |
| dynapm_v2_delta_0.001 | 0.001 | 0.5432 | 0.5707 | 0.6305 | 0.6842 | 2.5256 | 2.6905 | 8.17 | 6.0 | 1.11 | 1.11 |
| dynapm_v2 | 0.002 | 0.5452 | 0.5707 | 0.6387 | 0.6859 | 2.5168 | 2.6539 | 8.61 | 6.0 | 1.15 | 1.16 |
| dynapm_v2_delta_0.005 | 0.005 | 0.5590 | 0.6020 | 0.6431 | 0.7083 | 2.4446 | 2.5731 | 10.02 | 7.0 | 1.25 | 1.37 |
| dynapm_v2_delta_0.01 | 0.01 | 0.5560 | 0.6146 | 0.6483 | 0.7198 | 2.4222 | 2.5347 | 11.26 | 8.0 | 1.41 | 1.45 |
| periodic_6h_v2 | — | 0.7085 | 0.7304 | 0.7275 | 0.7376 | 1.5225 | 1.8248 | 141.22 | 148.0 | 9.92 | 10.96 |
| full_rewrite_v2 | — | 0.7329 | 0.7566 | 0.7497 | 0.7670 | 1.3715 | 1.6247 | 1293.46 | 1330.5 | 23.25 | 41.14 |

**Periodic-6h reference (Task 8):** mean `cpu_class_accuracy` **0.708545**, mean `total_updates` **141.22**.

## Pareto analysis
**(a) Highest mean `cpu_class_accuracy` (among sweep + references in this table):** **`full_rewrite_v2`** at **0.7329** mean CPU class accuracy (mean updates **1293.46**).

**(b) DynAPM `drift_delta` sweep vs Periodic-6h:**  
For each sweep delta, `(accuracy_gain_vs_periodic, update_ratio_vs_periodic) = (mean_cpu_acc − 0.708545, mean_total_updates / 141.22)`:

| delta | mean CPU acc | mean updates | acc gain vs Periodic | update ratio vs Periodic |
|---|---:|---:|---:|---:|
| 0.0001 | 0.5322 | 6.35 | −0.1763 | 0.0450 |
| 0.0005 | 0.5415 | 7.55 | −0.1671 | 0.0535 |
| 0.001 | 0.5432 | 8.17 | −0.1653 | 0.0579 |
| 0.005 | 0.5590 | 10.02 | −0.1495 | 0.0710 |
| 0.01 | 0.5560 | 11.26 | −0.1526 | 0.0797 |
| periodic (ref) | 0.7085 | 141.22 | 0.0000 | 1.0000 |

- **Strict “under budget” branch:** Among deltas with **update_ratio &lt; 1.0** (all sweep points), **δ=0.005** has the **highest** mean CPU accuracy (**0.5590** at **10.02** updates, ratio **0.071** vs Periodic).
- **“Near Periodic accuracy” branch:** No sweep point reaches Periodic−0.02 CPU accuracy (see Q1), so **`pareto_lowest_update_ratio_acc_within_002` is null** — there is no delta with mean CPU within **0.02** of Periodic’s mean at the lower end of the churn frontier in this run.

Within the sweep only, **δ=0.005** Pareto-dominates **δ=0.01** on this sample (higher mean CPU, lower mean updates).

## Per-tier breakdown for best sweep delta (`δ=0.005`)
Best sweep delta = highest mean CPU accuracy among the five new deltas (**δ=0.005**). Mean `cpu_class_accuracy` by tier vs Periodic-6h (from merged reference file).

| tier | DynAPM δ=0.005 | Periodic-6h |
|---|---:|---:|
| low | 0.4510 | 0.7149 |
| medium | 0.5778 | 0.7296 |
| high | 0.6456 | 0.6819 |

## Pairwise DynAPM vs Periodic-6h
This subtask is largely subsumed by the sweep table above. **Across the five tested deltas, none reaches Periodic’s mean CPU accuracy;** best gap to Periodic mean CPU is **−0.1495** at **δ=0.005** (still **~15pp** below Periodic on mean CPU accuracy).

## Checkpoint questions
Reference Periodic mean CPU: **0.708545**; reference mean updates: **141.22**. Best sweep mean CPU: **0.5590** at **δ=0.005**.

- **Q1:** MATCH threshold: any sweep delta has mean CPU **≥ 0.708545 − 0.02 = 0.688545** — **FAIL** (best sweep **0.5590**).
- **Q2:** Any sweep delta has mean CPU **>** **0.708545** — **FAIL**.
- **Q3:** With deltas ordered **decreasing** (`0.01 → 0.005 → 0.001 → 0.0005 → 0.0001`), mean updates are **11.26 → 10.02 → 8.17 → 7.55 → 6.35**. This sequence is **not** non-decreasing (moving to smaller `delta` reduces mean updates stepwise), so the monotonicity sanity check **FAIL** on this corpus sample.
- **Q4:** “Comparable to Periodic” interpreted as mean updates in **[0.5×141.22, 1.5×141.22] = [70.61, 211.83]** — **FAIL** (largest sweep mean updates **11.26**).
- **Q5:** For best-overall-accuracy sweep delta (**0.005**), high-tier mean CPU **0.6456** vs Periodic high-tier **0.6819** — **FAIL** (below **0.6819**).

## Figures
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_11/drift_delta_pareto.png` — mean updates (log *x*) vs mean CPU accuracy; sweep points annotated; Periodic, FullRewrite, Static references.
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_11/drift_delta_per_tier.png` — grouped bars of mean CPU accuracy by `drift_delta` and tier; dashed horizontal reference lines at Periodic-6h per-tier means.

## Interpretation
Across the five `drift_delta` values on this 100-VM sample, mean CPU class accuracy stays roughly **0.53–0.56**, far from Periodic-6h’s **~0.709**, while mean updates remain on the order of **6–11** per VM versus **~141** for Periodic. Tuning `delta` within this band does not make drift-only DynAPM-V2 match Periodic by the Task 11 “near match” criterion, and the empirical relation between smaller `delta` and higher update count is **inverted** here (smaller `delta` yields **fewer** mean updates in this run). The scatter / pareto view therefore shows DynAPM sweep points clustered at **low churn and low CPU accuracy** relative to Periodic, rather than a competitive accuracy–churn tradeoff near Periodic’s operating point. Whether that implies adding non–drift-only mechanisms is a design choice outside this sweep; the numbers only show that **within this pure drift-triggered design and delta range**, the sample does not exhibit Periodic-matching CPU accuracy.

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task11_delta_sweep.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_dynapm_delta_sweep.parquet` (created; 500 rows)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_delta_sweep_diagnostics.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/dynapm_v2_delta_<delta>__<trace>__<vm_id>.json` (created; 500 files)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_11/drift_delta_pareto.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_11/drift_delta_per_tier.png` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_11_drift_delta_sweep.md` (created)

No changes were made to `dynapm_v2.py`, `runner_v2.py`, or `baselines_v2.py`.

## Test results
```text
..........                                                               [100%]
10 passed in 2.90s
```
(Tests: `test_dynapm_v2`, `test_runner_v2`, `test_forecasting`.)

## Assumptions
- The locked sample is the unique `(trace, vm_id, volatility_tier)` rows from `eval_sample_v2_summary.parquet` (**100** VMs).
- Reference metrics for Periodic / Static / FullRewrite / Task 10 DynAPM are taken from `eval_sample_v2_summary_with_dynapm.parquet`.
- Q4 “within 50% of 141.22” is interpreted as mean updates lying in **[70.61, 211.83]** (±50% of Periodic’s mean updates).
- Q5 compares the **high-tier** mean CPU accuracy for the **best sweep delta by overall mean CPU** (**0.005**) against the cited Periodic high-tier mean **0.6819**.

## Surprises
- **Mean total_updates increased with larger `drift_delta`** in this run (e.g. **6.35** at `0.0001` vs **11.26** at `0.01`), so Q3’s monotonicity expectation (more updates as `delta` decreases) does not hold on this sample.
- The best sweep accuracy remained **~0.56**, only modestly above Task 10’s **~0.545** at `δ=0.002`, suggesting **delta tuning alone** does not unlock Periodic-like CPU accuracy here.

## Commands

```bash
dynapm/.venv/bin/python dynapm/scripts/task11_delta_sweep.py
# output snippet:
# "wall_seconds": 35.63, "best_delta": 0.005, "Q1": false, ...
```

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_dynapm_v2.py dynapm/tests/test_runner_v2.py dynapm/tests/test_forecasting.py -q
# output:
# ..........                                                               [100%]
# 10 passed in 2.90s
```

## Next-step suggestion
If the goal is to approach Periodic’s CPU accuracy without orders-of-magnitude more updates, the next experiments should isolate **why** smaller ADWIN `delta` associates with **fewer** commits on this sample (e.g., residual scale, detector dynamics, or lookahead alignment) before layering a periodic safety net; otherwise hybrid designs risk tuning around an unclear drift trigger rather than a measured failure mode.
