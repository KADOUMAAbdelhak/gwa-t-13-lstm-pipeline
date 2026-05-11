# DynAPM Task 17b — provisioning recommendation downstream eval

## What was done

Forecast p95 recommendations use `capacity = forecast_p95 * safety_factor` with `OVER_THRESHOLD=1.5` to flag waste. Corpus JSON lacks per-eval rows, so **`collect_verified_p95_pairs`** re-simulates each policy (`runner_v2`-identical stepping) once, joins GT `p95_*_next_24h`, persists `evaluate_provisioning` outcomes (`provisioning_eval_records.parquet`). Aggregates average eval-level rates inside each VM, then average across VMs (two-stage).

## Per-eval record availability

Existing corpus JSON omit per-eval payloads (only headline metrics): **replayed the runner_v2 stream inside `dynapm/eval/provisioning.collect_verified_p95_pairs`** rather than patching `runner_v2` (~20 min wall with 8 workers, one-time parquet cache). This is cheaper than a full runner format migration and reproducible offline.

## Per-method aggregate table
(Sorted by ascending `mean_cpu_under_rate`; **safety_factor=1.2**.)

| method | mean_cpu_under_rate | mean_cpu_over_rate | mean_cpu_well_rate | mean_cpu_provisioning_mae | mean_cpu_excess_when_over | mean_mem_under_rate | mean_mem_over_rate | mean_mem_well_rate | mean_mem_provisioning_mae | mean_mem_excess_when_over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dynapm_v3_v2 | 0.06113065 | 0.05732293 | 0.88154642 | 1.7893259 | 5.16673679 | 0.06991679 | 0.0178555 | 0.91222772 | 2.13799676 | 7.29854025 |
| periodic_predictor_6h_v2 | 0.06128626 | 0.06044954 | 0.8782642 | 1.8144606 | 5.3053297 | 0.07008894 | 0.01938884 | 0.91052221 | 2.16505127 | 7.80587903 |
| full_rewrite_v2 | 0.07512228 | 0.06113127 | 0.86374645 | 2.32612501 | 11.16696621 | 0.0663927 | 0.05045733 | 0.88314997 | 3.31699927 | 13.00596189 |
| periodic_6h_v2 | 0.08522219 | 0.06444185 | 0.85033595 | 2.41058932 | 11.28818256 | 0.08554197 | 0.05135854 | 0.86309948 | 3.4233083 | 13.7457389 |
| static_v2 | 0.10250519 | 0.15984525 | 0.73764956 | 3.64877099 | 12.38547703 | 0.0805421 | 0.11334272 | 0.80611518 | 4.42728259 | 15.52838677 |
| reactive_v2 | 0.16367618 | 0.06756353 | 0.76876029 | 2.87310141 | 11.19710453 | 0.1608412 | 0.06772562 | 0.77143318 | 3.54419729 | 11.01473295 |

## Pairwise deltas with bootstrap CI
`n_bootstrap=1000`, paired VMs (inner join on `trace`,`vm_id`).

### Periodic-Predictor minus Periodic-window
| metric | mean Δ | 95% bootstrap CI | Wilcoxon p |
|---|---:|---|---:|
| delta_cpu_under_rate | -0.023936 | [-0.030547, -0.016693] | 7.057e-62 |
| delta_mem_under_rate | -0.015453 | [-0.022482, -0.008054] | 1.858e-57 |
| delta_cpu_over_rate | -0.003992 | [-0.010862, 0.002831] | 2.527e-09 |
| delta_mem_over_rate | -0.031970 | [-0.036715, -0.027765] | 4.771e-83 |
| delta_cpu_provisioning_mae | -0.596129 | [-0.740731, -0.449824] | 8.899e-67 |

### DynAPM-V3 minus Periodic-Predictor
| metric | mean Δ | 95% bootstrap CI | Wilcoxon p |
|---|---:|---|---:|
| delta_cpu_under_rate | -0.000156 | [-0.000435, 0.000107] | 0.2072 |
| delta_mem_under_rate | -0.000172 | [-0.000410, 0.000050] | 0.2333 |
| delta_cpu_over_rate | -0.003127 | [-0.003615, -0.002609] | 1.242e-26 |
| delta_mem_over_rate | -0.001533 | [-0.001954, -0.001123] | 9.285e-13 |
| delta_cpu_provisioning_mae | -0.025135 | [-0.031748, -0.019337] | 1.218e-61 |

## Per-tier breakdown

| method | volatility_tier | mean_cpu_under_rate | mean_cpu_provisioning_mae |
| --- | --- | --- | --- |
| static_v2 | medium | 0.11965828 | 1.70609404 |
| static_v2 | low | 0.05294637 | 0.53173697 |
| static_v2 | high | 0.13484931 | 8.69886272 |
| periodic_6h_v2 | medium | 0.07936255 | 1.40538867 |
| periodic_6h_v2 | low | 0.02830881 | 0.39645878 |
| periodic_6h_v2 | high | 0.14787588 | 5.42418034 |
| reactive_v2 | medium | 0.141184 | 1.45167528 |
| reactive_v2 | low | 0.05437955 | 0.39609163 |
| reactive_v2 | high | 0.29521443 | 6.76412583 |
| full_rewrite_v2 | medium | 0.06898938 | 1.38288372 |
| full_rewrite_v2 | low | 0.02263688 | 0.38375262 |
| full_rewrite_v2 | high | 0.13362913 | 5.20625273 |
| periodic_predictor_6h_v2 | medium | 0.0623095 | 1.41377233 |
| periodic_predictor_6h_v2 | low | 0.03070731 | 0.49444287 |
| periodic_predictor_6h_v2 | high | 0.09078578 | 3.5318953 |
| dynapm_v3_v2 | medium | 0.0623558 | 1.40138557 |
| dynapm_v3_v2 | low | 0.03080079 | 0.49335963 |
| dynapm_v3_v2 | high | 0.09018003 | 3.47003115 |

## Safety-factor sensitivity

(Under/over rates recomputed analytically from stored forecasts/GT — no replay.)

| safety_factor | method | mean_cpu_under_rate | mean_cpu_over_rate |
| --- | --- | --- | --- |
| 1.1 | periodic_predictor_6h_v2 | 0.11502639 | 0.04700724 |
| 1.1 | dynapm_v3_v2 | 0.11549527 | 0.04431023 |
| 1.1 | full_rewrite_v2 | 0.11577017 | 0.04671187 |
| 1.1 | periodic_6h_v2 | 0.12882523 | 0.04969856 |
| 1.1 | static_v2 | 0.15330526 | 0.13238587 |
| 1.1 | reactive_v2 | 0.22371583 | 0.05217897 |
| 1.2 | dynapm_v3_v2 | 0.06113065 | 0.05732293 |
| 1.2 | periodic_predictor_6h_v2 | 0.06128626 | 0.06044954 |
| 1.2 | full_rewrite_v2 | 0.07512228 | 0.06113127 |
| 1.2 | periodic_6h_v2 | 0.08522219 | 0.06444185 |
| 1.2 | static_v2 | 0.10250519 | 0.15984525 |
| 1.2 | reactive_v2 | 0.16367618 | 0.06756353 |
| 1.5 | dynapm_v3_v2 | 0.02844161 | 0.31488314 |
| 1.5 | periodic_predictor_6h_v2 | 0.02858162 | 0.31789987 |
| 1.5 | full_rewrite_v2 | 0.03912966 | 0.40069466 |
| 1.5 | periodic_6h_v2 | 0.04616459 | 0.39769397 |
| 1.5 | static_v2 | 0.05818685 | 0.51784279 |
| 1.5 | reactive_v2 | 0.09898519 | 0.34191392 |

## Checkpoint questions

| Q | Result | Evidence |
|---|---|---|
| Q1 PP vs window ≥5 pp CPU under-rate gain | FAIL | PW=0.0852, PP=0.0613, Δ=0.0239 |
| Q2 CI excludes zero (PP−PW Δcpu_under) | PASS | mean=-0.0239, CI=[-0.030698,-0.016770] |
| Q3 DynAPM-V3 under ≤ Periodic-Predictor | PASS | V3=0.0611, PP=0.0613 |
| Q4 Rank match under vs cpu_p95_mae | FAIL | Spearman ρ≈**0.9429** |
| Q5 Q1 robust at SF 1.1 & 1.5 | FAIL | Δ@1.1=0.0138, Δ@1.5=0.0176 |

## Figures

- `dynapm/reports/task_17b/provisioning_outcomes_per_method.png`
- `dynapm/reports/task_17b/under_provisioning_rate_per_method.png`
- `dynapm/reports/task_17b/provisioning_mae_vs_p95_mae.png`
- `dynapm/reports/task_17b/under_rate_per_tier.png`

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/provisioning.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_provisioning.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task17b_provisioning.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/provisioning_eval_records.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b/provisioning_outcomes_per_method.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b/under_provisioning_rate_per_method.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b/provisioning_mae_vs_p95_mae.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b/under_rate_per_tier.png`

## Test results

`pytest dynapm/tests/test_provisioning.py -q` → **6 passed**.

## Assumptions

- Evaluation points with non-finite forecast or actual **p95 readings are omitted** for that modality’s provisioning flags (pandas `mean(skipna)` within VM).
- `actual_p95≈0` uses the same inequalities; no epsilon floor inserted.

## Surprises

- The periodic **window** baseline shows **0.0239** higher CPU under-rate than Periodic-Predictor (~**2.39** percentage points)—substantial, yet shy of the **0.05** absolute Q1 target.
- DynAPM-V3 trims mean CPU under-rate vs Periodic-Predictor by **0.0002** with paired Wilcoxon **p≈0.2072** (not significant even though provisioning MAE moves sharply).
- Ranking mean CPU under-rate vs Task 15 `cpu_p95_mae` shows Spearman ρ≈**0.9429** instead of a perfect match because `reactive_v2` keeps moderate MAE yet exhibits the worst under-provisioning.
- Varying safety factors {1.1, 1.2, 1.5} keeps predictor-first orderings but still fails the 5 pp PP-vs-window screen—scalar safety multipliers alone do not close the SLA gap articulated in Q1/Q5.

## Commands

```bash
PYTHONPATH=. python dynapm/scripts/task17b_provisioning.py --n-jobs 8
```
`--force-collect` rebuilds `provisioning_eval_records.parquet`; otherwise the script reloads the parquet and recomputes aggregates only. **Console diagnostics are truncated here for brevity.**

## Next-step suggestion

If SLA risk dominates cost, prioritize reducing **CPU under-rate** tied to predictor-source gains; if infra spend dominates, tighten `OVER_THRESHOLD` or learn safety factors per workload class (still without touching forecasting policies in this scaffold).
