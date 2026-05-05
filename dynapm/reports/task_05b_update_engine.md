# DynAPM Task 5B — Data-aware clock + update engine + baselines + 100-VM run

## Data-aware clock fix

Implemented `compute_data_aware_evaluation_timestamps(...)` in `dynapm/core/evaluation_clock.py` and kept `compute_evaluation_timestamps(...)` as the candidate-clock helper behavior.

### New function behavior

- Candidate eval timestamps: every 12h from `streaming_index[0]` to `streaming_index[-1] - 24h`
- For each candidate `t`, count rows in `[t, t+24h)`
- Drop if count `< min_observations_in_window` (set to 200 in ground-truth regeneration)
- Return `(kept_timestamps, info_dict)` where `info_dict` has:
  - `n_candidates`, `n_kept`, `n_dropped`, `dropped_fraction`

### Ground-truth regeneration with data-aware clock

Re-ran `dynapm/scripts/task04_ground_truth.py` (updated to use data-aware clock) and `dynapm/scripts/task04_ground_truth_stats.py`.

From `dynapm/data/ground_truth_run_meta.json`:

- `n_total`: **1576**
- `n_ok`: **1576**
- `n_failed`: **0**
- `wall_seconds`: **29.8151**
- `mean_eval_points`: **41.5419**
- `total_skipped_empty_windows`: **0**
- `n_candidates_total`: **66039**
- `n_kept_total`: **65470**
- `n_dropped_total`: **569**

From `dynapm/data/ground_truth_stats.json`:

- `n_points`: **65470** (reduced from 65933, as expected)

---

## Update engine implementation

Created:

- `dynapm/core/decision_head.py`
- `dynapm/core/churn.py`
- `dynapm/core/update_engine.py`

### Decision head

`UpdateDecisionHead` implemented with river online logistic regression and seeded first three decisions `(True, 1.0)` exactly as requested.

### Churn controller

`ChurnController` enforces:

- minimum inter-update interval
- categorical hysteresis over consecutive identical proposals
- bypasses hysteresis for continuous values (float)

### Update engine

`UpdateEngine` implements:

- drift-based trigger via ADWIN on residuals
- categorical update path with decision head + churn gating
- continuous update path via `step_continuous`
- update log

For continuous fields at eval timestamp `t`, runner supplies approximation:

- `predicted_*_p95_next_24h := empirical p95 of model y_pred over next 288 streaming points`

This approximation is documented and used consistently in runner + methods.

---

## Baseline implementations

Created:

- `dynapm/methods/__init__.py`
- `dynapm/methods/baselines.py`
- `dynapm/methods/dynapm.py`

Implemented baselines with common API (`step`, `step_continuous`):

- `StaticPolicy`
- `PeriodicPolicy` (`1h`, `6h`, `24h` in runner)
- `ReactiveThresholdPolicy`
- `FullRewritePolicy`

Implemented method factories:

- `dynapm_factory(alpha)`
- `dynapm_no_learner_factory(alpha)`
- `dynapm_no_churn_factory(alpha)`

---

## Evaluation runner

Created:

- `dynapm/eval/__init__.py`
- `dynapm/eval/runner.py`

`run_method_on_vm(trace, vm_id, method_factory, **kwargs)` performs:

- load CSV + split
- load thresholds/provisioning from `bootstrap_summary.parquet`
- initialize `ProfileInstance` (static + bootstrap-derived dynamic)
- load residual streams (`residuals_full`) and ground-truth series
- stream over timesteps:
  - `step()` each timestep
  - `step_continuous()` at evaluation timestamps
- snapshot/evaluate at eval timestamps
- return aggregate metrics + update counts + `skipped_eval_points` + `eval_density`

---

## Test results

### Task 5B/Regression tests run during implementation

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_evaluation_clock.py tests/test_update_engine.py tests/test_drift.py tests/test_profile_split.py tests/test_predictor.py -q
.........................                                                [100%]
25 passed in 5.39s
```

### Full suite final

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
............................                                             [100%]
28 passed in 5.34s
```

---

## 100-VM evaluation results

Script: `dynapm/scripts/task05_sample_eval.py`

- sample: 100 VMs (33/33/34 by trace, balanced by volatility tier within trace, seed 42)
- methods: 13
- persisted per-run JSON: `dynapm/results/eval_sample/<method>__<trace>__<vm_id>.json`
- summary parquet: `dynapm/data/eval_sample_summary.parquet`

Run metadata:

- wall-clock: **57.2016 s**
- rows in summary: **1300** (= 100 VMs × 13 methods)
- skipped eval points total: **0**

### Per-method CPU class accuracy (mean/median)

| method | mean | median |
|---|---:|---:|
| dynapm_alpha_0.1 | 0.3652 | 0.2906 |
| dynapm_alpha_0.3 | 0.3652 | 0.2906 |
| dynapm_alpha_0.5 | 0.3652 | 0.2906 |
| dynapm_alpha_0.7 | 0.3652 | 0.2906 |
| dynapm_alpha_0.9 | 0.3652 | 0.2906 |
| dynapm_no_churn_alpha_0.5 | 0.3815 | 0.3021 |
| dynapm_no_learner_alpha_0.5 | 0.3652 | 0.2906 |
| full_rewrite | 0.4102 | 0.3164 |
| periodic_1h | 0.4039 | 0.3101 |
| periodic_24h | 0.3990 | 0.3418 |
| periodic_6h | 0.4044 | 0.3164 |
| reactive_threshold | 0.3710 | 0.2518 |
| static | 0.3574 | 0.2375 |

### Per-method memory class accuracy (mean/median)

| method | mean | median |
|---|---:|---:|
| dynapm_alpha_0.1 | 0.2587 | 0.0507 |
| dynapm_alpha_0.3 | 0.2587 | 0.0507 |
| dynapm_alpha_0.5 | 0.2587 | 0.0507 |
| dynapm_alpha_0.7 | 0.2587 | 0.0507 |
| dynapm_alpha_0.9 | 0.2587 | 0.0507 |
| dynapm_no_churn_alpha_0.5 | 0.2865 | 0.1579 |
| dynapm_no_learner_alpha_0.5 | 0.2587 | 0.0507 |
| full_rewrite | 0.3354 | 0.2632 |
| periodic_1h | 0.3362 | 0.2632 |
| periodic_24h | 0.3120 | 0.2470 |
| periodic_6h | 0.3192 | 0.2500 |
| reactive_threshold | 0.1643 | 0.0000 |
| static | 0.2438 | 0.0417 |

### Per-method total updates (mean/median)

| method | mean | median |
|---|---:|---:|
| dynapm_alpha_0.1 | 81.55 | 82 |
| dynapm_alpha_0.3 | 81.55 | 82 |
| dynapm_alpha_0.5 | 81.55 | 82 |
| dynapm_alpha_0.7 | 81.55 | 82 |
| dynapm_alpha_0.9 | 81.55 | 82 |
| dynapm_no_churn_alpha_0.5 | 84.44 | 83 |
| dynapm_no_learner_alpha_0.5 | 81.56 | 82 |
| full_rewrite | 6588.26 | 6783 |
| periodic_1h | 525.71 | 552 |
| periodic_24h | 24.71 | 26 |
| periodic_6h | 97.65 | 105 |
| reactive_threshold | 82.53 | 83 |
| static | 0.00 | 0 |

### Per-method CPU p95 MAE (mean/median)

| method | mean | median |
|---|---:|---:|
| dynapm_alpha_0.1 | 1.3968 | 0.2405 |
| dynapm_alpha_0.3 | 1.3968 | 0.2405 |
| dynapm_alpha_0.5 | 1.3968 | 0.2405 |
| dynapm_alpha_0.7 | 1.3968 | 0.2405 |
| dynapm_alpha_0.9 | 1.3968 | 0.2405 |
| dynapm_no_churn_alpha_0.5 | 1.3968 | 0.2405 |
| dynapm_no_learner_alpha_0.5 | 1.3968 | 0.2405 |
| full_rewrite | 1.3968 | 0.2405 |
| periodic_1h | 3.1214 | 0.4072 |
| periodic_24h | 3.1214 | 0.4072 |
| periodic_6h | 3.1214 | 0.4072 |
| reactive_threshold | 1.3968 | 0.2405 |
| static | 3.1214 | 0.4072 |

### Per-method memory p95 MAE (mean/median)

| method | mean | median |
|---|---:|---:|
| dynapm_alpha_0.1 | 1.2111 | 0.6390 |
| dynapm_alpha_0.3 | 1.2111 | 0.6390 |
| dynapm_alpha_0.5 | 1.2111 | 0.6390 |
| dynapm_alpha_0.7 | 1.2111 | 0.6390 |
| dynapm_alpha_0.9 | 1.2111 | 0.6390 |
| dynapm_no_churn_alpha_0.5 | 1.2111 | 0.6390 |
| dynapm_no_learner_alpha_0.5 | 1.2111 | 0.6390 |
| full_rewrite | 1.2111 | 0.6390 |
| periodic_1h | 3.4834 | 0.9721 |
| periodic_24h | 3.4834 | 0.9721 |
| periodic_6h | 3.4834 | 0.9721 |
| reactive_threshold | 1.2111 | 0.6390 |
| static | 3.4834 | 0.9721 |

---

## DynAPM vs Static, by tier (headline)

Using `dynapm_alpha_0.5` vs `static` (means by tier):

- **Low volatility**: CPU acc 0.4144 vs 0.4299 (worse), Memory acc 0.3094 vs 0.3063 (slightly better)
- **Medium volatility**: CPU acc 0.3195 vs 0.3059 (better), Memory acc 0.2466 vs 0.2285 (better)
- **High volatility**: CPU acc 0.3619 vs 0.3371 (better), Memory acc 0.2214 vs 0.1978 (better)

Overall: DynAPM improves most in medium/high tiers, mixed in low tier.

---

## Decision-head ablation

Compare `dynapm_alpha_0.5` vs `dynapm_no_learner_alpha_0.5`:

- CPU acc: **identical** (0.3652 vs 0.3652)
- Memory acc: **identical** (0.2587 vs 0.2587)
- Updates: near-identical (81.55 vs 81.56)

No measurable effect in this run.

---

## Churn-controller ablation

Compare `dynapm_alpha_0.5` vs `dynapm_no_churn_alpha_0.5`:

- CPU acc: improves 0.3652 → **0.3815** without churn control
- Memory acc: improves 0.2587 → **0.2865** without churn control
- Updates: increases 81.55 → **84.44** without churn control

Observed trade-off: better accuracy at modestly higher churn.

---

## α sweep

Observed sweep (`dynapm_alpha_0.1`..`0.9`) produced identical metrics in this implementation/run (CPU acc, memory acc, updates, MAEs all equal to shown DynAPM row). This indicates the current online head path did not materially change commit decisions across alpha settings for this sample.

---

## Figures referenced

- `dynapm/reports/task_05/alpha_pareto.png`
- `dynapm/reports/task_05/dynapm_vs_static_delta.png`

---

## Files created or modified (absolute paths)

### Modified

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/evaluation_clock.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_evaluation_clock.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth.py`

### Created

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/decision_head.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/churn.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/update_engine.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/__init__.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/dynapm.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/__init__.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_update_engine.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task05_sample_eval.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05b_update_engine.md`

### Generated outputs

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_diagnostics.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05/alpha_pareto.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05/dynapm_vs_static_delta.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample/*.json` (1300 files)

---

## Assumptions

- Data-aware clock minimum coverage threshold set to `min_observations_in_window=200` as specified.
- For continuous-field update in runner, p95 approximation uses `y_pred` next 288 points from residual-stream predictor outputs.
- Ground truth is authoritative at data-aware eval timestamps (no interpolation).
- Volatility tiers are based on bootstrap CPU std tertiles recomputed over kept VMs.

---

## Surprises

- `n_points` was briefly observed stale (parallel stats call before regeneration finished); sequential re-run confirmed correct `65470`.
- Alpha sweep showed no sensitivity in this run (all DynAPM-alpha variants identical), suggesting head/training signal currently insufficient to separate decisions.
- Some baselines (`full_rewrite`, `periodic`) outperform DynAPM in aggregate class accuracy on this sample despite much higher churn.

---

## Commands (output truncated to <=20 lines)

```text
$ cd .../dynapm && .venv/bin/pytest tests/test_evaluation_clock.py tests/test_update_engine.py tests/test_drift.py tests/test_profile_split.py tests/test_predictor.py -q
.........................                                                [100%]
25 passed in 5.39s
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py
{
  "n_total": 1576,
  "n_ok": 1576,
  "n_failed": 0,
  "wall_seconds": 29.815128749934956,
  "mean_eval_points": 41.54187817258883,
  "total_skipped_empty_windows": 0,
  "n_candidates_total": 66039,
  "n_kept_total": 65470,
  "n_dropped_total": 569
}
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth_stats.py
{
  "stats_json": ".../dynapm/data/ground_truth_stats.json",
  "n_points": 65470
}
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task05_sample_eval.py
{
  "summary_parquet": ".../dynapm/data/eval_sample_summary.parquet",
  "diag_json": ".../dynapm/data/eval_sample_diagnostics.json",
  "wall_seconds": 57.20155250001699,
  "n_rows": 1300
}
```

```text
$ cd .../dynapm && .venv/bin/pytest tests/ -q
............................                                             [100%]
28 passed in 5.34s
```

```text
$ dynapm/.venv/bin/python - << 'PY' ... (summary extraction)
rows 1300 methods 13 vms 100
skipped_eval_points_total 0
... per-method table dumps ...
```

---

## Next-step suggestion

Before Task 6, calibrate the decision-head signal path (feature scaling, negative-label generation, and alpha sensitivity checks) so the α sweep becomes behaviorally distinct; then run a targeted sensitivity study on churn-vs-accuracy trade-offs per tier to pick a stable default DynAPM operating point.
