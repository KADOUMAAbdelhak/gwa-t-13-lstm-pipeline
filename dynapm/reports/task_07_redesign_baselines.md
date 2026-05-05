# Task 07 Redesign Baselines

## What changed
This task introduces a forecast-based evaluation protocol where profile snapshots represent explicit next-24h forecasts rather than current-state labels. I preserved the prior runner and baseline implementations as v1 copies, added canonical window-forecast utilities, implemented four forecast-based baseline policies with a unified window API, and created a new snapshot-and-verify runner (`runner_v2`) that defers verification until each snapshot has aged by 24h. I then validated the new stack with focused unit tests and a single-VM sanity run on `Materna-Trace-1/501`.

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v1.py` (created; copy of current runner)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v1.py` (created; copy of current baselines)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/forecasting.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_forecasting.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v2.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v2.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_runner_v2.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_07_single_vm_results.json` (created; sanity results)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_07_v1_compare_vm501.json` (created; v1 comparison extract)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_07_redesign_baselines.md` (created)

## Test results
Full pytest output:

```text
.......                                                                  [100%]
7 passed in 2.20s
```

## Sanity-check single-VM results
VM: `Materna-Trace-1/501` via `runner_v2`.

| method | cpu_class_accuracy | memory_class_accuracy | cpu_p95_mae | mem_p95_mae | total_updates | n_eval_points_verified | snapshots_dropped_unverified |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_v2 | 0.0571 | 0.3714 | 1.2556 | 2.6150 | 0 | 35 | 0 |
| periodic_6h_v2 | 0.7714 | 0.4000 | 0.4314 | 2.9097 | 145 | 35 | 0 |
| reactive_v2 | 0.3714 | 0.2286 | 0.7501 | 3.5759 | 606 | 35 | 0 |
| full_rewrite_v2 | 0.8000 | 0.4571 | 0.3738 | 2.7074 | 2127 | 35 | 0 |

Observation checks requested in spec:
- `full_rewrite_v2` CPU accuracy is `0.80` (not `>=0.95`, not `<0.20`), so no immediate leak/broken signal on this single VM from that criterion.

## Comparison vs runner_v1
Requested source: `dynapm/data/eval_sample_summary.parquet`, filtered to `trace='Materna-Trace-1'`, `vm_id='501'`, methods `static` and `full_rewrite`.

Result: no rows were present for this VM in that summary extract (`v1_rows=0`), so direct static/full_rewrite side-by-side numeric comparison for VM 501 from the task-5b sample summary is unavailable.

Available v2 values for the same methods:

| method | cpu_class_accuracy | memory_class_accuracy | cpu_p95_mae | mem_p95_mae | total_updates |
|---|---:|---:|---:|---:|---:|
| static_v2 | 0.0571 | 0.3714 | 1.2556 | 2.6150 | 0 |
| full_rewrite_v2 | 0.8000 | 0.4571 | 0.3738 | 2.7074 | 2127 |

## Assumptions
- Rolling windows passed into `step()` are length-based (`lookback_steps=288`) over aligned residual timestamps; this matches the task default `L=288`.
- Verification availability is determined by stream-time condition `current_time >= eval_timestamp + 24h`; only then snapshots are verified.
- Ground-truth lookup uses `gt_map` keyed by `eval_timestamp`; if a timestamp is missing there, it is skipped from verification records.
- Any queue entries still pending at stream end are counted as `snapshots_dropped_unverified` and excluded from metrics.
- Initial `DynamicFields` values in `runner_v2` are placeholders and are immediately set by policy constructors through `_set_initial_forecast`.
- For `PeriodicPolicyV2` initial forecast, bootstrap tail window is based on `period_hours` and bootstrap end timestamp.
- For `ReactiveThresholdPolicyV2`, candidate class checks are done on most recent 6h observed windows, and p95 fields are updated together when a class commit happens.
- `baselines_v2` intentionally uses observed windows only (no predictor-driven forecast logic), per task constraint.

## Surprises
- `eval_sample_summary.parquet` did not contain VM `Trace-1/501`, so the requested v1 side-by-side metric comparison from that exact artifact could not be numerically populated.
- On this VM, `reactive_v2` produced significantly more updates than `periodic_6h_v2` with lower class accuracy, indicating this reactive trigger may be too chatty in forecast mode.

## Commands

```bash
cp "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner.py" \
   "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v1.py" && \
cp "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines.py" \
   "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v1.py"
# output: (no stdout)
```

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_forecasting.py dynapm/tests/test_runner_v2.py -q
# output:
# .......                                                                  [100%]
# 7 passed in 2.20s
```

```bash
dynapm/.venv/bin/python - << 'PY'
# run single-VM v2 baselines and extract v1 comparison for Trace-1/501
PY
# output snippet:
# wrote .../dynapm/reports/task_07_single_vm_results.json
# wrote .../dynapm/reports/task_07_v1_compare_vm501.json
# v1_rows 0
```

## Next-step suggestion
Task 8 should wire DynAPM into this same `runner_v2` snapshot-and-verify framework (without changing v2 baselines), then re-run the controlled sample evaluation so DynAPM is compared under coherent forecast semantics; this will let alpha sensitivity, churn behavior, and ablations be interpreted against a consistent target definition.
