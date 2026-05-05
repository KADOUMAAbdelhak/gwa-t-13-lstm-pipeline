# Task 09 DynAPM v2 Single VM

## What was implemented
This task adds `DynAPMPolicyV2`, a minimal predictor-based forecast policy under the Task 7/8 redesign: ADWIN watches CPU and memory residuals (`y_true − y_pred` from the residual parquet windows), and when either detector fires the policy recommits that metric’s snapshot using the mean-class and empirical p95 over the next 288 predictor steps (`y_pred`) from the aligned streaming series. There is no decision head, churn controller, or alpha sweep; `alpha` is stored only for API compatibility. Initial forecast matches the static bootstrap aggregate over full bootstrap CPU/memory series.

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/dynapm_v2.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_dynapm_v2.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v2.py` (modified: load residuals before `method_factory`; pass `cpu_pred_full` / `mem_pred_full`; append `update_log` to return dict; drop unused `_norm` import)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_09/dynapm_v2_vm501.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_09/periodic_6h_v2_vm501.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_09_dynapm_v2_single_vm.md` (created)

## Test results
Full pytest output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.3, pluggy-1.6.0 -- .../dynapm/.venv/bin/python
cachedir: .pytest_cache
rootdir: .../dynapm
configfile: pytest.ini
collecting ... collected 10 items

dynapm/tests/test_dynapm_v2.py::test_dynapm_v2_initializes PASSED        [ 10%]
dynapm/tests/test_dynapm_v2.py::test_dynapm_v2_no_drift_no_updates PASSED [ 20%]
dynapm/tests/test_dynapm_v2.py::test_dynapm_v2_drift_commits_update PASSED [ 30%]
dynapm/tests/test_runner_v2.py::test_static_v2_runs PASSED               [ 40%]
dynapm/tests/test_runner_v2.py::test_full_rewrite_v2_runs PASSED         [ 50%]
dynapm/tests/test_runner_v2.py::test_periodic_v2_runs PASSED             [ 60%]
dynapm/tests/test_forecasting.py::test_forecast_class_simple PASSED      [ 70%]
dynapm/tests/test_forecasting.py::test_forecast_class_empty PASSED       [ 80%]
dynapm/tests/test_forecasting.py::test_forecast_p95_known PASSED         [ 90%]
dynapm/tests/test_forecasting.py::test_forecast_p95_empty PASSED         [100%]

============================== 10 passed in 2.79s ==============================
```

## Single-VM sanity check
VM: `Materna-Trace-1` / `501`. `PeriodicPolicyV2` was re-run under `runner_v2` for a fresh comparison (not copied from Task 7).

| Method | cpu_class_accuracy | memory_class_accuracy | cpu_p95_mae | mem_p95_mae | total_updates | n_eval_points_verified |
|---|---:|---:|---:|---:|---:|---:|
| DynAPMPolicyV2 | 0.8857 | 0.3143 | 0.4292 | 2.1648 | 21 | 35 |
| PeriodicPolicyV2 (6h) | 0.7714 | 0.4000 | 0.4314 | 2.9097 | 145 | 35 |

## Interpretation checkpoint
- **CHECK1** (≥1 update): **PASS** — `total_updates = 21`.
- **CHECK2** (DynAPM updates &lt; 50% of periodic): **PASS** — `21 < 72.5` (half of `145`).
- **CHECK3** (CPU accuracy within 0.10 of periodic): **FAIL** — `|0.8857 − 0.7714| = 0.1143` (&gt; `0.10`). DynAPM is better on CPU here but outside the stated tolerance band.

## Update-log inspection (DynAPM-V2, VM 501)
Total log entries: **21**. Counts by `field`:
- `cpu_class`: **2**
- `memory_class`: **2**
- `predicted_cpu_p95_next_24h`: **12**
- `predicted_mem_p95_next_24h`: **5**

Timestamps (first five log entries): `2015-11-16 20:10:00`, `2015-11-16 20:10:00`, `2015-11-17 09:30:00`, `2015-11-17 09:30:00`, `2015-11-17 17:30:00`.

Timestamps (last five log entries): `2015-11-26 06:50:00`, `2015-11-27 06:50:00`, `2015-11-28 06:50:00`, `2015-11-29 06:50:00`, `2015-12-03 17:30:00`.

## Assumptions
- Lookahead slices use `searchsorted(t)` on the streaming prediction index (same convention as legacy `_p95_pred_window` positioning).
- When CPU drift and memory drift coincide at one timestep, up to four log lines can be appended (two class lines if classes change, plus mandatory p95 lines per spec).
- Bootstrap initial forecast uses the entire bootstrap CPU/memory series (not a 24h tail), matching the Task 9 spec’s “same as Static fallback” wording.
- `runner_v2` passes full aligned `y_pred` series so policies can slice lookahead without reading parquet twice inside the policy.

## Surprises
- CHECK3 failed because DynAPM’s CPU class accuracy was **higher** than periodic’s by slightly more than 0.10, not lower — so the mechanism appears coherent but violates the symmetric “within 0.10” band as stated.
- Only **four** discrete class-change events occurred across **21** commits; most commits were p95-only refreshes tied to drift events where the mean-class stayed the same.

## Commands

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_dynapm_v2.py dynapm/tests/test_runner_v2.py dynapm/tests/test_forecasting.py -v
# ... (full session output shown in Test results above)
```

```bash
dynapm/.venv/bin/python - << 'PY'
# run_method_on_vm Trace-1/501 for DynAPMPolicyV2 and PeriodicPolicyV2; write JSON under dynapm/reports/task_09/
PY
# output snippet:
# dyn total_updates 21 per 145
# CHECK1 PASS
# CHECK2 PASS 21 72.5
# CHECK3 FAIL 0.11428571428571421
```

## Next-step suggestion
Before tuning drift sensitivity or adding the decision head, reconcile the CHECK3 criterion with directional outcomes (DynAPM can overshoot periodic in accuracy while still being “coherent”), then stress-test on a few high-volatility VMs from the Task 8 sample to see whether sparse drift-triggered predictor forecasts consistently track periodic-6h CPU accuracy within the intended tolerance band.
