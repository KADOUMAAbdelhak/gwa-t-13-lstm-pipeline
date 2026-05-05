# DynAPM Task 4 — Full-corpus predictor + drift detector wiring

## Predictor full-corpus run

Implemented **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_full_corpus_predictor.py`**:

- Reads `dynapm/data/kept_vms.json` (1,576 VMs)
- Per VM (parallel `joblib.Parallel(n_jobs=8)`):
  - load CSV
  - 30/70 split via `compute_split`
  - train `VMPredictor` for CPU and memory
  - predict streaming residuals
  - persist residuals to:
    - `dynapm/results/residuals_full/<trace>__<vm_id>__cpu.parquet`
    - `dynapm/results/residuals_full/<trace>__<vm_id>__mem.parquet`
  - compute CPU/memory streaming metrics: R2, RMSE, MAE
- Saves summary parquet:
  - `dynapm/data/predictor_summary.parquet`
- Saves failures JSON (per-VM errors):
  - `dynapm/data/predictor_failures.json`
- Saves run metadata:
  - `dynapm/data/predictor_run_meta.json`

### Full predictor run outcome

From `predictor_run_meta.json`:

- total VMs: **1576**
- successful: **1576**
- failed: **0**
- wall-clock: **782.94 s** (~13.05 min)

Residual artifacts:

- `results/residuals_full`: **3152 parquet files**, total **551,413,838 bytes** (~525.87 MiB)

---

## Predictor-result diagnostic

Implemented **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_predictor_diagnostics.py`**.

Generated figures:

- `dynapm/reports/task_04/predictor_r2_distribution.png`
- `dynapm/reports/task_04/predictor_r2_by_tier.png`
- `dynapm/reports/task_04/predictor_r2_by_trace.png`

Generated stats JSON:

- `dynapm/reports/task_04/predictor_diagnostics.json`

### Required stats (all 1,576 VMs)

- failed VM count: **0**

#### R2 quantiles

| Metric | min | p5 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | -122.7721 | -0.0454 | 0.6123 | 0.8540 | 0.9306 | 0.9792 | 0.9996 |
| Memory | -17.0585 | 0.4798 | 0.8417 | 0.8924 | 0.9328 | 0.9703 | 0.9912 |

#### RMSE quantiles

| Metric | min | p5 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0.0011 | 0.0067 | 0.0780 | 0.3394 | 0.9427 | 4.3376 | 58.7643 |
| Memory | 0.0132 | 0.1080 | 0.4134 | 0.7971 | 1.4062 | 3.6920 | 21.3730 |

#### MAE quantiles

| Metric | min | p5 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0.0002 | 0.0031 | 0.0198 | 0.1022 | 0.3670 | 2.1058 | 38.2984 |
| Memory | 0.0027 | 0.0597 | 0.2515 | 0.4358 | 0.7955 | 2.3439 | 16.5432 |

Counts:

- CPU `R2 < 0`: **89**
- Memory `R2 < 0`: **23**
- CPU `R2 < 0.5`: **314**
- Memory `R2 < 0.5`: **81**

---

## Ground-truth derivation

Implemented:

- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/evaluation_clock.py`**
  - `compute_evaluation_timestamps(streaming_index, period_hours=12)`
- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth.py`**
  - loads thresholds from `bootstrap_summary.parquet`
  - computes per-VM 12h evaluation timestamps in streaming segment
  - derives 24h-forward ground truth at each eval timestamp
  - writes per-VM files:
    - `dynapm/results/ground_truth/<trace>__<vm_id>.parquet`
- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth_stats.py`**
  - aggregates corpus-wide GT distributions

Ground-truth run metadata (`dynapm/data/ground_truth_run_meta.json`):

- total VMs: **1576**
- successful: **1576**
- failed: **0**
- wall-clock: **31.03 s**
- mean eval points per VM: **41.84**
- skipped empty windows: **106** (see assumptions/deviations)

Ground-truth artifacts:

- `results/ground_truth`: **1576 parquet files**, **8,019,108 bytes**

### Corpus-wide GT distribution stats

From `dynapm/data/ground_truth_stats.json` (`n_points = 65,933`):

- CPU class proportions:
  - burst: 0.3417
  - high: 0.2957
  - medium: 0.1774
  - idle: 0.0942
  - low: 0.0910
- Memory class proportions:
  - high: 0.4198
  - medium: 0.3617
  - burst: 0.1217
  - low: 0.0630
  - idle: 0.0338
- Diurnal pattern proportions:
  - irregular: 0.6776
  - flat: 0.3224
- `p95_cpu_next_24h`: mean **7.6478**, std **13.8987**
- `p95_mem_next_24h`: mean **13.6148**, std **15.2646**
- Distinct CPU classes observed per VM across streaming:
  - min 1, p25 2, median 3, p75 4, max 5
- Distinct memory classes observed per VM across streaming:
  - min 1, p25 2, median 2, p75 3, max 5

Interpretation required by spec: class changes are common (CPU median distinct=3; memory median distinct=2), so adaptive tracking has non-trivial headroom on many VMs.

---

## Drift detector implementation

Implemented **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/drift.py`**:

- `ADWINDetector(delta=0.002)`
- `PageHinkleyDetector(threshold=50.0, delta=0.005)`
- `run_detector(residual_series, detector_type="adwin", **kwargs)`
  - returns list of drift-detected timestamps

---

## Drift tests

Implemented tests in **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_drift.py`**:

- ADWIN no-drift on near-constant stream
- ADWIN drift on mean shift
- Page-Hinkley drift on mean shift
- `run_detector` timestamp return contract

Evaluation clock test in **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_evaluation_clock.py`**.

Pytest output:

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_drift.py tests/test_evaluation_clock.py -v
... collected 5 items
... all PASSED ...
============================== 5 passed in 1.08s ===============================
```

Full suite:

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
....................                                                     [100%]
20 passed in 4.85s
```

---

## Corpus-wide drift characterization

Implemented **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_drift_characterization.py`**:

- loads full CPU/memory residual streams per VM
- runs ADWIN for each metric
- writes per-VM drift timestamps:
  - `dynapm/results/drift_events/<trace>__<vm_id>__cpu.parquet`
  - `dynapm/results/drift_events/<trace>__<vm_id>__mem.parquet`
- aggregates to `dynapm/data/drift_summary.parquet` with required fields
- computes corpus stats in `dynapm/data/drift_stats.json`

Run result:

- wall-clock: **7.40 s**
- drift summary rows: **1576**
- drift event artifacts: **3152 parquet files**, **4,020,672 bytes**

### Required drift stats

From `drift_stats.json`:

#### Drift count distributions

| Metric | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| CPU drift count | 0 | 0 | 2 | 5 | 14 | 61 |
| Memory drift count | 0 | 0 | 1 | 4 | 11 | 43 |

- VMs with zero CPU drift: **429**
- VMs with zero memory drift: **547**

#### Distribution by volatility tier (CPU drift count)

- low: min 0, p25 0, median 1, p75 2, p95 6, max 29
- medium: min 0, p25 1, median 2, p75 4, p95 13.8, max 42
- high: min 0, p25 2, median 4, p75 8, p95 19, max 61

#### Distribution by volatility tier (memory drift count)

- low: min 0, p25 0, median 0, p75 1, p95 5, max 30
- medium: min 0, p25 0, median 1, p75 4, p95 12, max 40
- high: min 0, p25 1, median 3, p75 6, p95 13, max 43

#### Distribution by trace (CPU drift count)

- Trace-1: min 0, p25 0, median 2, p75 5, p95 13, max 54
- Trace-2: min 0, p25 0, median 1, p75 3, p95 14, max 61
- Trace-3: min 0, p25 1, median 3, p75 6, p95 17.85, max 42

#### Distribution by trace (memory drift count)

- Trace-1: min 0, p25 0, median 1, p75 3, p95 9.5, max 33
- Trace-2: min 0, p25 0, median 1, p75 3, p95 10, max 43
- Trace-3: min 0, p25 1, median 2, p75 5, p95 13, max 40

---

## Figures referenced

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04/predictor_r2_distribution.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04/predictor_r2_by_tier.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04/predictor_r2_by_trace.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04/drift_event_examples.png`

---

## Files created or modified (absolute paths)

### Created

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/conftest.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/_matplotlib_setup.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/evaluation_clock.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/drift.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_evaluation_clock.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_drift.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_full_corpus_predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_predictor_diagnostics.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth_stats.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_drift_characterization.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04_drift.md`

### Modified (explicitly allowed by task for matplotlib setup)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task03_sample_run.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/__init__.py` (added to support shared setup import)

### Generated artifacts (Task 4 runs)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/predictor_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/predictor_failures.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/predictor_run_meta.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_04/predictor_diagnostics.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_run_meta.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_failures.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_stats.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/drift_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/drift_stats.json`
- plus per-VM parquet outputs in:
  - `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/residuals_full/`
  - `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/ground_truth/`
  - `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/drift_events/`

---

## Test results

- Task-4-specific tests: **5 passed** (`test_drift.py`, `test_evaluation_clock.py`)
- Full dynapm suite: **20 passed**

---

## Assumptions

- For `compute_evaluation_timestamps`, max allowed timestamp is `streaming_end - 24h`, as specified.
- Volatility tiers (for predictor R2-by-tier and drift-by-tier) are based on bootstrap CPU std tertiles (q33/q67) recomputed over all kept VMs.
- Drift characterization uses ADWIN on residuals with default `delta=0.002`.
- Drift-event example figure uses **CPU residual stream** and CPU drift events to visualize low/high drift behavior.
- Predictor diagnostics use the full successful `predictor_summary.parquet` population (1576 rows).

---

## Surprises

- In sandboxed runs, `joblib` loky failed with semaphore permission error (`SC_SEM_NSEMS_MAX`), so heavy parallel scripts required reruns with `required_permissions=["all"]`.
- Ground-truth derivation initially produced failures from sparse windows (`empty window ...`); script was updated to skip only those specific evaluation timestamps and continue per VM.
- Predictor metric tails include extreme negative R2 outliers (CPU min -122.77, memory min -17.06), indicating a small subset of pathological fit scenarios even with per-VM RF.

---

## Commands (output truncated to <=20 lines each)

```text
$ mkdir -p .../dynapm/reports/task_04 .../dynapm/results/{residuals_full,ground_truth,drift_events}
(no output)
```

```text
$ cat > .../dynapm/scripts/_matplotlib_setup.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/conftest.py << 'PY' ...
(no output)
```

```text
$ touch .../dynapm/scripts/__init__.py
(no output)
```

```text
$ python3 - << 'PY' ... (update task03_sample_run import)
updated task03_sample_run import
```

```text
$ cat > .../dynapm/core/evaluation_clock.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/core/drift.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/tests/test_evaluation_clock.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/tests/test_drift.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/scripts/task04_full_corpus_predictor.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/scripts/task04_predictor_diagnostics.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/scripts/task04_ground_truth.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/scripts/task04_ground_truth_stats.py << 'PY' ...
(no output)
```

```text
$ cat > .../dynapm/scripts/task04_drift_characterization.py << 'PY' ...
(no output)
```

```text
$ cd .../dynapm && .venv/bin/pytest tests/test_evaluation_clock.py tests/test_drift.py tests/test_profile_split.py tests/test_predictor.py -q
.................                                                        [100%]
17 passed in 6.31s
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_full_corpus_predictor.py
PermissionError: [Errno 1] Operation not permitted
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_full_corpus_predictor.py   (required_permissions=all)
{
  "n_total": 1576,
  "n_ok": 1576,
  "n_failed": 0,
  "wall_seconds": 782.9420761250658,
  ...
}
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_predictor_diagnostics.py
ModuleNotFoundError: No module named 'dynapm'
```

```text
$ python3 - << 'PY' ... (fix import order in scripts)
patched import ordering
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_predictor_diagnostics.py
{
  "rows": 1576,
  "n_failed": 0,
  "stats_json": ".../predictor_diagnostics.json"
}
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py
PermissionError: [Errno 1] Operation not permitted
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py   (required_permissions=all)
{
  "n_total": 1576,
  "n_ok": 1497,
  "n_failed": 79,
  ...
}
```

```text
$ python3 - << 'PY' ... (normalize vm_id in task04_ground_truth)
patched norm vm id
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py   (required_permissions=all)
{
  "n_total": 1576,
  "n_ok": 1497,
  "n_failed": 79,
  ...
}
```

```text
$ python3 - << 'PY' ... (patch task04_ground_truth to skip empty windows)
patched skip empty windows
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py   (required_permissions=all)
{
  "n_total": 1576,
  "n_ok": 1576,
  "n_failed": 0,
  "wall_seconds": 31.031425834051333,
  "mean_eval_points": 41.83565989847716,
  "total_skipped_empty_windows": 106
}
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth_stats.py
{
  "stats_json": ".../ground_truth_stats.json",
  "n_points": 65933
}
```

```text
$ cd .../dynapm && .venv/bin/pytest tests/test_drift.py tests/test_evaluation_clock.py -v
... collected 5 items
... all PASSED ...
============================== 5 passed in 1.08s ===============================
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_drift_characterization.py
{
  "summary_path": ".../drift_summary.parquet",
  "stats_json": ".../drift_stats.json",
  "wall_seconds": 6.905595291987993
}
```

```text
$ python3 - << 'PY' ... (patch drift stats detail)
patched drift stats detail
```

```text
$ cd ... && dynapm/.venv/bin/python dynapm/scripts/task04_drift_characterization.py   (required_permissions=all)
{
  "summary_path": ".../drift_summary.parquet",
  "stats_json": ".../drift_stats.json",
  "wall_seconds": 7.395243333070539
}
```

```text
$ cd .../dynapm && .venv/bin/pytest tests/ -q
....................                                                     [100%]
20 passed in 4.85s
```

```text
$ python3 - << 'PY' ... (artifact counts)
results/residuals_full files 3152 total_bytes 551413838
results/ground_truth files 1576 total_bytes 8019108
results/drift_events files 3152 total_bytes 4020672
```

```text
$ dynapm/.venv/bin/python - << 'PY' ... (parquet row checks)
predictor_rows 1576
gt_files_expected 1576
drift_rows 1576
```

---

## Next-step suggestion

Task 5 should consume the outputs now wired in Task 4 (`predictor_summary`, per-VM residual streams, per-VM ground-truth time series, and drift summaries) to implement the DynAPM update engine and decision head end-to-end: define when profile fields are updated at evaluation timestamps, compare static-vs-adaptive policies against the same ground-truth windows, and quantify update cost/churn trade-offs before introducing any clustering or LLM-assisted labeling.
