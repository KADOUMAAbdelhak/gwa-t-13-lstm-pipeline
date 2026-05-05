# DynAPM Task 3 — Per-VM predictor pipeline and residual streams

## Schema update

### Changes applied

- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py`**
  - Removed `DynamicFields.burst_profile`.
  - Removed module constant `BURST_PROFILES`.
  - Updated `ProfileInstance.to_json()` to stop emitting `dynamic.burst_profile`.
- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py`**
  - Removed `burst_profile` from `derive_ground_truth_at_t()` return payload.
  - Kept all other fields unchanged.
- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/pattern.py`**
  - Kept `derive_burst_profile()` as requested.
  - Added docstring note that it is an analytical utility and not in the active dynamic-field set.
- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_profile_split.py`**
  - Updated ground-truth key expectations to 5 keys.
  - Kept burst utility unit test intact.

### Test result right after schema update

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py -q
.......                                                                  [100%]
7 passed in 0.87s
```

---

## Predictor implementation

### Feature engineering

Created **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/features.py`** with:

- `build_features(df, target)` target-aware for `CPU usage [%]` or `Memory usage [%]`
- Target lags: 1, 2, 3, 6, 12, 24, 48
- Rolling windows (6, 12, 24, 48, 96): mean/std/min/max/range
- Time features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- Diffs: `target_diff_1`, `target_diff_12`
- Drops NaNs and enforces output starting from row offset 96 (`n_input - 96` rows)

### Predictor class

Created **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py`**:

- `VMPredictor(target, n_estimators=200, max_depth=20, random_state=42)`
- Uses `sklearn.ensemble.RandomForestRegressor` with:
  - `n_jobs=1`
  - `max_features="sqrt"`
- `fit(df_bootstrap)` trains only on bootstrap-derived aligned features
- `predict_streaming(df_streaming)` returns DataFrame indexed by streaming feature timestamps:
  - `y_true`, `y_pred`, `residual = y_true - y_pred`

### Task 3 tests

Created **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_predictor.py`**:

- Feature column completeness and row-count checks for CPU/memory
- Predictor train/predict shape checks
- Residual finiteness checks
- Train/test timestamp separation checks (`residual.index >= streaming_start`)

---

## Sample run results (30 VMs)

### Pipeline and artifacts

Created **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task03_sample_run.py`** to:

1. Scan all 1,576 kept VMs, compute bootstrap CPU std
2. Compute global tertiles (`q33`, `q67`) for volatility tiers
3. Sample 30 VMs with seed 42, stratified as:
   - 10 per trace
   - per trace target mix low/medium/high = 4/3/3 (with fallback fill if needed)
4. Train CPU+memory predictors per VM (bootstrap only), predict on streaming
5. Save residual parquet files:
   - `dynapm/results/residuals_sample/<trace>__<vm_id>__cpu.parquet`
   - `dynapm/results/residuals_sample/<trace>__<vm_id>__mem.parquet`
6. Compute metrics and runtime summaries
7. Save summary JSON and residual diagnostic plot

Produced files:

- **Summary JSON:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_03/sample_run_summary.json`
- **Residual files:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/residuals_sample/` (60 parquet files)

### Stratification details

- **CPU std tertiles (bootstrap, all kept VMs):**
  - q33 = **0.4485869950**
  - q67 = **1.7509051749**
- **Sample counts by trace:** Trace-1=10, Trace-2=10, Trace-3=10
- **Sample counts by tier:** low=12, medium=9, high=9

### Runtime

- Bootstrap-std scan (all 1,576 kept): **101.148 s**
- 30-VM predictor run total: **58.373 s**
- Per-VM wall-clock:
  - min = **0.700 s**
  - median = **1.988 s**
  - max = **2.631 s**

### Metrics distributions across 30 VMs

#### R2

| Metric | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| CPU% | -0.3840 | 0.6493 | 0.8549 | 0.9005 | 0.9617 |
| Memory% | 0.6145 | 0.8538 | 0.8700 | 0.9187 | 0.9772 |

#### RMSE

| Metric | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| CPU% | 0.0028 | 0.0995 | 0.2922 | 0.6382 | 7.9756 |
| Memory% | 0.0494 | 0.2910 | 0.6808 | 0.9740 | 5.9717 |

#### MAE

| Metric | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| CPU% | 0.0003 | 0.0252 | 0.1156 | 0.2813 | 2.0258 |
| Memory% | 0.0245 | 0.2253 | 0.4167 | 0.5693 | 1.4919 |

### Median R2 by volatility tier

| Tier | CPU% median R2 | Memory% median R2 |
|---|---:|---:|
| low | 0.6501 | 0.8631 |
| medium | 0.8620 | 0.8643 |
| high | 0.9033 | 0.9172 |

---

## Full-corpus runtime estimate

Based on sample per-VM median time (`1.988 s`) and kept corpus size (1,576):

- **Single-thread wall-clock estimate:**
  - `1.988 * 1576 = 3133.66 s` ≈ **52.23 min**
- **Parallel estimate (`n_jobs=8`)** (idealized scaling):
  - `3133.66 / 8 = 391.71 s` ≈ **6.53 min**
- **Residual storage estimate for full corpus:**
  - sample residual bytes = **10,648,333 B** for 30 VMs
  - bytes per VM ≈ **354,944 B**
  - full corpus estimate = **559,392,427 B** ≈ **533.48 MiB** (≈0.56 GB)

*(No full-corpus predictor run was executed, per instructions.)*

---

## Residual diagnostic

Generated figure:

- **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_03/residual_distributions.png`**

Figure content:

- Histograms of residuals faceted by volatility tier (low/medium/high)
- 2 rows: CPU residuals (top), memory residuals (bottom)
- Zero reference vertical line per panel

---

## Files created or modified

### Modified (explicitly allowed in subtask 1)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/pattern.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_profile_split.py`

### Created (Task 3)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/features.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task03_sample_run.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_03/residual_distributions.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_03/sample_run_summary.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/residuals_sample/*.parquet` (60 files)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_03_predictor.md`

---

## Test results

### Schema-update check

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py -q
.......                                                                  [100%]
7 passed in 0.87s
```

### Task 3 tests

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py tests/test_predictor.py -v
... (12 tests listed) ...
============================= 12 passed in 22.50s ==============================
```

### Full dynapm tests

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
...............                                                          [100%]
15 passed in 4.58s
```

---

## Assumptions

- **Tertiles:** CPU volatility tiers are based on bootstrap `std(ddof=0)` over `CPU usage [%]`, with boundaries at global 33rd/67th percentiles across all 1,576 kept VMs.
- **Tier assignment rule:** `std <= q33 -> low`, `q33 < std <= q67 -> medium`, `std > q67 -> high`.
- **Sampling protocol:** per trace target allocation is 4 low / 3 medium / 3 high (seed 42); if any tier has insufficient candidates in a trace, remaining slots are filled from other tiers within the same trace via deterministic shuffled fallback.
- **Feature row count contract:** row count forced to `n_input - 96` by explicit context trimming at index offset 96, matching the requested test expectation.
- **Runtime extrapolation:** `n_jobs=8` estimate assumes near-ideal parallel scaling and negligible orchestration overhead.
- **Disk extrapolation:** estimated from sample residual parquet bytes per VM (CPU+mem files together) multiplied by 1,576 VMs.
- **Predictor defaults:** RandomForest uses fixed defaults requested (`n_estimators=200`, `max_depth=20`, `max_features="sqrt"`, `n_jobs=1`, `random_state=42`).

---

## Surprises

- Initial sample run exited with code **134** due matplotlib config/cache path issues under the sandbox (`/Users/fgtek002/.matplotlib` not writable). Resolved by forcing `MPLCONFIGDIR` inside `dynapm/` and `Agg` backend in the script.
- One CPU sample VM produced negative R2 (min -0.384), while memory R2 stayed consistently high; indicates heterogeneity in CPU predictability even with per-VM models.
- Full test run became faster after caches/warmup (`15 passed in 4.58s` vs earlier task-specific run 22.50s).

---

## Commands

*(Every shell command run for this task; outputs truncated to <=20 lines.)*

```text
$ python3 - << 'PY' ... (bulk text replacements for subtask1 files)
updated subtask1 files
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py -q
.......                                                                  [100%]
7 passed in 0.87s
```

```text
$ cat > /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/features.py << 'PY' ...
(no stdout)
```

```text
$ cat > /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py << 'PY' ...
(no stdout)
```

```text
$ cat > /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_predictor.py << 'PY' ...
(no stdout)
```

```text
$ cat > /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task03_sample_run.py << 'PY' ...
(no stdout)
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py tests/test_predictor.py -v
... collected 12 items
... all PASSED ...
============================= 12 passed in 22.50s ==============================
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task03_sample_run.py
/Users/fgtek002/.matplotlib is not a writable directory
Matplotlib created a temporary cache directory ...
Matplotlib is building the font cache; this may take a moment.
(exit 134)
```

```text
$ python3 - << 'PY' ... (patch task03_sample_run.py for MPLCONFIGDIR + Agg)
patched
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task03_sample_run.py
Matplotlib is building the font cache; this may take a moment.
{
  "summary_json": ".../dynapm/reports/task_03/sample_run_summary.json",
  "plot_path": ".../dynapm/reports/task_03/residual_distributions.png",
  "sample_size": 30,
  "total_wall_s": 58.373013374977745
}
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
...............                                                          [100%]
15 passed in 4.58s
```

```text
$ python3 - << 'PY' ... (inspect residual sample output files)
n_files 60
first5
Materna-Trace-1__120__cpu.parquet 173373
Materna-Trace-1__120__mem.parquet 170464
Materna-Trace-1__145__cpu.parquet 168367
Materna-Trace-1__145__mem.parquet 169435
Materna-Trace-1__179__cpu.parquet 162492
total_bytes 10648333
```

---

## Next-step suggestion

Task 4 should run the same frozen per-VM predictor pipeline over the **full 1,576 kept VMs** (optionally with joblib parallelization), persist complete residual streams for both metrics, and then implement the first drift detector over those residuals (with reproducible thresholds and per-VM alert traces), while keeping model training frozen to bootstrap only and avoiding any online retraining.
