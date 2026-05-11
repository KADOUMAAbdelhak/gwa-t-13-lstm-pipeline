# Task 13 Leakage Audit

## What was audited
This report documents a static review of **feature construction**, **VMPredictor** train/test separation, **`runner_v2`** data flow, **absence of redundant raw columns** in the model inputs, and a **100-VM null control** (`PeriodicNullPredictorPolicyV2`) that replaces the real predictor lookahead with **constant** bootstrap-mean series while keeping the **same 6h refresh schedule** as `PeriodicPredictorPolicyV2`. The goal is to detect accidental **target leakage** or **structural bias** that would inflate Task 12’s **~+18pp** CPU-class gain from using real `y_pred`.

## Feature directionality audit
Source: `dynapm/core/features.py`, `build_features`.

| Feature group | Columns / rule | Temporal direction | Notes |
|---------------|----------------|--------------------|--------|
| **Lag** | `target_lag_1` … `target_lag_48` via `s.shift(lag)` | **Backward (past only)** — positive `lag` uses strictly older rows | No future rows. |
| **Rolling** | `target_roll_mean/std/min/max/range_{6,12,24,48,96}` | **Backward** — `s.rolling(window=w, min_periods=w)` default **trailing** window | `center` not set (**False** / default). |
| **Diff** | `target_diff_1`, `target_diff_12` | **Backward** — `s.diff(periods)` uses current minus past | Standard pandas `diff`. |
| **Time** | `hour_sin/cos`, `dow_sin/cos` from row index | **Current row timestamp only** | No series values from other columns; not future data. |
| **Trim** | `out.iloc[MAX_CONTEXT:]` then `dropna` | Drops rows **without** enough **history** | Longest lookback **96** steps. |

**Verdict:** **PASS** — no constructed feature uses values from timestamps **strictly after** the current row; lags/shifts and rolling windows are backward-only; time features are calendar encodings of the current index.

## Predictor train/predict separation audit
Source: `dynapm/core/predictor.py`.

- **`fit(df_bootstrap)`** builds `build_features(df_bootstrap, target)` and fits on rows **only** inside that bootstrap frame.
- **`predict_streaming(df_streaming)`** builds features **only** from `df_streaming`; labels/residuals align to `features.index` within streaming.

**Verdict:** **PASS** — training does not use streaming rows; prediction does not use bootstrap rows.

## Runner pipeline audit
Source: `dynapm/eval/runner_v2.py`.

- `cpu_pred_full` / `mem_pred_full` are the **`y_pred`** columns read from  
  `dynapm/results/residuals_full/<trace>__<vm_id>__cpu.parquet` and `__mem.parquet`.
- The runner **does not** call `VMPredictor.fit` or rebuild models; it only **reads** precomputed residuals (Task 4 pipeline: fit on bootstrap, predict on streaming).
- No path in `runner_v2` retrains or merges bootstrap targets into live prediction at eval time.

**Verdict:** **PASS**.

## Redundant-target feature check
`build_features(df, target)` operates **only** on `df[target]` for target in **`"CPU usage [%]"`** or **`"Memory usage [%]"`**. There is **no** use of `"CPU usage [MHZ]"`, `"Memory usage [KB]"`, or other CSV columns.

**Column list produced by `build_features`** (identical naming pattern for either target — all values are built from the **single** selected target series `s`):

- Lags: `target_lag_1`, `target_lag_2`, `target_lag_3`, `target_lag_6`, `target_lag_12`, `target_lag_24`, `target_lag_48`
- Rolling: `target_roll_mean_*`, `target_roll_std_*`, `target_roll_min_*`, `target_roll_max_*`, `target_roll_range_*` for `*` in `{6,12,24,48,96}`
- Diff: `target_diff_1`, `target_diff_12`
- Time: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

**Verdict:** **PASS** — no redundant raw scaling column from the CSV is injected as a feature.

## Null-predictor control
Implementation: `dynapm/methods/baselines_v3_audit.py` — `PeriodicNullPredictorPolicyV2` (constant lookahead = bootstrap **mean** of each metric; `cpu_pred_full` / `mem_pred_full` **ignored**).

100-VM run: `joblib.Parallel(n_jobs=8)`, wall **~6.32 s** (second invocation).

### Three-method comparison (+ Static reference)
Means over **100** locked VMs. **`periodic_6h_v2`** / **`static_v2`** from `dynapm/data/eval_sample_v2_summary.parquet`; **`periodic_predictor_6h_v2`** from `dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet` (Task 12); **`periodic_null_pred_6h_v2`** from this task’s parquet.

| method | mean cpu_class_accuracy | mean total_updates |
|--------|-------------------------:|-------------------:|
| static_v2 | 0.4207 | 0.00 |
| periodic_6h_v2 (observation window) | 0.7085 | 141.22 |
| periodic_predictor_6h_v2 (real y_pred) | 0.8878 | 189.72 |
| periodic_null_pred_6h_v2 (constant lookahead) | 0.4207 | 172.31 |

**Interpretation:** The null policy’s **mean CPU class accuracy** matches **`static_v2`’s mean to floating-point equality (0.420682…)** — i.e. **no uplift** from “predictor-shaped” mechanics alone; the large gain appears only when **real** `y_pred` is used (Task 12). Per-VM checks (e.g. Trace-1 / VM 53) show **identical** `cpu_class_accuracy` between null and static, consistent with the aggregate match.

## Audit verdict
**Classification: `AUDIT_PASS`**

**Justification:** Subtasks **1–4** are **PASS** under the code review above (strictly backward features, bootstrap/streaming separation in `VMPredictor`, precomputed residuals in `runner_v2`, no MHZ/KB features). The **null control** yields **mean CPU accuracy equal to Static** (difference **0.0**, well within **0.05**), and **far below** both **Periodic-6h (window)** and **Periodic-Predictor (real)**. That pattern matches **“no structural inflation from predictor-source machinery alone”**; the **+18pp** lift is therefore **not** explained by the control as spurious bias from the evaluation recipe, under this test’s assumptions.

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v3_audit.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task13_leakage_control.py` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_periodic_null_predictor_summary.parquet` (created; 100 rows)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/task13_leakage_audit.json` (created)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_sample_v2/periodic_null_pred_6h_v2__<trace>__<vm_id>.json` (created; 100 files)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_13_leakage_audit.md` (created)

**Not modified (per constraints):** `dynapm/core/features.py`, `dynapm/core/predictor.py`, `dynapm/eval/runner_v2.py`.

## Test results
```text
...........                                                              [100%]
11 passed in 3.30s
```
(`dynapm/tests/test_dynapm_v2.py`, `test_runner_v2.py`, `test_forecasting.py`)

## Assumptions
- Residual parquets in `dynapm/results/residuals_full/` reflect Task 4’s bootstrap-only training and streaming-only prediction.
- **AUDIT_PASS** “null comparable to Static (within 0.05)” is evaluated on **mean** `cpu_class_accuracy` over 100 VMs.
- The null policy uses **bootstrap mean** constants per spec (full bootstrap series mean for CPU and memory targets).

## Surprises
- **Mean CPU accuracy for the null predictor policy equals the Static mean exactly** (bit-for-bit at aggregate precision), and sample spot-checks show **per-VM equality** with Static on checked IDs — strong evidence that **constant lookahead** under this schedule does not introduce extra “predictor pathway” lift.

## Commands

```bash
dynapm/.venv/bin/python dynapm/scripts/task13_leakage_control.py
# output snippet:
# "periodic_null_pred_6h_v2": 0.4206819897554261  (matches static)
# "periodic_predictor_6h_v2": 0.8877743179437688
```

```bash
dynapm/.venv/bin/python -m pytest dynapm/tests/test_dynapm_v2.py dynapm/tests/test_runner_v2.py dynapm/tests/test_forecasting.py -q
# 11 passed in 3.30s
```

## Next-step suggestion
Document explicitly in any paper draft that the **null control matches Static** on mean CPU-class accuracy, so readers do not conflate **6h refresh activity** (non-zero `total_updates` for null vs zero for Static) with **forecast information**: the informative ingredient in Task 12 is the **content** of real `y_pred`, not the refresh cadence alone.
