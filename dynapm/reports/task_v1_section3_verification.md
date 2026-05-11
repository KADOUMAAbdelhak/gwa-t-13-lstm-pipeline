# Task V1 — Section 3 technical verification (read-only)

## D1: Class quantile boundaries

**Primary file:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/discretization.py`

**Support:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/forecasting.py` (`forecast_class_from_window` applies thresholds to the **mean** of a window of values.)

Percentile computation (same file):

```9:17:dynapm/core/discretization.py
def compute_thresholds(df_bootstrap: pd.DataFrame, metric: str) -> dict[str, float]:
    """Percentiles p20..p95 on bootstrap segment for ``metric``."""
    s = df_bootstrap[metric].astype(float)
    out = {
        "p20": float(np.percentile(s, 20)),
        "p40": float(np.percentile(s, 40)),
        "p60": float(np.percentile(s, 60)),
        "p80": float(np.percentile(s, 80)),
        "p95": float(np.percentile(s, 95)),
    }
```

Bin mapping:

```21:35:dynapm/core/discretization.py
def classify_value(value: float, thresholds: dict[str, float]) -> str:
    """Map scalar to discrete class using frozen percentile gates."""
    p20 = thresholds["p20"]
    p40 = thresholds["p40"]
    p60 = thresholds["p60"]
    p80 = thresholds["p80"]
    if value < p20:
        return "idle"
    if p20 <= value < p40:
        return "low"
    if p40 <= value < p60:
        return "medium"
    if p60 <= value < p80:
        return "high"
    return "burst"
```

**Exact quantile *indices*:** empirical percentiles at **20, 40, 60, 80, and 95** of the **per-VM bootstrap segment** for each metric (not universal numeric constants).

**CPU vs memory:** thresholds are **computed separately** for **`"CPU usage [%]"`** and **`"Memory usage [%]"`** (see `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/run_bootstrap_characterization.py`, which calls `compute_thresholds` twice per VM). Gate values **differ** between CPU and memory for the same VM.

**Class labels:** **five** names — `idle`, `low`, `medium`, `high`, `burst` (see also `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py`). **`p95`** appears in `compute_thresholds` but **`classify_value` only uses `p20`–`p80`**; values **≥ `p80`** fall in **`burst`**.

---

## D2: `reactive_v2` trigger

**File:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v2.py` — class `ReactiveThresholdPolicyV2`.

**What signal is watched?** At each streaming timestep `t`, the policy takes the **observed** CPU and memory series (`cpu_obs_window`, `mem_obs_window` passed in by the runner) and restricts them to the sub-window **`[t − 6 hours, t]`** (inclusive). It does **not** use instantaneous single-tick samples in isolation; it uses whatever observations fall in that **6-hour** window (implementation: `_window_last_hours(..., 6.0)`).

**How is “class” derived from that signal?** For each modality, `forecast_class_from_window` computes the **arithmetic mean** of the values in that 6-hour window, then `classify_value(mean, thresholds)` maps that mean into `idle` / `low` / `medium` / `high` / `burst` using **per-VM bootstrap thresholds** (`cpu_thresholds` / `mem_thresholds`).

**What triggers an update?** An update is triggered when **either** the candidate CPU class **or** the candidate memory class **differs** from the **currently committed** `profile.dynamic.cpu_class` or `profile.dynamic.memory_class`. There is **no** separate threshold-cross rule beyond this inequality with committed state.

**Hysteresis / cooldown:** None in `step` itself—every timestep re-evaluates. (Committed profile fields only change when `_commit_forecast` runs.)

**CPU vs memory coupling:** The condition is an **OR** across modalities, but when it fires, `_commit_forecast(t, cpu_candidate, mem_candidate)` runs **once** and may append **multiple** log entries (one per changed forecast field inside `_commit_forecast`), i.e. CPU and memory forecasts are refreshed **together** on that event.

**Verbatim snippet (17 lines):**

```python
    def step(self, t, cpu_obs_window, mem_obs_window, cpu_pred_window, mem_pred_window):
        t = pd.Timestamp(t)
        cpu_candidate = _window_last_hours(cpu_obs_window, t, 6.0)
        mem_candidate = _window_last_hours(mem_obs_window, t, 6.0)

        candidate_cpu_class = forecast_class_from_window(cpu_candidate, self.cpu_thresholds)
        candidate_mem_class = forecast_class_from_window(mem_candidate, self.mem_thresholds)
        if (
            candidate_cpu_class != self.profile.dynamic.cpu_class
            or candidate_mem_class != self.profile.dynamic.memory_class
        ):
            self._commit_forecast(t, cpu_candidate, mem_candidate)
```

*(Source: `ReactiveThresholdPolicyV2.step` in `baselines_v2.py`.)*

---

## D3: Bootstrap segment length (forecaster training)

**Split definition:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py`

**Literal value:** `LOCKED_BOOTSTRAP_FRACTION = 0.30` — the bootstrap segment is the **first** `floor(n_total * 0.30)` **rows** of each VM time series (sorted time index). This is a **fixed fraction**, not a fixed calendar duration.

**Forecaster training:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_full_corpus_predictor.py` calls `compute_split(df)` then `VMPredictor.fit(df_b)` on that bootstrap frame (`dynapm/core/predictor.py`). So training uses exactly this **same** 30% prefix.

**Fixed vs adaptive:** **Per-VM**: every VM has its own `n_total` and thus its own `n_bootstrap` row count. The **0.30** proportion is global and fixed.

**Units:** defined in **observation count** (rows). `compute_split` also reports **`bootstrap_duration_days`** derived from timestamps (`bootstrap_duration_days` in the returned `info` dict), which is **derived** and varies by sampling cadence and series length—it is not the primary definition of the split.

---

## Discrepancies

1. **Class count / naming:** Implementation uses **five** workload classes (`idle`, `low`, `medium`, `high`, `burst`), matching `dynapm/core/schema.py`—not a four-way **low/medium/high/critical** partition, and there is **no** label `critical`.
2. **`p95` in classification:** `p95` is computed and stored for each VM/metric, but **classification** only uses **p20–p80** gates; the top bin is **`burst` for values ≥ p80**.
3. **Quantiles are not universal constants:** Paper text must describe **per-VM empirical percentiles** on the bootstrap window (CPU vs memory differ), not a single global `(q₁,q₂,q₃)`.
4. **`reactive_v2`:** Watches **rolling 6 h of observations** and compares **mean-based** discrete classes to **committed** classes—**not** a pure instantaneous threshold cross, and **no** explicit cooldown in code.

---

## Files inspected (absolute paths)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/discretization.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/forecasting.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/baselines_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_full_corpus_predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/run_bootstrap_characterization.py`

*(No `bootstrap_classes.py` or `profile_schema.py` exists under `dynapm/`; behavior lives in the files above.)*
