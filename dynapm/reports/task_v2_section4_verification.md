# Task V2 — Section 4 experimental setup verification (read-only)

This note records **what the repository actually implements**, with file paths for drafting Section 4. No code was executed for this task beyond reading sources and existing task reports.

---

## E1: Dataset filter

**Filtering implementation:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/filter.py` (no `build_kept_vms.py` in this repo; `filter_corpus` writes `kept_vms.json`).

- **Function:** `filter_corpus(min_timesteps: int = 2000)`
- **Criterion:** a VM is kept iff `len(df) >= min_timesteps` after `load_vm_csv` (full parse). There is **no** missing-rate cutoff or additional quality gate in this module—only **minimum row count**.

```13:17:dynapm/data/filter.py
def vm_passes_min_timesteps(df, min_timesteps: int) -> bool:
    """Return True iff the VM series has at least ``min_timesteps`` rows."""
    return len(df) >= min_timesteps
```

**Iteration over raw traces:** `iter_corpus` in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/loader.py` walks `Materna-Trace-1`, `Materna-Trace-2`, `Materna-Trace-3` under `old/data/raw/`, yielding every `*.csv` except `00-Trace-Info.txt`.

**Documented filter statistics** (from Task 1 write-up, consistent with `stats` returned by `filter_corpus`):

| Stage | Count | Source |
|------|------:|--------|
| `total_seen` (all VM CSVs across 3 traces) | **1,592** | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_01_data_layer.md` §5 |
| `dropped` (`len(df) < 2000`) | **16** | same |
| `kept` | **1,576** | same; persisted as `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json` |

**Per-trace kept counts** (bootstrap summary / Task 2): Materna-Trace-1 **511**, Materna-Trace-2 **521**, Materna-Trace-3 **544** (sum **1,576**). Raw per-trace totals before/drop are not split in the cited report—the authoritative **raw total** for the three-trace crawl is **1,592** with **16** short series removed.

**Final corpus size:** **1,576** VMs (verified in multiple asserts, e.g. `task15_full_corpus.py`).

---

## E2: Forecaster hyperparameters and features (`VMPredictor`)

**Class definition:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py`  
**Features:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/features.py`

### RandomForestRegressor hyperparameters explicitly set

| Parameter | Value |
|-----------|------:|
| `n_estimators` | **200** |
| `max_depth` | **20** |
| `random_state` | **42** |
| `n_jobs` | **1** |
| `max_features` | **`"sqrt"`** |

`min_samples_split`, `min_samples_leaf`, etc. are **not** passed—**scikit-learn defaults** apply (for sklearn 1.4+, `min_samples_split=2`, `min_samples_leaf=1` unless overridden by the installed version).

### Complete feature set (`build_features`)

Built only from the **target** column (`CPU usage [%]` or `Memory usage [%]`) plus the **timestamp index**:

1. **Lags** (7): `target_lag_k` for **k ∈ {1, 2, 3, 6, 12, 24, 48}** — value of the target **k** steps back.
2. **Rolling statistics** over windows **w ∈ {6, 12, 24, 48, 96}** (5 windows × 5 stats = 25 columns): for each `w`, `rolling(window=w, min_periods=w)` on the target produces **mean**, **std (ddof=0)**, **min**, **max**, and **range** (`max−min`), named `target_roll_mean_w`, `target_roll_std_w`, `target_roll_min_w`, `target_roll_max_w`, `target_roll_range_w`.
3. **Differences:** `target_diff_1`, `target_diff_12`.
4. **Cyclical time:** `hour_sin`, `hour_cos` (hour-of-day), `dow_sin`, `dow_cos` (day-of-week) from the index.

After construction, rows are sliced with **`out = out.iloc[MAX_CONTEXT:]`** where `MAX_CONTEXT = 96`, then **`dropna(how="any")`**—the first **96** timesteps of (lag/roll) history are removed before training or prediction.

### Target variable

`VMPredictor.fit` sets **`y = df_bootstrap.loc[features.index, self.target].values`** — i.e. the **raw next-step (same-timestamp) target column** `%` usage aligned with each feature row. This is **one-step regression** of the scalar target from the engineered history, **not** a temporally aggregated label over a multi-day horizon.

### Preprocessing

- **No** explicit scaling or normalization in `predictor`/`features`.
- **NA handling:** rows with any NaN in the feature matrix after the `iloc[96:]` cut are **dropped** (`dropna`).
- **Outlier clipping:** none in these modules.

### Training data and validation

- **Training:** `fit` uses only **`df_bootstrap`** — the first **`floor(0.30 × n_total)`** rows from `compute_split` (`LOCKED_BOOTSTRAP_FRACTION = 0.30` in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py`).
- **Held-out validation:** **none** in code—`predict_streaming` runs on the **streaming** segment after fit; that is the evaluation stream for residuals, not an internal validation fold.
- **Overfitting control:** RF depth/capacity limits (`max_depth=20`, `max_features="sqrt"`, `n_estimators=200`) only; no early stopping or CV.

```10:25:dynapm/core/predictor.py
class VMPredictor:
    def __init__(
        self,
        target: str,
        n_estimators: int = 200,
        max_depth: int = 20,
        random_state: int = 42,
    ):
        self.target = target
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
        )
```

---

## E3: Volatility tier definition

**Where:** Task 15 (and related scripts) build tiers inline—e.g. `_tier_map_from_kept` in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task15_full_corpus.py`. There is **no** separate `stratification.py` in the tree.

**Definition of “volatility”:**

- For each kept VM, load the CSV, run `compute_split(df)`, take the **bootstrap** segment `b`, compute **`std = float(b["CPU usage [%]"].std(ddof=0))`** — i.e. **population std of CPU % on the bootstrap window**, not coefficient of variation and not memory.

**Tier cutoffs:**

- Build a Series of these std values across all **1,576** VMs.
- **q1** = `quantile(1/3)`, **q2** = `quantile(2/3)` of that distribution.
- Assign **`low`** if `std <= q1`, **`medium`** if `q1 < std <= q2`, **`high`** if `std > q2`.

Thresholds are **data-dependent** (tertiles of the empirical std distribution), **not** fixed numeric constants.

**Counts (reported in Task 15):**

| Tier | `n_vms_in_tier` |
|------|----:|
| low | **525** |
| medium | **525** |
| high | **526** |

(Source: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15_full_corpus.md`.)

---

## E4: Evaluation cadence and horizon

**Ground-truth timestamps:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/evaluation_clock.py` + `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth.py`.

- **Candidate clock:** from the **start of the streaming index**, a grid every **`period_hours=12`** (`pd.date_range(..., freq="12h")`) up to **`last_index - 24h`** so a **24 h** forward window fits in the trace.

```7:22:dynapm/core/evaluation_clock.py
def _compute_candidate_timestamps(
    streaming_index: pd.DatetimeIndex,
    period_hours: int = 12,
    window_hours: int = 24,
) -> pd.DatetimeIndex:
    ...
    ts = pd.date_range(start=start, end=last_allowed, freq=f"{period_hours}h")
    return pd.DatetimeIndex(ts).sort_values()
```

- **Data-aware filter:** a candidate **`t`** is **kept** only if the full VM series `df` has **≥ `min_observations_in_window` (200)** rows in **`[t, t+24h)`** (half-open). Others are dropped (`compute_data_aware_evaluation_timestamps`).

**Labels at each eval time (`derive_ground_truth_at_t_v2`):** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py`

- **cpu_class / memory_class:** mean CPU% / Memory% over **`[t − 1h, t]`** (backward window default **1.0 h**), then `classify_value` vs frozen bootstrap thresholds.
- **p95_cpu_next_24h / p95_mem_next_24h:** empirical **0.95 quantile** over **`[t, t+24h)`** (forward **24 h**).

**`runner_v2`:** reads precomputed GT parquet per VM; it evaluates policies only at timestamps present in that file—so the **operational eval grid follows the 12 h (filtered) clock**, not policy refresh cadence (e.g. 6 h periodic policies still **score** on the 12 h GT grid when it exists).

**Warm-up / cool-down:** No separate “warm-up” period in the runner beyond the predictor’s **96-step** feature context on the streaming segment. **`lookback_steps=288`** in `run_method_on_vm` corresponds to **288 × 5 min = 24 h** of history windows for policy stepping, separate from the **12 h** eval grid spacing.

---

## E5: Provisioning task parameters

**Module:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/provisioning.py`

```16:17:dynapm/eval/provisioning.py
SAFETY_FACTOR = 1.2
OVER_THRESHOLD = 1.5
```

**Primary reporting SF:** **1.2** (Task 17b headline table).

**Sensitivity:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task17b_provisioning.py` recomputes aggregates at **safety factors `{1.1, 1.2, 1.5}`** (see sensitivity table in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md`).

**Cadence vs profile:** Provisioning records are built by **replaying** the same stream as `runner_v2` and joining **GT `p95_*_next_24h`** at eval steps—so per-VM **eval timestamp cadence matches the ground-truth / evaluation grid** (12 h candidates after data-aware filtering), **not** the 6 h policy refresh period.

---

## E6: Computing environment

**Pinned dependency floors:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/requirements.txt` lists **minimum** versions (e.g. `numpy>=1.26`, `pandas>=2.1`, `scikit-learn>=1.4`, `scipy>=1.12`, `joblib>=1.3`, `matplotlib>=3.8`, `river>=0.21`, `pyarrow>=14`, `pytest>=7.4`) but **no** exact resolved versions and **no** Python interpreter pin.

**Root** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/requirements.txt` is a looser alternate list (incl. TensorFlow, notebook)—not version-pinned.

**`pyproject.toml` / `setup.py`:** **not** present at repo root in this workspace.

**`README`:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/README.md` documents `venv` + `pip install -r dynapm/requirements.txt` but **does not** record Python version or hardware.

**Wall-clock excerpts from task reports ( illustrative, not reproducible hardware specs):**

| Artifact | Wall time noted |
|----------|----------------:|
| Task 2 bootstrap characterization | **~102.2 s** (`task_02_profile_and_split.md`) |
| Task 15 full corpus eval (9456 runs, `n_jobs=8`) | **1162.0 s** (`task_15_full_corpus.md`) |
| Task 17b provisioning collect + aggregates | **~20 min** narrative for parallel replay (`task_17b_provisioning.md`) |

**Machine specs (CPU model, RAM):** **not** recorded in the inspected README/requirements; task markdown does not cite hardware beyond wall times.

---

## E7: Statistical methodology

**`dynapm/eval/metrics.py`:** defines **`cohen_d_paired(x, y)`** as **mean(x−y) / std(x−y, ddof=1)** on finite pairs—paired Cohen’s **d** on differences.

**Bootstrap CI & Wilcoxon** are implemented in task scripts (e.g. Task 15/17b), not centrally in `metrics.py`. Example from `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task17b_provisioning.py`:

- **Bootstrap:** **`n_boot = 1000`** resamples of **paired per-VM deltas** (with replacement over VM index set); percentile **2.5 / 97.5** on bootstrap means → **95% CI**.
- **Wilcoxon:** `scipy.stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")` → **paired signed-rank**, **two-sided** p-value.
- **Significance:** Results report **exact** Wilcoxon **p**; there is **no** hard-coded **α = 0.05** gate in these utilities.

Task 15 report text mirrors the same bootstrap/Wilcoxon pattern for headline deltas (`n_bootstrap=1000`, paired VMs).

---

## Discrepancies or surprises

1. **Volatility metric** is **bootstrap-segment CPU-usage standard deviation**, **not** coefficient of variation and **not** memory-based.
2. **Evaluation grid for profile metrics** is **12-hour** spacing (with 24 h forward window and density filtering), while several baselines **refresh every 6 h**—the paper should distinguish **policy update cadence** vs **evaluation / GT grid**.
3. **RandomForest** omits explicit `min_samples_*`; defaults are **version-dependent**.
4. **Training** uses **30% prefix rows** only—**no** inner validation split inside the bootstrap segment.
5. **Corpus filter** is **only** `len(df) ≥ 2000`; no missing-data rate threshold appears in `filter.py`.
6. **`metrics.py`** does **not** implement bootstrap or Wilcoxon—those live in **`task*_*.py` scripts**.

---

## Files inspected (absolute paths)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/filter.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/loader.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/features.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_full_corpus_predictor.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task15_full_corpus.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/evaluation_clock.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/provisioning.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task17b_provisioning.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/metrics.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/requirements.txt`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/requirements.txt`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/README.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_01_data_layer.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_02_profile_and_split.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15_full_corpus.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md`
