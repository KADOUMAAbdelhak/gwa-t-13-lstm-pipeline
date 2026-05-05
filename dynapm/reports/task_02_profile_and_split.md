# DynAPM Task 2 — Profile schema, bootstrap split, ground-truth derivation

## What was implemented

1. **Profile schema** (`dynapm/core/schema.py`): `StaticFields` and `DynamicFields` (frozen), `ProfileInstance` with `last_updated` per dynamic field, `version=1`, and `to_json()` (timestamps as ISO-8601 strings). Module constants `CPU_CLASSES`, `MEMORY_CLASSES`, `DIURNAL_PATTERNS`, `BURST_PROFILES`.

2. **Bootstrap split** (`dynapm/data/bootstrap.py`): `compute_split(df, bootstrap_fraction=0.30)` with `n_bootstrap = floor(n * 0.30)`; `df_bootstrap = iloc[:n_bootstrap]`, `df_streaming` the remainder. `info` includes `n_total`, `n_bootstrap`, `n_streaming`, boundary timestamps, and duration in days. **Contract:** `LOCKED_BOOTSTRAP_FRACTION = 0.30`; passing any other value raises `ValueError`. `bootstrap_end` and `streaming_start` are the **same** cut timestamp (first row of the streaming segment).

3. **Provisioning** (`dynapm/data/provisioning.py`): `extract_provisioning` returns modal `CPU cores` and modal `Memory capacity provisioned [KB]` (RAM in GB = KB / 1024²). Warnings to stderr if either modal share is &lt; 99% of rows.

4. **Discretization** (`dynapm/core/discretization.py`): `compute_thresholds(df_bootstrap, metric)` → p20…p95; `classify_value` maps to idle/low/medium/high/burst per locked boundaries.

5. **Patterns** (`dynapm/core/pattern.py`): `derive_diurnal_pattern` (flat → weekly vs diurnal via autocorrelations at lags 288 and 2016 for 5‑minute data); `derive_burst_profile` vs bootstrap `p95`.

6. **Ground truth** (`dynapm/core/ground_truth.py`): `derive_ground_truth_at_t(df, t, …)` uses the **forward** window `[t, t + window_hours)` (default 24h). Labels: mean CPU% / Memory% classified with **frozen** bootstrap thresholds; `diurnal_pattern` from CPU% in the window; `burst_profile` from CPU% vs bootstrap CPU `p95`; empirical `p95_cpu_next_24h` / `p95_mem_next_24h` inside that window.

7. **Tests** (`dynapm/tests/test_profile_split.py`): seven tests covering split, provisioning, thresholds, classification boundaries, synthetic diurnal/burst, and ground-truth keys/vocabularies.

8. **Corpus pass** (`dynapm/scripts/run_bootstrap_characterization.py`): loads each kept VM, split, provisioning, thresholds, bootstrap diurnal + CPU/memory burst labels → `bootstrap_summary.parquet`.

---

## Files created or modified (absolute paths)

| Path |
|------|
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/provisioning.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/discretization.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/pattern.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_profile_split.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/run_bootstrap_characterization.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap_summary.parquet` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_02_profile_and_split.md` |

*(Existing empty package `__init__.py` files unchanged; `dynapm/data/loader.py` not modified.)*

---

## Test results

Full suite (`dynapm/tests/`):

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
..........                                                               [100%]
10 passed in 0.70s
```

*(Includes Task 1 tests + Task 2 tests.)*

Task 2-only run (`tests/test_profile_split.py`), verbose:

```text
tests/test_profile_split.py::test_compute_split_70_30 PASSED
tests/test_profile_split.py::test_provisioning_extraction PASSED
tests/test_profile_split.py::test_thresholds_monotone PASSED
tests/test_profile_split.py::test_classify_boundaries PASSED
tests/test_profile_split.py::test_derive_diurnal_synthetic PASSED
tests/test_profile_split.py::test_derive_burst_profile_thresholds PASSED
tests/test_profile_split.py::test_ground_truth_returns_all_fields PASSED
7 passed in 0.75s
```

---

## Bootstrap summary statistics

**Run:** `bootstrap_summary.parquet`, **1,576** rows (one per kept VM). **Wall-clock:** ~**102.2 s** for the full pass on this machine.

### Provisioned CPU cores (count per value)

| Cores | VMs |
|------:|----:|
| 1 | 209 |
| 2 | 992 |
| 4 | 263 |
| 6 | 39 |
| 8 | 73 |

### Provisioned RAM (GB, count per value)

| GB | VMs |
|---:|----:|
| 1.0 | 16 |
| 2.0 | 308 |
| 3.0 | 10 |
| 4.0 | 367 |
| 5.0 | 3 |
| 6.0 | 55 |
| 8.0 | 428 |
| 10.0 | 19 |
| 12.0 | 81 |
| 14.0 | 12 |
| 16.0 | 192 |
| 20.0 | 34 |
| 24.0 | 31 |
| 32.0 | 12 |
| 48.0 | 2 |
| 128.0 | 6 |

### Bootstrap `derive_diurnal_pattern` (CPU bootstrap segment)

| Pattern | VMs |
|---------|----:|
| irregular | 919 |
| weekly | 284 |
| flat | 201 |
| diurnal | 172 |

### Bootstrap burst (CPU / Memory), vs bootstrap `p95`

| Label | CPU VMs | Mem VMs |
|-------|--------:|--------:|
| frequent | 1548 | 1575 |
| occasional | 21 | 1 |
| none | 7 | 0 |

### Kept VMs per trace (from summary file)

| Trace | VMs |
|-------|----:|
| Materna-Trace-1 | 511 |
| Materna-Trace-2 | 521 |
| Materna-Trace-3 | 544 |

### Distribution of per-VM bootstrap **p80** thresholds (variation across VMs)

| Metric | min | p25 | median | p75 | max |
|--------|-----|-----|--------|-----|-----|
| `cpu_p80` | 0.07 | 0.54 | 1.94 | 4.87 | 100.00 |
| `mem_p80` | 0.00 | 2.32 | 5.64 | 13.79 | 75.59 |

---

## Assumptions

- **Time index:** Rows are in chronological order after `load_vm_csv`; split uses positional `iloc`, consistent with Task 1 sorting.
- **Bootstrap boundary:** `bootstrap_end` / `streaming_start` refer to the **first timestamp of the streaming segment** (cut instant); bootstrap duration uses span from first to **last** bootstrap row.
- **Provisioning ties:** Modal value ties resolved by taking the **smallest** numeric candidate so outputs are deterministic.
- **`derive_diurnal_pattern`:** Spec text for weekly vs diurnal was interpreted as: **weekly** if both autocorrelations valid and `ac2016 > 0.5` and `ac2016 > ac24`; **diurnal** if `ac24 > 0.5` and (`ac2016` invalid **or** `ac24 > ac2016`). Short series without lag‑2016 still allow **diurnal** when daily autocorr dominates (needed for tests and short windows).
- **`derive_ground_truth_at_t`:** Single field `burst_profile` follows **CPU** usage vs bootstrap CPU `p95` (aligned with using CPU series for diurnal in the window).
- **Ground-truth window:** Half-open `[t, t + window_hours)` so exactly one week of 5‑minute samples ≈ 2016 steps when fully contained.

---

## Surprises

- Many **stderr provisioning warnings** during the corpus pass: RAM (and sometimes CPU) provisioning columns are **not strictly constant** on some VMs (dynamic resizing in data), so modal mass &lt; 99% is common—warnings are expected, not errors.
- **`bootstrap_burst_cpu`** is almost always **frequent** because exceedances vs bootstrap `p95` are common by construction; memory burst is almost always frequent as well.
- **`cpu_p80` max = 100** on at least one VM (saturation / spike behavior).
- Reading Parquet under sandbox triggered **PyArrow sysctl cache-size warnings** (harmless IO noise).

---

## Commands

*(Outputs truncated to ≤20 lines where applicable.)*

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/ -q
..........                                                               [100%]
10 passed in 0.70s
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/run_bootstrap_characterization.py
[dynapm.provisioning] warning: mode coverage <99% ...
(repeated for multiple VMs)
wrote .../dynapm/data/bootstrap_summary.parquet rows=1576 wall_s=102.176
```

```text
$ cd .../gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python << 'PY'
import pandas as pd
df = pd.read_parquet("dynapm/data/bootstrap_summary.parquet")
...
PY
(rows 1576; value_counts / quantiles as in tables above)
```

---

## Next-step suggestion (advisory)

**Task 3** should introduce the **forecasting targets and training loop** on the streaming segment only: build supervised examples that use the frozen bootstrap thresholds for labeling or features, predict `CPU usage [%]` and `Memory usage [%]` with agreed horizons, and evaluate against simple baselines—without yet wiring drift detection or clustering, so the experiment matches the locked 70/30 temporal split and the ground-truth dictionaries defined here.
