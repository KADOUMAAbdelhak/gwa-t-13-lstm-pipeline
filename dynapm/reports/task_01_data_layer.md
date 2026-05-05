# DynAPM Task 1 — Environment, data loader, and corpus filter

## Findings

### 1. Python environment setup

**Choice:** `venv` plus `dynapm/requirements.txt`.

**Rationale:** Standard library tooling needs no extra tooling (Poetry/uv), keeps dependencies explicit in one file, and matches the requested layout `dynapm/.venv`.

**Location:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/.venv`

**Resolved versions** (import verified for each listed package):

| Package | Version |
|---------|---------|
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| matplotlib | 3.10.9 |
| scipy | 1.17.1 |
| river | 0.24.2 |
| hdbscan | 0.8.42 |
| pyarrow | 24.0.0 |
| joblib | 1.5.3 |
| pytest | 9.0.3 |

Constraint lines in `requirements.txt` use the requested minimum versions (`>=`); pip resolved to the versions above on Python 3.13.2 / macOS arm64.

---

### 2. Data loader module

**Files:**

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/__init__.py` (empty)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/__init__.py` (empty)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/loader.py`

**`load_vm_csv(path)`:** Reads with `sep=';'`, `decimal=','`, `quotechar='"'`, `dtype=str`; strips column names; parses `Timestamp` with format `%d.%m.%Y %H:%M:%S`; coerces the twelve non-timestamp fields to float (comma→dot on strings); sets `Timestamp` as the index (`DatetimeIndex`, ascending sort); returns a DataFrame with **12** columns and named index `Timestamp`.

**`iter_corpus(root=None)`:** Default `root` is the parent of `Materna-Trace-{1,2,3}`, i.e.  
`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw`. Yields `(trace_name, vm_id, df)` for each `*.csv`, skipping `00-Trace-Info.txt` and non-CSV files.

---

### 3. Filter module

**File:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/filter.py`

**`filter_corpus(min_timesteps=2000)`:** Iterates `iter_corpus()`, keeps VMs with `len(df) >= min_timesteps`, returns `(kept_tuples, stats_dict)` with keys `total_seen`, `kept`, `dropped`, `dropped_list` (each drop: `trace`, `vm_id`, `row_count`).

**Helper:** `vm_passes_min_timesteps(df, min_timesteps)` — used by tests and filter logic.

**Persistence:** `kept` is written to  
`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json` as a JSON array of `{"trace": ..., "vm_id": ...}` objects (equivalent to the tuple list for downstream consumers).

---

### 4. Unit tests

**File:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_loader.py`

**Config:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/pytest.ini` sets `pythonpath = ..` so `dynapm` imports resolve when pytest runs from `dynapm/`.

**Result:** **3 passed** (see Commands section).

---

### 5. Corpus-wide filter run (`min_timesteps=2000`)

| Metric | Value |
|--------|------:|
| total_seen | 1,592 |
| kept | 1,576 |
| dropped | 16 |

**Full dropped list `(trace_name, vm_id, row_count)`** (row counts match data rows after header in source CSVs; consistent with `len(load_vm_csv(...))`):

1. `(Materna-Trace-1, 211, 1004)`
2. `(Materna-Trace-1, 247, 652)`
3. `(Materna-Trace-1, 268, 129)`
4. `(Materna-Trace-1, 280, 527)`
5. `(Materna-Trace-1, 282, 94)`
6. `(Materna-Trace-1, 435, 82)`
7. `(Materna-Trace-1, 505, 1440)`
8. `(Materna-Trace-2, 125, 92)`
9. `(Materna-Trace-2, 229, 1152)`
10. `(Materna-Trace-2, 231, 1152)`
11. `(Materna-Trace-2, 235, 1152)`
12. `(Materna-Trace-2, 402, 1440)`
13. `(Materna-Trace-2, 520, 1445)`
14. `(Materna-Trace-3, 267, 582)`
15. `(Materna-Trace-3, 541, 1147)`
16. `(Materna-Trace-3, 542, 1147)`

**Saved kept list:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json`

---

### 6. Sample load stress test (10 VMs, seed 42)

Using `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json`, `random.Random(42).sample(kept, 10)`, then `load_vm_csv` per VM.

**Load times (seconds):**

| Stat | Value (s) |
|------|-----------|
| min | 0.0175 |
| median | 0.0574 |
| max | 0.0834 |

**Sampled VMs:**  
`(Materna-Trace-3, 344)`, `(Materna-Trace-1, 304)`, `(Materna-Trace-1, 139)`, `(Materna-Trace-3, 534)`, `(Materna-Trace-2, 14)`, `(Materna-Trace-1, 90)`, `(Materna-Trace-1, 515)`, `(Materna-Trace-1, 357)`, `(Materna-Trace-3, 525)`, `(Materna-Trace-1, 287)`.

---

## Assumptions

- **Repository layout:** `dynapm` lives directly under the repo root so `_repo_root()` in `loader.py` resolves to `.../gwa-t-13-lstm-pipeline` and `old/data/raw` contains the three Materna trace folders.
- **Schema:** All VM CSVs share the 13-column header in `SCHEMA_COLUMNS` in `loader.py` (locked schema from Task 0).
- **Row count:** Filtering uses `len(df)` after full parse; dropped VM row counts were cross-checked with raw line counts minus header for reporting convenience.
- **JSON persistence:** `kept_vms.json` uses objects with `trace` / `vm_id` keys instead of nested tuples; semantically equivalent to a list of `(trace_name, vm_id)` pairs.

---

## Deviations from specification

- **`kept_vms.json` format:** Stored as JSON array of `{"trace", "vm_id"}` rather than serializing Python tuple literals; easier for non-Python readers.
- **Running imports:** `python -c "from dynapm..."` must be executed with the **repository root** on `PYTHONPATH` (i.e. run from repo root or set `PYTHONPATH`); running from inside `dynapm/` without adjusting path fails to resolve package `dynapm` (pytest works because `pytest.ini` sets `pythonpath = ..`).

---

## Errors / surprises

- **Full `filter_corpus()` runtime:** ~100 seconds wall-clock for 1,592 full-file parses on this machine (Apple M3, SSD)—acceptable but worth noting for batch pipelines.
- **`ModuleNotFoundError: dynapm`** when invoking the interpreter from `cd dynapm` without `PYTHONPATH`; resolved by running from repo root.

---

## Commands

*(Output truncated to ≤20 lines per command where applicable.)*

**Create venv and install**

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
[... many lines ...]
Successfully installed contourpy-1.3.3 cycler-0.12.1 fonttools-4.62.1 hdbscan-0.8.42 [...] xgboost-3.2.0
```

**Pytest**

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_loader.py -v
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.3, pluggy-1.6.0 -- .../dynapm/.venv/bin/python3.13
rootdir: .../dynapm
configfile: pytest.ini
collecting ... collected 3 items

tests/test_loader.py::test_load_one_csv PASSED                           [ 33%]
tests/test_loader.py::test_load_two_csvs_different_traces PASSED         [ 66%]
tests/test_loader.py::test_filter_drops_short PASSED                     [100%]

============================== 3 passed in 8.59s ===============================
```

**Corpus filter (from repo root)**

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python -c "
from dynapm.data.filter import filter_corpus
kept, stats = filter_corpus(2000)
print('kept', len(kept), 'dropped', stats['dropped'], 'total', stats['total_seen'])
"
kept 1576 dropped 16 total 1592
```

**Import check (resolved versions)**

```text
$ dynapm/.venv/bin/python -c "import importlib.metadata as m; pkgs=[...]; ..."
numpy 2.4.4
pandas 2.3.3
scikit-learn 1.8.0
xgboost 3.2.0
matplotlib 3.10.9
scipy 1.17.1
river 0.24.2
hdbscan 0.8.42
pyarrow 24.0.0
joblib 1.5.3
pytest 9.0.3
```

**Stress test**

```text
$ cd .../gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python << 'PY'
[ script: load 10 VMs seed 42 ]
PY
seconds min/median/max: 0.0175... 0.0573... 0.0834...
paths sampled:
Materna-Trace-3 344
Materna-Trace-1 304
...
```

**Dropped-list verification (line-count sanity check, not used in production code)**

```text
$ cd .../gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python << 'PY'
[ line-count script ]
PY
count 16
('Materna-Trace-1', '211', 1004)
...
```

---

## Absolute paths — files created or modified (Task 1)

| Path |
|------|
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/__init__.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/requirements.txt` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/pytest.ini` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/__init__.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/__init__.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/loader.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/filter.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_loader.py` |
| `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_01_data_layer.md` |

Virtualenv directory: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/.venv/` (generated; not hand-edited).

---

## Next-step suggestion (advisory)

**Task 2** should define the **forecasting experiment protocol** on the **1,576 kept VMs**: fixed train/validation/test splits over time (and optionally per-trace strata), baseline models for `CPU usage [%]` and `Memory usage [%]` (e.g. seasonal naive, sklearn regressors, optional sequence model interface), and evaluation metrics plus a standard **batch runner** that streams VMs via `iter_corpus`/`kept_vms.json` so runs stay reproducible without reloading the full corpus filter each time.
