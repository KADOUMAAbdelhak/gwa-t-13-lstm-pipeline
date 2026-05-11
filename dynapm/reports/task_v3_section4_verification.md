# Task V3 — Remaining Section 4 numbers (read-only verification)

This report fills gaps for Section 4 prose using **repository files and a live-environment Python check** only. No source files were modified; **no experiment pipelines were re-executed** (predictor training, runner batch jobs, or provisioning drivers).

---

## V3-A: Python runtime version

**Pinned in repo:** None found.

| Check | Result |
|--------|--------|
| `dynapm/.python-version` | **Absent** |
| `pyvenv.cfg` under repo | **Absent** |
| `.github/workflows/*.yml` | **Absent** |
| `dynapm/pytest.ini` | Sets `pythonpath` / `testpaths` only — **no** Python version pin |
| `tox.ini` | **Absent** |
| `pyproject.toml` (repo root) | **Absent** |
| Shebangs (`dynapm/scripts/*.py`) | Often `#!/usr/bin/env python3` (e.g. `task15_full_corpus.py`, `task18b_corpus_churn_rerun.py`, …) — specifies **Python 3**, **not** a minor/patch level |

**Environment used for this verification (read-only `python3 -c` / ad-hoc analysis):**

```text
3.13.2 (main, Feb  4 2025, 14:51:09) [Clang 16.0.0 (clang-1600.0.26.6)]
```

**Interpretation:** The codebase does not lock an interpreter version in-repo. The number above is the **currently active `python3` on the machine where this verification ran**; the **original** Task 4 / 15 / 17b / 18b runs may have used a different version (typically whatever their `venv` was created with). For the paper, either record **the interpreter from the venv that will ship with artifacts** or state **“Python 3”** plus `dynapm/requirements.txt` floors unless you reproduce the full environment lockfile.

---

## V3-B: Average evaluation points per VM

**Source file:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`

This table has **9,456** rows (**1,576** VMs × **6** methods). Field **`n_eval_points_verified`** is **identical across all six methods for every `(trace, vm_id)`** (0 VMs disagree). **Evaluation timestamps** come from the precomputed ground-truth parquet per VM; the runner scores only at those times (see Task V2 / `evaluation_clock` / GT construction).

**Take one method’s slice** (e.g. `static_v2`) to avoid counting each VM six times — statistics below are **per VM**, **N = 1,576**.

| Statistic | `n_eval_points_verified` |
|-----------|-------------------------:|
| Mean | **40.496** (40.4961929…) |
| Median | **41** |
| Min | **7** |
| Max | **48** |
| Standard deviation (population, ddof=0) | **6.970** |
| Standard deviation (sample-style, ddof=1) | **6.973** |

**Note:** These are **verified evaluation points** (`n_eval_points_verified`), not raw pre-filter grid length—consistent with the data-aware **≥ 200 observations** rule in the evaluation clock / GT pipeline.

---

## V3-C: Bootstrap segment duration range

**Field `bootstrap_duration_days`** is **not** a column in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap_summary.parquet` (that file stores boundaries and `n_bootstrap`, etc., but not duration; see `run_bootstrap_characterization.py`).

**Definition (canonical):** `dynapm/data/bootstrap.py` — `bootstrap_duration_days` = `(last timestamp of bootstrap index − bootstrap_start) / 86400` (span of the bootstrap **segment rows**, not `(bootstrap_end − bootstrap_start)`; `bootstrap_end` is the **cut** timestamp shared with `streaming_start`).

**Procedure for this verification:** For each of the **1,576** entries in `kept_vms.json`, load `old/data/raw/<trace>/<vm_id>.csv` via `load_vm_csv`, call `compute_split(df)`, and read `info["bootstrap_duration_days"]`. (Purely **re-stating** the locked split on frozen inputs—**not** re-running predictor or evaluation jobs.)

| Statistic | `bootstrap_duration_days` |
|-----------|--------------------------:|
| Mean | **9.312** (9.31235…) |
| Median | **9.295** (9.29514…) |
| Min | **2.094** |
| Max | **15.649** |
| Standard deviation (ddof=0) | **1.501** |

---

## V3-D: Memory class accuracy CI (PP vs Periodic-window)

**File:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table2_pairwise_deltas.csv`

The CSV uses comparison label **`Periodic-Predictor vs Periodic-window`** (same as **PP vs periodic-window** in prose).

**Row:** `metric == mem_acc`

| Column | Value | ×100 (percentage points) |
|--------|------:|-------------------------:|
| `mean_delta` | 0.14542521 | **14.5425** |
| `ci_low` | 0.13714282 | **13.7143** |
| `ci_high` | 0.15309575 | **15.3096** |
| `n_pairs` | 1576 | — |

**Draft introduction claim:** “14.5 percentage points (95% CI [13.8, 15.3])” **vs** table:

- **Mean:** 14.5 vs **14.54** — **one-decimal agreement**; slight rounding.
- **CI:** **[13.8, 15.3]** vs **[~13.71, ~15.31]** — **rounded to one decimal**, the lower bound **13.7** rounds to **13.7** or **14** depending on rule; **13.8** is **0.1 pp** above the tabulated low. **Recommend aligning prose to the CSV** (e.g. **14.54 pp, 95% CI [13.71, 15.31]** or round consistently to one decimal: **14.5 pp, CI [13.7, 15.3]**).

---

## V3-E: Total wall-clock (major task reports)

Wall times are **as logged in task artifacts / markdown** on the machine that ran each job—they are **not additive** across different hardware sessions in a scientifically strict sense, but the **sum** is useful for order-of-magnitude “compute budget” narrative.

| Task | Role | Reported wall-clock | Primary citation |
|------|------|-------------------|------------------|
| **Task 4 — full-corpus predictor** | RF residual training, all kept VMs | **782.94 s** | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/predictor_run_meta.json` → `"wall_seconds": 782.9420761250658` |
| **Task 15** | Full-corpus profile eval (6 methods × 1,576 VMs) | **1162.0 s** | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15_full_corpus.md` L7 |
| **Task 17b** | Provisioning replay + aggregates | **~20 min** (narrative only) | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md` L9 — **no** `wall_seconds` figure in that file |
| **Task 18b** | Corpus churn rerun (PP + V3, with `update_log`) | **296.87 s** | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_18b_corpus_normalized_churn.md` L5 |

**Partial sum (Tasks 4 + 15 + 18b, exact):**  
782.942 + 1162.0 + 296.87 = **2241.81 s** ≈ **37.4 min**.

**If** Task 17b is approximated as **20 min → 1200 s**, rough total ≈ **3442 s** ≈ **57.4 min** — treat as **illustrative** until a numeric 17b log is archived.

---

## Files inspected (absolute paths)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/pytest.ini`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/bootstrap.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/run_bootstrap_characterization.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/predictor_run_meta.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table2_pairwise_deltas.csv`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15_full_corpus.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_18b_corpus_normalized_churn.md`
- Representative shebangs: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task15_full_corpus.py`, `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task18b_corpus_churn_rerun.py`
