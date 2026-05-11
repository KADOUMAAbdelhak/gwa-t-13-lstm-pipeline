# DynAPM Task 18B — corpus rerun for normalized churn

## What was rerun

`runner_v2.run_method_on_vm` was executed for **only** `periodic_predictor_6h_v2` and `dynapm_v3_v2` on the **1,576** VMs in `dynapm/data/kept_vms.json` (**3,152** runs). Each run wrote a **full** result dict (including **`update_log`**) to **`dynapm/results/eval_full_corpus_with_log/`** using filenames `method__trace__vm_id.json`. The original **`dynapm/results/eval_full_corpus/`** tree was **not** touched, so Task 15 headline accuracy/MAE aggregates remain valid. Driver: **`dynapm/scripts/task18b_corpus_churn_rerun.py`** (`joblib`, `n_jobs=8`). Wall-clock for the full batch: **296.87 s** (~4.9 min) on this machine (all runs fresh; **0** failures).

## Corpus normalized churn table

| method | mean_refresh_events | median | mean_periodic | mean_drift |
| ------ | ------------------: | -----: | ------------: | ---------: |
| PP | 85.3902 | 86.0 | 85.3902 | 0 |
| V3 | 91.3458 | 89.0 | 85.3902 | 5.9556 |

- **`mean_refresh_events`**: `count_unique_refresh_timestamps(update_log)` per VM, then mean across VMs.
- **V3 `mean_periodic` / `mean_drift`**: from runner fields `n_periodic_refreshes` / `n_drift_refreshes` (distinct timestamp×reason summaries in `runner_v2`).
- **PP**: per spec, `n_periodic_refreshes = n_refresh_events` and `n_drift_refreshes = 0`.

Per-VM paired delta **V3 − PP** (refresh counts): **mean = 5.955584**, **95% bootstrap CI** [5.583740, 6.333931], **Wilcoxon p ≈ 1.54 × 10⁻²¹²**, **n_pairs = 1576**.

## Comparison vs sample numbers from Task 18

Task 18’s **`normalized_refresh_counts_sample.parquet`** (100‑VM Task 12/14 JSON) reported means **dynapm_v3 ≈ 91.15**, **periodic_predictor ≈ 85.57** with medians **88 / 86**. Corpus means **91.35** and **85.39** with medians **89** and **86** are essentially the same (~0.2 refresh events off on the PP mean), confirming the sample was representative of the corpus mix.

## Updated Table 1 cells

Only **`mean_n_refresh_events`** and **`median_n_refresh_events`** were edited in **`dynapm/reports/paper/table1_method_comparison.csv`**; all other numeric columns unchanged.

| row | column | before | after |
|-----|--------|-------:|------:|
| `dynapm_v3_v2` | `mean_n_refresh_events` | 365.4 | 91.3 |
| `dynapm_v3_v2` | `median_n_refresh_events` | 356.0 | 89.0 |
| `periodic_predictor_6h_v2` | `mean_n_refresh_events` | 187.4 | 85.4 |
| `periodic_predictor_6h_v2` | `median_n_refresh_events` | 187.0 | 86.0 |

**Left unchanged:** `static_v2`, `periodic_6h_v2`, `reactive_v2`, **`full_rewrite_v2`** still use corpus **`total_updates`** from Task 15, which aligns with telemetry row counts closely enough here (they do **not** share the PP/V3 pattern of asymmetric multi-field periodic logging versus drift-only deltas).

## Checkpoint questions

| Q | PASS/FAIL | Evidence |
|---|-----------|---------|
| **Q1** | **PASS** | V3 corpus **mean_refresh_events = 91.35** ∈ [80, 110]. |
| **Q2** | **PASS** | PP corpus **mean = 85.39** ∈ [75, 100]. |
| **Q3** | **PASS** | Mean paired **Δ(V3 − PP) = 5.956** ∈ [3, 15]. |
| **Q4** | **PASS** | Bootstrap CI **[5.584, 6.334]** excludes **0** (Wilcoxon **p ≈ 1.5 × 10⁻²¹²**). |
| **Q5** | **PASS** | `pandas.read_csv` loads updated Table 1 with **17** columns and **6** rows OK. |

## Files created or modified (absolute paths)

**Created**

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task18b_corpus_churn_rerun.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/normalized_refresh_counts_corpus.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_full_corpus_with_log/` (**3,152** JSON files; not enumerated here)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_18b_corpus_normalized_churn.md`

**Modified**

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v2.py` (module docstring clarifying serialized **`update_log`** and related audit keys; return payload already included **`update_log`**)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table1_method_comparison.csv` (four churn cells only)

## Test results

```text
66 passed, 1 skipped in ~8.8s
```

## Assumptions

- **Kept list** (`kept_vms.json`) matches Task 15’s evaluated population; parallel join on `(trace, vm_id)` yields **1576** pairs for both methods.
- **PP** has no drift rows; periodic-only normalized counts match **`n_periodic_refreshes`** policy from the runner.
- **JSON** round-trip via `json.dumps(..., default=str)` preserves timestamps in a form **`count_unique_refresh_timestamps`** can parse.

## Surprises

- **`mean_periodic` for V3** across the corpus equals **`mean_refresh_events` for PP** (both **85.39** to four decimals), matching the shared 6‑hour periodic shell; **all corpus-wide lift in normalized churn** for V3 is carried by **`mean_drift ≈ 5.96`** extra distinct refresh timestamps on average.
- End-to-end **~5 min** wall time for **3,152** full streaming replays was faster than a conservative prior estimate (I/O + `loky` locality on this host).

## Commands

```bash
cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline
PYTHONPATH=. pytest dynapm/tests -q --tb=no
PYTHONPATH=. python dynapm/scripts/task18b_corpus_churn_rerun.py --n-jobs 8
# JSON summary: {"wall_clock_s_total": 296.87, "fresh_ok": 3152, "cached": 0, "failed": 0}
# Truncated: no per-VM logs.
python3 -c "import pandas as pd; pd.read_csv('dynapm/reports/paper/table1_method_comparison.csv')"
```

## Next-step suggestion

If the manuscript body still cites “**churn ≈ 365 vs 187 rows**”, update prose to normalized **(~91 vs ~85 refresh events)** and cite **`normalized_refresh_counts_corpus.parquet`**; optionally add one sentence that **`total_updates`** in legacy Task 15 JSON was dropped specifically to shrink artifacts, while this rerun directory hosts auditable **`update_log`** slices for PP/V3 only—no figure or Table 2/3 regeneration is required unless the narrative crosses those assets.
