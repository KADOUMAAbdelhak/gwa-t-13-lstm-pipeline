# Task V4 — Section 5 headline numbers verification (read-only)

All **paper CSV** values below are **verbatim** from `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/*.csv` (no rounding applied here). Supplementary statistics were derived **read-only** from `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`, `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/normalized_refresh_counts_corpus.parquet`, and `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/provisioning_eval_records.parquet` using the same `safety_factor` as Task 17b / Task 18 packaging (`dynapm.eval.provisioning.SAFETY_FACTOR = 1.2`).

---

## V4-A — Table 1 extraction (`table1_method_comparison.csv`)

Verbatim rows from the repository CSV (no rounding):

| method | n_vms | mean_cpu_class_acc | ci_low_cpu_acc | ci_high_cpu_acc | mean_mem_class_acc | ci_low_mem_acc | ci_high_mem_acc | mean_cpu_p95_mae | ci_low_cpu_mae | ci_high_cpu_mae | mean_mem_p95_mae | ci_low_mem_mae | ci_high_mem_mae | mean_n_refresh_events | median_n_refresh_events | mean_cpu_under_rate |
|--------|------:|---------------------:|---------------:|----------------:|-------------------:|---------------:|----------------:|-----------------:|---------------:|----------------:|-----------------:|---------------:|----------------:|----------------------:|------------------------:|--------------------:|
| dynapm_v3_v2 | 1576 | 0.8698 | 0.8626 | 0.8783 | 0.8929 | 0.8868 | 0.8992 | 1.042 | 0.885 | 1.218 | 1.127 | 1.028 | 1.227 | 91.3 | 89.0 | 0.0611 |
| full_rewrite_v2 | 1576 | 0.7577 | 0.7485 | 0.7671 | 0.7631 | 0.7539 | 0.7723 | 1.29 | 1.152 | 1.44 | 1.499 | 1.368 | 1.642 | 1252.9 | 1217.5 | 0.0751 |
| periodic_6h_v2 | 1576 | 0.737 | 0.7278 | 0.7468 | 0.7443 | 0.7353 | 0.7536 | 1.4 | 1.248 | 1.565 | 1.668 | 1.52 | 1.825 | 140.1 | 146.5 | 0.0852 |
| periodic_predictor_6h_v2 | 1576 | 0.8675 | 0.8603 | 0.8759 | 0.8897 | 0.8834 | 0.8963 | 1.062 | 0.903 | 1.242 | 1.143 | 1.042 | 1.245 | 85.4 | 86.0 | 0.0613 |
| reactive_v2 | 1576 | 0.6445 | 0.6328 | 0.6567 | 0.63 | 0.6183 | 0.6426 | 2.239 | 1.965 | 2.543 | 2.377 | 2.169 | 2.597 | 552.0 | 568.0 | 0.1637 |
| static_v2 | 1576 | 0.4745 | 0.4568 | 0.4925 | 0.5889 | 0.5722 | 0.6051 | 2.56 | 2.2 | 2.97 | 2.446 | 2.187 | 2.705 | 0.0 | 0.0 | 0.1025 |

---

## V4-B — Table 2 extraction (`table2_pairwise_deltas.csv`)

| comparison | metric | mean_delta | ci_low | ci_high | wilcoxon_p | cohen_d | n_pairs |
|------------|--------|------------|--------|---------|------------|---------|---------|
| Periodic-Predictor vs Periodic-window | cpu_acc | 0.13048404 | 0.12082397 | 0.14032743 | 5.17154e-148 | 0.691773 | 1576 |
| Periodic-Predictor vs Periodic-window | mem_acc | 0.14542521 | 0.13714282 | 0.15309575 | 1.50055e-190 | 0.891774 | 1576 |
| Periodic-Predictor vs Periodic-window | cpu_p95_mae | -0.3383894 | -0.48713828 | -0.18758474 | 1.19551e-42 | -0.115704 | 1576 |
| Periodic-Predictor vs Periodic-window | mem_p95_mae | -0.52455803 | -0.62634316 | -0.42372903 | 2.37783e-22 | -0.24798 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | cpu_acc | 0.00236181 | 0.00186906 | 0.00284539 | 2.75927e-18 | 0.234422 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | mem_acc | 0.00318647 | 0.00257673 | 0.00373098 | 2.37303e-25 | 0.277161 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | cpu_p95_mae | -0.01980611 | -0.02505921 | -0.01514227 | 1.53029e-39 | -0.198382 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | mem_p95_mae | -0.01661361 | -0.02171819 | -0.0125437 | 1.8689e-09 | -0.170987 | 1576 |
| Periodic-Predictor vs Periodic-window | cpu_under_rate | -0.02393593 | -0.03069289 | -0.01671844 | 7.05737e-62 | -0.165622 | 1576 |
| Periodic-Predictor vs Periodic-window | mem_under_rate | -0.01545303 | -0.02239241 | -0.00841579 | 1.858e-57 | -0.109174 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | cpu_under_rate | -0.00015561 | -0.00042888 | 0.00010243 | 0.207246 | -0.02761 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | cpu_over_rate | -0.00312661 | -0.00365901 | -0.00261682 | 1.2418e-26 | -0.300567 | 1576 |

---

## V4-C — Table 3 extraction (`table3_per_tier.csv`)

| volatility_tier | method | mean_cpu_class_acc | mean_cpu_under_rate |
|-----------------|--------|--------------------|--------------------|
| high | dynapm_v3_v2 | 0.8802 | 0.0902 |
| high | full_rewrite_v2 | 0.7616 | 0.1336 |
| high | periodic_6h_v2 | 0.7384 | 0.1479 |
| high | periodic_predictor_6h_v2 | 0.876 | 0.0908 |
| high | reactive_v2 | 0.6181 | 0.2952 |
| high | static_v2 | 0.5354 | 0.1348 |
| low | dynapm_v3_v2 | 0.8768 | 0.0308 |
| low | full_rewrite_v2 | 0.7561 | 0.0226 |
| low | periodic_6h_v2 | 0.7358 | 0.0283 |
| low | periodic_predictor_6h_v2 | 0.876 | 0.0307 |
| low | reactive_v2 | 0.6957 | 0.0544 |
| low | static_v2 | 0.4134 | 0.0529 |
| medium | dynapm_v3_v2 | 0.8524 | 0.0624 |
| medium | full_rewrite_v2 | 0.7555 | 0.069 |
| medium | periodic_6h_v2 | 0.7367 | 0.0794 |
| medium | periodic_predictor_6h_v2 | 0.8504 | 0.0623 |
| medium | reactive_v2 | 0.6197 | 0.1412 |
| medium | static_v2 | 0.4747 | 0.1197 |

---

## V4-D — Figure 5 statistics (per-VM V3 − PP on CPU class accuracy)

**Data source:** Inner-join on `(trace, vm_id)` of `dynapm_v3_v2` and `periodic_predictor_6h_v2` rows in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`, column `cpu_class_accuracy`. Delta = **acc(V3) − acc(PP)** (matches `figure_v3_vs_pp_scatter` in `dynapm/scripts/task18_paper_packaging.py`, which plots x = PP, y = V3).

| Quantity | Value |
|----------|------:|
| VM count | 1576 |
| V3 > PP (above y = x) | 135 |
| V3 = PP (on diagonal) | 1427 |
| V3 < PP (below y = x) | 14 |
| Mean Δ (V3 − PP) | 0.002361806012331522 |
| Median Δ | 0.0 |
| Min Δ | -0.052631578947368474 |
| Max Δ | 0.125 |

---

## V4-E — Provisioning sensitivity (`safety_factor` ∈ {1.1, 1.2, 1.5})

Source: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md` §“Safety-factor sensitivity” (lines 71–90), **verbatim**.

Sorted by **safety_factor** ascending, then **mean_cpu_under_rate** ascending within group:

| safety_factor | method | mean_cpu_under_rate | mean_cpu_over_rate |
|---------------|--------|--------------------:|-------------------:|
| 1.1 | periodic_predictor_6h_v2 | 0.11502639 | 0.04700724 |
| 1.1 | dynapm_v3_v2 | 0.11549527 | 0.04431023 |
| 1.1 | full_rewrite_v2 | 0.11577017 | 0.04671187 |
| 1.1 | periodic_6h_v2 | 0.12882523 | 0.04969856 |
| 1.1 | static_v2 | 0.15330526 | 0.13238587 |
| 1.1 | reactive_v2 | 0.22371583 | 0.05217897 |
| 1.2 | dynapm_v3_v2 | 0.06113065 | 0.05732293 |
| 1.2 | periodic_predictor_6h_v2 | 0.06128626 | 0.06044954 |
| 1.2 | full_rewrite_v2 | 0.07512228 | 0.06113127 |
| 1.2 | periodic_6h_v2 | 0.08522219 | 0.06444185 |
| 1.2 | static_v2 | 0.10250519 | 0.15984525 |
| 1.2 | reactive_v2 | 0.16367618 | 0.06756353 |
| 1.5 | dynapm_v3_v2 | 0.02844161 | 0.31488314 |
| 1.5 | periodic_predictor_6h_v2 | 0.02858162 | 0.31789987 |
| 1.5 | full_rewrite_v2 | 0.03912966 | 0.40069466 |
| 1.5 | periodic_6h_v2 | 0.04616459 | 0.39769397 |
| 1.5 | static_v2 | 0.05818685 | 0.51784279 |
| 1.5 | reactive_v2 | 0.09898519 | 0.34191392 |

---

## V4-F — Provisioning deltas with CI (paper Table 2 + gaps)

### F.1 — Rows present in `table2_pairwise_deltas.csv` (verbatim)

**Periodic-Predictor vs Periodic-window**

| metric | mean_delta | ci_low | ci_high | wilcoxon_p | cohen_d | n_pairs |
|--------|------------|--------|---------|------------|---------|---------|
| cpu_under_rate | -0.02393593 | -0.03069289 | -0.01671844 | 7.05737e-62 | -0.165622 | 1576 |
| mem_under_rate | -0.01545303 | -0.02239241 | -0.00841579 | 1.858e-57 | -0.109174 | 1576 |

**DynAPM-V3 vs Periodic-Predictor**

| metric | mean_delta | ci_low | ci_high | wilcoxon_p | cohen_d | n_pairs |
|--------|------------|--------|---------|------------|---------|---------|
| cpu_under_rate | -0.00015561 | -0.00042888 | 0.00010243 | 0.207246 | -0.02761 | 1576 |
| cpu_over_rate | -0.00312661 | -0.00365901 | -0.00261682 | 1.2418e-26 | -0.300567 | 1576 |

Interpretation sign (`dynapm/scripts/task18_paper_packaging.py`): per-VM delta = **xa − xb** where the first named method is **xa** (e.g. PP − PW for the PP vs window row).

### F.2 — Not in Table 2: `cpu_over_rate` / `mem_over_rate` (PP vs window) and `mem_*` (V3 vs PP)

These comparisons are **not** exported to `table2_pairwise_deltas.csv` by `build_table_pairwise`. Statistics below were recomputed once from **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/provisioning_eval_records.parquet`** using `dynapm/scripts/task17b_provisioning.py` helpers (`per_vm_metric`, `SAFETY_FACTOR`, `bootstrap_mean_ci` with `n_boot=1000`, `safe_wilcoxon`, `cohen_d_paired`), with bootstrap seeds **312–315** to mirror ad-hoc offsets (paired **n = 1576**). They align with the narrative table in `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md` §“Pairwise deltas” (mean / CI / Wilcoxon).

| comparison | metric | mean_delta | ci_low | ci_high | wilcoxon_p | cohen_d | n_pairs |
|------------|--------|------------|--------|---------|------------|---------|---------|
| Periodic-Predictor vs Periodic-window | cpu_over_rate | -0.003992310702069391 | -0.010689141798415252 | 0.0032238815175606003 | 2.5272054081754596e-09 | -0.028443 | 1576 |
| Periodic-Predictor vs Periodic-window | mem_over_rate | -0.031969699471961655 | -0.03638967778325799 | -0.027267728788432484 | 4.7710910957226434e-83 | -0.360704 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | mem_under_rate | -0.00017215876883192507 | -0.0004273740402285514 | 7.019814980441947e-05 | 0.23328607534420864 | -0.036400 | 1576 |
| DynAPM-V3 vs Periodic-Predictor | mem_over_rate | -0.0015333447377495182 | -0.0019522298044696946 | -0.0011356493723014359 | 9.285286025879505e-13 | -0.184370 | 1576 |

*(Task 17b markdown lists slightly different CI rounding for the PP vs window rows, e.g. cpu_over CI [-0.010862, 0.002831]; differences are **≤ about 1e‑3** in CI endpoints versus the one-off recompute above.)*

---

## V4-G — Per-tier PP vs periodic-window (CPU under-rate)

**Source:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md` §“Per-tier breakdown” — **Periodic-window** = `periodic_6h_v2`, **Periodic-Predictor** = `periodic_predictor_6h_v2`. Tier population sizes **525 / 525 / 526** (low / medium / high) from `eval_full_corpus_summary.parquet` (`static_v2` slice).

Let **PW** = `mean_cpu_under_rate` for `periodic_6h_v2`, **PP** = same for `periodic_predictor_6h_v2`.

| volatility_tier | periodic-window (PW) | periodic-predictor (PP) | Δ = PP − PW | Relative reduction (PW − PP) / PW × 100 (%) |
|------------------|-----------------------|-------------------------|------------:|----------------------------------------------:|
| low | 0.02830881 | 0.03070731 | 0.0023985 | -8.4710 |
| medium | 0.07936255 | 0.0623095 | -0.01705305 | 21.4876 |
| high | 0.14787588 | 0.09078578 | -0.05709010 | 38.5935 |

**Corpus-wide (Table 1 provisioning column, SAFETY_FACTOR 1.2):**  
PW = **0.0852**, PP = **0.0613**, Δ = **-0.0239**, relative reduction **(0.0852 − 0.0613) / 0.0852 × 100 = 28.05164319248827%**.

---

## V4-H — Normalized churn (Task 18b parquet)

**File:** `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/normalized_refresh_counts_corpus.parquet`

| method | mean_n_refresh_events | median_n_refresh_events | mean_n_periodic_refreshes | mean_n_drift_refreshes |
|--------|------------------------:|------------------------:|--------------------------:|------------------------:|
| periodic_predictor_6h_v2 | 85.39022842639594 | 86.0 | 85.39022842639594 | 0.0 |
| dynapm_v3_v2 | 91.34581218274111 | 89.0 | 85.39022842639594 | 5.9555837563451774 |

**Per-VM paired delta (V3 − PP) on `n_refresh_events`:**

| Statistic | Value |
|-----------|------:|
| mean Δ | 5.9555837563451774 |
| 95% bootstrap CI (n_boot=1000, seed 42) | [5.57160533, 6.33442259] |
| Wilcoxon p (two-sided, `zero_method="wilcox"`) | 1.5411503497260375e-212 |
| n_pairs | 1576 |

*(Task 18b report cites CI **[5.583740, 6.333931]** and **p ≈ 1.54×10⁻²¹²** — same order; sub‑minor differences vs the one-liner bootstrap above are **seed / tie-handling display** only.)*

---

## V4-I — Sanity checks

| Check | Result |
|-------|--------|
| Tier sizes sum to 1576 | **PASS:** low **525** + medium **525** + high **526** = **1576** (from `eval_full_corpus_summary` / Table 3 tier rows). |
| `n_pairs` in Table 2 | **PASS:** **1576** on every row. |
| `n_vms` in Table 1 | **PASS:** **1576** for every method row. |
| Table 1 `mean_cpu_under_rate` vs tier-weighted Table 3 | **PASS** within **4e‑4** for all six methods (Table 1 rounds to **4** decimals; largest gap **`dynapm_v3_v2`:** Table 1 **0.0611** vs tier-weighted **0.06115**). |

---

## Surprises or anomalies

1. **Informal “+13.0 pp” CPU lift (PP vs window):** Table 2 gives **0.13048404** → **≈ 13.05 percentage points**, not **13.00** exactly.
2. **“+14.5 pp” memory lift:** **0.14542521** → **≈ 14.54 pp** (consistent with **14.5** at one decimal).
3. **“~28% relative under-rate reduction” (window → PP):** Corpus means **0.0852 → 0.0613** yields **28.05164319248827%** relative reduction — consistent with informal **28%**.
4. **Figure 5 dominance on-diagonal:** Only **135 / 1576** VMs have **V3 > PP**; **1427** ties exactly — the **mean** lift **0.00236181** on CPU accuracy (Table 2) is compatible with **sparse** per-VM wins despite many ties.
5. **Table 2 is incomplete for provisioning extras:** **`cpu_over_rate` / `mem_over_rate` for PP vs window** and **`mem_under_rate` / `mem_over_rate` for V3 vs PP** are absent from the CSV but documented in **Task 17b** / recomputed in **§V4-F.2**.
6. **`dynapm_v3_v2` CPU under-rate:** Tier-weighted mean **≈ 0.06115** vs rounded Table 1 **0.0611** — document as **rounding**, not protocol drift.

---

## Files inspected (absolute paths)

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table1_method_comparison.csv`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table2_pairwise_deltas.csv`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/table3_per_tier.csv`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/paper/figure1_method_progression.pdf` *(listed in task spec; numbers verified via Table 1 / packaging script, not pixel-read)*
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/normalized_refresh_counts_corpus.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/provisioning_eval_records.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17b_provisioning.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_18b_corpus_normalized_churn.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task18_paper_packaging.py`
