# DynAPM Task 17 — D1 retrieval (**STOP: degenerate clustering**)

## What was done

Streaming-segment metrics (12 features × 1 576 VMs), `StandardScaler`, and HDBSCAN (`min_cluster_size=20`, `min_samples=5`) were computed. The partition failed the non-degeneracy gate (**<3 non-noise clusters**; noise share was also checked against 30%). Per specification, **sample and corpus retrieval were not run**.

## Clustering result

- **Non-noise clusters found:** 2
- **Noise VMs:** 285 (18.084%)
- **Observed cluster populations:** {0: 1269, 1: 22}

### Per-cluster raw-feature means (non-noise only)

| cluster_id | size | mean_mean_cpu | mean_std_cpu | mean_p95_cpu | mean_p20_cpu | mean_mean_mem | mean_std_mem | mean_p95_mem | mean_p20_mem | mean_cpu_autocorr_lag12 | mean_mem_autocorr_lag12 | mean_cpu_skewness | mean_mem_skewness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1269.0 | 2.0504024266086924 | 1.1393393472316566 | 3.234351851851851 | 1.5913695823483058 | 5.271371452685346 | 2.6105234647603694 | 8.976580378250588 | 3.67228999211978 | 0.3200257325825649 | 0.22979675691964294 | 13.978269751224541 | 5.615338799963282 |
| 1.0 | 22.0 | 10.004561675511598 | 7.768149770361782 | 26.104409090909087 | 5.619545454545456 | 20.812990548525327 | 11.048107832237026 | 46.55020454545454 | 13.320636363636362 | 0.23985628891941 | 0.4751830471821515 | 2.998988979260686 | 1.9094724116401165 |

## Sample retrieval result

**Skipped (clustering STOP).**

## Corpus retrieval result

**Skipped (clustering STOP).**

## Pairwise deltas with bootstrap CI

**Not applicable** — retrieval never executed.

## Checkpoint questions

| Question | Verdict | Evidence |
|---|---|---|
| Q1 | N/A | Retrieval skipped |
| Q2 | N/A | Retrieval skipped |
| Q3 | N/A | Retrieval skipped |
| Q4 | N/A | Retrieval skipped |
| Q5 | N/A | Retrieval skipped |

## Figures

- `dynapm/reports/task_17/cluster_characteristics.png` (only artifact generated)

## Files created or modified

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/streaming_features.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/vm_clusters.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17/cluster_characteristics.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task17_d1_retrieval.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_task17_d1_retrieval.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_17_d1_retrieval.md`

## Test results

`pytest dynapm/tests/test_task17_d1_retrieval.py -q` → **4 passed**.

## Assumptions

- Streaming features use the locked 30/70 bootstrap/stream split via `compute_split`.
- `lag12` autocorrelation uses twelve 5 min steps (≈1 h horizon) per column naming convention.

## Surprises

- HDBSCAN returned only **2** dense clusters (<3 threshold) with **18.1%** noise; corpus-wide retrieval experiment stopped per task contract.

## Commands

```bash
PYTHONPATH=. python dynapm/scripts/task17_d1_retrieval.py
```

## Next-step suggestion

- Relax HDBSCAN constraints, engineer additional discriminative features (e.g., burst counts), or evaluate an alternative mixture model so at least three stable behavioral clusters emerge before revisiting D1 retrieval.
