# Task 16: Decision head + churn + alpha sweep

## What was implemented

Added River-backed `DecisionHeadV2`, `ChurnControllerV2`, and `DynAPMPolicyV4` wrapping the v3
predictor/drift cadence with retrospective online learning invoked by the new evaluator
`runner_v4.run_method_on_vm`. **Task constraints forbid editing `runner_v2.py`, so that file still lacks
`verify_pending_commits`; the v4 runner is a parallel copy that adds only the verification hook.** Drift commits
pass churn (min 2h) and head gating; periodic commits unchanged.

## Sample alpha sweep

Sorted by alpha ascending:

| alpha | cpu_class_accuracy | memory_class_accuracy | total_updates | n_periodic_refreshes | n_drift_refreshes | commit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.892273 | 0.87676 | 364.6 | 85.57 | 5.58 | 0.84 |
| 0.3 | 0.892065 | 0.87676 | 361.16 | 85.57 | 4.72 | 0.787099 |
| 0.5 | 0.890203 | 0.875063 | 354.68 | 85.57 | 3.1 | 0.655765 |
| 0.7 | 0.890203 | 0.875063 | 354.44 | 85.57 | 3.04 | 0.647551 |
| 0.9 | 0.890203 | 0.875063 | 354.44 | 85.57 | 3.04 | 0.647551 |

**Reference (same 100-VM rows):** DynAPM-V3 mean CPU **0.892273** | Periodic-Predictor mean CPU **0.887774**.

### Best-alpha selection rationale
**Best accuracy alpha** (maximum mean CPU accuracy on the 100-VM ladder): **0.0**. **Efficiency-
preferred alpha** (lowest mean `total_updates` while mean CPU stays within **0.005** of V3’s sample mean
**0.8923**): **0.7**.

## Corpus run with best alpha

| method | cpu mean | cpu median | total_updates mean | total_updates median |
|---|---|---|---|---|
| **static_v2** | 0.474523 | 0.416667 | 0.00 | 0.00 |
| **dynapm_v3_v2** | 0.869813 | 0.916667 | 365.38 | 356.00 |
| **periodic_predictor_6h_v2** | 0.867451 | 0.916667 | 187.43 | 187.00 |
| **dynapm_v4 (α=0.0)** | 0.869813 | 0.916667 | 365.38 | 356.00 |

### Pairwise V4 vs V3 corpus deltas (CPU accuracy)

- **n VMs:** **1576**
- **Mean Δ (V4−V3):** **0.000000**
- **Bootstrap 95% CI:** **[0.000000, 0.000000]**
- **Wilcoxon signed-rank p:** **nan**
- **Mean drift bundles:** V3 **heuristic 4.93** vs V4 **measured 5.96** ⇒ **≈ -20.9%** reduction (heuristic defined in Assumptions).

### Per-tier corpus breakdown (mean CPU accuracy)

| tier | static_cpu | predictor_cpu | v3_cpu | v4_cpu |
| --- | --- | --- | --- | --- |
| low | 0.413412 | 0.876007 | 0.876814 | 0.876814 |
| medium | 0.474675 | 0.850358 | 0.852385 | 0.852385 |
| high | 0.535366 | 0.875972 | 0.880221 | 0.880221 |

## Checkpoint questions
| Q | Result | Notes |
|---|---|---|
| Q1 α=0 V4 CPU ≈ V3 sample |Δ| < **0.005**? | **PASS** | gap **0.000000** |
| Q2 CPU spread α=0 vs α=**0.9** ≥ **0.005**? | **FAIL** | |Δ| **0.002070** |
| Q3 Corpus V4 mean CPU ≥ V3 + **0.001**? | **FAIL** | V4 **0.869813**, V3 **0.869813** |
| Q4 Bootstrap CI (V4−V3 ΔCPU) excludes **0**? | **FAIL** | **[0.000000, 0.000000]** |
| Q5 Fewer drift refreshes than V3? | **FAIL** | **≈ -20.9%** reduction |

## Figures
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_16/alpha_sweep_pareto.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_16/alpha_sweep_per_tier.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_16/v4_vs_v3_corpus.png`

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/decision_head_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/churn_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/methods/dynapm_v4.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/eval/runner_v4.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_decision_head_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_churn_v2.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_dynapm_v4.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task16_alpha_sweep.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_sample_v2_dynapm_v4_alpha_sweep.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_dynapm_v4_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_16_decision_head_alpha_sweep.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_16`

## Test results
Run from repo root (`PYTHONPATH=.`): `pytest dynapm/tests/test_decision_head_v2.py dynapm/tests/test_churn_v2.py dynapm/tests/test_dynapm_v4.py`

## Assumptions
- DynAPM-V3 corpus JSON omits per-reason refresh tallies; drift bundle counts for Q5 use `total_updates/4 − round(stream_days·24/6)` as a **heuristic** proxy for drift-only bundles.
- `runner_v4` calls `verify_pending_commits` only at evaluation timestamps present in the ground-truth grid.

## Surprises
_Populated automatically from checkpoints (see Diagnostics block in script stdout)._

## Commands
```text
$ PYTHONPATH=. python dynapm/scripts/task16_alpha_sweep.py
```

## Next-step suggestion
Tighten the retrospective label wiring to exact `t_commit` ground-truth rows if available, explore
corpus-optimal alphas disjoint from accuracy-max, and quantify memory-class trade-offs under tighter
churn.
