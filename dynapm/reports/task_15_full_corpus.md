# Task 15: Full-corpus evaluation (1,576 VMs × 6 methods)

## What was ran
Locked kept list `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/kept_vms.json` (**1576** VMs) was evaluated under `runner_v2.run_method_on_vm` for six methods (**9456** runs) with `joblib.Parallel(n_jobs=8)`. Volatility tiers are tertiles of bootstrap-segment CPU% std (same construction as Task 5B’s sample strata). Per-run JSON fragments are stored under `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_full_corpus/` **without** the heavy `update_log` field (metrics are duplicated in parquet). Consolidated **`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`** includes `run_status` ∈ {`ok`,`failed`,`skipped`} and `failure_reason`. The driver script reloads existing valid per-run JSON so interrupted batches can resume without recomputing finished pairs.

## Run-status diagnostic
- **Total runs attempted:** 9456 (expected **9456**).
- **Total wall-clock:** 1162.0 s.
- **static_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).
- **periodic_6h_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).
- **reactive_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).
- **full_rewrite_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).
- **periodic_predictor_6h_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).
- **dynapm_v3_v2:** ok **1576**, failed **0**, skipped **0** (success fraction **1.0000**).

**Failure reason counts (failed only):**
- *(none)*

**Up to five example failures per method** `(trace, vm_id)`:
- **static_v2:** []
- **periodic_6h_v2:** []
- **reactive_v2:** []
- **full_rewrite_v2:** []
- **periodic_predictor_6h_v2:** []
- **dynapm_v3_v2:** []

## Per-method aggregates (headline)
Successful runs only; **fraction_corpus** = successful / 1,576. Sorted by **mean** `cpu_class_accuracy` **ascending**.
| method | mean cpu_acc | median cpu_acc | mean mem_acc | median mem_acc | mean cpu_p95_mae | median cpu_p95_mae | mean mem_p95_mae | median mem_p95_mae | mean total_updates | median total_updates | n_ok | frac_corpus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static_v2 | 0.4745 | 0.4167 | 0.5889 | 0.6458 | 2.5597 | 0.2524 | 2.4456 | 0.6628 | 0.00 | 0.00 | 1576 | 1.0000 |
| reactive_v2 | 0.6445 | 0.6585 | 0.6300 | 0.6316 | 2.2389 | 0.2569 | 2.3771 | 0.8325 | 551.98 | 568.00 | 1576 | 1.0000 |
| periodic_6h_v2 | 0.7370 | 0.7561 | 0.7443 | 0.7561 | 1.4001 | 0.1666 | 1.6679 | 0.5972 | 140.05 | 146.50 | 1576 | 1.0000 |
| full_rewrite_v2 | 0.7577 | 0.7760 | 0.7631 | 0.7708 | 1.2900 | 0.1356 | 1.4989 | 0.5162 | 1252.85 | 1217.50 | 1576 | 1.0000 |
| periodic_predictor_6h_v2 | 0.8675 | 0.9167 | 0.8897 | 0.9167 | 1.0617 | 0.1636 | 1.1434 | 0.5234 | 187.43 | 187.00 | 1576 | 1.0000 |
| dynapm_v3_v2 | 0.8698 | 0.9167 | 0.8929 | 0.9211 | 1.0419 | 0.1605 | 1.1268 | 0.5237 | 365.38 | 356.00 | 1576 | 1.0000 |

## Per-tier breakdown
*Mean CPU / memory class accuracy; tier counts reflect the full kept list.*
| volatility_tier | n_vms_in_tier | static_v2_cpu | static_v2_mem | periodic_6h_v2_cpu | periodic_6h_v2_mem | reactive_v2_cpu | reactive_v2_mem | full_rewrite_v2_cpu | full_rewrite_v2_mem | periodic_predictor_6h_v2_cpu | periodic_predictor_6h_v2_mem | dynapm_v3_v2_cpu | dynapm_v3_v2_mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | 525 | 0.4134123021404823 | 0.6975408781859836 | 0.7357822225716083 | 0.7845006059562294 | 0.6956575480144958 | 0.7097271133179245 | 0.7560807025056974 | 0.8005920180432378 | 0.8760071258749661 | 0.9126269768463217 | 0.8768138130342116 | 0.9136675779155589 |
| medium | 525 | 0.47467499812516184 | 0.5754920758931468 | 0.7367280037631266 | 0.7389221669380704 | 0.6197430895313154 | 0.6012707868869911 | 0.7554925697401417 | 0.7573439564317628 | 0.8503580314061779 | 0.8751642569240543 | 0.8523845284896603 | 0.8791697739235989 |
| high | 526 | 0.5353660370758812 | 0.49373981169923875 | 0.7383883527125827 | 0.7094257002541264 | 0.6180704777840095 | 0.5792141190622151 | 0.7615689581390582 | 0.7313114147910483 | 0.8759720281313732 | 0.8812825190867368 | 0.8802206679564729 | 0.8857932819163943 |

## Pairwise deltas with bootstrap CIs
*Paired exclusively on VMs where **both** methods succeeded.*

### (A) Periodic-Predictor vs Periodic-Window
- **n_pairs:** 1576
- **Δ CPU:** mean **0.130484**, bootstrap 95% CI **[0.121514, 0.139817]**, Wilcoxon *p=* **5.172e-148**
- **Δ Memory:** mean **0.145425**, CI **[0.137547, 0.153082]**, Wilcoxon *p=* **1.501e-190**

Per-tier mean ΔCPU / ΔMem:

- **low:** ΔCPU **0.140225**, ΔMem **0.128126**
- **medium:** ΔCPU **0.113630**, ΔMem **0.136242**
- **high:** ΔCPU **0.137584**, ΔMem **0.171857**
### (B) DynAPM-V3 vs Periodic-Predictor
- **n_pairs:** 1576
- **Δ CPU:** mean **0.002362**, bootstrap 95% CI **[0.001885, 0.002887]**, Wilcoxon *p=* **2.759e-18**
- **Δ Memory:** mean **0.003186**, CI **[0.002624, 0.003788]**, Wilcoxon *p=* **2.373e-25**

Per-tier mean ΔCPU / ΔMem:

- **low:** ΔCPU **0.000807**, ΔMem **0.001041**
- **medium:** ΔCPU **0.002026**, ΔMem **0.004006**
- **high:** ΔCPU **0.004249**, ΔMem **0.004511**

## Near-idle VM subset analysis
Non-idle if `bootstrap_summary.cpu_p80` **>** **5**. Count **non-idle VMs:** **386** / **1576**.

| method | n_vms | cpu_class_accuracy_mean | memory_class_accuracy_mean |
| --- | --- | --- | --- |
| static_v2 | 386 | 0.489080340042171 | 0.5097131322500632 |
| periodic_6h_v2 | 386 | 0.7326594809499801 | 0.7323071731905813 |
| reactive_v2 | 386 | 0.627879317981722 | 0.6095023754516832 |
| full_rewrite_v2 | 386 | 0.7548584425818544 | 0.7570229757478435 |
| periodic_predictor_6h_v2 | 386 | 0.8729240320348377 | 0.876163025722873 |
| dynapm_v3_v2 | 386 | 0.8761807715327355 | 0.8802017404841247 |

## Memory-class diagnostic (DynAPM-V3, successful runs)
- **Mean** `n_memory_class_updates`: **91.3458**
- **Median:** **89.0**
- **VMs with `n_memory_class_updates == 0`:** **0** / **1576**

## Checkpoint questions
| Q | Result | Numbers |
|---|--------|--------|
| Q1 PP mean CPU acc − Window mean ≥ **0.10**? | **PASS** | PP **0.867451**, Window **0.736967**, gap **0.130484** |
| Q2 PP−Window Δ CPU bootstrap 95% CI excludes **0**? | **PASS** | CI **[0.121514, 0.139817]** |
| Q3 V3 mean CPU acc ≥ PP mean? | **PASS** | V3 **0.869813**, PP **0.867451** |
| Q4 V3−PP Δ CPU CI excludes negatives (low ≥ 0)? | **PASS** | CI **[0.001885, 0.002887]** |
| Q5 High tier: V3 mean CPU > PP mean? | **PASS** | V3 **0.880221**, PP **0.875972** |
| Q6 Failures per method < **5%**? | **PASS** | `static_v2` **0.00%**, `periodic_6h_v2` **0.00%**, `reactive_v2` **0.00%**, `full_rewrite_v2` **0.00%**, `periodic_predictor_6h_v2` **0.00%**, `dynapm_v3_v2` **0.00%** |

## Figures
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15/method_progression_corpus.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15/method_progression_per_tier.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15/dynapm_v3_vs_periodic_predictor_per_vm_corpus.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15/predictor_lift_distribution.png`

## Files created or modified
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_summary.parquet`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/eval_full_corpus_diagnostics.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_15_full_corpus.md`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/results/eval_full_corpus`

## Test results
- `pytest dynapm/tests/test_dynapm_v3.py -q` — **3 passed** (post–Task 15 smoke).

## Assumptions
- **Volatility tiers** follow Task 5B: tertiles of bootstrap CPU% standard deviation across the locked kept list (not a column shipped in `bootstrap_summary.parquet`).
- Corpus JSON omit `update_log` to limit disk footprint; parquet holds all headline metrics.
- **`task15_full_corpus.py` resume:** existing per-run JSON with finite `cpu_class_accuracy` reloads that result instead of recomputing the VM–method pair.
- **Skipped** runs are precondition failures (`failure_reason` e.g. `MissingResidualCpu`).
- **Wilcoxon** uses `zero_method="wilcox"`; ill-conditioned cases return `nan` *p*-value.

## Surprises
- Full-corpus predictor lift (PP−Window) mean ΔCPU **0.1305**; V3−PP mean **0.0024**.
- Highest mean CPU class accuracy on corpus: **dynapm_v3_v2** (0.8698).

## Commands
```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && . .venv/bin/activate
$ PYTHONPATH=. pytest dynapm/tests/test_dynapm_v3.py -q
... 3 passed in ~2s ...
$ PYTHONPATH=. python dynapm/scripts/task15_full_corpus.py
Computing volatility tiers ...
Per-tier VM counts: {'high': 526, 'medium': 525, 'low': 525}
{ "wall_seconds": 1162.0, "checkpoint": { ... all Q1–Q6 PASS ... } }
```

## Next-step suggestion
If failure or skip counts cluster on particular traces or missing artifacts, backfill residuals /
ground-truth parquet for those VMs and re-run failed rows only; alternatively publish both full-
corpus and artifact-complete subsets explicitly in the paper.
