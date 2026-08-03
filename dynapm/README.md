# DynAPM package

Forecasting, profile-maintenance policies, evaluation, and provisioning simulation for the DynAPM study.

## Layout

| Path | Role |
|------|------|
| `dynapm/core/` | Profiles, update engine, ground truth, predictors, schema |
| `dynapm/methods/` | Policies (`baselines_v2`, Periodic-Predictor, `dynapm_v3`, legacy variants) |
| `dynapm/eval/` | Runners (`runner_v2` canonical), provisioning helpers, metrics |
| `dynapm/data/` | Parquet summaries, `kept_vms.json`, loaders |
| `dynapm/scripts/` | Task drivers (bootstrap, GT, Task 15 full-corpus, Task 17b provisioning, Task 18 paper) |
| `dynapm/results/` | Per-run artifacts (full-corpus JSON, ground-truth caches, residuals) |
| `dynapm/tests/` | Unit and integration tests |
| `dynapm/reports/paper/` | Table 1–3 CSVs and Figures 1–5 (PNG+PDF) |
| `data/raw/` | Materna VM CSV traces at repo root (not under this package) |

## Install

From the **repository root**:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -r dynapm/requirements.txt
export PYTHONPATH=$(pwd)
```

## Headline pipeline

1. **Bootstrap & thresholds** — `run_bootstrap_characterization.py`
2. **Predictor residuals** — `task04_full_corpus_predictor.py`
3. **Ground truth snapshots** — `task04_ground_truth.py` (forward-24h class + p95)
4. **Class-level eval (1576 VMs × 6 methods)** — `task15_full_corpus.py`
5. **Provisioning** — `task17b_provisioning.py`
6. **Paper packaging** — `task18_paper_packaging.py` → `dynapm/reports/paper/`

## Reproduce tables and figures

```bash
PYTHONPATH=. python dynapm/scripts/task18_paper_packaging.py
```

## Tests

```bash
PYTHONPATH=. pytest dynapm/tests -q
```

Canonical evaluation path: `dynapm.eval.runner_v2.run_method_on_vm`.

## Data

Place Materna traces under `data/raw/Materna-Trace-{1,2,3}/`, or use the copy already present in this artifact branch. Public mirrors:

- https://www.kaggle.com/datasets/kpiyush04/maternaworkloadtraces
- Grid Workloads Archive (GWA-T-13 Materna)

## Citation

Cite the associated DynAPM paper (forthcoming). Replace this note when a DOI is assigned.
