# DynAPM Artifact

Reproduction package for **DynAPM: Drift-Aware Profile Maintenance with Predictor-Grounded Forecasting in the Cloud-Edge Continuum**.

This branch (`dynapm`) contains only the DynAPM evaluation code, regenerated paper tables/figures, and the Materna VM traces used by the experiments. Legacy LSTM training pipelines and unrelated baseline experiments have been removed.

## Artifact availability

| Resource | Location |
|----------|----------|
| Code + evaluation results (this repo) | https://github.com/KADOUMAAbdelhak/gwa-t-13-lstm-pipeline/tree/dynapm |
| Materna workload traces (Kaggle mirror) | https://www.kaggle.com/datasets/kpiyush04/maternaworkloadtraces |
| Materna / GWA-T-13 (canonical archive) | [Grid Workloads Archive](http://gwa.ewi.tudelft.nl) |
| Manuscript source | `manuscript_updated.tex` |
| Autoscaling bibliography additions | `cas-refs_autoscaling_additions.bib` (append into Overleaf `cas-refs.bib`) |
| Paper tables & figures | `dynapm/reports/paper/` |

## Repository layout

```
.
├── data/raw/                 # Materna-Trace-{1,2,3} VM CSVs
├── dynapm/                   # DynAPM package (core, methods, eval, scripts, tests)
│   ├── core/                 # Ground truth, predictors, profiles, drift helpers
│   ├── methods/              # Periodic-Predictor, DynAPM-V3, observation baselines
│   ├── eval/                 # Canonical runner_v2 + provisioning helpers
│   ├── scripts/              # Task drivers (bootstrap → Task 18 paper packaging)
│   ├── data/                 # Kept-VM lists, corpus summaries, parquet intermediates
│   ├── results/              # Per-VM eval JSON, ground-truth caches, residuals
│   ├── reports/paper/        # Tables 1–3 and Figures 1–5
│   └── tests/
├── manuscript_updated.tex
├── cas-refs_autoscaling_additions.bib
├── requirements.txt          # Mirrors dynapm/requirements.txt
└── README.md
```

## Install

```bash
git clone https://github.com/KADOUMAAbdelhak/gwa-t-13-lstm-pipeline.git
cd gwa-t-13-lstm-pipeline
git checkout dynapm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

Traces are expected under `data/raw/Materna-Trace-{1,2,3}/`. If you clone without the CSVs, download them from Kaggle or the Grid Workloads Archive and place them in that layout.

## Reproduce headline paper outputs

```bash
export PYTHONPATH=$(pwd)
# Regenerate Tables 1–3 and Figures 1–5 from checked-in corpus summaries:
python dynapm/scripts/task18_paper_packaging.py
```

Full corpus re-evaluation (slow) follows the pipeline documented in `dynapm/README.md` (bootstrap → predictors → ground truth → Task 15 → Task 17b → Task 18).

## Tests

```bash
export PYTHONPATH=$(pwd)
pytest dynapm/tests -q
```

## Citation

Please cite the DynAPM manuscript when using this artifact. Update the citation once a DOI is assigned.
