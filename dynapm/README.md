# DynAPM (Dynamic APM) — reproducibility

This package contains the forecasting, policy, evaluation, and provisioning simulation code for the DynAPM study. The layout below matches the recommended **Zenodo** code drop (after you attach data and fill license/citation metadata).

## Layout

| Path | Role |
|------|------|
| `dynapm/core/` | Profiles, update engine, churn, decision head, schema |
| `dynapm/methods/` | Policies (`baselines_v2`, `dynapm_v3`, `dynapm_v4`, legacy `dynapm`) |
| `dynapm/eval/` | Runners (`runner_v2` canonical, legacy `runner`), provisioning helpers, `metrics.py` |
| `dynapm/data/` | Parquet summaries, `kept_vms.json`, loaders, packaging outputs |
| `dynapm/scripts/` | Task drivers (bootstrap, GT, full-corpus eval Task 15, provisioning Task 17b, paper Task 18) |
| `dynapm/results/` | Per-run artifacts (e.g. full-corpus JSON without `update_log`, sample JSON with logs) |
| `dynapm/tests/` | Unit and integration tests |
| `dynapm/reports/` | Task write-ups; `reports/paper/` holds Table 1–3 CSVs and Figures 1–5 (PNG+PDF) |
| `old/data/raw/` | VM CSV traces (not redistributed here; place your copy to match paths in runners) |

Other repo roots hold the LSTM predictor training stack; this README focuses on the DynAPM evaluation path.

## Install

From the **repository root** (parent of `dynapm/`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r dynapm/requirements.txt
```

Extras: joblib for parallel eval scripts; CUDA is only needed for training outside this package.

## Environment

Drivers expect `PYTHONPATH` to include the repo root so `import dynapm…` resolves:

```bash
export PYTHONPATH=$(pwd)
```

## Headline pipeline (full corpus + paper tables)

Rough order (scripts live under `dynapm/scripts/`):

1. **Bootstrap & thresholds** — `run_bootstrap_characterization.py` → `dynapm/data/bootstrap_summary.parquet`.
2. **Predictor residuals** — predictor tasking (see `task04_full_corpus_predictor.py`) → `dynapm/results/residuals_full/` and `predictor_summary.parquet`.
3. **Ground truth snapshots** — `task04_ground_truth.py` → `dynapm/results/ground_truth/`.
4. **Class-level eval (1576 VMs × 6 methods)** — `task15_full_corpus.py` → `dynapm/data/eval_full_corpus_summary.parquet` plus per-run JSON under `dynapm/results/eval_full_corpus/` (JSON omits `update_log` to save space).
5. **Provisioning records** — `task17b_provisioning.py` (reloads cached `provisioning_eval_records.parquet` unless `--force-collect`) → aggregates in the Task 17b report.
6. **Paper packaging** — `task18_paper_packaging.py` reads the Task 15 summary and Task 17b provisioning parquet only (no re-eval) → `dynapm/reports/paper/` CSVs and figures, and `dynapm/data/normalized_refresh_counts_sample.parquet` from the retained 100-VM JSON logs.

Step counts and wall times depend on CPU cores (`--n-jobs` on the heavy scripts).

## Reproduce a single table or figure

- **Table 1–3 & Figures 1–5** — `PYTHONPATH=. python dynapm/scripts/task18_paper_packaging.py` writes `dynapm/reports/paper/table*.csv` plus `figure*_*.png`/`.pdf`.
- **Normalized churn sample** — same script regenerates `dynapm/data/normalized_refresh_counts_sample.parquet` from `dynapm/results/eval_sample_v2/*` JSON (must still contain `update_log`).
- **Task 15 aggregates only** — load `dynapm/data/eval_full_corpus_summary.parquet` in pandas; per-VM detail is in JSON shards if present.

## Tests

```bash
PYTHONPATH=. pytest dynapm/tests -q
```

Canonical evaluation path for new policies: `dynapm.eval.runner_v2.run_method_on_vm`. `runner_v4` re-exports this name for backward compatibility and is deprecated.

## Citation

If you use this artifact, please cite the associated paper (forthcoming). **Placeholder:** `(Author et al., YEAR) — replace when DOI is assigned.`

## License

**Placeholder.** Add a root `LICENSE` file before public release; the checklist in `dynapm/RELEASE_CHECKLIST.md` calls this out explicitly.

## Contact

See `RELEASE_CHECKLIST.md` for pre-release contributor and packaging checks.
