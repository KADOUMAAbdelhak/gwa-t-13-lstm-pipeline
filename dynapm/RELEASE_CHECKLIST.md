# Zenodo / public release checklist (DynAPM)

Use this list before publishing the code bundle. This is a **pre-flight** only; it does not perform the release.

## Quality

- [ ] Full test suite passes: `PYTHONPATH=. pytest dynapm/tests -q`
- [ ] Paper packaging script runs cleanly: `PYTHONPATH=. python dynapm/scripts/task18_paper_packaging.py`
- [ ] No unintended large binaries under version control (purge caches, accidental Parquet/PNG dumps)

## Code hygiene

- [ ] All scripts intended for users have a module docstring stating purpose and inputs/outputs
- [ ] No hard-coded absolute paths such as `/Users/...` in code, JSON metadata, or diagnostics (search the tree)
- [ ] `runner_v4` imports resolve to the canonical `runner_v2` behavior; no duplicate evaluator logic left stale in docs

## Data & size

- [ ] Sample traces or a tiny public subset are documented; full `data/raw/` may be omitted from Zenodo with download instructions (Kaggle / GWA)
- [ ] Total downloadable artifact is under **100 MB** unless a split-archive plan is documented (Git LFS, Zenodo multiple files, etc.)
- [ ] `normalized_refresh_counts_sample.parquet` and `reports/paper/` artifacts are reproducible from checked-in intermediates **or** explicitly listed as generated-only

## Legal & attribution

- [ ] **`LICENSE` file present** at repo root with chosen terms
- [ ] **Citation / DOI** updated in `dynapm/README.md` once the manuscript is registered
- [ ] **Contributors** section (Zenodo metadata or `CONTRIBUTORS.md`) lists authors and institutional affiliations consistent with authorship rules

## Science integrity

- [ ] Task numbers in `dynapm/reports/` match frozen parquets checked in or archived alongside the Zenodo bundle
- [ ] Random seeds documented for bootstrap-heavy scripts (`task18`, `task17b`) if strict bit-for-bit reproducibility is claimed
