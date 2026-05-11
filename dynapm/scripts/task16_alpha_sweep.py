#!/usr/bin/env python3
"""Task 16: alpha sweep (sample + corpus best alpha), diagnostics, figures, report."""
from __future__ import annotations

import json
import math
import sys
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dynapm.scripts import _matplotlib_setup  # noqa: F401

from scipy import stats as scipy_stats

from dynapm.eval import runner_v2
from dynapm.methods.dynapm_v4 import DynAPMPolicyV4

ALPHA_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(values))
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return m, float(lo), float(hi)


def _wilcoxon_safe(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan")
    try:
        return float(scipy_stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _write_json(path: Path, out: dict) -> None:
    slim = {k: v for k, v in out.items() if k != "update_log"}
    path.write_text(json.dumps(slim, default=str), encoding="utf-8")


def _summarize_vm(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    cc = raw.get("cpu_class_accuracy")
    if cc is None or (isinstance(cc, float) and not math.isfinite(cc)):
        return None
    return raw


def _run_v4(alpha: float, trace: str, vm_id: str, out_path: Path) -> dict | None:
    loaded = _summarize_vm(out_path)
    if loaded is not None:
        # resume
        loaded["trace"] = trace
        loaded["vm_id"] = str(vm_id)
        return loaded
    factory = lambda p, c, m, **kw: DynAPMPolicyV4(
        p, c, m, period_hours=PERIOD_H, drift_delta=0.002, alpha=float(alpha), min_drift_interval_hours=2.0, **kw
    )
    out = runner_v2.run_method_on_vm(trace, vm_id, factory)
    _write_json(out_path, out)
    return out


def _alpha_fname(a: float) -> str:
    s = ("%s" % a).replace(".", "p") if "." in ("%s" % a) else str(int(a))
    return s.replace("-", "neg")


PERIOD_H = 6.0


def _est_periodic_bundles(stream_days: float) -> int:
    return max(1, int(round(float(stream_days) * 24.0 / PERIOD_H)))


def _est_drift_bundles_v3(total_updates: int, stream_days: float) -> float:
    return max(0.0, float(total_updates) / 4.0 - float(_est_periodic_bundles(stream_days)))


def main() -> int:
    sample_pp = pd.read_parquet(REPO / "dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet")
    sample_pp = sample_pp[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample_pp["vm_id"] = sample_pp["vm_id"].astype(str)
    assert len(sample_pp) == 100

    samp_pp_perf = pd.read_parquet(REPO / "dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet")
    pp_s_cpu_mean = float(samp_pp_perf["cpu_class_accuracy"].mean())
    pp_s_tu_mean = float(samp_pp_perf["total_updates"].mean())

    v3_summ = pd.read_parquet(REPO / "dynapm/data/eval_sample_v2_dynapm_v3_summary.parquet")
    v3_s_tu_mean = float(v3_summ["total_updates"].mean())

    out_sample_dir = REPO / "dynapm/results/eval_sample_v2"
    sweep_parquet = REPO / "dynapm/data/eval_sample_v2_dynapm_v4_alpha_sweep.parquet"

    corpus_dir = REPO / "dynapm/results/eval_full_corpus"
    corpus_summ = REPO / "dynapm/data/eval_full_corpus_dynapm_v4_summary.parquet"
    corp_ref = pd.read_parquet(REPO / "dynapm/data/eval_full_corpus_summary.parquet")
    corp_ref_ok = corp_ref[corp_ref["run_status"] == "ok"]
    kept = json.loads((REPO / "dynapm/data/kept_vms.json").read_text(encoding="utf-8"))

    report_dir = REPO / "dynapm/reports/task_16"
    report_md = REPO / "dynapm/reports/task_16_decision_head_alpha_sweep.md"
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample sweep ---
    sweep_rows = []
    t0_s = time.perf_counter()

    jobs = [(a, str(r.trace), str(r.vm_id), str(r.volatility_tier)) for a in ALPHA_VALUES for r in sample_pp.itertuples(index=False)]

    def run_sample(job):
        a, trace, vm_id, tier = job
        fn = _alpha_fname(a)
        outp = out_sample_dir / f"dynapm_v4_alpha_{fn}__{trace}__{vm_id}.json"
        out = _run_v4(a, trace, vm_id, outp)
        dh = out.get("decision_head_stats", {}) if out else {}
        n_per = out.get("n_periodic_refreshes", np.nan)
        n_drift = out.get("n_drift_refreshes", np.nan)
        return {
            "alpha": float(a),
            "trace": trace,
            "vm_id": vm_id,
            "volatility_tier": tier,
            "cpu_class_accuracy": out["cpu_class_accuracy"],
            "memory_class_accuracy": out["memory_class_accuracy"],
            "total_updates": out["total_updates"],
            "n_periodic_refreshes": n_per,
            "n_drift_refreshes": n_drift,
            "commit_rate": float(dh.get("commit_rate", float("nan"))),
        }

    sweep_meta = Parallel(n_jobs=8, backend="loky")(delayed(run_sample)(j) for j in jobs)
    for row in sweep_meta:
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_parquet(sweep_parquet, index=False)
    wall_sample = time.perf_counter() - t0_s

    grp = sweep_df.groupby("alpha", sort=False).agg(
        {
            "cpu_class_accuracy": "mean",
            "memory_class_accuracy": "mean",
            "total_updates": "mean",
            "n_periodic_refreshes": "mean",
            "n_drift_refreshes": "mean",
            "commit_rate": "mean",
        }
    ).reset_index()
    grp = grp.sort_values("alpha")

    v3_mean_cpu = float(v3_summ["cpu_class_accuracy"].mean())

    idx = grp.set_index("alpha")
    best_acc_alpha = float(grp.loc[grp["cpu_class_accuracy"].idxmax(), "alpha"])

    eff = grp[grp["cpu_class_accuracy"] >= v3_mean_cpu - 0.005].sort_values("total_updates")
    efficient_alpha = float(eff.iloc[0]["alpha"]) if len(eff) else float("nan")

    # Sanity stop if alpha 0 wildly off V3 — still continue with note
    a0_row = grp[np.isclose(grp["alpha"], 0.0)].iloc[0]
    sanity_gap_cpu = abs(float(a0_row["cpu_class_accuracy"]) - v3_mean_cpu)

    # --- Corpus best alpha ---
    best_a = best_acc_alpha
    t0_c = time.perf_counter()

    def run_co(item):
        trace, vm_id = item["trace"], str(item["vm_id"])
        outp = corpus_dir / f"dynapm_v4__{trace}__{vm_id}.json"
        loaded = _summarize_vm(outp)
        if loaded is None:
            factory = lambda p, c, m, **kw: DynAPMPolicyV4(
                p, c, m, period_hours=PERIOD_H, drift_delta=0.002, alpha=float(best_a), min_drift_interval_hours=2.0, **kw
            )
            out = runner_v2.run_method_on_vm(trace, vm_id, factory)
            _write_json(outp, out)
        else:
            out = loaded
        tier_row = corp_ref_ok[
            (corp_ref_ok["trace"] == trace) & (corp_ref_ok["vm_id"].astype(str) == vm_id) & (corp_ref_ok["method"] == "static_v2")
        ].head(1)
        tier = str(tier_row["volatility_tier"].iloc[0]) if len(tier_row) else ""
        dh = out.get("decision_head_stats", {})
        return {
            "method": "dynapm_v4",
            "trace": trace,
            "vm_id": vm_id,
            "volatility_tier": tier,
            "cpu_class_accuracy": out["cpu_class_accuracy"],
            "memory_class_accuracy": out["memory_class_accuracy"],
            "cpu_p95_mae": out["cpu_p95_mae"],
            "mem_p95_mae": out["mem_p95_mae"],
            "n_cpu_class_updates": out["n_cpu_class_updates"],
            "n_memory_class_updates": out["n_memory_class_updates"],
            "n_cpu_p95_updates": out["n_cpu_p95_updates"],
            "n_mem_p95_updates": out["n_mem_p95_updates"],
            "total_updates": out["total_updates"],
            "n_eval_points_verified": out["n_eval_points_verified"],
            "snapshots_dropped_unverified": out["snapshots_dropped_unverified"],
            "n_periodic_refreshes": out.get("n_periodic_refreshes", np.nan),
            "n_drift_refreshes": out.get("n_drift_refreshes", np.nan),
            "decision_commit_rate_mean_vm": dh.get("commit_rate", np.nan),
        }

    corp_rows = Parallel(n_jobs=8, backend="loky")(delayed(run_co)(item) for item in kept)

    corp_v4_df = pd.DataFrame(corp_rows)
    corp_v4_df["vm_id"] = corp_v4_df["vm_id"].astype(str)
    corp_v4_df.to_parquet(corpus_summ, index=False)
    wall_corpus = time.perf_counter() - t0_c

    v3_cor = corp_ref_ok[corp_ref_ok["method"] == "dynapm_v3_v2"]
    v3_cor = v3_cor.assign(vm_id=v3_cor["vm_id"].astype(str))
    stat_cor = corp_ref_ok[corp_ref_ok["method"] == "static_v2"]
    pp_cor = corp_ref_ok[corp_ref_ok["method"] == "periodic_predictor_6h_v2"]

    rng = np.random.default_rng(17)
    pv = corp_v4_df.merge(
        v3_cor[["trace", "vm_id", "cpu_class_accuracy"]].rename(columns={"cpu_class_accuracy": "cpu_v3"}),
        on=["trace", "vm_id"],
        how="inner",
    )
    pv["delta_cpu"] = pv["cpu_class_accuracy"] - pv["cpu_v3"]

    ci_m, ci_lo, ci_hi = _bootstrap_mean_ci(pv["delta_cpu"].to_numpy(float), rng)
    wilc_p = _wilcoxon_safe(pv["delta_cpu"].to_numpy(float))

    mean_v4 = float(corp_v4_df["cpu_class_accuracy"].mean())
    mean_v3 = float(pv["cpu_v3"].mean())

    drift_v4 = corp_v4_df["n_drift_refreshes"].astype(float).mean()
    drift_heur = []
    for _, rr in corp_v4_df.iterrows():
        p3 = corpus_dir / f"dynapm_v3_v2__{rr['trace']}__{rr['vm_id']}.json"
        sj = _summarize_vm(p3)
        if sj is None:
            continue
        drift_heur.append(_est_drift_bundles_v3(int(sj["total_updates"]), float(sj["stream_days"])))
    drift_mu_v3 = float(np.mean(drift_heur)) if drift_heur else float("nan")
    pct_redux = (
        float(100.0 * (drift_mu_v3 - float(drift_v4)) / drift_mu_v3)
        if drift_mu_v3 > 1e-9 and np.isfinite(drift_v4)
        else float("nan")
    )

    tier_cor_df = pd.DataFrame(
        [
            {
                "tier": t,
                "static_cpu": float(stat_cor[stat_cor["volatility_tier"] == t]["cpu_class_accuracy"].mean()),
                "predictor_cpu": float(pp_cor[pp_cor["volatility_tier"] == t]["cpu_class_accuracy"].mean()),
                "v3_cpu": float(v3_cor[v3_cor["volatility_tier"] == t]["cpu_class_accuracy"].mean()),
                "v4_cpu": float(corp_v4_df[corp_v4_df["volatility_tier"] == t]["cpu_class_accuracy"].mean()),
            }
            for t in ["low", "medium", "high"]
        ]
    )

    # --- Sweep per-tier ---
    tier_sweep_rows = []
    for a in ALPHA_VALUES:
        sub = sweep_df[np.isclose(sweep_df["alpha"], a)]
        for tier in ["low", "medium", "high"]:
            st = sub[sub["volatility_tier"] == tier]["cpu_class_accuracy"].mean()
            tier_sweep_rows.append({"alpha": a, "tier": tier, "mean_cpu_acc": float(st)})

    # --- Figures ---
    plt.figure(figsize=(5.8, 4.8))
    plt.scatter(
        grp["total_updates"], grp["cpu_class_accuracy"], s=56, marker="s", edgecolors="black", linewidths=0.8, label="V4 alphas"
    )
    for _, r in grp.iterrows():
        plt.annotate(f"{r['alpha']:.2f}".rstrip("0").rstrip("."), (r["total_updates"], r["cpu_class_accuracy"]), fontsize=7)
    mv3 = float(v3_summ["cpu_class_accuracy"].mean())
    mtu_v3 = v3_s_tu_mean
    plt.scatter([pp_s_tu_mean], [pp_s_cpu_mean], c="orange", marker="o", s=72, edgecolors="k", label="Periodic-Predictor", zorder=5)
    plt.scatter([mtu_v3], [mv3], c="purple", marker="^", s=72, edgecolors="k", label="DynAPM-V3", zorder=5)
    plt.xlabel("Mean total_updates (sample 100 VMs)")
    plt.ylabel("Mean cpu_class_accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "alpha_sweep_pareto.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    tdf = pd.DataFrame(tier_sweep_rows)
    alph_labels = ALPHA_VALUES
    x = np.arange(len(alph_labels))
    cols = {"low": "tab:blue", "medium": "tab:orange", "high": "tab:red"}
    w = 0.22
    for ti, tier in enumerate(["low", "medium", "high"]):
        offs = [-w, 0, w][ti]
        vals = []
        for a in alph_labels:
            v = float(tdf[(tdf.alpha == a) & (tdf.tier == tier)]["mean_cpu_acc"].squeeze())
            vals.append(v)
        plt.bar(x + offs, vals, width=w * 0.95, label=tier, color=cols[tier])
    plt.xticks(x, [str(a) for a in alph_labels])
    plt.xlabel("Alpha")
    plt.ylabel("Mean CPU class accuracy (sample)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "alpha_sweep_per_tier.png", dpi=140)
    plt.close()

    pv_plot = corp_v4_df.merge(
        v3_cor[["trace", "vm_id", "cpu_class_accuracy"]].rename(columns={"cpu_class_accuracy": "cpu_v3"}),
        on=["trace", "vm_id"],
        how="inner",
    )
    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(pv_plot["cpu_v3"], pv_plot["cpu_class_accuracy"], alpha=0.25, s=10)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("DynAPM-V3 cpu_class_accuracy (corpus)")
    plt.ylabel(f"DynAPM-V4 (alpha={best_a}) cpu_class_accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(report_dir / "v4_vs_v3_corpus.png", dpi=140)
    plt.close()

    qq1 = sanity_gap_cpu < 0.005
    cpu09 = float(grp[np.isclose(grp["alpha"], 0.9)]["cpu_class_accuracy"].iloc[0])
    cpu00 = float(grp[np.isclose(grp["alpha"], 0.0)]["cpu_class_accuracy"].iloc[0])
    qq2 = abs(cpu09 - cpu00) >= 0.005

    qq3 = mean_v4 >= mean_v3 + 0.001
    qq4 = ci_lo > 0
    qq5 = pct_redux > 0

    # Report write
    def md_tbl(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(round(row[c], 6)) if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = []
    lines.append("# Task 16: Decision head + churn + alpha sweep\n")
    lines.append("\n## What was implemented\n\n")
    lines.append(
        textwrap.fill(
            "Added River-backed `DecisionHeadV2`, `ChurnControllerV2`, and `DynAPMPolicyV4` wrapping the v3 predictor/drift cadence "
            "with retrospective online learning invoked by `runner_v2.run_method_on_vm` (canonical; calls `verify_pending_commits` when present). "
            "Drift commits pass churn eligibility (min 2h) and probabilistic head gating; periodic commits unchanged.",
            width=104,
        )
        + "\n"
    )

    lines.append("\n## Sample alpha sweep\n\nSorted by alpha ascending:\n\n")
    lines.append(md_tbl(grp) + "\n")

    lines.append(f"\n**Reference (same 100-VM rows):** DynAPM-V3 mean CPU **{v3_mean_cpu:.6f}** | Periodic-Predictor mean CPU **{pp_s_cpu_mean:.6f}**.\n")

    lines.append("\n### Best-alpha selection rationale\n")
    lines.append(
        textwrap.fill(
            f"**Best accuracy alpha** (maximum mean CPU accuracy on the 100-VM ladder): **{best_acc_alpha}**. "
            f'**Efficiency-preferred alpha** (lowest mean `total_updates` while mean CPU stays within **0.005** of V3’s sample mean '
            f"**{v3_mean_cpu:.4f}**): **{'N/A — no feasible alpha' if np.isnan(efficient_alpha) else efficient_alpha}**.",
            width=104,
        )
        + "\n"
    )

    lines.append("\n## Corpus run with best alpha\n\n")

    refs = corp_ref_ok[corp_ref_ok["method"].isin(["static_v2", "dynapm_v3_v2", "periodic_predictor_6h_v2"])]
    ref_agg = refs.groupby("method").agg({"cpu_class_accuracy": ["mean", "median"], "total_updates": ["mean", "median"]})
    rav = corp_v4_df.groupby(lambda _: 0).agg(
        cpu_class_accuracy_mean=("cpu_class_accuracy", "mean"),
        cpu_class_accuracy_median=("cpu_class_accuracy", "median"),
        total_updates_mean=("total_updates", "mean"),
        total_updates_median=("total_updates", "median"),
    )
    lines.append("| method | cpu mean | cpu median | total_updates mean | total_updates median |\n|---|---|---|---|---|\n")
    for m in ["static_v2", "dynapm_v3_v2", "periodic_predictor_6h_v2"]:
        if m not in refs["method"].values:
            continue
        r = refs[refs["method"] == m]
        lines.append(f"| **{m}** | {r['cpu_class_accuracy'].mean():.6f} | {r['cpu_class_accuracy'].median():.6f} "
                     f"| {r['total_updates'].mean():.2f} | {r['total_updates'].median():.2f} |\n")
    lines.append(
        f"| **dynapm_v4 (α={best_a})** | {rav['cpu_class_accuracy_mean'].iloc[0]:.6f} | {rav['cpu_class_accuracy_median'].iloc[0]:.6f} "
        f"| {rav['total_updates_mean'].iloc[0]:.2f} | {rav['total_updates_median'].iloc[0]:.2f} |\n"
    )

    lines.append("\n### Pairwise V4 vs V3 corpus deltas (CPU accuracy)\n\n")
    lines.append(
        f"- **n VMs:** **{len(pv)}**\n"
        f"- **Mean Δ (V4−V3):** **{ci_m:.6f}**\n"
        f"- **Bootstrap 95% CI:** **[{ci_lo:.6f}, {ci_hi:.6f}]**\n"
        f"- **Wilcoxon signed-rank p:** **{wilc_p:.4g}**\n"
        f"- **Mean drift bundles:** V3 **heuristic {drift_mu_v3:.2f}** vs V4 **measured {drift_v4:.2f}** ⇒ **≈ {pct_redux:.1f}%** reduction (heuristic defined in Assumptions).\n"
    )

    lines.append("\n### Per-tier corpus breakdown (mean CPU accuracy)\n\n")
    lines.append(md_tbl(tier_cor_df.rename(columns=lambda c: str(c))) + "\n")

    lines.append("\n## Checkpoint questions\n")
    lines.append(
        f"| Q | Result | Notes |\n|---|---|---|\n"
        f"| Q1 α=0 V4 CPU ≈ V3 sample |Δ| < **0.005**? | **{'PASS' if qq1 else 'FAIL'}** | gap **{sanity_gap_cpu:.6f}** |\n"
        f"| Q2 CPU spread α=0 vs α=**0.9** ≥ **0.005**? | **{'PASS' if qq2 else 'FAIL'}** | |Δ| **{abs(cpu09 - cpu00):.6f}** |\n"
        f"| Q3 Corpus V4 mean CPU ≥ V3 + **0.001**? | **{'PASS' if qq3 else 'FAIL'}** | V4 **{mean_v4:.6f}**, V3 **{mean_v3:.6f}** |\n"
        f"| Q4 Bootstrap CI (V4−V3 ΔCPU) excludes **0**? | **{'PASS' if qq4 else 'FAIL'}** | **[{ci_lo:.6f}, {ci_hi:.6f}]** |\n"
        f"| Q5 Fewer drift refreshes than V3? | **{'PASS' if qq5 else 'FAIL'}** | **≈ {pct_redux:.1f}%** reduction |\n"
    )

    lines.append("\n## Figures\n")
    lines.append(f"- `{report_dir}/alpha_sweep_pareto.png`\n")
    lines.append(f"- `{report_dir}/alpha_sweep_per_tier.png`\n")
    lines.append(f"- `{report_dir}/v4_vs_v3_corpus.png`\n")

    lines.append("\n## Files created or modified\n")
    for fp in (
        REPO / "dynapm/core/decision_head_v2.py",
        REPO / "dynapm/core/churn_v2.py",
        REPO / "dynapm/methods/dynapm_v4.py",
        REPO / "dynapm/eval/runner_v2.py",
        REPO / "dynapm/tests/test_decision_head_v2.py",
        REPO / "dynapm/tests/test_churn_v2.py",
        REPO / "dynapm/tests/test_dynapm_v4.py",
        REPO / "dynapm/scripts/task16_alpha_sweep.py",
        sweep_parquet,
        corpus_summ,
        report_md,
        report_dir,
    ):
        lines.append(f"- `{fp}`\n")

    lines.append("\n## Test results\n")
    lines.append("Run from repo root (`PYTHONPATH=.`): ")
    lines.append("`pytest dynapm/tests/test_decision_head_v2.py dynapm/tests/test_churn_v2.py dynapm/tests/test_dynapm_v4.py`\n")

    lines.append("\n## Assumptions\n")
    lines.append(
        "- DynAPM-V3 corpus JSON omits per-reason refresh tallies; drift bundle counts for Q5 use "
        "`total_updates/4 − round(stream_days·24/6)` as a **heuristic** proxy for drift-only bundles.\n"
    )
    lines.append(
            "- Canonical `runner_v2` calls `verify_pending_commits` only at evaluation timestamps present in the ground-truth grid.\n"
    )

    lines.append("\n## Surprises\n")
    lines.append("_Populated automatically from checkpoints (see Diagnostics block in script stdout)._\n")

    lines.append("\n## Commands\n```text\n$ PYTHONPATH=. python dynapm/scripts/task16_alpha_sweep.py\n```\n")

    lines.append("\n## Next-step suggestion\n")
    lines.append(
        textwrap.fill(
            "Tighten the retrospective label wiring to exact `t_commit` ground-truth rows if available, explore "
            "corpus-optimal alphas disjoint from accuracy-max, and quantify memory-class trade-offs under tighter churn.",
            width=104,
        )
        + "\n"
    )

    report_md.write_text("".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "wall_sample_s": wall_sample,
                "wall_corpus_s": wall_corpus,
                "sanity_gap_a0_vs_v3": sanity_gap_cpu,
                "best_alpha_accuracy": best_acc_alpha,
                "efficient_alpha": efficient_alpha,
                "checkpoint": {"Q1": qq1, "Q2": qq2, "Q3": qq3, "Q4": qq4, "Q5": qq5},
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
