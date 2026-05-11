#!/usr/bin/env python3
"""Task 11: drift_delta sweep on 100-VM sample — writes parquet, diagnostics JSON, figures."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dynapm.scripts import _matplotlib_setup  # noqa: F401

from dynapm.eval.runner_v2 import run_method_on_vm  # noqa: E402
from dynapm.methods.dynapm_v2 import DynAPMPolicyV2  # noqa: E402

DELTAS = [0.0001, 0.0005, 0.001, 0.005, 0.01]
METRICS_AGG = [
    "cpu_class_accuracy",
    "memory_class_accuracy",
    "cpu_p95_mae",
    "mem_p95_mae",
    "total_updates",
    "n_cpu_class_updates",
    "n_memory_class_updates",
]


def main() -> int:
    sample_path = REPO / "dynapm/data/eval_sample_v2_summary.parquet"
    ref_path = REPO / "dynapm/data/eval_sample_v2_summary_with_dynapm.parquet"
    out_parquet = REPO / "dynapm/data/eval_sample_v2_dynapm_delta_sweep.parquet"
    out_json_dir = REPO / "dynapm/results/eval_sample_v2"
    report_dir = REPO / "dynapm/reports/task_11"
    diag_path = REPO / "dynapm/data/eval_sample_v2_delta_sweep_diagnostics.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(sample_path)
    sample = base[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample["vm_id"] = sample["vm_id"].astype(str)
    assert len(sample) == 100, len(sample)

    def run_one(delta: float, trace: str, vm_id: str, tier: str) -> dict:
        out = run_method_on_vm(
            trace,
            vm_id,
            lambda p, c, m, **k: DynAPMPolicyV2(p, c, m, drift_delta=delta, alpha=0.5, **k),
        )
        fname = f"dynapm_v2_delta_{delta}__{trace}__{vm_id}.json"
        (out_json_dir / fname).write_text(json.dumps(out, default=str), encoding="utf-8")
        mname = f"dynapm_v2_delta_{delta}"
        return {
            "drift_delta": float(delta),
            "method": mname,
            "trace": trace,
            "vm_id": str(vm_id),
            "volatility_tier": tier,
            "cpu_class_accuracy": float(out["cpu_class_accuracy"]),
            "memory_class_accuracy": float(out["memory_class_accuracy"]),
            "cpu_p95_mae": float(out["cpu_p95_mae"]),
            "mem_p95_mae": float(out["mem_p95_mae"]),
            "n_cpu_class_updates": int(out["n_cpu_class_updates"]),
            "n_memory_class_updates": int(out["n_memory_class_updates"]),
            "n_cpu_p95_updates": int(out["n_cpu_p95_updates"]),
            "n_mem_p95_updates": int(out["n_mem_p95_updates"]),
            "total_updates": int(out["total_updates"]),
            "n_eval_points_verified": int(out["n_eval_points_verified"]),
        }

    jobs = [(d, str(r.trace), str(r.vm_id), str(r.volatility_tier)) for d in DELTAS for r in sample.itertuples(index=False)]

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(delayed(run_one)(d, tr, vm, tier) for d, tr, vm, tier in jobs)
    wall = time.perf_counter() - t0

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_parquet(out_parquet, index=False)

    ref = pd.read_parquet(ref_path)
    ref_subset = ref[ref["method"].isin(["dynapm_v2", "periodic_6h_v2", "full_rewrite_v2", "static_v2"])].copy()
    ref_subset["drift_delta"] = np.where(ref_subset["method"] == "dynapm_v2", 0.002, np.nan)

    per_cpu = float(ref_subset[ref_subset["method"] == "periodic_6h_v2"]["cpu_class_accuracy"].mean())
    per_up = float(ref_subset[ref_subset["method"] == "periodic_6h_v2"]["total_updates"].mean())

    by_d = []
    for d in DELTAS:
        sub = sweep_df[sweep_df["drift_delta"] == d]
        row = {"method": f"dynapm_v2_delta_{d}", "drift_delta": d, "kind": "sweep"}
        for m in METRICS_AGG:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_median"] = float(sub[m].median())
        by_d.append(row)
    by_d = pd.DataFrame(by_d)

    ref_rows = []
    for method in ["dynapm_v2", "periodic_6h_v2", "full_rewrite_v2", "static_v2"]:
        sub = ref_subset[ref_subset["method"] == method]
        row = {"method": method, "drift_delta": (0.002 if method == "dynapm_v2" else np.nan), "kind": "reference"}
        for m in METRICS_AGG:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_median"] = float(sub[m].median())
        ref_rows.append(row)
    ref_agg = pd.DataFrame(ref_rows)

    combo = pd.concat([by_d, ref_agg], ignore_index=True).sort_values("total_updates_mean", ascending=True)

    best_acc_delta = max(
        DELTAS,
        key=lambda dd: float(sweep_df[sweep_df["drift_delta"] == dd]["cpu_class_accuracy"].mean()),
    )
    best_acc_val = float(sweep_df[sweep_df["drift_delta"] == best_acc_delta]["cpu_class_accuracy"].mean())

    pareto_pts = []
    for d in DELTAS:
        sub = sweep_df[sweep_df["drift_delta"] == d]
        mc = float(sub["cpu_class_accuracy"].mean())
        mu = float(sub["total_updates"].mean())
        pareto_pts.append(
            {
                "delta": d,
                "mean_cpu": mc,
                "mean_updates": mu,
                "acc_gain_vs_periodic": mc - per_cpu,
                "update_ratio_vs_periodic": mu / per_up,
            }
        )
    dyn_only = pareto_pts.copy()
    set_a = [p for p in dyn_only if p["update_ratio_vs_periodic"] < 1.0]
    opt_a = max(set_a, key=lambda x: x["mean_cpu"]) if set_a else None
    set_b = [p for p in dyn_only if p["mean_cpu"] >= per_cpu - 0.02]
    opt_b = min(set_b, key=lambda x: x["update_ratio_vs_periodic"]) if set_b else None

    best_sub = sweep_df[sweep_df["drift_delta"] == best_acc_delta]
    per_tier_dyn = best_sub.groupby("volatility_tier")["cpu_class_accuracy"].mean()
    per_tier_per = ref_subset[ref_subset["method"] == "periodic_6h_v2"].groupby("volatility_tier")["cpu_class_accuracy"].mean()

    per_mean_cpu_exact = per_cpu
    sweep_means = {d: float(sweep_df[sweep_df["drift_delta"] == d]["cpu_class_accuracy"].mean()) for d in DELTAS}
    sweep_up = {d: float(sweep_df[sweep_df["drift_delta"] == d]["total_updates"].mean()) for d in DELTAS}

    Q1 = any(sweep_means[d] >= per_mean_cpu_exact - 0.02 for d in DELTAS)
    Q2 = any(sweep_means[d] > per_mean_cpu_exact for d in DELTAS)
    order_desc = sorted(DELTAS, reverse=True)
    up_series = [sweep_up[d] for d in order_desc]
    Q3 = all(up_series[i] <= up_series[i + 1] for i in range(len(up_series) - 1))
    Q4 = any(0.5 * per_up <= sweep_up[d] <= 1.5 * per_up for d in DELTAS)
    Q5 = float(per_tier_dyn.get("high", np.nan)) > 0.6819

    fig1 = report_dir / "drift_delta_pareto.png"
    pts = sweep_df.groupby("drift_delta").agg(cpu=("cpu_class_accuracy", "mean"), upd=("total_updates", "mean")).reset_index()
    ref_static = ref_subset[ref_subset["method"] == "static_v2"].agg({"cpu_class_accuracy": "mean", "total_updates": "mean"})
    ref_fr = ref_subset[ref_subset["method"] == "full_rewrite_v2"].agg({"cpu_class_accuracy": "mean", "total_updates": "mean"})

    plt.figure(figsize=(7, 5))
    plt.xscale("log")
    for _, r in pts.iterrows():
        plt.scatter(r["upd"], r["cpu"], s=80)
        plt.annotate(f'{r["drift_delta"]:.4g}', (r["upd"], r["cpu"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.scatter(per_up, per_cpu, s=120, marker="s", label="periodic_6h_v2")
    plt.scatter(ref_fr["total_updates"], ref_fr["cpu_class_accuracy"], s=100, marker="^", label="full_rewrite_v2")
    plt.scatter(ref_static["total_updates"], ref_static["cpu_class_accuracy"], s=100, marker="v", label="static_v2")
    plt.xlabel("Mean total_updates (log x)")
    plt.ylabel("Mean cpu_class_accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig1, dpi=140)
    plt.close()

    fig2 = report_dir / "drift_delta_per_tier.png"
    tier_order = ["low", "medium", "high"]
    x = np.arange(len(DELTAS))
    w = 0.25
    plt.figure(figsize=(9, 5))
    for i, tier in enumerate(tier_order):
        vals = [
            float(sweep_df[(sweep_df["drift_delta"] == d) & (sweep_df["volatility_tier"] == tier)]["cpu_class_accuracy"].mean())
            for d in DELTAS
        ]
        plt.bar(x + (i - 1) * w, vals, width=w, label=f"tier {tier}")
    plt.axhline(float(per_tier_per["low"]), linestyle="--", linewidth=0.8, alpha=0.5, color="C0")
    plt.axhline(float(per_tier_per["medium"]), linestyle="--", linewidth=0.8, alpha=0.5, color="C1")
    plt.axhline(float(per_tier_per["high"]), linestyle="--", linewidth=0.8, alpha=0.5, color="C2")
    plt.xticks(x, [str(d) for d in DELTAS], rotation=15)
    plt.ylabel("Mean cpu_class_accuracy")
    plt.xlabel("drift_delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=140)
    plt.close()

    out = {
        "wall_seconds": float(wall),
        "n_runs": len(rows),
        "per_mean_cpu_periodic_exact": per_mean_cpu_exact,
        "per_mean_up_periodic": per_up,
        "sweep_means_cpu": {str(k): v for k, v in sweep_means.items()},
        "sweep_means_updates": {str(k): v for k, v in sweep_up.items()},
        "best_acc_delta": float(best_acc_delta),
        "best_acc_mean_cpu": best_acc_val,
        "pareto_highest_acc_update_ratio_lt_1": opt_a,
        "pareto_lowest_update_ratio_acc_within_002": opt_b,
        "pareto_points": pareto_pts,
        "combo_sorted_by_mean_updates": combo[["method", "drift_delta", "total_updates_mean", "cpu_class_accuracy_mean"]].to_dict(orient="records"),
        "Q1": Q1,
        "Q2": Q2,
        "Q3_monotonic_updates_as_delta_decreases": Q3,
        "Q4": Q4,
        "Q5": Q5,
        "per_tier_best_delta": per_tier_dyn.to_dict(),
        "per_tier_periodic": per_tier_per.to_dict(),
        "full_aggregate_table": combo.to_dict(orient="records"),
    }
    diag_path.write_text(json.dumps(out, default=str, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "wall_seconds": wall,
                "out_parquet": str(out_parquet),
                "best_delta": best_acc_delta,
                "Q1": Q1,
                "Q2": Q2,
                "Q3": Q3,
                "Q4": Q4,
                "Q5": Q5,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
