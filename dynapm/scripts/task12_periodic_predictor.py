#!/usr/bin/env python3
"""Task 12: PeriodicPredictorPolicyV2 on 100-VM sample + diagnostics + figures."""
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
from dynapm.methods.baselines_v2 import PeriodicPredictorPolicyV2  # noqa: E402

COLS = [
    "method",
    "trace",
    "vm_id",
    "volatility_tier",
    "cpu_class_accuracy",
    "memory_class_accuracy",
    "cpu_p95_mae",
    "mem_p95_mae",
    "n_cpu_class_updates",
    "n_memory_class_updates",
    "n_cpu_p95_updates",
    "n_mem_p95_updates",
    "total_updates",
    "n_eval_points_verified",
    "snapshots_dropped_unverified",
]


def row_from_out(out: dict, method: str, trace: str, vm_id: str, tier: str) -> dict:
    r = {k: out[k] for k in COLS if k in out}
    r["method"] = method
    r["trace"] = trace
    r["vm_id"] = str(vm_id)
    r["volatility_tier"] = tier
    return r


def main() -> int:
    sample_path = REPO / "dynapm/data/eval_sample_v2_summary.parquet"
    out_dir = REPO / "dynapm/results/eval_sample_v2"
    out_parquet = REPO / "dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet"
    report_dir = REPO / "dynapm/reports/task_12"
    diag_path = REPO / "dynapm/data/eval_sample_v2_periodic_predictor_diagnostics.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(sample_path)
    sample = base[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample["vm_id"] = sample["vm_id"].astype(str)
    assert len(sample) == 100

    def run_one(trace: str, vm_id: str, tier: str) -> dict:
        out = run_method_on_vm(
            trace,
            vm_id,
            lambda p, c, m, **k: PeriodicPredictorPolicyV2(p, c, m, period_hours=6.0, **k),
        )
        fn = out_dir / f"periodic_predictor_6h_v2__{trace}__{vm_id}.json"
        fn.write_text(json.dumps(out, default=str), encoding="utf-8")
        return row_from_out(out, "periodic_predictor_6h_v2", trace, vm_id, tier)

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(
        delayed(run_one)(str(r.trace), str(r.vm_id), str(r.volatility_tier)) for r in sample.itertuples(index=False)
    )
    wall = time.perf_counter() - t0

    pp = pd.DataFrame(rows)
    pp.to_parquet(out_parquet, index=False)

    per = base[base["method"] == "periodic_6h_v2"].copy()
    pair = pp[["trace", "vm_id", "volatility_tier", "cpu_class_accuracy", "memory_class_accuracy"]].rename(
        columns={"cpu_class_accuracy": "cpu_pp", "memory_class_accuracy": "mem_pp"}
    ).merge(
        per[["trace", "vm_id", "cpu_class_accuracy", "memory_class_accuracy"]].rename(
            columns={"cpu_class_accuracy": "cpu_per", "memory_class_accuracy": "mem_per"}
        ),
        on=["trace", "vm_id"],
        how="inner",
    )
    pair["delta_cpu_acc"] = pair["cpu_pp"] - pair["cpu_per"]
    pair["delta_mem_acc"] = pair["mem_pp"] - pair["mem_per"]

    metrics = [
        "cpu_class_accuracy",
        "memory_class_accuracy",
        "cpu_p95_mae",
        "mem_p95_mae",
        "total_updates",
    ]
    agg_pp = {f"{m}_mean": float(pp[m].mean()) for m in metrics}
    agg_pp.update({f"{m}_median": float(pp[m].median()) for m in metrics})
    agg_per = {f"{m}_mean": float(per[m].mean()) for m in metrics}
    agg_per.update({f"{m}_median": float(per[m].median()) for m in metrics})

    tier_pp = pp.groupby("volatility_tier")["cpu_class_accuracy"].mean()
    tier_per = per.groupby("volatility_tier")["cpu_class_accuracy"].mean()

    mean_cpu_gap = float(pp["cpu_class_accuracy"].mean() - per["cpu_class_accuracy"].mean())
    mean_mem_gap = float(pp["memory_class_accuracy"].mean() - per["memory_class_accuracy"].mean())

    Q1 = mean_cpu_gap >= 0.02
    Q2 = mean_mem_gap >= 0.02
    Q3 = int((pair["delta_cpu_acc"] >= 0).sum()) >= 60
    Q4 = float(tier_pp.get("high", np.nan)) > float(tier_per.get("high", np.nan))
    Q5 = abs(mean_cpu_gap) < 0.02

    fig1 = report_dir / "predictor_vs_window_per_vm.png"
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(pair["cpu_per"], pair["cpu_pp"], alpha=0.5, s=15)
    lo, hi = 0.0, 1.0
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("periodic_6h_v2 CPU class accuracy")
    plt.ylabel("periodic_predictor_6h_v2 CPU class accuracy")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(fig1, dpi=140)
    plt.close()

    fig2 = report_dir / "predictor_vs_window_per_tier.png"
    tiers = ["low", "medium", "high"]
    x = np.arange(len(tiers))
    w = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w / 2, [float(tier_per[t]) for t in tiers], width=w, label="periodic_6h_v2")
    plt.bar(x + w / 2, [float(tier_pp[t]) for t in tiers], width=w, label="periodic_predictor_6h_v2")
    plt.xticks(x, tiers)
    plt.ylabel("Mean CPU class accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=140)
    plt.close()

    out = {
        "wall_seconds": float(wall),
        "summary_parquet": str(out_parquet),
        "agg_periodic_predictor": agg_pp,
        "agg_periodic_6h_ref": agg_per,
        "mean_gap_cpu": mean_cpu_gap,
        "mean_gap_mem": mean_mem_gap,
        "pairwise": {
            "mean_delta_cpu": float(pair["delta_cpu_acc"].mean()),
            "median_delta_cpu": float(pair["delta_cpu_acc"].median()),
            "mean_delta_mem": float(pair["delta_mem_acc"].mean()),
            "median_delta_mem": float(pair["delta_mem_acc"].median()),
            "count_cpu_gt_0": int((pair["delta_cpu_acc"] > 0).sum()),
            "count_cpu_lt_neg_005": int((pair["delta_cpu_acc"] < -0.05).sum()),
            "count_mem_gt_0": int((pair["delta_mem_acc"] > 0).sum()),
            "count_mem_lt_neg_005": int((pair["delta_mem_acc"] < -0.05).sum()),
        },
        "per_tier": {"periodic_predictor": tier_pp.to_dict(), "periodic_6h": tier_per.to_dict()},
        "checkpoint": {"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4, "Q5": Q5},
    }
    diag_path.write_text(json.dumps(out, default=str, indent=2), encoding="utf-8")

    print(json.dumps({"wall_seconds": wall, "mean_cpu_gap": mean_cpu_gap, "checkpoint": out["checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
