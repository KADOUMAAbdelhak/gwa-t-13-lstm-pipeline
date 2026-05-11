#!/usr/bin/env python3
"""Task 14: DynAPMPolicyV3 — 100-VM eval, aggregates, figures, checkpoints."""
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
from dynapm.methods.dynapm_v3 import DynAPMPolicyV3  # noqa: E402

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


def refresh_reason_counts(update_log: list) -> tuple[int, int]:
    """Count refresh events (each refresh logs 4 rows with same timestamp+reason)."""
    if not update_log:
        return 0, 0
    # Group by (timestamp, reason) — one row per refresh event
    df = pd.DataFrame(update_log)
    if "reason" not in df.columns:
        return 0, 0
    sub = df[["timestamp", "reason"]].drop_duplicates()
    n_per = int((df["field"] == "cpu_class").sum())  # sanity: should be #refreshes
    peri = int((sub["reason"] == "periodic").sum())
    drift = int((sub["reason"] == "drift").sum())
    return peri, drift


def main() -> int:
    sample_path = REPO / "dynapm/data/eval_sample_v2_summary.parquet"
    merged_path = REPO / "dynapm/data/eval_sample_v2_summary_with_dynapm.parquet"
    pp_path = REPO / "dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet"
    out_dir = REPO / "dynapm/results/eval_sample_v2"
    out_parquet = REPO / "dynapm/data/eval_sample_v2_dynapm_v3_summary.parquet"
    report_dir = REPO / "dynapm/reports/task_14"
    vm501_report = report_dir / "dynapm_v3_vm501.json"
    diag_path = REPO / "dynapm/data/eval_sample_v2_dynapm_v3_diagnostics.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(sample_path)
    sample = base[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample["vm_id"] = sample["vm_id"].astype(str)
    assert len(sample) == 100

    # --- Single-VM sanity ---
    trace501, vm501 = "Materna-Trace-1", "501"
    out_v3 = run_method_on_vm(
        trace501,
        vm501,
        lambda p, c, m, **k: DynAPMPolicyV3(
            p, c, m, period_hours=6.0, drift_delta=0.002, **k
        ),
    )
    vm501_report.write_text(json.dumps(out_v3, default=str), encoding="utf-8")

    out_pp_vm = run_method_on_vm(
        trace501,
        vm501,
        lambda p, c, m, **k: PeriodicPredictorPolicyV2(p, c, m, period_hours=6.0, **k),
    )

    peri501 = json.loads((REPO / "dynapm/reports/task_09/periodic_6h_v2_vm501.json").read_text())
    dv2_501 = json.loads((REPO / "dynapm/reports/task_09/dynapm_v2_vm501.json").read_text())

    r_peri501, r_drift501 = refresh_reason_counts(out_v3.get("update_log", []))

    single_table = pd.DataFrame(
        [
            {
                "method": "periodic_6h_v2 (window)",
                "cpu_class_accuracy": peri501["cpu_class_accuracy"],
                "memory_class_accuracy": peri501["memory_class_accuracy"],
                "total_updates": peri501["total_updates"],
                "n_periodic_refreshes": None,
                "n_drift_refreshes": None,
            },
            {
                "method": "periodic_predictor_6h_v2",
                "cpu_class_accuracy": out_pp_vm["cpu_class_accuracy"],
                "memory_class_accuracy": out_pp_vm["memory_class_accuracy"],
                "total_updates": out_pp_vm["total_updates"],
                "n_periodic_refreshes": None,
                "n_drift_refreshes": None,
            },
            {
                "method": "dynapm_v2",
                "cpu_class_accuracy": dv2_501["cpu_class_accuracy"],
                "memory_class_accuracy": dv2_501["memory_class_accuracy"],
                "total_updates": dv2_501["total_updates"],
                "n_periodic_refreshes": None,
                "n_drift_refreshes": None,
            },
            {
                "method": "dynapm_v3",
                "cpu_class_accuracy": out_v3["cpu_class_accuracy"],
                "memory_class_accuracy": out_v3["memory_class_accuracy"],
                "total_updates": out_v3["total_updates"],
                "n_periodic_refreshes": r_peri501,
                "n_drift_refreshes": r_drift501,
            },
        ]
    )

    # --- 100-VM parallel ---
    def run_one(trace: str, vm_id: str, tier: str) -> dict:
        out = run_method_on_vm(
            trace,
            vm_id,
            lambda p, c, m, **k: DynAPMPolicyV3(
                p, c, m, period_hours=6.0, drift_delta=0.002, **k
            ),
        )
        fn = out_dir / f"dynapm_v3__{trace}__{vm_id}.json"
        fn.write_text(json.dumps(out, default=str), encoding="utf-8")
        return row_from_out(out, "dynapm_v3", trace, vm_id, tier)

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(
        delayed(run_one)(str(r.trace), str(r.vm_id), str(r.volatility_tier)) for r in sample.itertuples(index=False)
    )
    wall = time.perf_counter() - t0

    v3 = pd.DataFrame(rows)
    v3.to_parquet(out_parquet, index=False)

    merged = pd.read_parquet(merged_path)
    pp = pd.read_parquet(pp_path)
    pp = pp.drop(columns=["volatility_tier", "method"], errors="ignore")
    pp = sample[["trace", "vm_id", "volatility_tier"]].merge(
        pp, on=["trace", "vm_id"], how="inner"
    )
    pp["method"] = "periodic_predictor_6h_v2"

    methods_needed = ["static_v2", "periodic_6h_v2", "full_rewrite_v2", "dynapm_v2"]
    ref = merged[merged["method"].isin(methods_needed)].copy()

    def agg_block(df: pd.DataFrame, name: str) -> dict:
        mets = ["cpu_class_accuracy", "memory_class_accuracy", "cpu_p95_mae", "mem_p95_mae", "total_updates"]
        return {
            "method": name,
            **{f"{m}_mean": float(df[m].mean()) for m in mets},
            **{f"{m}_median": float(df[m].median()) for m in mets},
        }

    headline_rows = []
    for m in methods_needed:
        sub = ref[ref["method"] == m]
        headline_rows.append(agg_block(sub, m))
    headline_rows.append(agg_block(pp, "periodic_predictor_6h_v2"))
    headline_rows.append(agg_block(v3, "dynapm_v3"))

    headline_df = pd.DataFrame(headline_rows)
    headline_df = headline_df.sort_values("cpu_class_accuracy_mean", ascending=True)

    # Per-tier CPU accuracy
    tier_methods = {
        "static_v2": merged[merged["method"] == "static_v2"],
        "periodic_6h_v2": merged[merged["method"] == "periodic_6h_v2"],
        "periodic_predictor_6h_v2": pp,
        "dynapm_v2": merged[merged["method"] == "dynapm_v2"],
        "dynapm_v3": v3,
        "full_rewrite_v2": merged[merged["method"] == "full_rewrite_v2"],
    }
    tier_table = []
    for tier in ["low", "medium", "high"]:
        row = {"volatility_tier": tier}
        for name, sdf in tier_methods.items():
            sub = sdf[sdf["volatility_tier"] == tier]
            row[name] = float(sub["cpu_class_accuracy"].mean()) if len(sub) else float("nan")
        tier_table.append(row)
    tier_df = pd.DataFrame(tier_table)

    # Pairwise v3 vs pp
    pair = v3.merge(
        pp[["trace", "vm_id", "cpu_class_accuracy", "memory_class_accuracy"]].rename(
            columns={
                "cpu_class_accuracy": "cpu_pp",
                "memory_class_accuracy": "mem_pp",
            }
        ),
        on=["trace", "vm_id"],
        how="inner",
    )
    pair["delta_cpu_acc"] = pair["cpu_class_accuracy"] - pair["cpu_pp"]
    pair["delta_mem_acc"] = pair["memory_class_accuracy"] - pair["mem_pp"]

    d_cpu = pair["delta_cpu_acc"]
    d_mem = pair["delta_mem_acc"]
    pairwise = {
        "mean_delta_cpu": float(d_cpu.mean()),
        "median_delta_cpu": float(d_cpu.median()),
        "mean_delta_mem": float(d_mem.mean()),
        "median_delta_mem": float(d_mem.median()),
        "count_cpu_gt_0": int((d_cpu > 0).sum()),
        "count_cpu_lt_neg_002": int((d_cpu < -0.02).sum()),
        "count_cpu_tied_0_to_001": int(((d_cpu >= 0) & (d_cpu < 0.01)).sum()),
        "count_cpu_ge_001": int((d_cpu >= 0.01).sum()),
        "count_mem_gt_0": int((d_mem > 0).sum()),
        "count_mem_lt_neg_002": int((d_mem < -0.02).sum()),
        "count_mem_tied_0_to_001": int(((d_mem >= 0) & (d_mem < 0.01)).sum()),
        "count_mem_ge_001": int((d_mem >= 0.01).sum()),
    }

    mean_v3_cpu = float(v3["cpu_class_accuracy"].mean())
    mean_pp_cpu = float(pp["cpu_class_accuracy"].mean())
    mean_v3_mem = float(v3["memory_class_accuracy"].mean())
    mean_pp_mem = float(pp["memory_class_accuracy"].mean())
    mean_v3_tu = float(v3["total_updates"].mean())
    mean_pp_tu = float(pp["total_updates"].mean())
    mean_fr_cpu = float(merged[merged["method"] == "full_rewrite_v2"]["cpu_class_accuracy"].mean())

    tier_high_v3 = float(v3[v3["volatility_tier"] == "high"]["cpu_class_accuracy"].mean())
    tier_high_pp = float(pp[pp["volatility_tier"] == "high"]["cpu_class_accuracy"].mean())

    Q1 = mean_v3_cpu >= mean_pp_cpu + 0.01
    Q2 = mean_v3_mem >= mean_pp_mem + 0.01
    Q3 = int((d_cpu >= 0).sum()) >= 50
    Q4 = tier_high_v3 > tier_high_pp
    Q5 = mean_v3_tu <= 1.5 * mean_pp_tu
    Q6 = abs(mean_v3_cpu - mean_fr_cpu) <= 0.05

    checkpoint = {
        "Q1_headline_cpu_ge_pp_1pp": {"pass": Q1, "mean_v3": mean_v3_cpu, "mean_pp": mean_pp_cpu, "gap": mean_v3_cpu - mean_pp_cpu},
        "Q2_headline_mem_ge_pp_1pp": {"pass": Q2, "mean_v3": mean_v3_mem, "mean_pp": mean_pp_mem, "gap": mean_v3_mem - mean_pp_mem},
        "Q3_at_least_50vm_delta_cpu_ge_0": {"pass": Q3, "count": int((d_cpu >= 0).sum())},
        "Q4_high_tier_cpu_v3_gt_pp": {"pass": Q4, "tier_high_v3": tier_high_v3, "tier_high_pp": tier_high_pp},
        "Q5_total_updates_within_50pct_pp": {"pass": Q5, "mean_v3_tu": mean_v3_tu, "mean_pp_tu": mean_pp_tu, "ratio": mean_v3_tu / mean_pp_tu if mean_pp_tu else float("nan")},
        "Q6_v3_within_05_of_fullrewrite": {"pass": Q6, "mean_v3_cpu": mean_v3_cpu, "mean_fr_cpu": mean_fr_cpu, "abs_gap": abs(mean_v3_cpu - mean_fr_cpu)},
    }

    # Per-VM refresh reasons for figure
    refresh_rows = []
    for _, r in sample.iterrows():
        p = out_dir / f"dynapm_v3__{r['trace']}__{r['vm_id']}.json"
        raw = json.loads(p.read_text())
        peri, drift = refresh_reason_counts(raw.get("update_log", []))
        refresh_rows.append(
            {"trace": r["trace"], "vm_id": str(r["vm_id"]), "periodic": peri, "drift": drift}
        )
    refresh_df = pd.DataFrame(refresh_rows)

    # Figures
    order = ["static_v2", "periodic_6h_v2", "periodic_predictor_6h_v2", "dynapm_v2", "dynapm_v3", "full_rewrite_v2"]
    labels = [
        "static_v2",
        "periodic_6h_v2\n(window)",
        "periodic_predictor\n6h_v2",
        "dynapm_v2",
        "dynapm_v3",
        "full_rewrite_v2",
    ]
    means_map = {r["method"]: r["cpu_class_accuracy_mean"] for r in headline_rows}
    heights = [means_map[m] for m in order]
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(order)), heights, color="steelblue")
    plt.xticks(range(len(order)), labels, fontsize=8)
    plt.ylabel("Mean CPU class accuracy (100 VMs)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(report_dir / "method_progression.png", dpi=140)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(pair["cpu_pp"], pair["cpu_class_accuracy"], alpha=0.5, s=18)
    lo, hi = 0.0, 1.0
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("Periodic-Predictor cpu_class_accuracy")
    plt.ylabel("DynAPM-V3 cpu_class_accuracy")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(report_dir / "dynapm_v3_vs_periodic_predictor_per_vm.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 4))
    x = np.arange(len(refresh_df))
    plt.bar(x, refresh_df["periodic"], label="periodic", color="tab:blue")
    plt.bar(x, refresh_df["drift"], bottom=refresh_df["periodic"], label="drift", color="tab:orange")
    plt.ylabel("Refresh events")
    plt.xlabel("VM index (sample order)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "refresh_reason_breakdown.png", dpi=140)
    plt.close()

    diag = {
        "wall_seconds_100vm": float(wall),
        "summary_parquet": str(out_parquet),
        "vm501_json": str(vm501_report),
        "headline_table_sorted": headline_df.to_dict(orient="records"),
        "single_vm_table": single_table.to_dict(orient="records"),
        "per_tier_cpu": tier_df.to_dict(orient="records"),
        "pairwise": pairwise,
        "refresh_reasons_vm501": {"periodic": r_peri501, "drift": r_drift501},
        "checkpoint": checkpoint,
    }
    diag_path.write_text(json.dumps(diag, default=str, indent=2), encoding="utf-8")

    print(json.dumps({"wall_100vm": wall, "checkpoint": checkpoint}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
