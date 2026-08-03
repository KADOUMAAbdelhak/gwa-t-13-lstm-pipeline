#!/usr/bin/env python3
"""Task 3 sample predictor run on 30 VMs with residual outputs and diagnostics."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from dynapm.scripts import _matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dynapm.core.predictor import VMPredictor
from dynapm.data.bootstrap import compute_split
from dynapm.data.loader import load_vm_csv

CPU_T = "CPU usage [%]"
MEM_T = "Memory usage [%]"

KEPT_PATH = REPO / "dynapm/data/kept_vms.json"
RESULTS_DIR = REPO / "dynapm/results/residuals_sample"
REPORT_DIR = REPO / "dynapm/reports/task_03"
SUMMARY_JSON = REPO / "dynapm/reports/task_03/sample_run_summary.json"


def _load_kept() -> list[dict]:
    with open(KEPT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _vm_path(item: dict) -> Path:
    return REPO / "data/raw" / item["trace"] / f'{item["vm_id"]}.csv'


def _tier_from_std(v: float, q1: float, q2: float) -> str:
    if v <= q1:
        return "low"
    if v <= q2:
        return "medium"
    return "high"


def _pick_per_trace_by_tier(annot: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    targets = {"low": 4, "medium": 3, "high": 3}
    traces = sorted({x["trace"] for x in annot})
    for tr in traces:
        pool = [x for x in annot if x["trace"] == tr]
        by_tier = {k: [x for x in pool if x["tier"] == k] for k in ("low", "medium", "high")}
        picked: list[dict] = []
        for tier, need in targets.items():
            cand = by_tier[tier][:]
            rng.shuffle(cand)
            take = cand[: min(need, len(cand))]
            picked.extend(take)
        if len(picked) < 10:
            chosen_ids = {(x["trace"], x["vm_id"]) for x in picked}
            rest = [x for x in pool if (x["trace"], x["vm_id"]) not in chosen_ids]
            rng.shuffle(rest)
            picked.extend(rest[: 10 - len(picked)])
        out.extend(picked[:10])
    return out


def _dist(vals: list[float]) -> dict[str, float]:
    s = pd.Series(vals, dtype=float)
    return {
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    kept = _load_kept()

    # Bootstrap CPU std for all kept VMs -> tertiles
    std_rows = []
    t_std0 = time.perf_counter()
    for item in kept:
        df = load_vm_csv(str(_vm_path(item)))
        df_b, _, _ = compute_split(df)
        std_rows.append(
            {
                "trace": item["trace"],
                "vm_id": item["vm_id"],
                "cpu_boot_std": float(df_b[CPU_T].std(ddof=0)),
            }
        )
    std_elapsed = time.perf_counter() - t_std0
    std_df = pd.DataFrame(std_rows)
    q1 = float(std_df["cpu_boot_std"].quantile(1 / 3))
    q2 = float(std_df["cpu_boot_std"].quantile(2 / 3))

    annot = []
    for r in std_rows:
        tier = _tier_from_std(r["cpu_boot_std"], q1, q2)
        annot.append({**r, "tier": tier})

    sample30 = _pick_per_trace_by_tier(annot, seed=42)

    # run predictors on sample
    per_vm_seconds = []
    metrics_rows = []
    resid_for_plot = []

    t0 = time.perf_counter()
    for item in sample30:
        trace = item["trace"]
        vm_id = item["vm_id"]
        tier = item["tier"]

        vm_t0 = time.perf_counter()
        df = load_vm_csv(str(_vm_path(item)))
        df_b, df_s, _ = compute_split(df)

        # CPU predictor
        p_cpu = VMPredictor(target=CPU_T)
        p_cpu.fit(df_b)
        out_cpu = p_cpu.predict_streaming(df_s)

        # Memory predictor
        p_mem = VMPredictor(target=MEM_T)
        p_mem.fit(df_b)
        out_mem = p_mem.predict_streaming(df_s)

        cpu_path = RESULTS_DIR / f"{trace}__{vm_id}__cpu.parquet"
        mem_path = RESULTS_DIR / f"{trace}__{vm_id}__mem.parquet"
        out_cpu.to_parquet(cpu_path, index=True)
        out_mem.to_parquet(mem_path, index=True)

        for metric_name, out in (("cpu", out_cpu), ("mem", out_mem)):
            y_true = out["y_true"].to_numpy(dtype=float)
            y_pred = out["y_pred"].to_numpy(dtype=float)
            metrics_rows.append(
                {
                    "trace": trace,
                    "vm_id": vm_id,
                    "tier": tier,
                    "metric": metric_name,
                    "r2": float(r2_score(y_true, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                }
            )
            resid_for_plot.append(
                {
                    "trace": trace,
                    "vm_id": vm_id,
                    "tier": tier,
                    "metric": metric_name,
                    "residuals": out["residual"].to_numpy(dtype=float),
                }
            )

        per_vm_seconds.append(time.perf_counter() - vm_t0)

    total_elapsed = time.perf_counter() - t0

    mdf = pd.DataFrame(metrics_rows)

    # distributions
    cpu_m = mdf[mdf["metric"] == "cpu"]
    mem_m = mdf[mdf["metric"] == "mem"]

    # tier medians
    tier_med_r2 = (
        mdf.groupby(["metric", "tier"])["r2"]
        .median()
        .reset_index()
        .pivot(index="tier", columns="metric", values="r2")
        .to_dict(orient="index")
    )

    # residual diagnostics plot: 2 rows (cpu/mem) x 3 cols (tiers)
    tiers = ["low", "medium", "high"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for col, tier in enumerate(tiers):
        for row, metric in enumerate(["cpu", "mem"]):
            ax = axes[row, col]
            arrs = [x["residuals"] for x in resid_for_plot if x["tier"] == tier and x["metric"] == metric]
            vals = np.concatenate(arrs) if arrs else np.array([])
            ax.hist(vals, bins=50, alpha=0.8)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_title(f"{metric.upper()} residuals — {tier}")
    plt.tight_layout()
    plot_path = REPORT_DIR / "residual_distributions.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)

    # storage estimate based on sample residual parquet bytes
    sample_bytes = sum(p.stat().st_size for p in RESULTS_DIR.glob("*.parquet"))
    sample_vm_count = len(sample30)
    bytes_per_vm = sample_bytes / sample_vm_count if sample_vm_count else 0.0
    est_full_bytes = bytes_per_vm * len(kept)

    summary = {
        "n_kept_total": len(kept),
        "cpu_std_tertiles": {"q33": q1, "q67": q2},
        "std_scan_wall_seconds": std_elapsed,
        "sample_count": len(sample30),
        "sample_by_trace": pd.Series([x["trace"] for x in sample30]).value_counts().to_dict(),
        "sample_by_tier": pd.Series([x["tier"] for x in sample30]).value_counts().to_dict(),
        "total_wall_seconds": total_elapsed,
        "per_vm_seconds": {
            "min": float(np.min(per_vm_seconds)),
            "median": float(np.median(per_vm_seconds)),
            "max": float(np.max(per_vm_seconds)),
        },
        "cpu_r2": _dist(cpu_m["r2"].tolist()),
        "mem_r2": _dist(mem_m["r2"].tolist()),
        "cpu_rmse": _dist(cpu_m["rmse"].tolist()),
        "mem_rmse": _dist(mem_m["rmse"].tolist()),
        "cpu_mae": _dist(cpu_m["mae"].tolist()),
        "mem_mae": _dist(mem_m["mae"].tolist()),
        "tier_median_r2": tier_med_r2,
        "sample_residual_bytes": int(sample_bytes),
        "bytes_per_vm_estimate": float(bytes_per_vm),
        "est_full_residual_bytes": float(est_full_bytes),
        "single_thread_est_seconds": float(np.median(per_vm_seconds) * len(kept)),
        "n_jobs8_est_seconds": float(np.median(per_vm_seconds) * len(kept) / 8.0),
        "plot_path": str(plot_path),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "summary_json": str(SUMMARY_JSON),
        "plot_path": str(plot_path),
        "sample_size": len(sample30),
        "total_wall_s": total_elapsed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
