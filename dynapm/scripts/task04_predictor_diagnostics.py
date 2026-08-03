#!/usr/bin/env python3
"""Task 4 predictor diagnostics and plots from full-corpus summary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.insert(0, str(REPO))
from dynapm.scripts import _matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt


from dynapm.data.bootstrap import compute_split
from dynapm.data.loader import load_vm_csv

SUMMARY_PATH = REPO / "dynapm/data/predictor_summary.parquet"
FAIL_PATH = REPO / "dynapm/data/predictor_failures.json"
KEPT_PATH = REPO / "dynapm/data/kept_vms.json"
OUT_DIR = REPO / "dynapm/reports/task_04"
STATS_JSON = OUT_DIR / "predictor_diagnostics.json"

CPU_T = "CPU usage [%]"


def _dist(s: pd.Series) -> dict[str, float]:
    return {
        "min": float(s.min()),
        "p5": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def _compute_vol_tiers(df_summary: pd.DataFrame) -> pd.DataFrame:
    with open(KEPT_PATH, encoding="utf-8") as f:
        kept = json.load(f)
    rows = []
    for item in kept:
        trace = item["trace"]
        vm_id = item["vm_id"]
        p = REPO / "data/raw" / trace / f"{vm_id}.csv"
        df = load_vm_csv(str(p))
        df_b, _, _ = compute_split(df)
        rows.append({"trace": trace, "vm_id": vm_id, "cpu_boot_std": float(df_b[CPU_T].std(ddof=0))})
    s = pd.DataFrame(rows)
    q1 = float(s["cpu_boot_std"].quantile(1 / 3))
    q2 = float(s["cpu_boot_std"].quantile(2 / 3))

    def tier(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    s["tier"] = s["cpu_boot_std"].map(tier)
    m = df_summary.merge(s[["trace", "vm_id", "tier"]], on=["trace", "vm_id"], how="left")
    return m


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SUMMARY_PATH)
    with open(FAIL_PATH, encoding="utf-8") as f:
        fail = json.load(f)

    df = _compute_vol_tiers(df)

    # r2 distribution histogram figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["cpu_r2"], bins=40)
    axes[0].set_title("CPU R2 distribution")
    axes[1].hist(df["mem_r2"], bins=40)
    axes[1].set_title("Memory R2 distribution")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "predictor_r2_distribution.png", dpi=120)
    plt.close(fig)

    # R2 by tier, faceted by metric
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tiers = ["low", "medium", "high"]
    axes[0].boxplot([df.loc[df["tier"] == t, "cpu_r2"] for t in tiers], tick_labels=tiers)
    axes[0].set_title("CPU R2 by volatility tier")
    axes[1].boxplot([df.loc[df["tier"] == t, "mem_r2"] for t in tiers], tick_labels=tiers)
    axes[1].set_title("Memory R2 by volatility tier")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "predictor_r2_by_tier.png", dpi=120)
    plt.close(fig)

    # R2 by trace
    traces = sorted(df["trace"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([df.loc[df["trace"] == t, "cpu_r2"] for t in traces], tick_labels=traces)
    axes[0].set_title("CPU R2 by trace")
    axes[1].boxplot([df.loc[df["trace"] == t, "mem_r2"] for t in traces], tick_labels=traces)
    axes[1].set_title("Memory R2 by trace")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "predictor_r2_by_trace.png", dpi=120)
    plt.close(fig)

    stats = {
        "n_failed_vms": len(fail),
        "cpu_r2": _dist(df["cpu_r2"]),
        "mem_r2": _dist(df["mem_r2"]),
        "cpu_rmse": _dist(df["cpu_rmse"]),
        "mem_rmse": _dist(df["mem_rmse"]),
        "cpu_mae": _dist(df["cpu_mae"]),
        "mem_mae": _dist(df["mem_mae"]),
        "cpu_r2_lt_0": int((df["cpu_r2"] < 0).sum()),
        "mem_r2_lt_0": int((df["mem_r2"] < 0).sum()),
        "cpu_r2_lt_0_5": int((df["cpu_r2"] < 0.5).sum()),
        "mem_r2_lt_0_5": int((df["mem_r2"] < 0.5).sum()),
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({"rows": len(df), "n_failed": len(fail), "stats_json": str(STATS_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
