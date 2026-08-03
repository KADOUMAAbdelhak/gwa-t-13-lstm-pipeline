#!/usr/bin/env python3
"""Task 4 corpus-wide drift characterization over full residual streams."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.insert(0, str(REPO))
from dynapm.scripts import _matplotlib_setup  # noqa: F401
import matplotlib.pyplot as plt


from dynapm.core.drift import run_detector
from dynapm.data.bootstrap import compute_split
from dynapm.data.loader import load_vm_csv

KEPT_PATH = REPO / "dynapm/data/kept_vms.json"
RESID_DIR = REPO / "dynapm/results/residuals_full"
OUT_DIR = REPO / "dynapm/results/drift_events"
SUMMARY_PATH = REPO / "dynapm/data/drift_summary.parquet"
STATS_JSON = REPO / "dynapm/data/drift_stats.json"
FIG_PATH = REPO / "dynapm/reports/task_04/drift_event_examples.png"

CPU_T = "CPU usage [%]"


def _dist(s: pd.Series) -> dict[str, float]:
    return {
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.5)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def _tier_map(kept: list[dict]) -> dict[tuple[str, str], str]:
    rows = []
    for item in kept:
        trace = item["trace"]
        vm_id = item["vm_id"]
        p = REPO / "data/raw" / trace / f"{vm_id}.csv"
        df = load_vm_csv(str(p))
        df_b, _, _ = compute_split(df)
        rows.append((trace, vm_id, float(df_b[CPU_T].std(ddof=0))))
    s = pd.DataFrame(rows, columns=["trace", "vm_id", "std"])
    q1 = float(s["std"].quantile(1 / 3))
    q2 = float(s["std"].quantile(2 / 3))
    out = {}
    for _, r in s.iterrows():
        v = float(r["std"])
        if v <= q1:
            t = "low"
        elif v <= q2:
            t = "medium"
        else:
            t = "high"
        out[(r["trace"], str(r["vm_id"]))] = t
    return out


def _process(item: dict) -> dict:
    trace = item["trace"]
    vm_id = item["vm_id"]
    cpu_path = RESID_DIR / f"{trace}__{vm_id}__cpu.parquet"
    mem_path = RESID_DIR / f"{trace}__{vm_id}__mem.parquet"

    cpu_df = pd.read_parquet(cpu_path)
    mem_df = pd.read_parquet(mem_path)

    cpu_series = pd.Series(cpu_df["residual"].to_numpy(dtype=float), index=pd.to_datetime(cpu_df.index))
    mem_series = pd.Series(mem_df["residual"].to_numpy(dtype=float), index=pd.to_datetime(mem_df.index))

    cpu_events = run_detector(cpu_series, detector_type="adwin", delta=0.002)
    mem_events = run_detector(mem_series, detector_type="adwin", delta=0.002)

    pd.DataFrame({"drift_timestamp": cpu_events}).to_parquet(OUT_DIR / f"{trace}__{vm_id}__cpu.parquet", index=False)
    pd.DataFrame({"drift_timestamp": mem_events}).to_parquet(OUT_DIR / f"{trace}__{vm_id}__mem.parquet", index=False)

    stream_start = pd.Timestamp(cpu_series.index[0])
    stream_end = pd.Timestamp(cpu_series.index[-1])
    stream_days = max((stream_end - stream_start).total_seconds() / 86400.0, 1e-9)

    cpu_first_h = None
    if cpu_events:
        cpu_first_h = float((pd.Timestamp(cpu_events[0]) - stream_start).total_seconds() / 3600.0)
    mem_first_h = None
    if mem_events:
        mem_first_h = float((pd.Timestamp(mem_events[0]) - stream_start).total_seconds() / 3600.0)

    return {
        "trace": trace,
        "vm_id": vm_id,
        "cpu_drift_count": int(len(cpu_events)),
        "mem_drift_count": int(len(mem_events)),
        "cpu_first_drift_offset_h": cpu_first_h,
        "mem_first_drift_offset_h": mem_first_h,
        "cpu_drift_rate_per_day": float(len(cpu_events) / stream_days),
        "mem_drift_rate_per_day": float(len(mem_events) / stream_days),
    }


def _plot_examples(df: pd.DataFrame) -> None:
    score = df["cpu_drift_count"] + df["mem_drift_count"]
    high = df.loc[score.nlargest(3).index]
    low = df.loc[score.nsmallest(3).index]
    pick = pd.concat([high, low], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    axes = axes.flatten()
    for ax, (_, r) in zip(axes, pick.iterrows()):
        trace, vm_id = r["trace"], r["vm_id"]
        resid_path = RESID_DIR / f"{trace}__{vm_id}__cpu.parquet"
        drift_path = OUT_DIR / f"{trace}__{vm_id}__cpu.parquet"
        rdf = pd.read_parquet(resid_path)
        ddf = pd.read_parquet(drift_path)
        idx = pd.to_datetime(rdf.index)
        ax.plot(idx, rdf["residual"].to_numpy(dtype=float), linewidth=0.8)
        for ts in pd.to_datetime(ddf["drift_timestamp"]):
            ax.axvline(ts, color="red", alpha=0.5, linewidth=0.8)
        ax.set_title(f"{trace} {vm_id} (cpu drifts={int(r['cpu_drift_count'])})")
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(KEPT_PATH, encoding="utf-8") as f:
        kept = json.load(f)

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(delayed(_process)(item) for item in kept)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.to_parquet(SUMMARY_PATH, index=False)

    tier_map = _tier_map(kept)
    df["tier"] = [tier_map[(r["trace"], str(r["vm_id"]))] for _, r in df.iterrows()]

    _plot_examples(df)

    def grp_dist(group_col: str, val_col: str) -> dict:
        out = {}
        for k, g in df.groupby(group_col):
            out[str(k)] = _dist(g[val_col])
        return out

    stats = {
        "wall_seconds": elapsed,
        "cpu_drift_count_dist": _dist(df["cpu_drift_count"]),
        "mem_drift_count_dist": _dist(df["mem_drift_count"]),
        "n_zero_cpu_drift": int((df["cpu_drift_count"] == 0).sum()),
        "n_zero_mem_drift": int((df["mem_drift_count"] == 0).sum()),
        "by_tier_cpu_drift_dist": grp_dist("tier", "cpu_drift_count"),
        "by_tier_mem_drift_dist": grp_dist("tier", "mem_drift_count"),
        "by_trace_cpu_drift_dist": grp_dist("trace", "cpu_drift_count"),
        "by_trace_mem_drift_dist": grp_dist("trace", "mem_drift_count"),
        "figure_path": str(FIG_PATH),
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({"summary_path": str(SUMMARY_PATH), "stats_json": str(STATS_JSON), "wall_seconds": elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
