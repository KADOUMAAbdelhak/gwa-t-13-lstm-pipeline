#!/usr/bin/env python3
"""
Materna corpus characterization (Task 0). Requires: Python 3, numpy, pandas, matplotlib, scipy.
Reads CSVs from three Materna trace directories; writes JSON summaries and PNG plots
under dynapm/reports/task_00/ (relative to project root next to dynapm/ or CWD).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# --- project layout: .../gwa-t-13-lstm-pipeline/dynapm/scripts/thisfile.py
SCRIPT_DIR = Path(__file__).resolve().parent
DYNAPM = SCRIPT_DIR.parent
REPO_ROOT = DYNAPM.parent
OUT_DIR = DYNAPM / "reports" / "task_00"
TRACES = [
    REPO_ROOT / "old" / "data" / "raw" / "Materna-Trace-1",
    REPO_ROOT / "old" / "data" / "raw" / "Materna-Trace-2",
    REPO_ROOT / "old" / "data" / "raw" / "Materna-Trace-3",
]
INFO_NAME = "00-Trace-Info.txt"
SEED = 42
SAMPLE_N = 100
PLOT_VM_N = 6

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


def _parse_float(s: str) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float("nan")
    t = str(s).strip().strip('"')
    return float(t.replace(",", "."))


def read_vm_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        quotechar='"',
        engine="python",
    )
    # normalize column names (strip quotes/whitespace)
    df.columns = [c.strip().strip('"') for c in df.columns]
    ts_col = "Timestamp"
    if ts_col not in df.columns:
        raise ValueError(f"No Timestamp column in {path}")
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True, format="%d.%m.%Y %H:%M:%S")
    for c in df.columns:
        if c == ts_col:
            continue
        if df[c].dtype == object:
            df[c] = df[c].map(_parse_float)
    return df


def list_vm_csvs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.name != INFO_NAME
    )


def directory_inventory() -> list[dict]:
    rows = []
    for root in TRACES:
        if not root.is_dir():
            rows.append({
                "path": str(root),
                "error": "not a directory",
            })
            continue
        all_files = [p for p in root.iterdir() if p.is_file()]
        exts = Counter(p.suffix.lower() or "(no ext)" for p in all_files)
        total_bytes = sum(p.stat().st_size for p in all_files)
        info = (root / INFO_NAME).is_file()
        rows.append({
            "path": str(root.resolve()),
            "n_files": len(all_files),
            "extensions": dict(exts),
            "size_mb": total_bytes / (1024 * 1024),
            "has_trace_info": info,
        })
    return rows


def column_tuple_hash(cols: tuple[str, ...]) -> str:
    """Hash the sorted tuple of column names (spec: distinct structures by sorted names)."""
    h = hashlib.sha256()
    for c in sorted(cols):
        h.update(c.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def stratified_sample_paths(vm_paths_by_trace: list[list[Path]], counts: list[int]) -> list[Path]:
    rng = random.Random(SEED)
    out: list[Path] = []
    for paths, k in zip(vm_paths_by_trace, counts):
        if k <= 0:
            continue
        if k >= len(paths):
            out.extend(paths)
        else:
            out.extend(rng.sample(paths, k))
    return out


def stratified_counts(sizes: list[int], n_total: int) -> list[int]:
    total = sum(sizes)
    base = [int(n_total * s / total) for s in sizes]
    rem = n_total - sum(base)
    order = sorted(range(len(sizes)), key=lambda i: (n_total * sizes[i] / total) - base[i], reverse=True)
    i = 0
    while rem > 0:
        base[order[i % len(sizes)]] += 1
        rem -= 1
        i += 1
    return base


def trace_label(parent: Path) -> int:
    m = {
        "Materna-Trace-1": 1,
        "Materna-Trace-2": 2,
        "Materna-Trace-3": 3,
    }
    return m.get(parent.name, 0)


def sample_plot_vms(all_paths: list[Path]) -> list[tuple[int, Path]]:
    rng = random.Random(SEED)
    n = min(PLOT_VM_N, len(all_paths))
    chosen = rng.sample(all_paths, n)
    return [(trace_label(p.parent), p) for p in chosen]


def main() -> int:
    for r in TRACES:
        if not r.is_dir():
            print(f"Missing trace directory: {r}", file=sys.stderr)
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inv = directory_inventory()
    with open(OUT_DIR / "directory_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

    vm_by_trace = [list_vm_csvs(r) for r in TRACES]
    all_vm_paths = [p for group in vm_by_trace for p in group]

    # Per-file metadata
    col_struct_counts: Counter[str] = Counter()
    col_struct_example: dict[str, tuple[str, ...]] = {}
    row_counts: list[int] = []
    duration_days: list[float] = []
    interval_seconds: list[float] = []
    # Fixed exemplars (one per trace) for verbatim header/row reporting
    triplet = [
        TRACES[0] / "501.csv",
        TRACES[1] / "58.csv",
        TRACES[2] / "238.csv",
    ]
    for p in triplet:
        if not p.is_file():
            print(f"Expected sample file missing: {p}", file=sys.stderr)
            return 1

    for path in all_vm_paths:
        try:
            df = read_vm_csv(path)
        except Exception:
            continue
        cols = tuple(df.columns)
        ch = column_tuple_hash(cols)
        col_struct_counts[ch] += 1
        col_struct_example.setdefault(ch, cols)
        n = len(df)
        row_counts.append(n)
        ts = df["Timestamp"]
        if len(ts) >= 2:
            deltas = ts.diff().dropna().dt.total_seconds().astype(float)
            med_i = float(np.median(deltas))
            interval_seconds.append(med_i)
        else:
            med_i = float("nan")
        t0, t1 = ts.min(), ts.max()
        duration_days.append((t1 - t0).total_seconds() / 86400.0)

    rc = np.array(row_counts, dtype=float)
    dd = np.array(duration_days, dtype=float)
    q = lambda x, p: float(np.quantile(x, p))

    struct_summary = [
        {"hash": h, "count": c, "columns": list(col_struct_example[h])}
        for h, c in col_struct_counts.most_common()
    ]

    counts_100 = stratified_counts([len(x) for x in vm_by_trace], SAMPLE_N)
    sampled_paths = stratified_sample_paths(vm_by_trace, counts_100)

    cpu_col = "CPU usage [%]"
    mem_col = "Memory usage [%]"

    per_vm_stats: list[dict] = []
    nan_vm = 0
    const_cpu = 0
    const_mem = 0
    short_vm = 0

    for path in sampled_paths:
        df = read_vm_csv(path)
        cpu = df[cpu_col].astype(float).to_numpy()
        mem = df[mem_col].astype(float).to_numpy()
        if np.any(np.isnan(cpu)) or np.any(np.isnan(mem)):
            nan_vm += 1
        if np.nanstd(cpu) == 0:
            const_cpu += 1
        if np.nanstd(mem) == 0:
            const_mem += 1
        if len(cpu) < 1000:
            short_vm += 1

        def pct(a, ps):
            return [float(np.nanpercentile(a, p)) for p in ps]

        def skew_arr(a):
            a = a[~np.isnan(a)]
            if len(a) < 3:
                return float("nan")
            return float(scipy_stats.skew(a, bias=False))

        per_vm_stats.append({
            "path": str(path.resolve()),
            "cpu_mean": float(np.nanmean(cpu)),
            "cpu_std": float(np.nanstd(cpu)),
            "cpu_p5": pct(cpu, [5])[0],
            "cpu_p50": pct(cpu, [50])[0],
            "cpu_p95": pct(cpu, [95])[0],
            "cpu_p99": pct(cpu, [99])[0],
            "cpu_skew": skew_arr(cpu),
            "mem_mean": float(np.nanmean(mem)),
            "mem_std": float(np.nanstd(mem)),
            "mem_p5": pct(mem, [5])[0],
            "mem_p50": pct(mem, [50])[0],
            "mem_p95": pct(mem, [95])[0],
            "mem_p99": pct(mem, [99])[0],
            "mem_skew": skew_arr(mem),
        })

    def agg_quantiles(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        return {
            "min": float(np.min(a)),
            "p25": q(a, 0.25),
            "median": q(a, 0.50),
            "p75": q(a, 0.75),
            "max": float(np.max(a)),
        }

    cpu_means = [x["cpu_mean"] for x in per_vm_stats]
    mem_means = [x["mem_mean"] for x in per_vm_stats]
    cpu_stds = [x["cpu_std"] for x in per_vm_stats]
    mem_stds = [x["mem_std"] for x in per_vm_stats]
    cpu_skews = [x["cpu_skew"] for x in per_vm_stats]
    mem_skews = [x["mem_skew"] for x in per_vm_stats]

    sample_summary = {
        "seed": SEED,
        "stratified_counts": dict(zip(["trace1", "trace2", "trace3"], counts_100)),
        "nan_fraction": nan_vm / SAMPLE_N,
        "const_cpu_fraction": const_cpu / SAMPLE_N,
        "const_mem_fraction": const_mem / SAMPLE_N,
        "short_lt_1000_fraction": short_vm / SAMPLE_N,
        "cpu_mean_dist": agg_quantiles(cpu_means),
        "mem_mean_dist": agg_quantiles(mem_means),
        "cpu_std_dist": agg_quantiles(cpu_stds),
        "mem_std_dist": agg_quantiles(mem_stds),
        "cpu_skew_dist": agg_quantiles(cpu_skews),
        "mem_skew_dist": agg_quantiles(mem_skews),
    }

    # Triplet detail for report
    triplet_detail = []
    for p in triplet:
        df = read_vm_csv(p)
        raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        header = raw_lines[0] if raw_lines else ""
        data_lines = [ln for ln in raw_lines[1:] if ln.strip()]
        first3 = data_lines[:3]
        last2 = data_lines[-2:] if len(data_lines) >= 2 else data_lines
        ts = df["Timestamp"]
        deltas = ts.diff().dropna().dt.total_seconds().astype(float)
        med_int = float(np.median(deltas)) if len(deltas) else float("nan")
        triplet_detail.append({
            "path": str(p.resolve()),
            "headers_verbatim": header,
            "first_3_data_rows_verbatim": first3,
            "last_2_data_rows_verbatim": last2,
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp_column": "Timestamp",
            "first_ts": ts.min().isoformat(),
            "last_ts": ts.max().isoformat(),
            "median_interval_seconds": med_int,
        })

    corpus = {
        "total_vm_csv": len(all_vm_paths),
        "row_count_dist": {
            "min": float(np.min(rc)),
            "q25": q(rc, 0.25),
            "median": q(rc, 0.50),
            "q75": q(rc, 0.75),
            "max": float(np.max(rc)),
            "mean": float(np.mean(rc)),
            "std": float(np.std(rc, ddof=1)),
        },
        "distinct_column_structures": len(col_struct_counts),
        "structure_counts": [{"hash": h, "count": col_struct_counts[h]} for h in col_struct_counts],
        "duration_days_dist_from_timestamps": {
            "min": float(np.min(dd)),
            "q25": q(dd, 0.25),
            "median": q(dd, 0.50),
            "q75": q(dd, 0.75),
            "max": float(np.max(dd)),
            "mean": float(np.mean(dd)),
        },
        "note_5min_sampling": "Durations are (last_ts - first_ts) in days. With ~5 min steps, row_count*5min is consistent for spot checks.",
        "median_sampling_intervals_across_files": {
            "min": float(np.nanmin(interval_seconds)) if interval_seconds else None,
            "median_of_medians": float(np.nanmedian(interval_seconds)) if interval_seconds else None,
            "max": float(np.nanmax(interval_seconds)) if interval_seconds else None,
        },
    }

    # Plots: six random VMs (seed 42) from full corpus
    plot_vms = sample_plot_vms(all_vm_paths)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for ax, (ti, pth) in zip(axes, plot_vms):
        df = read_vm_csv(pth)
        ax.plot(df["Timestamp"], df[cpu_col])
        ax.set_title(f"Trace {ti} — {pth.name}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cpu_usage_examples.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for ax, (ti, pth) in zip(axes, plot_vms):
        df = read_vm_csv(pth)
        ax.plot(df["Timestamp"], df[mem_col])
        ax.set_title(f"Trace {ti} — {pth.name}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "memory_usage_examples.png", dpi=120)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(cpu_means, bins=20, edgecolor="black")
    ax1.set_title("CPU mean per VM (sample 100)")
    ax2.hist(mem_means, bins=20, edgecolor="black")
    ax2.set_title("Memory mean per VM (sample 100)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_vm_mean_distribution.png", dpi=120)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(cpu_skews, bins=20, edgecolor="black")
    ax1.set_title("CPU skewness per VM (sample 100)")
    ax2.hist(mem_skews, bins=20, edgecolor="black")
    ax2.set_title("Memory skewness per VM (sample 100)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_vm_skewness_distribution.png", dpi=120)
    plt.close()

    out_blob = {
        "directory_inventory": inv,
        "corpus": corpus,
        "triplet_csv_detail": triplet_detail,
        "per_vm_sample": per_vm_stats,
        "sample_summary": sample_summary,
        "cpu_column_target": cpu_col,
        "mem_column_target": mem_col,
        "column_structure_detail": struct_summary,
    }
    with open(OUT_DIR / "characterization_full.json", "w", encoding="utf-8") as f:
        json.dump(out_blob, f, indent=2)

    print(json.dumps({"ok": True, "out_dir": str(OUT_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
