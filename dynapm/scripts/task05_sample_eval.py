#!/usr/bin/env python3
"""Task 5B: 100-VM sample evaluation across baselines and DynAPM variants."""
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

from dynapm.eval.runner import run_method_on_vm
from dynapm.methods.baselines import FullRewritePolicy, PeriodicPolicy, ReactiveThresholdPolicy, StaticPolicy
from dynapm.methods.dynapm import dynapm_factory, dynapm_no_churn_factory, dynapm_no_learner_factory
from dynapm.data.bootstrap import compute_split
from dynapm.data.loader import load_vm_csv

OUT_DIR = REPO / "dynapm/results/eval_sample"
REPORT_DIR = REPO / "dynapm/reports/task_05"
SUMMARY_PATH = REPO / "dynapm/data/eval_sample_summary.parquet"
DIAG_JSON = REPO / "dynapm/data/eval_sample_diagnostics.json"


def _load_kept():
    with open(REPO / "dynapm/data/kept_vms.json", encoding="utf-8") as f:
        return json.load(f)


def _tier_map(kept: list[dict]) -> dict[tuple[str, str], str]:
    rows = []
    for item in kept:
        p = REPO / "data/raw" / item["trace"] / f"{item['vm_id']}.csv"
        df = load_vm_csv(str(p))
        b, _, _ = compute_split(df)
        rows.append({"trace": item["trace"], "vm_id": item["vm_id"], "std": float(b["CPU usage [%]"].std(ddof=0))})
    s = pd.DataFrame(rows)
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
        out[(r["trace"], r["vm_id"])] = t
    return out


def _sample_100(kept: list[dict], tier_map: dict, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    targets = {"Materna-Trace-1": 33, "Materna-Trace-2": 33, "Materna-Trace-3": 34}
    out = []
    for tr, n in targets.items():
        pool = [x for x in kept if x["trace"] == tr]
        by_t = {k: [x for x in pool if tier_map[(x["trace"], x["vm_id"])] == k] for k in ("low", "medium", "high")}
        k_low = n // 3
        k_med = n // 3
        k_high = n - k_low - k_med
        chosen = []
        for k, kk in [("low", k_low), ("medium", k_med), ("high", k_high)]:
            arr = by_t[k]
            idx = rng.choice(len(arr), size=min(kk, len(arr)), replace=False)
            chosen.extend([arr[int(i)] for i in idx])
        if len(chosen) < n:
            used = {(x["trace"], x["vm_id"]) for x in chosen}
            rest = [x for x in pool if (x["trace"], x["vm_id"]) not in used]
            idx = rng.choice(len(rest), size=n - len(chosen), replace=False)
            chosen.extend([rest[int(i)] for i in idx])
        out.extend(chosen[:n])
    return out


def _factories():
    return {
        "static": lambda p, c, m: StaticPolicy(p, c, m),
        "periodic_1h": lambda p, c, m: PeriodicPolicy(p, c, m, period_hours=1.0),
        "periodic_6h": lambda p, c, m: PeriodicPolicy(p, c, m, period_hours=6.0),
        "periodic_24h": lambda p, c, m: PeriodicPolicy(p, c, m, period_hours=24.0),
        "reactive_threshold": lambda p, c, m: ReactiveThresholdPolicy(p, c, m),
        "full_rewrite": lambda p, c, m: FullRewritePolicy(p, c, m),
        "dynapm_alpha_0.1": dynapm_factory(alpha=0.1),
        "dynapm_alpha_0.3": dynapm_factory(alpha=0.3),
        "dynapm_alpha_0.5": dynapm_factory(alpha=0.5),
        "dynapm_alpha_0.7": dynapm_factory(alpha=0.7),
        "dynapm_alpha_0.9": dynapm_factory(alpha=0.9),
        "dynapm_no_learner_alpha_0.5": dynapm_no_learner_factory(alpha=0.5),
        "dynapm_no_churn_alpha_0.5": dynapm_no_churn_factory(alpha=0.5),
    }


def _run_one(method: str, factory, item: dict, tier: str):
    trace, vm = item["trace"], item["vm_id"]
    res = run_method_on_vm(trace, vm, factory)
    res["method"] = method
    res["volatility_tier"] = tier
    out_file = OUT_DIR / f"{method}__{trace}__{vm}.json"
    out_file.write_text(json.dumps(res, default=str), encoding="utf-8")
    row = {
        "method": method,
        "trace": trace,
        "vm_id": vm,
        "volatility_tier": tier,
        "cpu_class_accuracy": res["cpu_class_accuracy"],
        "memory_class_accuracy": res["memory_class_accuracy"],
        "cpu_p95_mae": res["cpu_p95_mae"],
        "mem_p95_mae": res["mem_p95_mae"],
        "n_cpu_class_updates": res["n_cpu_class_updates"],
        "n_memory_class_updates": res["n_memory_class_updates"],
        "n_cpu_p95_updates": res["n_cpu_p95_updates"],
        "n_mem_p95_updates": res["n_mem_p95_updates"],
        "total_updates": res["total_updates"],
        "n_eval_points": res["n_eval_points"],
        "eval_density": res["eval_density"],
        "skipped_eval_points": res["skipped_eval_points"],
    }
    return row


def _agg(df: pd.DataFrame, metric: str):
    g = df.groupby("method")[metric]
    return pd.DataFrame({"mean": g.mean(), "median": g.median()}).reset_index()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    kept = _load_kept()
    tmap = _tier_map(kept)
    sample = _sample_100(kept, tmap, seed=42)

    fac = _factories()
    jobs = []
    for method, f in fac.items():
        for item in sample:
            jobs.append((method, f, item, tmap[(item["trace"], item["vm_id"])]))

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(
        delayed(_run_one)(m, f, i, tier) for (m, f, i, tier) in jobs
    )
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.to_parquet(SUMMARY_PATH, index=False)

    # diagnostics tables
    out = {
        "wall_seconds": elapsed,
        "cpu_class_accuracy": _agg(df, "cpu_class_accuracy").to_dict(orient="records"),
        "memory_class_accuracy": _agg(df, "memory_class_accuracy").to_dict(orient="records"),
        "total_updates": _agg(df, "total_updates").to_dict(orient="records"),
        "cpu_p95_mae": _agg(df, "cpu_p95_mae").to_dict(orient="records"),
        "mem_p95_mae": _agg(df, "mem_p95_mae").to_dict(orient="records"),
    }

    # per-tier breakdown
    tier_rows = []
    for (method, tier), g in df.groupby(["method", "volatility_tier"]):
        tier_rows.append(
            {
                "method": method,
                "volatility_tier": tier,
                "cpu_class_accuracy_mean": float(g["cpu_class_accuracy"].mean()),
                "memory_class_accuracy_mean": float(g["memory_class_accuracy"].mean()),
                "total_updates_mean": float(g["total_updates"].mean()),
                "cpu_p95_mae_mean": float(g["cpu_p95_mae"].mean()),
                "mem_p95_mae_mean": float(g["mem_p95_mae"].mean()),
            }
        )
    out["per_tier"] = tier_rows

    # alpha pareto plot
    alpha_methods = [m for m in fac.keys() if m.startswith("dynapm_alpha_")]
    pareto = df[df["method"].isin(alpha_methods + ["static", "full_rewrite"])].groupby("method", as_index=False).agg(
        cpu_acc=("cpu_class_accuracy", "mean"),
        updates=("total_updates", "mean"),
    )
    fig = plt.figure(figsize=(7, 5))
    for _, r in pareto.iterrows():
        plt.scatter(r["updates"], r["cpu_acc"], label=r["method"])
    plt.xlabel("mean total updates")
    plt.ylabel("mean cpu_class_accuracy")
    plt.title("Alpha Pareto")
    plt.legend(fontsize=7)
    plt.tight_layout()
    pareto_path = REPORT_DIR / "alpha_pareto.png"
    plt.savefig(pareto_path, dpi=120)
    plt.close(fig)

    # dynapm vs static delta histogram
    a05 = df[df["method"] == "dynapm_alpha_0.5"].set_index(["trace", "vm_id"])
    st = df[df["method"] == "static"].set_index(["trace", "vm_id"])
    common = a05.index.intersection(st.index)
    delta = (a05.loc[common, "cpu_class_accuracy"] - st.loc[common, "cpu_class_accuracy"]).to_numpy(dtype=float)
    fig = plt.figure(figsize=(7, 4))
    plt.hist(delta, bins=30)
    plt.axvline(0.0, color="black")
    plt.title("DynAPM alpha=0.5 vs Static (CPU accuracy delta)")
    plt.tight_layout()
    delta_path = REPORT_DIR / "dynapm_vs_static_delta.png"
    plt.savefig(delta_path, dpi=120)
    plt.close(fig)

    out["figure_alpha_pareto"] = str(pareto_path)
    out["figure_dynapm_vs_static_delta"] = str(delta_path)

    DIAG_JSON.parent.mkdir(parents=True, exist_ok=True)
    DIAG_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "summary_parquet": str(SUMMARY_PATH),
        "diag_json": str(DIAG_JSON),
        "wall_seconds": elapsed,
        "n_rows": len(df),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
