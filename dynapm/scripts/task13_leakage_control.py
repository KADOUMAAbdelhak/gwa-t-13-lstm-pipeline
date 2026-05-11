#!/usr/bin/env python3
"""Task 13: null-predictor 100-VM control + audit diagnostics."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from joblib import Parallel, delayed

from dynapm.eval.runner_v2 import run_method_on_vm  # noqa: E402
from dynapm.methods.baselines_v3_audit import PeriodicNullPredictorPolicyV2  # noqa: E402

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


def main() -> int:
    sample_path = REPO / "dynapm/data/eval_sample_v2_summary.parquet"
    pp_path = REPO / "dynapm/data/eval_sample_v2_periodic_predictor_summary.parquet"
    out_parquet = REPO / "dynapm/data/eval_sample_v2_periodic_null_predictor_summary.parquet"
    out_json_dir = REPO / "dynapm/results/eval_sample_v2"
    diag_path = REPO / "dynapm/data/task13_leakage_audit.json"

    base = pd.read_parquet(sample_path)
    sample = base[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample["vm_id"] = sample["vm_id"].astype(str)
    assert len(sample) == 100

    def run_one(trace: str, vm_id: str, tier: str) -> dict:
        out = run_method_on_vm(
            trace,
            vm_id,
            lambda p, c, m, **k: PeriodicNullPredictorPolicyV2(p, c, m, period_hours=6.0, **k),
        )
        fn = out_json_dir / f"periodic_null_pred_6h_v2__{trace}__{vm_id}.json"
        fn.write_text(json.dumps(out, default=str), encoding="utf-8")
        row = {k: out[k] for k in COLS if k in out}
        row["method"] = "periodic_null_pred_6h_v2"
        row["trace"] = trace
        row["vm_id"] = str(vm_id)
        row["volatility_tier"] = tier
        return row

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=8, backend="loky")(
        delayed(run_one)(str(r.trace), str(r.vm_id), str(r.volatility_tier)) for r in sample.itertuples(index=False)
    )
    wall = time.perf_counter() - t0

    null_df = pd.DataFrame(rows)
    null_df.to_parquet(out_parquet, index=False)

    per = base[base["method"] == "periodic_6h_v2"]
    st = base[base["method"] == "static_v2"]
    pp = pd.read_parquet(pp_path)

    m_null = float(null_df["cpu_class_accuracy"].mean())
    m_per = float(per["cpu_class_accuracy"].mean())
    m_st = float(st["cpu_class_accuracy"].mean())
    m_pp = float(pp["cpu_class_accuracy"].mean())

    u_null = float(null_df["total_updates"].mean())
    u_per = float(per["total_updates"].mean())
    u_pp = float(pp["total_updates"].mean())
    u_st = float(st["total_updates"].mean())

    null_vs_static = abs(m_null - m_st)

    audit_1_4 = True
    if null_vs_static <= 0.05 and m_null < m_per and m_null < m_pp:
        verdict = "AUDIT_PASS"
    elif m_null < m_pp and (m_null > m_st or null_vs_static > 0.05) and m_per <= m_null < m_pp:
        verdict = "AUDIT_PARTIAL_PASS"
    elif m_null >= m_pp - 1e-9:
        verdict = "AUDIT_FAIL"
    elif null_vs_static > 0.05 and m_null < m_per:
        verdict = "AUDIT_PARTIAL_PASS"
    else:
        verdict = "AUDIT_PARTIAL_PASS"

    if m_null >= m_pp:
        verdict = "AUDIT_FAIL"

    out = {
        "wall_seconds_null_run": float(wall),
        "mean_cpu_acc": {
            "static_v2": m_st,
            "periodic_6h_v2": m_per,
            "periodic_predictor_6h_v2": m_pp,
            "periodic_null_pred_6h_v2": m_null,
        },
        "mean_total_updates": {
            "static_v2": u_st,
            "periodic_6h_v2": u_per,
            "periodic_predictor_6h_v2": u_pp,
            "periodic_null_pred_6h_v2": u_null,
        },
        "null_minus_static_cpu_acc_abs": float(null_vs_static),
        "verdict_automated": verdict,
    }
    diag_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
