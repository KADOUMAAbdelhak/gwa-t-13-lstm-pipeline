"""Tests for minimal DynAPM v2 policy."""
from __future__ import annotations

import pandas as pd

from dynapm.core.schema import DynamicFields, ProfileInstance, StaticFields
from dynapm.methods.dynapm_v2 import DynAPMPolicyV2


def _minimal_profile():
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    return ProfileInstance(
        static=StaticFields(
            vm_id="vm",
            trace_name="TestTrace",
            provisioned_cpu_cores=2,
            provisioned_ram_gb=4.0,
            bootstrap_start=t0,
            bootstrap_end=t0,
            n_total_timesteps=500,
        ),
        dynamic=DynamicFields(
            cpu_class="idle",
            memory_class="idle",
            predicted_cpu_p95_next_24h=float("nan"),
            predicted_mem_p95_next_24h=float("nan"),
        ),
        last_updated={
            "cpu_class": t0,
            "memory_class": t0,
            "predicted_cpu_p95_next_24h": t0,
            "predicted_mem_p95_next_24h": t0,
        },
    )


def _thresholds_medium_band():
    return {"p20": 10.0, "p40": 30.0, "p60": 60.0, "p80": 90.0, "p95": 95.0}


def test_dynapm_v2_initializes():
    profile = _minimal_profile()
    th = _thresholds_medium_band()
    idx_boot = pd.date_range("2021-01-01", periods=50, freq="5min")
    boot_cpu = pd.Series([45.0] * 50, index=idx_boot)
    boot_mem = pd.Series([45.0] * 50, index=idx_boot)
    idx_full = pd.date_range("2021-01-02", periods=400, freq="5min")
    pred_cpu = pd.Series([50.0] * 400, index=idx_full)
    pred_mem = pd.Series([50.0] * 400, index=idx_full)

    pol = DynAPMPolicyV2(
        profile,
        th,
        th,
        bootstrap_cpu_series=boot_cpu,
        bootstrap_mem_series=boot_mem,
        cpu_pred_full=pred_cpu,
        mem_pred_full=pred_mem,
        drift_delta=0.002,
        alpha=0.5,
    )
    assert pol.profile.dynamic.cpu_class == "medium"
    assert pol.profile.dynamic.memory_class == "medium"
    assert not pd.isna(pol.profile.dynamic.predicted_cpu_p95_next_24h)
    assert not pd.isna(pol.profile.dynamic.predicted_mem_p95_next_24h)


def test_dynapm_v2_no_drift_no_updates():
    profile = _minimal_profile()
    th = _thresholds_medium_band()
    idx_boot = pd.date_range("2021-01-01", periods=20, freq="5min")
    boot_cpu = pd.Series([20.0] * 20, index=idx_boot)
    boot_mem = pd.Series([20.0] * 20, index=idx_boot)
    idx_full = pd.date_range("2021-01-02", periods=800, freq="5min")
    pred_cpu = pd.Series([10.0] * 800, index=idx_full)
    pred_mem = pd.Series([10.0] * 800, index=idx_full)

    pol = DynAPMPolicyV2(
        profile,
        th,
        th,
        bootstrap_cpu_series=boot_cpu,
        bootstrap_mem_series=boot_mem,
        cpu_pred_full=pred_cpu,
        mem_pred_full=pred_mem,
        drift_delta=0.002,
    )

    for ts in idx_full[:500]:
        obs = pd.Series([10.0], index=[ts])
        pred_w = pd.Series([10.0], index=[ts])
        pol.step(ts, obs, obs, pred_w, pred_w)

    assert len(pol.update_log) == 0


def test_dynapm_v2_drift_commits_update():
    profile = _minimal_profile()
    th = _thresholds_medium_band()
    idx_boot = pd.date_range("2021-01-01", periods=20, freq="5min")
    boot_cpu = pd.Series([20.0] * 20, index=idx_boot)
    boot_mem = pd.Series([20.0] * 20, index=idx_boot)
    idx_full = pd.date_range("2021-01-02", periods=1200, freq="5min")
    # Constant predictions so residuals mirror observation shifts (obs - pred).
    pred_cpu = pd.Series([10.0] * len(idx_full), index=idx_full)
    pred_mem = pred_cpu.copy()

    pol = DynAPMPolicyV2(
        profile,
        th,
        th,
        bootstrap_cpu_series=boot_cpu,
        bootstrap_mem_series=boot_mem,
        cpu_pred_full=pred_cpu,
        mem_pred_full=pred_mem,
        drift_delta=0.002,
    )

    fields = {"cpu_class", "memory_class", "predicted_cpu_p95_next_24h", "predicted_mem_p95_next_24h"}

    for i, ts in enumerate(idx_full):
        if i < 600:
            y = 10.0
        else:
            y = 90.0
        obs = pd.Series([y], index=[ts])
        pc = pd.Series([float(pred_cpu.loc[ts])], index=[ts])
        pm = pd.Series([float(pred_mem.loc[ts])], index=[ts])
        pol.step(ts, obs, obs, pc, pm)

    assert len(pol.update_log) > 0
    assert any(r["field"] in fields for r in pol.update_log)


def test_periodic_predictor_v2_runs():
    from dynapm.eval.runner_v2 import run_method_on_vm
    from dynapm.methods.baselines_v2 import PeriodicPredictorPolicyV2

    out = run_method_on_vm(
        "Materna-Trace-1",
        "501",
        lambda p, c, m, **k: PeriodicPredictorPolicyV2(p, c, m, period_hours=6, **k),
    )
    assert out["n_eval_points_verified"] > 10
    assert out["total_updates"] > 0

