"""Tests for DynAPMPolicyV4."""
from __future__ import annotations

import pandas as pd

from dynapm.core.schema import DynamicFields, ProfileInstance, StaticFields
from dynapm.methods.dynapm_v4 import DynAPMPolicyV4


def _profile():
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    return ProfileInstance(
        static=StaticFields(
            vm_id="vm",
            trace_name="T",
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


def _th():
    return {"p20": 10.0, "p40": 30.0, "p60": 60.0, "p80": 90.0, "p95": 95.0}


def test_v4_initializes():
    p = _profile()
    th = _th()
    idx_boot = pd.date_range("2021-01-01", periods=50, freq="5min")
    boot_cpu = pd.Series([40.0] * 50, index=idx_boot)
    boot_mem = boot_cpu.copy()
    idx_full = pd.date_range("2021-01-02", periods=400, freq="5min")
    pred = pd.Series([40.0] * len(idx_full), index=idx_full)
    pol = DynAPMPolicyV4(p, th, th, boot_cpu, boot_mem, pred, pred, alpha=0.5)
    assert pol.profile.dynamic.cpu_class == "medium"
    assert pol.decision_head.stats()["n_seen"] == 0


def test_v4_periodic_only_no_drift_no_head_calls():
    p = _profile()
    th = _th()
    idx_boot = pd.date_range("2021-01-01", periods=20, freq="5min")
    boot_cpu = pd.Series([25.0] * 20, index=idx_boot)
    boot_mem = boot_cpu.copy()
    idx_full = pd.date_range("2021-01-02", periods=800, freq="5min")
    pred = pd.Series([25.0] * len(idx_full), index=idx_full)
    pol = DynAPMPolicyV4(p, th, th, boot_cpu, boot_mem, pred, pred, period_hours=1.0, drift_delta=0.002)
    for ts in idx_full:
        obs = pd.Series([25.0], index=[ts])
        pr = pd.Series([25.0], index=[ts])
        pol.step(ts, obs, obs, pr, pr)
    assert pol.decision_head.stats()["n_seen"] == 0


def test_v4_drift_event_triggers_head():
    p = _profile()
    th = _th()
    idx_boot = pd.date_range("2021-01-01", periods=30, freq="5min")
    boot_cpu = pd.Series([20.0] * 30, index=idx_boot)
    boot_mem = boot_cpu.copy()
    idx_full = pd.date_range("2021-01-02", periods=500, freq="5min")
    pred = pd.Series([10.0] * len(idx_full), index=idx_full)
    pol = DynAPMPolicyV4(
        p,
        th,
        th,
        boot_cpu,
        boot_mem,
        pred,
        pred,
        period_hours=10_000.0,
        drift_delta=0.002,
        alpha=0.0,
        min_drift_interval_hours=0.0,
    )
    for i, ts in enumerate(idx_full):
        y_obs = 10.0 if i < 250 else 90.0
        obs = pd.Series([y_obs], index=[ts])
        pc = pd.Series([10.0], index=[ts])
        pol.step(ts, obs, obs, pc, pc)
    assert pol.decision_head.stats()["n_seen"] >= 1


def test_v4_alpha_zero_commits_everything():
    p = _profile()
    th = _th()
    idx_boot = pd.date_range("2021-01-01", periods=25, freq="5min")
    boot_cpu = pd.Series([25.0] * 25, index=idx_boot)
    boot_mem = boot_cpu.copy()
    idx_full = pd.date_range("2021-01-02", periods=320, freq="5min")
    pred = pd.Series([25.0] * len(idx_full), index=idx_full)
    pol = DynAPMPolicyV4(
        p,
        th,
        th,
        boot_cpu,
        boot_mem,
        pred,
        pred,
        period_hours=10_000.0,
        drift_delta=0.002,
        alpha=0.0,
        min_drift_interval_hours=0.0,
    )
    for i, ts in enumerate(idx_full):
        if i >= 5:
            # Force repeated drift-flag timesteps independent of residual path.
            pol._cpu_drift.update = lambda r: True
            pol._mem_drift.update = lambda r: False
        obs = pd.Series([25.0], index=[ts])
        pr = pd.Series([25.0], index=[ts])
        pol.step(ts, obs, obs, pr, pr)
    st = pol.decision_head.stats()
    assert st["n_seen"] >= 25
    assert st["commit_rate"] > 0.9


def test_v4_alpha_one_commits_seed_only():
    p = _profile()
    th = _th()
    idx_boot = pd.date_range("2021-01-01", periods=25, freq="5min")
    boot_cpu = pd.Series([25.0] * 25, index=idx_boot)
    boot_mem = boot_cpu.copy()
    idx_full = pd.date_range("2021-01-02", periods=360, freq="5min")
    pred = pd.Series([25.0] * len(idx_full), index=idx_full)
    pol = DynAPMPolicyV4(
        p,
        th,
        th,
        boot_cpu,
        boot_mem,
        pred,
        pred,
        period_hours=10_000.0,
        drift_delta=0.002,
        alpha=1.0,
        min_drift_interval_hours=0.0,
    )
    for i, ts in enumerate(idx_full):
        if i >= 5:
            pol._cpu_drift.update = lambda r: True
            pol._mem_drift.update = lambda r: False
        obs = pd.Series([25.0], index=[ts])
        pr = pd.Series([25.0], index=[ts])
        pol.step(ts, obs, obs, pr, pr)
    st = pol.decision_head.stats()
    assert st["n_seen"] >= 40
    assert st["n_committed"] <= 5
