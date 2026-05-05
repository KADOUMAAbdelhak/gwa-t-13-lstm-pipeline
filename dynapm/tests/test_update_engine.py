from __future__ import annotations

import pandas as pd

from dynapm.core.churn import ChurnController
from dynapm.core.decision_head import UpdateDecisionHead
from dynapm.core.schema import DynamicFields, ProfileInstance, StaticFields
from dynapm.core.update_engine import UpdateEngine
from dynapm.eval.runner import run_method_on_vm
from dynapm.methods.baselines import FullRewritePolicy, StaticPolicy
from dynapm.methods.dynapm import dynapm_factory


def test_static_policy_no_updates():
    out = run_method_on_vm("Materna-Trace-1", "501", lambda p, c, m: StaticPolicy(p, c, m))
    assert out["total_updates"] == 0


def test_full_rewrite_many_updates():
    out = run_method_on_vm("Materna-Trace-1", "501", lambda p, c, m: FullRewritePolicy(p, c, m))
    assert out["total_updates"] > 100


def test_dynapm_runs_without_error():
    out = run_method_on_vm("Materna-Trace-1", "501", dynapm_factory(alpha=0.5))
    for k in [
        "cpu_class_accuracy",
        "memory_class_accuracy",
        "cpu_p95_mae",
        "mem_p95_mae",
        "total_updates",
        "n_eval_points",
    ]:
        assert k in out


def test_decision_head_seeds_first_three():
    h = UpdateDecisionHead(alpha=0.5)
    feat = {
        "drift_window_size": 10.0,
        "residual_magnitude": 1.2,
        "hours_since_last_update": 3.0,
        "churn_rate_recent_24h": 1.0,
    }
    r1 = h.predict_commit(feat)
    r2 = h.predict_commit(feat)
    r3 = h.predict_commit(feat)
    assert r1 == (True, 1.0)
    assert r2 == (True, 1.0)
    assert r3 == (True, 1.0)


def test_churn_controller_min_interval():
    c = ChurnController(min_interval_hours=2.0, hysteresis_consecutive=1)
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    assert c.can_update("cpu_class", t0, "low")
    assert not c.can_update("cpu_class", t0 + pd.Timedelta(hours=1), "medium")


def test_churn_controller_hysteresis():
    c = ChurnController(min_interval_hours=0.0, hysteresis_consecutive=3)
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    assert not c.can_update("cpu_class", t0, "medium")
    assert not c.can_update("cpu_class", t0 + pd.Timedelta(minutes=1), "high")
    assert not c.can_update("cpu_class", t0 + pd.Timedelta(minutes=2), "medium")
    assert not c.can_update("cpu_class", t0 + pd.Timedelta(minutes=3), "medium")
    assert c.can_update("cpu_class", t0 + pd.Timedelta(minutes=4), "medium")


def _make_profile() -> ProfileInstance:
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    return ProfileInstance(
        static=StaticFields(
            vm_id="x",
            trace_name="t",
            provisioned_cpu_cores=2,
            provisioned_ram_gb=4.0,
            bootstrap_start=t0,
            bootstrap_end=t0,
            n_total_timesteps=100,
        ),
        dynamic=DynamicFields(
            cpu_class="low",
            memory_class="low",
            predicted_cpu_p95_next_24h=50.0,
            predicted_mem_p95_next_24h=50.0,
        ),
        last_updated={
            "cpu_class": t0,
            "memory_class": t0,
            "predicted_cpu_p95_next_24h": t0,
            "predicted_mem_p95_next_24h": t0,
        },
    )


def test_decision_head_receives_negative_labels():
    profile = _make_profile()
    th = {"p20": 20.0, "p40": 40.0, "p60": 60.0, "p80": 80.0, "p95": 95.0}
    eng = UpdateEngine(
        profile,
        cpu_thresholds=th,
        mem_thresholds=th,
        alpha=0.5,
        drift_delta=1.0,
        min_interval_hours=0.0,
        hysteresis_n=1,
        use_decision_head=True,
        use_churn_control=False,
    )
    # force a committed proposal into pending verification
    feat = {
        "drift_window_size": 1.0,
        "residual_magnitude": 1.0,
        "hours_since_last_update": 1.0,
        "churn_rate_recent_24h": 0.0,
    }
    eng._pending_commits.append(
        {
            "field": "cpu_class",
            "features": feat,
            "old_value": "low",
            "new_value": "burst",
            "commit_t": pd.Timestamp("2021-01-01 00:00:00"),
        }
    )

    calls = []

    def _learn_one(features, label):
        calls.append(bool(label))

    eng._head.learn_one = _learn_one  # type: ignore[method-assign]
    eng.verify_pending_commits(pd.Timestamp("2021-01-01 06:00:00"), {"cpu_class": "low", "memory_class": "low"})
    assert any(lbl is False for lbl in calls)


def test_alpha_sensitivity_after_fix():
    out_low = run_method_on_vm("Materna-Trace-1", "501", dynapm_factory(alpha=0.1))
    out_high = run_method_on_vm("Materna-Trace-1", "501", dynapm_factory(alpha=0.9))
    a = float(out_low["total_updates"])
    b = float(out_high["total_updates"])
    denom = max(a, b, 1.0)
    rel_diff = abs(a - b) / denom
    assert rel_diff >= 0.05
