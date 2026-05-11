"""Audit-only baselines (leakage / control experiments). Not part of production baselines_v2."""
from __future__ import annotations

import pandas as pd

from dynapm.core.forecasting import forecast_class_from_window, forecast_p95_from_window
from dynapm.methods.baselines_v2 import _BasePolicyV2, _window_last_hours


class PeriodicNullPredictorPolicyV2(_BasePolicyV2):
    """
    Audit baseline: same 6h refresh cadence as PeriodicPredictorPolicyV2, but the
    lookahead is a constant series equal to the bootstrap mean of each metric.
    Ignores real ``cpu_pred_full`` / ``mem_pred_full``.
    """

    def __init__(self, profile, cpu_thresholds, mem_thresholds, **kwargs):
        super().__init__(profile, cpu_thresholds, mem_thresholds, **kwargs)
        self._period_hours = float(kwargs["period_hours"])
        self._period = pd.Timedelta(hours=self._period_hours)
        cpu_boot = kwargs["bootstrap_cpu_series"]
        mem_boot = kwargs["bootstrap_mem_series"]
        t0 = pd.Timestamp(cpu_boot.index[-1])
        cpu_init = _window_last_hours(cpu_boot, t0, self._period_hours)
        mem_init = _window_last_hours(mem_boot, t0, self._period_hours)
        self._set_initial_forecast(cpu_init, mem_init, t=t0)
        self._last = t0
        self._cpu_const = float(cpu_boot.astype(float).mean())
        self._mem_const = float(mem_boot.astype(float).mean())

    def step(self, t, cpu_obs_window, mem_obs_window, cpu_pred_window, mem_pred_window):
        t = pd.Timestamp(t)
        if (t - self._last) < self._period:
            return

        lookahead_cpu = pd.Series([self._cpu_const] * 288)
        lookahead_mem = pd.Series([self._mem_const] * 288)

        old_cpu_cls = self.profile.dynamic.cpu_class
        old_mem_cls = self.profile.dynamic.memory_class
        old_cpu_p95 = self.profile.dynamic.predicted_cpu_p95_next_24h
        old_mem_p95 = self.profile.dynamic.predicted_mem_p95_next_24h

        new_cpu_cls = forecast_class_from_window(lookahead_cpu, self.cpu_thresholds)
        new_mem_cls = forecast_class_from_window(lookahead_mem, self.mem_thresholds)
        new_cpu_p95 = forecast_p95_from_window(lookahead_cpu)
        new_mem_p95 = forecast_p95_from_window(lookahead_mem)

        self.profile.dynamic.cpu_class = new_cpu_cls
        self.profile.dynamic.memory_class = new_mem_cls
        self.profile.dynamic.predicted_cpu_p95_next_24h = new_cpu_p95
        self.profile.dynamic.predicted_mem_p95_next_24h = new_mem_p95

        self.profile.last_updated["cpu_class"] = t
        self.profile.last_updated["memory_class"] = t
        self.profile.last_updated["predicted_cpu_p95_next_24h"] = t
        self.profile.last_updated["predicted_mem_p95_next_24h"] = t

        self._last = t

        if old_cpu_cls != new_cpu_cls:
            self._update_log.append(
                {
                    "timestamp": t,
                    "field": "cpu_class",
                    "old_value": old_cpu_cls,
                    "new_value": new_cpu_cls,
                    "p_commit": 1.0,
                }
            )
        if old_mem_cls != new_mem_cls:
            self._update_log.append(
                {
                    "timestamp": t,
                    "field": "memory_class",
                    "old_value": old_mem_cls,
                    "new_value": new_mem_cls,
                    "p_commit": 1.0,
                }
            )
        self._update_log.append(
            {
                "timestamp": t,
                "field": "predicted_cpu_p95_next_24h",
                "old_value": old_cpu_p95,
                "new_value": new_cpu_p95,
                "p_commit": 1.0,
            }
        )
        self._update_log.append(
            {
                "timestamp": t,
                "field": "predicted_mem_p95_next_24h",
                "old_value": old_mem_p95,
                "new_value": new_mem_p95,
                "p_commit": 1.0,
            }
        )
