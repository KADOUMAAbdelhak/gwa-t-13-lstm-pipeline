"""Minimal DynAPM policy v2: predictor-based forecast, drift-triggered commits."""
from __future__ import annotations

import pandas as pd

from dynapm.core.drift import ADWINDetector
from dynapm.core.forecasting import forecast_class_from_window, forecast_p95_from_window
from dynapm.methods.baselines_v2 import _BasePolicyV2


class DynAPMPolicyV2(_BasePolicyV2):
    """
    Minimal predictor-based, drift-triggered DynAPM policy.

    Forecast source: predictor's y_pred values over next 288 timesteps (24h),
    classified via bootstrap thresholds.

    Update trigger: ADWIN drift event on residual stream.

    No decision head, no churn controller, no alpha logic.
    Every drift event commits a new forecast.
    """

    def __init__(
        self,
        profile,
        cpu_thresholds,
        mem_thresholds,
        bootstrap_cpu_series,
        bootstrap_mem_series,
        cpu_pred_full,
        mem_pred_full,
        drift_delta: float = 0.002,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(profile, cpu_thresholds, mem_thresholds, **kwargs)
        self._cpu_pred_full = cpu_pred_full
        self._mem_pred_full = mem_pred_full
        self._cpu_drift = ADWINDetector(delta=drift_delta)
        self._mem_drift = ADWINDetector(delta=drift_delta)
        self._alpha = alpha

        self._set_initial_forecast(bootstrap_cpu_series, bootstrap_mem_series)

    def _set_initial_forecast(self, boot_cpu, boot_mem):
        t0 = pd.Timestamp(self.profile.static.bootstrap_end)
        self.profile.dynamic.cpu_class = forecast_class_from_window(boot_cpu, self.cpu_thresholds)
        self.profile.dynamic.memory_class = forecast_class_from_window(boot_mem, self.mem_thresholds)
        self.profile.dynamic.predicted_cpu_p95_next_24h = forecast_p95_from_window(boot_cpu)
        self.profile.dynamic.predicted_mem_p95_next_24h = forecast_p95_from_window(boot_mem)
        self.profile.last_updated["cpu_class"] = t0
        self.profile.last_updated["memory_class"] = t0
        self.profile.last_updated["predicted_cpu_p95_next_24h"] = t0
        self.profile.last_updated["predicted_mem_p95_next_24h"] = t0

    def step(self, t, cpu_obs_window, mem_obs_window, cpu_pred_window, mem_pred_window):
        t = pd.Timestamp(t)

        cpu_resid = float(cpu_obs_window.iloc[-1]) - float(cpu_pred_window.iloc[-1])
        mem_resid = float(mem_obs_window.iloc[-1]) - float(mem_pred_window.iloc[-1])

        cpu_drift = self._cpu_drift.update(cpu_resid)
        mem_drift = self._mem_drift.update(mem_resid)

        if not (cpu_drift or mem_drift):
            return

        pos_cpu = int(self._cpu_pred_full.index.searchsorted(t))
        pos_mem = int(self._mem_pred_full.index.searchsorted(t))
        cpu_lookahead = self._cpu_pred_full.iloc[pos_cpu : pos_cpu + 288]
        mem_lookahead = self._mem_pred_full.iloc[pos_mem : pos_mem + 288]

        if cpu_drift and len(cpu_lookahead) > 0:
            new_class = forecast_class_from_window(cpu_lookahead, self.cpu_thresholds)
            new_p95 = forecast_p95_from_window(cpu_lookahead)
            old_class = self.profile.dynamic.cpu_class
            old_p95 = self.profile.dynamic.predicted_cpu_p95_next_24h
            self.profile.dynamic.cpu_class = new_class
            self.profile.dynamic.predicted_cpu_p95_next_24h = new_p95
            self.profile.last_updated["cpu_class"] = t
            self.profile.last_updated["predicted_cpu_p95_next_24h"] = t
            if old_class != new_class:
                self._update_log.append(
                    {
                        "timestamp": t,
                        "field": "cpu_class",
                        "old_value": old_class,
                        "new_value": new_class,
                        "p_commit": 1.0,
                    }
                )
            self._update_log.append(
                {
                    "timestamp": t,
                    "field": "predicted_cpu_p95_next_24h",
                    "old_value": old_p95,
                    "new_value": new_p95,
                    "p_commit": 1.0,
                }
            )

        if mem_drift and len(mem_lookahead) > 0:
            new_class = forecast_class_from_window(mem_lookahead, self.mem_thresholds)
            new_p95 = forecast_p95_from_window(mem_lookahead)
            old_class = self.profile.dynamic.memory_class
            old_p95 = self.profile.dynamic.predicted_mem_p95_next_24h
            self.profile.dynamic.memory_class = new_class
            self.profile.dynamic.predicted_mem_p95_next_24h = new_p95
            self.profile.last_updated["memory_class"] = t
            self.profile.last_updated["predicted_mem_p95_next_24h"] = t
            if old_class != new_class:
                self._update_log.append(
                    {
                        "timestamp": t,
                        "field": "memory_class",
                        "old_value": old_class,
                        "new_value": new_class,
                        "p_commit": 1.0,
                    }
                )
            self._update_log.append(
                {
                    "timestamp": t,
                    "field": "predicted_mem_p95_next_24h",
                    "old_value": old_p95,
                    "new_value": new_p95,
                    "p_commit": 1.0,
                }
            )
