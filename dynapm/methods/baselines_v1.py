"""Baseline update policies with UpdateEngine-compatible API."""
from __future__ import annotations

import pandas as pd

from dynapm.core.discretization import classify_value


class _BasePolicy:
    def __init__(self, profile, cpu_thresholds, mem_thresholds):
        self.profile = profile
        self.cpu_thresholds = cpu_thresholds
        self.mem_thresholds = mem_thresholds
        self._update_log = []

    @property
    def update_log(self):
        return self._update_log


class StaticPolicy(_BasePolicy):
    """Profile is set at bootstrap end; never updated during streaming."""

    def step(self, t, cpu_residual, mem_residual, cpu_obs, mem_obs, cpu_pred, mem_pred):
        return

    def step_continuous(self, t, p95_cpu_pred, p95_mem_pred):
        return


class PeriodicPolicy(_BasePolicy):
    """Refresh all dynamic fields every N hours."""

    def __init__(self, profile, cpu_thresholds, mem_thresholds, period_hours: float):
        super().__init__(profile, cpu_thresholds, mem_thresholds)
        self._period = pd.Timedelta(hours=period_hours)
        self._last = None

    def step(self, t, cpu_residual, mem_residual, cpu_obs, mem_obs, cpu_pred, mem_pred):
        t = pd.Timestamp(t)
        if self._last is not None and (t - self._last) < self._period:
            return
        old_cpu = self.profile.dynamic.cpu_class
        old_mem = self.profile.dynamic.memory_class
        self.profile.dynamic.cpu_class = classify_value(float(cpu_obs), self.cpu_thresholds)
        self.profile.dynamic.memory_class = classify_value(float(mem_obs), self.mem_thresholds)
        self.profile.last_updated["cpu_class"] = t
        self.profile.last_updated["memory_class"] = t
        self._last = t
        if old_cpu != self.profile.dynamic.cpu_class:
            self._update_log.append({"timestamp": t, "field": "cpu_class", "old_value": old_cpu, "new_value": self.profile.dynamic.cpu_class, "p_commit": 1.0})
        if old_mem != self.profile.dynamic.memory_class:
            self._update_log.append({"timestamp": t, "field": "memory_class", "old_value": old_mem, "new_value": self.profile.dynamic.memory_class, "p_commit": 1.0})

    def step_continuous(self, t, p95_cpu_pred, p95_mem_pred):
        t = pd.Timestamp(t)
        if self._last is None or (t - self._last) >= self._period:
            for field, val in [("predicted_cpu_p95_next_24h", float(p95_cpu_pred)), ("predicted_mem_p95_next_24h", float(p95_mem_pred))]:
                old = getattr(self.profile.dynamic, field)
                setattr(self.profile.dynamic, field, val)
                self.profile.last_updated[field] = t
                self._update_log.append({"timestamp": t, "field": field, "old_value": old, "new_value": val, "p_commit": 1.0})


class ReactiveThresholdPolicy(_BasePolicy):
    """Update when observed metric crosses fixed bootstrap-p80 boundary."""

    def step(self, t, cpu_residual, mem_residual, cpu_obs, mem_obs, cpu_pred, mem_pred):
        t = pd.Timestamp(t)
        # Reactive policy should classify each metric using its own frozen thresholds.
        cpu_new = classify_value(float(cpu_obs), self.cpu_thresholds)
        mem_new = classify_value(float(mem_obs), self.mem_thresholds)
        if cpu_new != self.profile.dynamic.cpu_class:
            old = self.profile.dynamic.cpu_class
            self.profile.dynamic.cpu_class = cpu_new
            self.profile.last_updated["cpu_class"] = t
            self._update_log.append({"timestamp": t, "field": "cpu_class", "old_value": old, "new_value": cpu_new, "p_commit": 1.0})
        if mem_new != self.profile.dynamic.memory_class:
            old = self.profile.dynamic.memory_class
            self.profile.dynamic.memory_class = mem_new
            self.profile.last_updated["memory_class"] = t
            self._update_log.append({"timestamp": t, "field": "memory_class", "old_value": old, "new_value": mem_new, "p_commit": 1.0})

    def step_continuous(self, t, p95_cpu_pred, p95_mem_pred):
        t = pd.Timestamp(t)
        for field, val in [("predicted_cpu_p95_next_24h", float(p95_cpu_pred)), ("predicted_mem_p95_next_24h", float(p95_mem_pred))]:
            old = getattr(self.profile.dynamic, field)
            setattr(self.profile.dynamic, field, val)
            self.profile.last_updated[field] = t
            self._update_log.append({"timestamp": t, "field": field, "old_value": old, "new_value": val, "p_commit": 1.0})


class FullRewritePolicy(_BasePolicy):
    """Update every step (worst-case churn baseline)."""

    def step(self, t, cpu_residual, mem_residual, cpu_obs, mem_obs, cpu_pred, mem_pred):
        t = pd.Timestamp(t)
        new_cpu = classify_value(float(cpu_obs), self.cpu_thresholds)
        new_mem = classify_value(float(mem_obs), self.mem_thresholds)
        for field, val in [("cpu_class", new_cpu), ("memory_class", new_mem)]:
            old = getattr(self.profile.dynamic, field)
            if old != val:
                setattr(self.profile.dynamic, field, val)
                self.profile.last_updated[field] = t
                self._update_log.append({"timestamp": t, "field": field, "old_value": old, "new_value": val, "p_commit": 1.0})

    def step_continuous(self, t, p95_cpu_pred, p95_mem_pred):
        t = pd.Timestamp(t)
        for field, val in [("predicted_cpu_p95_next_24h", float(p95_cpu_pred)), ("predicted_mem_p95_next_24h", float(p95_mem_pred))]:
            old = getattr(self.profile.dynamic, field)
            setattr(self.profile.dynamic, field, val)
            self.profile.last_updated[field] = t
            self._update_log.append({"timestamp": t, "field": field, "old_value": old, "new_value": val, "p_commit": 1.0})
