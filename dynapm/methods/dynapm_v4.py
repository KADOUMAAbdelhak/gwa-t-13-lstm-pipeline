"""DynAPM v4: v3 predictor + drift, with churn-gated logistic decision head."""
from __future__ import annotations

import pandas as pd

from dynapm.core.churn_v2 import ChurnControllerV2
from dynapm.core.decision_head_v2 import DecisionHeadV2
from dynapm.core.drift import ADWINDetector
from dynapm.core.forecasting import forecast_class_from_window, forecast_p95_from_window
from dynapm.methods.baselines_v2 import _BasePolicyV2


class DynAPMPolicyV4(_BasePolicyV2):
    """
    DynAPM v4 = periodic predictor refreshes + optional drift commits gated by churn
    and an online logistic head.

    Periodic path matches v3 (always commits, advances `_last_periodic`).
    Drift commits require churn eligibility and ``predict_commit`` from DecisionHeadV2.

    Pending drift commits obtain retrospective labels via ``verify_pending_commits``.
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
        period_hours: float = 6.0,
        drift_delta: float = 0.002,
        alpha: float = 0.5,
        min_drift_interval_hours: float = 2.0,
    ):
        super().__init__(profile, cpu_thresholds, mem_thresholds)
        self._cpu_pred_full = cpu_pred_full
        self._mem_pred_full = mem_pred_full
        self._period = pd.Timedelta(hours=float(period_hours))
        self._cpu_drift = ADWINDetector(delta=float(drift_delta))
        self._mem_drift = ADWINDetector(delta=float(drift_delta))
        self._last_periodic: pd.Timestamp | None = None
        self._head = DecisionHeadV2(alpha=float(alpha))
        self._churn = ChurnControllerV2(min_drift_interval_hours=float(min_drift_interval_hours))
        self._pending_verify: list = []
        self._set_initial_forecast(bootstrap_cpu_series, bootstrap_mem_series)

    @property
    def decision_head(self) -> DecisionHeadV2:
        return self._head

    def _set_initial_forecast(self, boot_cpu, boot_mem):
        t0 = pd.Timestamp(self.profile.static.bootstrap_end)
        self.profile.dynamic.cpu_class = forecast_class_from_window(boot_cpu, self.cpu_thresholds)
        self.profile.dynamic.memory_class = forecast_class_from_window(boot_mem, self.mem_thresholds)
        self.profile.dynamic.predicted_cpu_p95_next_24h = forecast_p95_from_window(boot_cpu)
        self.profile.dynamic.predicted_mem_p95_next_24h = forecast_p95_from_window(boot_mem)
        for f in ("cpu_class", "memory_class", "predicted_cpu_p95_next_24h", "predicted_mem_p95_next_24h"):
            self.profile.last_updated[f] = t0

    def _refresh_from_predictor(self, t: pd.Timestamp, reason: str) -> None:
        t = pd.Timestamp(t)
        pos_cpu = int(self._cpu_pred_full.index.searchsorted(t))
        pos_mem = int(self._mem_pred_full.index.searchsorted(t))
        cpu_lookahead = self._cpu_pred_full.iloc[pos_cpu : pos_cpu + 288]
        mem_lookahead = self._mem_pred_full.iloc[pos_mem : pos_mem + 288]
        if len(cpu_lookahead) == 0:
            return

        old = (
            self.profile.dynamic.cpu_class,
            self.profile.dynamic.memory_class,
            self.profile.dynamic.predicted_cpu_p95_next_24h,
            self.profile.dynamic.predicted_mem_p95_next_24h,
        )

        new_cpu_cls = forecast_class_from_window(cpu_lookahead, self.cpu_thresholds)
        new_mem_cls = forecast_class_from_window(mem_lookahead, self.mem_thresholds)
        new_cpu_p95 = forecast_p95_from_window(cpu_lookahead)
        new_mem_p95 = forecast_p95_from_window(mem_lookahead)

        self.profile.dynamic.cpu_class = new_cpu_cls
        self.profile.dynamic.memory_class = new_mem_cls
        self.profile.dynamic.predicted_cpu_p95_next_24h = new_cpu_p95
        self.profile.dynamic.predicted_mem_p95_next_24h = new_mem_p95

        for f in ("cpu_class", "memory_class", "predicted_cpu_p95_next_24h", "predicted_mem_p95_next_24h"):
            self.profile.last_updated[f] = t

        meta = {"timestamp": t, "p_commit": 1.0, "reason": reason}
        self._update_log.append({**meta, "field": "cpu_class", "old_value": old[0], "new_value": new_cpu_cls})
        self._update_log.append({**meta, "field": "memory_class", "old_value": old[1], "new_value": new_mem_cls})
        self._update_log.append(
            {**meta, "field": "predicted_cpu_p95_next_24h", "old_value": old[2], "new_value": new_cpu_p95}
        )
        self._update_log.append(
            {**meta, "field": "predicted_mem_p95_next_24h", "old_value": old[3], "new_value": new_mem_p95}
        )

    def _build_drift_features(self, t, recent_resid_cpu_abs, recent_resid_mem_abs) -> dict:
        t = pd.Timestamp(t)
        hours_since_last = 0.0
        if self._last_periodic is not None:
            hours_since_last = float((t - self._last_periodic).total_seconds() / 3600.0)
        pos = int(self._cpu_pred_full.index.searchsorted(t))
        lookahead = self._cpu_pred_full.iloc[pos : pos + 288]
        if len(lookahead) > 0:
            proposed_p95 = float(lookahead.quantile(0.95))
        else:
            proposed_p95 = float(self.profile.dynamic.predicted_cpu_p95_next_24h)
        cur_p95 = float(self.profile.dynamic.predicted_cpu_p95_next_24h)
        if not pd.notna(cur_p95):
            cur_p95 = proposed_p95
        delta_p95 = abs(proposed_p95 - cur_p95)
        return {
            "resid_cpu_abs": float(recent_resid_cpu_abs),
            "resid_mem_abs": float(recent_resid_mem_abs),
            "hours_since_last_periodic": float(hours_since_last),
            "delta_p95": float(delta_p95),
        }

    def step(self, t, cpu_obs_window, mem_obs_window, cpu_pred_window, mem_pred_window):
        t = pd.Timestamp(t)

        cpu_resid = float(cpu_obs_window.iloc[-1]) - float(cpu_pred_window.iloc[-1])
        mem_resid = float(mem_obs_window.iloc[-1]) - float(mem_pred_window.iloc[-1])
        cpu_drift = self._cpu_drift.update(cpu_resid)
        mem_drift = self._mem_drift.update(mem_resid)
        drift_event = cpu_drift or mem_drift

        periodic_due = self._last_periodic is None or (t - self._last_periodic) >= self._period
        if periodic_due:
            self._refresh_from_predictor(t, reason="periodic")
            self._last_periodic = t
            return

        if drift_event:
            if not self._churn.can_commit_drift(t):
                return
            rc = cpu_obs_window.sub(cpu_pred_window, fill_value=0).abs().iloc[-24:].mean()
            rm = mem_obs_window.sub(mem_pred_window, fill_value=0).abs().iloc[-24:].mean()
            recent_cpu = float(rc) if pd.notna(rc) else 0.0
            recent_mem = float(rm) if pd.notna(rm) else 0.0
            features = self._build_drift_features(t, recent_cpu, recent_mem)
            commit, _p = self._head.predict_commit(features)
            if not commit:
                return
            old_cpu_cls = self.profile.dynamic.cpu_class
            old_mem_cls = self.profile.dynamic.memory_class
            self._refresh_from_predictor(t, reason="drift")
            self._churn.record_drift_commit(t)
            self._pending_verify.append(
                {
                    "t_commit": t,
                    "features": features,
                    "old_cpu_cls": old_cpu_cls,
                    "old_mem_cls": old_mem_cls,
                    "new_cpu_cls": self.profile.dynamic.cpu_class,
                    "new_mem_cls": self.profile.dynamic.memory_class,
                }
            )

    def verify_pending_commits(self, t: pd.Timestamp, ground_truth: dict) -> None:
        t = pd.Timestamp(t)
        remaining = []
        for entry in self._pending_verify:
            tc = pd.Timestamp(entry["t_commit"])
            if tc + pd.Timedelta(hours=24) > t:
                remaining.append(entry)
                continue
            gt_cpu = ground_truth.get("cpu_class")
            gt_mem = ground_truth.get("memory_class")

            cpu_better = (entry["new_cpu_cls"] == gt_cpu) and (entry["old_cpu_cls"] != gt_cpu)
            mem_better = (entry["new_mem_cls"] == gt_mem) and (entry["old_mem_cls"] != gt_mem)

            cpu_worse = (entry["new_cpu_cls"] != gt_cpu) and (entry["old_cpu_cls"] == gt_cpu)
            mem_worse = (entry["new_mem_cls"] != gt_mem) and (entry["old_mem_cls"] == gt_mem)

            label = bool(cpu_better or mem_better) and not (cpu_worse or mem_worse)
            self._head.learn_one(entry["features"], label)
        self._pending_verify = remaining
