"""Update engine: drift detector + decision head + churn controller."""
from __future__ import annotations

import pandas as pd

from dynapm.core.churn import ChurnController
from dynapm.core.decision_head import UpdateDecisionHead
from dynapm.core.discretization import classify_value
from dynapm.core.drift import ADWINDetector
from dynapm.core.schema import ProfileInstance


class UpdateEngine:
    """Orchestrates profile updates from streaming residual/observation events."""

    def __init__(
        self,
        profile: ProfileInstance,
        cpu_thresholds: dict,
        mem_thresholds: dict,
        alpha: float = 0.5,
        drift_delta: float = 0.002,
        min_interval_hours: float = 2.0,
        hysteresis_n: int = 3,
        use_decision_head: bool = True,
        use_churn_control: bool = True,
    ):
        self.profile = profile
        self.cpu_thresholds = cpu_thresholds
        self.mem_thresholds = mem_thresholds
        self._cpu_drift = ADWINDetector(delta=drift_delta)
        self._mem_drift = ADWINDetector(delta=drift_delta)
        self._head = UpdateDecisionHead(alpha=alpha) if use_decision_head else None
        self._churn = ChurnController(min_interval_hours, hysteresis_n) if use_churn_control else None
        self._update_log = []
        self._pending_uncommitted = []
        self._pending_commits = []

    def step(
        self,
        t: pd.Timestamp,
        cpu_residual: float,
        mem_residual: float,
        cpu_obs: float,
        mem_obs: float,
        cpu_pred: float,
        mem_pred: float,
    ) -> None:
        cpu_drift = self._cpu_drift.update(cpu_residual)
        mem_drift = self._mem_drift.update(mem_residual)
        if cpu_drift:
            self._consider_update(pd.Timestamp(t), "cpu_class", cpu_obs, cpu_residual)
        if mem_drift:
            self._consider_update(pd.Timestamp(t), "memory_class", mem_obs, mem_residual)

    def _consider_update(self, t: pd.Timestamp, field: str, observed: float, residual: float):
        proposed = classify_value(float(observed), self.cpu_thresholds if field == "cpu_class" else self.mem_thresholds)
        current = getattr(self.profile.dynamic, field)
        if proposed == current:
            return

        features = {
            "drift_window_size": float(self._adwin_window_size(field)),
            "residual_magnitude": float(abs(residual)),
            "hours_since_last_update": self._hours_since(field, t),
            "churn_rate_recent_24h": self._churn_rate_last_24h(t),
        }

        if self._head is not None:
            commit, p = self._head.predict_commit(features)
        else:
            commit, p = True, 1.0

        if not commit:
            self._pending_uncommitted.append({"field": field, "features": features, "proposed": proposed, "t": t})
            return

        if self._churn is not None and not self._churn.can_update(field, t, proposed):
            self._pending_uncommitted.append({"field": field, "features": features, "proposed": proposed, "t": t})
            return

        self._apply_update(field, proposed, t)
        self._update_log.append(
            {
                "timestamp": pd.Timestamp(t),
                "field": field,
                "old_value": current,
                "new_value": proposed,
                "p_commit": p,
            }
        )
        if self._head is not None:
            self._pending_commits.append(
                {
                    "field": field,
                    "features": features,
                    "old_value": current,
                    "new_value": proposed,
                    "commit_t": pd.Timestamp(t),
                }
            )

    def verify_pending_commits(self, t: pd.Timestamp, ground_truth_dict: dict) -> None:
        if self._head is None:
            return
        now = pd.Timestamp(t)
        lb = now - pd.Timedelta(hours=12)
        remain = []
        for row in self._pending_commits:
            if row["commit_t"] < lb:
                # too old to verify in the requested window
                continue
            field = row["field"]
            gt_val = ground_truth_dict.get(field)
            if gt_val is None:
                remain.append(row)
                continue
            if gt_val == row["new_value"]:
                self._head.learn_one(row["features"], True)
            elif gt_val == row["old_value"]:
                self._head.learn_one(row["features"], False)
            else:
                self._head.learn_one(row["features"], False)
        self._pending_commits = remain

    def step_continuous(self, t, p95_cpu_predicted, p95_mem_predicted):
        t = pd.Timestamp(t)
        for field, val in [
            ("predicted_cpu_p95_next_24h", float(p95_cpu_predicted)),
            ("predicted_mem_p95_next_24h", float(p95_mem_predicted)),
        ]:
            if self._churn is not None and not self._churn.can_update(field, t, val):
                continue
            current = getattr(self.profile.dynamic, field)
            setattr(self.profile.dynamic, field, val)
            self.profile.last_updated[field] = t
            self._update_log.append(
                {
                    "timestamp": t,
                    "field": field,
                    "old_value": current,
                    "new_value": val,
                    "p_commit": 1.0,
                }
            )

        if self._head is not None and self._pending_uncommitted:
            for row in self._pending_uncommitted:
                field = row["field"]
                label = (row["proposed"] != getattr(self.profile.dynamic, field))
                self._head.learn_one(row["features"], bool(label))
            self._pending_uncommitted.clear()

    def _adwin_window_size(self, field: str) -> float:
        det = self._cpu_drift if field == "cpu_class" else self._mem_drift
        w = getattr(getattr(det, "_d", None), "width", None)
        if w is None:
            return 0.0
        return float(w)

    def _hours_since(self, field: str, t: pd.Timestamp) -> float:
        last = self.profile.last_updated.get(field)
        if last is None:
            return 1e9
        return float((pd.Timestamp(t) - pd.Timestamp(last)).total_seconds() / 3600.0)

    def _churn_rate_last_24h(self, t: pd.Timestamp) -> float:
        t = pd.Timestamp(t)
        lb = t - pd.Timedelta(hours=24)
        n = sum(1 for x in self._update_log if lb <= pd.Timestamp(x["timestamp"]) <= t)
        return float(n)

    def _apply_update(self, field: str, value, t: pd.Timestamp):
        setattr(self.profile.dynamic, field, value)
        self.profile.last_updated[field] = pd.Timestamp(t)

    @property
    def update_log(self):
        return self._update_log
