"""Minimum-interval guard for drift-triggered commits (v2 protocol)."""
from __future__ import annotations

import pandas as pd


class ChurnControllerV2:
    """
    Enforces a minimum spacing between drift-triggered commits.

    Periodic commits bypass this controller (handled upstream).
    """

    def __init__(self, min_drift_interval_hours: float = 2.0):
        self._min_interval = pd.Timedelta(hours=float(min_drift_interval_hours))
        self._last_drift_commit: pd.Timestamp | None = None

    def can_commit_drift(self, t: pd.Timestamp) -> bool:
        t = pd.Timestamp(t)
        if self._last_drift_commit is None:
            return True
        return (t - self._last_drift_commit) >= self._min_interval

    def record_drift_commit(self, t: pd.Timestamp) -> None:
        self._last_drift_commit = pd.Timestamp(t)
