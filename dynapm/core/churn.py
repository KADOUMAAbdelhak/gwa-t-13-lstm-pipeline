"""Churn controller for update rate limiting and categorical hysteresis."""
from __future__ import annotations

import pandas as pd


class ChurnController:
    """Enforces minimum inter-update interval and hysteresis per dynamic field."""

    def __init__(self, min_interval_hours: float = 2.0, hysteresis_consecutive: int = 3):
        self._min_interval = pd.Timedelta(hours=min_interval_hours)
        self._hyst_n = hysteresis_consecutive
        self._last_update = {}
        self._pending = {}

    def can_update(self, field: str, t: pd.Timestamp, proposed_value) -> bool:
        last = self._last_update.get(field)
        if last is not None and (pd.Timestamp(t) - pd.Timestamp(last)) < self._min_interval:
            return False

        if isinstance(proposed_value, str):
            prev = self._pending.get(field, (None, 0))
            if prev[0] == proposed_value:
                new_count = prev[1] + 1
            else:
                new_count = 1
            self._pending[field] = (proposed_value, new_count)
            if new_count < self._hyst_n:
                return False
            self._pending[field] = (proposed_value, 0)

        self._last_update[field] = pd.Timestamp(t)
        return True
