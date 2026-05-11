"""Tests for ChurnControllerV2."""
from __future__ import annotations

import pandas as pd

from dynapm.core.churn_v2 import ChurnControllerV2


def test_first_commit_allowed():
    c = ChurnControllerV2(min_drift_interval_hours=2.0)
    t0 = pd.Timestamp("2024-01-01 12:00:00")
    assert c.can_commit_drift(t0) is True


def test_too_soon_blocks():
    c = ChurnControllerV2(min_drift_interval_hours=2.0)
    t0 = pd.Timestamp("2024-01-01 12:00:00")
    c.record_drift_commit(t0)
    t1 = t0 + pd.Timedelta(hours=1.0)
    assert c.can_commit_drift(t1) is False


def test_after_interval_allowed():
    c = ChurnControllerV2(min_drift_interval_hours=2.0)
    t0 = pd.Timestamp("2024-01-01 12:00:00")
    c.record_drift_commit(t0)
    t2 = t0 + pd.Timedelta(hours=2.1)
    assert c.can_commit_drift(t2) is True
