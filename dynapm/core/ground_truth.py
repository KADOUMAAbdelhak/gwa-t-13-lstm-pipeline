"""Ground-truth labels for profile evaluation at a snapshot time *t*."""
from __future__ import annotations

import pandas as pd

from dynapm.core.discretization import classify_value

CPU_METRIC = "CPU usage [%]"
MEM_METRIC = "Memory usage [%]"


def derive_ground_truth_at_t(
    df: pd.DataFrame,
    t: pd.Timestamp,
    window_hours: int = 24,
    cpu_thresholds: dict | None = None,
    mem_thresholds: dict | None = None,
) -> dict:
    """
    DEPRECATED: v1 semantics using a forward 24h window for all fields.

    Kept for reference. Prefer ``derive_ground_truth_at_t_v2``.
    """
    if cpu_thresholds is None or mem_thresholds is None:
        raise ValueError("cpu_thresholds and mem_thresholds are required")

    t0 = pd.Timestamp(t)
    t1 = t0 + pd.Timedelta(hours=window_hours)
    w = df.loc[(df.index >= t0) & (df.index < t1)]
    if len(w) == 0:
        raise ValueError(f"empty window for t={t0}, window_hours={window_hours}")

    cpu_s = w[CPU_METRIC]
    mem_s = w[MEM_METRIC]
    mean_cpu = float(cpu_s.mean())
    mean_mem = float(mem_s.mean())

    return {
        "cpu_class": classify_value(mean_cpu, cpu_thresholds),
        "memory_class": classify_value(mean_mem, mem_thresholds),
        "p95_cpu_next_24h": float(cpu_s.quantile(0.95)),
        "p95_mem_next_24h": float(mem_s.quantile(0.95)),
    }


def derive_ground_truth_at_t_v2(
    df: pd.DataFrame,
    t: pd.Timestamp,
    backward_window_hours: float = 1.0,
    forward_window_hours: int = 24,
    cpu_thresholds: dict | None = None,
    mem_thresholds: dict | None = None,
) -> dict:
    """
    v2 semantics: categorical fields from backward 1h state, continuous fields from forward 24h forecast.

    - cpu_class/memory_class: classify mean over [t-backward_window_hours, t]
    - p95_cpu_next_24h/p95_mem_next_24h: p95 over [t, t+forward_window_hours)
    If backward window is empty, falls back to smallest available backward window (1 row at/before t).
    """
    if cpu_thresholds is None or mem_thresholds is None:
        raise ValueError("cpu_thresholds and mem_thresholds are required")

    t0 = pd.Timestamp(t)

    b0 = t0 - pd.Timedelta(hours=float(backward_window_hours))
    wb = df.loc[(df.index >= b0) & (df.index <= t0)]
    if len(wb) == 0:
        wb = df.loc[df.index <= t0].tail(1)
    if len(wb) == 0:
        raise ValueError(f"empty backward window for t={t0}")

    f1 = t0 + pd.Timedelta(hours=int(forward_window_hours))
    wf = df.loc[(df.index >= t0) & (df.index < f1)]
    if len(wf) == 0:
        raise ValueError(f"empty forward window for t={t0}, window_hours={forward_window_hours}")

    cpu_b = wb[CPU_METRIC]
    mem_b = wb[MEM_METRIC]
    cpu_f = wf[CPU_METRIC]
    mem_f = wf[MEM_METRIC]

    return {
        "cpu_class": classify_value(float(cpu_b.mean()), cpu_thresholds),
        "memory_class": classify_value(float(mem_b.mean()), mem_thresholds),
        "p95_cpu_next_24h": float(cpu_f.quantile(0.95)),
        "p95_mem_next_24h": float(mem_f.quantile(0.95)),
    }
