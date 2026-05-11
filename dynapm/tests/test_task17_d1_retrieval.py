"""Unit tests for Task 17 retrieval helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = _REPO / "dynapm/scripts/task17_d1_retrieval.py"
_spec = importlib.util.spec_from_file_location("task17_d1_retrieval_mod", _MOD_PATH)
assert _spec and _spec.loader
t17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t17)


def test_profile_distance_increases_with_category_mismatch():
    std_cpu, std_mem = 10.0, 5.0
    base = dict(
        p_cpu=np.array(["a"]),
        p_mem=np.array(["x"]),
        p_pc=np.array([50.0]),
        p_pm=np.array([80.0]),
    )
    d0 = t17.profile_distance_matrix(
        np.array(["a"]),
        np.array(["x"]),
        np.array([50.0]),
        np.array([80.0]),
        **base,
        std_cpu=std_cpu,
        std_mem=std_mem,
    )[0, 0]
    d_cat = t17.profile_distance_matrix(
        np.array(["b"]),
        np.array(["x"]),
        np.array([50.0]),
        np.array([80.0]),
        **base,
        std_cpu=std_cpu,
        std_mem=std_mem,
    )[0, 0]
    assert d_cat > d0


def test_predict_cluster_majority_knn_exact():
    pool = pd.DataFrame(
        {
            "vm_id": ["1", "2", "3", "4", "6"],
            "cpu_class": ["low"] * 5,
            "memory_class": ["low"] * 5,
            "predicted_cpu_p95_next_24h": [1.0, 2.0, 3.0, 4.0, 99.0],
            "predicted_mem_p95_next_24h": [1.0, 2.0, 3.0, 4.0, 99.0],
            "cluster_id": [10, 10, 11, 10, 10],
        }
    )
    labs = pool["cluster_id"].to_numpy()
    pred = t17.predict_cluster_majority_knn(pool, 4, labs, std_cpu=1.0, std_mem=1.0, k=3)
    assert pred == 10


def test_bootstrap_delta_stats_constant():
    d = np.ones(20)
    m, lo, hi = t17.bootstrap_delta_stats(d, n_boot=200, seed=3)
    assert m == 1.0
    assert lo == 1.0 and hi == 1.0


def test_streaming_feature_row_simple():
    idx = pd.date_range("2020-01-01", periods=100, freq="5min")
    df = pd.DataFrame(
        {
            "CPU usage [%]": np.linspace(0, 50, 100),
            "Memory usage [%]": np.linspace(10, 30, 100),
        },
        index=idx,
    )
    row = t17.compute_streaming_feature_row(df, "T", "1")
    assert row["mean_cpu"] > 0
    assert row["p95_cpu"] >= row["p20_cpu"]
    assert "cpu_autocorr_lag12" in row
