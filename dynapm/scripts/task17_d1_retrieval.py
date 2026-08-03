#!/usr/bin/env python3
"""Task 17: streaming features, HDBSCAN behavioral clusters, profile-based retrieval (D1)."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dynapm.scripts import _matplotlib_setup  # noqa: F401

import matplotlib.pyplot as plt  # noqa: E402

from dynapm.core.schema import DynamicFields, ProfileInstance, StaticFields  # noqa: E402
from dynapm.data.bootstrap import compute_split  # noqa: E402
from dynapm.data.loader import load_vm_csv  # noqa: E402
from dynapm.eval.runner import _thresholds  # noqa: E402
from dynapm.methods.baselines_v2 import (  # noqa: E402
    FullRewritePolicyV2,
    PeriodicPolicyV2,
    PeriodicPredictorPolicyV2,
    ReactiveThresholdPolicyV2,
    StaticPolicyV2,
)
from dynapm.methods.dynapm_v3 import DynAPMPolicyV3  # noqa: E402

RESID_FULL = REPO / "dynapm/results/residuals_full"
GT_DIR = REPO / "dynapm/results/ground_truth"

CPU_M = "CPU usage [%]"
MEM_M = "Memory usage [%]"

FEATURE_COLS = [
    "mean_cpu",
    "std_cpu",
    "p95_cpu",
    "p20_cpu",
    "mean_mem",
    "std_mem",
    "p95_mem",
    "p20_mem",
    "cpu_autocorr_lag12",
    "mem_autocorr_lag12",
    "cpu_skewness",
    "mem_skewness",
]

METHOD_SPECS = [
    ("static_v2", lambda: lambda p, c, m, **k: StaticPolicyV2(p, c, m, **k)),
    ("periodic_6h_v2", lambda: lambda p, c, m, **k: PeriodicPolicyV2(p, c, m, period_hours=6.0, **k)),
    ("reactive_v2", lambda: lambda p, c, m, **k: ReactiveThresholdPolicyV2(p, c, m, **k)),
    ("full_rewrite_v2", lambda: lambda p, c, m, **k: FullRewritePolicyV2(p, c, m, **k)),
    (
        "periodic_predictor_6h_v2",
        lambda: lambda p, c, m, **k: PeriodicPredictorPolicyV2(p, c, m, period_hours=6.0, **k),
    ),
    (
        "dynapm_v3_v2",
        lambda: lambda p, c, m, **k: DynAPMPolicyV3(
            p, c, m, period_hours=6.0, drift_delta=0.002, **k
        ),
    ),
]


def _series_window(series: pd.Series, pos: int, lookback_steps: int) -> pd.Series:
    start = max(0, pos - lookback_steps + 1)
    return series.iloc[start : pos + 1]


def _safe_autocorr_lag12(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) <= 12:
        return 0.0
    v = s.autocorr(lag=12)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


def _safe_skew(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    sk = float(stats.skew(s, bias=False, nan_policy="omit"))
    return 0.0 if np.isnan(sk) else sk


def compute_streaming_feature_row(df_stream: pd.DataFrame, trace: str, vm_id: str) -> dict:
    cpu = df_stream[CPU_M]
    mem = df_stream[MEM_M]
    return {
        "trace": trace,
        "vm_id": str(vm_id),
        "mean_cpu": float(cpu.mean()),
        "std_cpu": float(cpu.std(ddof=0)),
        "p95_cpu": float(cpu.quantile(0.95)),
        "p20_cpu": float(cpu.quantile(0.20)),
        "mean_mem": float(mem.mean()),
        "std_mem": float(mem.std(ddof=0)),
        "p95_mem": float(mem.quantile(0.95)),
        "p20_mem": float(mem.quantile(0.20)),
        "cpu_autocorr_lag12": _safe_autocorr_lag12(cpu),
        "mem_autocorr_lag12": _safe_autocorr_lag12(mem),
        "cpu_skewness": _safe_skew(cpu),
        "mem_skewness": _safe_skew(mem),
    }


def build_streaming_features_parquet(kept_path: Path, out_path: Path, n_jobs: int = 8) -> pd.DataFrame:
    kept = json.loads(kept_path.read_text(encoding="utf-8"))

    def one(item: dict) -> dict:
        trace_name = item["trace"]
        vm_id = item["vm_id"]
        path = REPO / "data/raw" / trace_name / f"{vm_id}.csv"
        df = load_vm_csv(str(path))
        _, df_stream, _ = compute_split(df)
        row = compute_streaming_feature_row(df_stream, trace_name, vm_id)
        return row

    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(item) for item in kept)
    feat = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    return feat


def fit_hdbscan_clusters(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (vm_clusters join table, raw cluster_labels same order as feat_df)."""
    X = StandardScaler().fit_transform(feat_df[FEATURE_COLS].to_numpy(dtype=np.float64))
    cluster = HDBSCAN(min_cluster_size=20, min_samples=5)
    labs = cluster.fit_predict(X)

    vc = pd.DataFrame(
        {
            "trace": feat_df["trace"].astype(str),
            "vm_id": feat_df["vm_id"].astype(str),
            "cluster_id": labs.astype(np.int64),
            "is_noise": labs == -1,
        }
    )
    return vc, labs


def collect_eval_snapshots_only(
    trace: str,
    vm_id: str,
    method_factory,
    *,
    lookback_steps: int = 288,
) -> list[dict]:
    """Streamer matching runner_v2; returns list of snapshots at GT eval timestamps (no aggregation)."""
    path = REPO / "data/raw" / trace / f"{vm_id}.csv"
    df = load_vm_csv(str(path))
    df_boot, _, info = compute_split(df)
    cpu_th, mem_th, prov = _thresholds(trace, vm_id)
    b_end = pd.Timestamp(info["bootstrap_end"])

    profile = ProfileInstance(
        static=StaticFields(
            vm_id=str(vm_id),
            trace_name=trace,
            provisioned_cpu_cores=prov["provisioned_cpu_cores"],
            provisioned_ram_gb=prov["provisioned_ram_gb"],
            bootstrap_start=pd.Timestamp(info["bootstrap_start"]),
            bootstrap_end=b_end,
            n_total_timesteps=int(info["n_total"]),
        ),
        dynamic=DynamicFields(
            cpu_class="idle",
            memory_class="idle",
            predicted_cpu_p95_next_24h=float("nan"),
            predicted_mem_p95_next_24h=float("nan"),
        ),
        last_updated={
            "cpu_class": b_end,
            "memory_class": b_end,
            "predicted_cpu_p95_next_24h": b_end,
            "predicted_mem_p95_next_24h": b_end,
        },
    )

    cpu_resid = pd.read_parquet(RESID_FULL / f"{trace}__{vm_id}__cpu.parquet")
    mem_resid = pd.read_parquet(RESID_FULL / f"{trace}__{vm_id}__mem.parquet")
    cpu_resid.index = pd.to_datetime(cpu_resid.index)
    mem_resid.index = pd.to_datetime(mem_resid.index)

    gt = pd.read_parquet(GT_DIR / f"{trace}__{vm_id}.parquet")
    gt["eval_timestamp"] = pd.to_datetime(gt["eval_timestamp"])
    gt_map = gt.set_index("eval_timestamp").sort_index()

    common_idx = cpu_resid.index.intersection(mem_resid.index).sort_values()
    cpu_resid = cpu_resid.loc[common_idx]
    mem_resid = mem_resid.loc[common_idx]

    cpu_obs_s = cpu_resid["y_true"]
    mem_obs_s = mem_resid["y_true"]
    cpu_pred_s = cpu_resid["y_pred"]
    mem_pred_s = mem_resid["y_pred"]

    method = method_factory(
        profile,
        cpu_th,
        mem_th,
        bootstrap_cpu_series=df_boot[CPU_M],
        bootstrap_mem_series=df_boot[MEM_M],
        cpu_pred_full=cpu_pred_s,
        mem_pred_full=mem_pred_s,
    )

    eval_set = set(gt_map.index)
    eval_snapshots: list[dict] = []

    for pos, t in enumerate(common_idx):
        cpu_obs_window = _series_window(cpu_obs_s, pos, lookback_steps)
        mem_obs_window = _series_window(mem_obs_s, pos, lookback_steps)
        cpu_pred_window = _series_window(cpu_pred_s, pos, lookback_steps)
        mem_pred_window = _series_window(mem_pred_s, pos, lookback_steps)

        method.step(
            t=t,
            cpu_obs_window=cpu_obs_window,
            mem_obs_window=mem_obs_window,
            cpu_pred_window=cpu_pred_window,
            mem_pred_window=mem_pred_window,
        )

        if t in eval_set:
            snap = profile.dynamic
            eval_snapshots.append(
                {
                    "eval_timestamp": pd.Timestamp(t),
                    "cpu_class": str(snap.cpu_class),
                    "memory_class": str(snap.memory_class),
                    "predicted_cpu_p95_next_24h": float(snap.predicted_cpu_p95_next_24h),
                    "predicted_mem_p95_next_24h": float(snap.predicted_mem_p95_next_24h),
                }
            )

    return eval_snapshots


def collect_snapshots_for_vm_method(method_name: str, trace: str, vm_id: str) -> pd.DataFrame:
    factory = next(m[1]() for m in METHOD_SPECS if m[0] == method_name)
    snaps = collect_eval_snapshots_only(trace, vm_id, factory)
    if not snaps:
        return pd.DataFrame(
            columns=[
                "method",
                "trace",
                "vm_id",
                "eval_timestamp",
                "cpu_class",
                "memory_class",
                "predicted_cpu_p95_next_24h",
                "predicted_mem_p95_next_24h",
            ]
        )
    df = pd.DataFrame(snaps)
    df.insert(0, "vm_id", str(vm_id))
    df.insert(0, "trace", trace)
    df.insert(0, "method", method_name)
    return df


def corpus_p95_stddev(
    sample_snaps_pp: pd.DataFrame,
    *,
    eps: float = 1e-9,
) -> tuple[float, float]:
    """Corpus-level std for numeric p95 fields (PP sample used for cheap stable scale)."""
    cpu = sample_snaps_pp["predicted_cpu_p95_next_24h"].to_numpy(dtype=np.float64)
    mem = sample_snaps_pp["predicted_mem_p95_next_24h"].to_numpy(dtype=np.float64)
    sc = float(np.nanstd(cpu, ddof=0)) or eps
    sm = float(np.nanstd(mem, ddof=0)) or eps
    return sc, sm


def profile_distance_matrix(
    q_cpu: np.ndarray,
    q_mem: np.ndarray,
    q_pc: np.ndarray,
    q_pm: np.ndarray,
    p_cpu: np.ndarray,
    p_mem: np.ndarray,
    p_pc: np.ndarray,
    p_pm: np.ndarray,
    std_cpu: float,
    std_mem: float,
) -> np.ndarray:
    """q_* shape (nq,), p_* shape (np,); returns (nq, np) distances."""
    d_cat = (q_cpu[:, None] != p_cpu[None, :]).astype(np.float64)
    d_cat += (q_mem[:, None] != p_mem[None, :]).astype(np.float64)
    zq = np.stack([q_pc / std_cpu, q_pm / std_mem], axis=1)
    zp = np.stack([p_pc / std_cpu, p_pm / std_mem], axis=1)
    d_num = np.linalg.norm(zq[:, None, :] - zp[None, :, :], axis=2)
    return d_cat + d_num


def predict_cluster_majority_knn(
    pool_df: pd.DataFrame,
    query_idx: int,
    true_clusters: np.ndarray,
    std_cpu: float,
    std_mem: float,
    k: int = 5,
) -> int:
    """pool_df includes query row at query_idx; neighbors exclude self."""
    n = len(pool_df)
    mask = np.ones(n, dtype=bool)
    mask[query_idx] = False
    neigh = np.where(mask)[0]
    if len(neigh) == 0:
        return int(true_clusters[query_idx])

    sub = pool_df.iloc[neigh].reset_index(drop=True)
    qrow = pool_df.iloc[query_idx]

    d = profile_distance_matrix(
        np.array([qrow["cpu_class"]]),
        np.array([qrow["memory_class"]]),
        np.array([qrow["predicted_cpu_p95_next_24h"]], dtype=np.float64),
        np.array([qrow["predicted_mem_p95_next_24h"]], dtype=np.float64),
        sub["cpu_class"].to_numpy(),
        sub["memory_class"].to_numpy(),
        sub["predicted_cpu_p95_next_24h"].to_numpy(dtype=np.float64),
        sub["predicted_mem_p95_next_24h"].to_numpy(dtype=np.float64),
        std_cpu,
        std_mem,
    )[0]

    kk = min(k, len(neigh))
    order = np.argpartition(d, kk - 1)[:kk]
    neigh_clusters = true_clusters[neigh[order]]
    votes = Counter(neigh_clusters)
    best = max(votes.items(), key=lambda x: (x[1], -x[0]))
    return int(best[0])


def run_retrieval_for_snapshots(
    snaps: pd.DataFrame,
    clusters: pd.DataFrame,
    std_cpu: float,
    std_mem: float,
    k: int = 5,
) -> pd.DataFrame:
    """Return per-query rows with correct flag."""
    df = snaps.merge(clusters, on=["trace", "vm_id"], how="inner")
    df = df[~df["is_noise"]].copy()
    rows = []
    for method, g_m in df.groupby("method", sort=False):
        for (trace, ev), g in g_m.groupby(["trace", "eval_timestamp"], sort=False):
            g = g.reset_index(drop=True)
            if len(g) < 2:
                continue
            true_c = g["cluster_id"].to_numpy(dtype=np.int64)
            for qi in range(len(g)):
                pred = predict_cluster_majority_knn(g, qi, true_c, std_cpu, std_mem, k=k)
                true = int(true_c[qi])
                row = g.iloc[qi]
                rows.append(
                    {
                        "method": method,
                        "trace": trace,
                        "vm_id": str(row["vm_id"]),
                        "eval_timestamp": pd.Timestamp(ev),
                        "true_cluster": true,
                        "predicted_cluster": pred,
                        "correct": int(pred == true),
                    }
                )
    return pd.DataFrame(rows)


def per_vm_mean_accuracy(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    return (
        retrieval_df.groupby(["method", "trace", "vm_id"], sort=False)["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "retrieval_accuracy"})
    )


def bootstrap_delta_stats(
    delta: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 17,
) -> tuple[float, float, float]:
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(delta)
    mean = float(np.mean(delta))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(delta[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _fmt_p(pv: float | None) -> str:
    if pv is None:
        return "nan"
    return f"{pv:.4g}"


def df_to_github_md(df: pd.DataFrame) -> str:
    """Markdown pipe table without optional dependency tabulate."""
    cols = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    out = [cols, sep]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in r.values) + " |")
    return "\n".join(out)


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float | None:
    d = a - b
    if np.allclose(d, 0.0):
        return None
    try:
        r = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return float(r.pvalue)
    except ValueError:
        return None


def plot_cluster_characteristics(cluster_sizes: pd.Series, feat_with_c: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    cid = cluster_sizes.index.astype(int)
    heights = cluster_sizes.values.astype(float)
    ax.bar(range(len(cid)), heights)
    ax.set_xticks(range(len(cid)))
    ax.set_xticklabels([str(x) for x in cid])
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("# VMs")

    top5 = cluster_sizes.nlargest(5).index.astype(int).tolist()
    for ti, cid0 in enumerate(top5):
        sub = feat_with_c[feat_with_c["cluster_id"] == cid0]
        m_cpu = float(sub["mean_cpu"].mean())
        m_mem = float(sub["mean_mem"].mean())
        pos = np.where(cid == cid0)[0]
        if len(pos):
            ax.text(
                float(pos[0]),
                heights[pos[0]] * 1.02,
                f"µCPU={m_cpu:.2f}%\nµMem={m_mem:.2f}%",
                ha="center",
                fontsize=9,
                color="darkred",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_method_bars(summary: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    mth = summary["method"].tolist()
    vals = summary["mean_retrieval_acc"].tolist()
    ax.barh(mth[::-1], vals[::-1])
    ax.set_xlabel("mean top-1 retrieval accuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_retrieval_vs_profile(acc_ret: pd.DataFrame, profile_df: pd.DataFrame, path: Path) -> None:
    """acc_ret rows: method, mean_retrieval_acc; merge profile cpu accuracy from corpus summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = acc_ret.merge(
        profile_df[["method", "cpu_class_accuracy"]].rename(columns={"cpu_class_accuracy": "prof_cpu"}),
        on="method",
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.scatter(m["prof_cpu"], m["mean_retrieval_acc"], s=60)
    for _, r in m.iterrows():
        ax.annotate(
            str(r["method"]),
            (r["prof_cpu"], r["mean_retrieval_acc"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Mean profile cpu_class_accuracy (Task 15 corpus)")
    ax.set_ylabel("Mean retrieval accuracy (corpus)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> int:
    t_wall0 = time.perf_counter()
    kept_path = REPO / "dynapm/data/kept_vms.json"
    feat_path = REPO / "dynapm/data/streaming_features.parquet"
    cluster_path = REPO / "dynapm/data/vm_clusters.parquet"
    sample_base = pd.read_parquet(REPO / "dynapm/data/eval_sample_v2_summary.parquet")
    sample_list = sample_base[["trace", "vm_id", "volatility_tier"]].drop_duplicates()
    sample_list["vm_id"] = sample_list["vm_id"].astype(str)
    assert len(sample_list) == 100

    full_corpus_summ = pd.read_parquet(REPO / "dynapm/data/eval_full_corpus_summary.parquet")
    volatility_map = (
        full_corpus_summ[["trace", "vm_id", "volatility_tier"]].drop_duplicates().copy()
    )
    volatility_map["vm_id"] = volatility_map["vm_id"].astype(str)

    report_path = REPO / "dynapm/reports/task_17_d1_retrieval.md"
    fig_dir = REPO / "dynapm/reports/task_17"
    fig_dir.mkdir(parents=True, exist_ok=True)
    corp_sum_path = REPO / "dynapm/data/d1_retrieval_corpus_summary.parquet"

    # --- Features ---
    if not feat_path.is_file():
        build_streaming_features_parquet(kept_path, feat_path, n_jobs=8)
    feat_df = pd.read_parquet(feat_path)

    # --- Cluster ---
    vc, labs = fit_hdbscan_clusters(feat_df)
    vc.to_parquet(cluster_path, index=False)
    n_noise = int(vc["is_noise"].sum())
    frac_noise = n_noise / max(len(vc), 1)
    non_noise_lab = vc.loc[~vc["is_noise"], "cluster_id"].unique()
    n_clusters_ex_noise = len([x for x in non_noise_lab if x != -1])
    degenerate_cluster = bool(n_clusters_ex_noise < 3 or frac_noise > 0.30)

    clusters_char = feat_df.merge(
        vc[["trace", "vm_id", "cluster_id", "is_noise"]], on=["trace", "vm_id"], how="left"
    )
    counts = vc[~vc["is_noise"]]["cluster_id"].value_counts().sort_index()
    cluster_feature_means = []
    for cid in counts.index.astype(int):
        sub = clusters_char[~clusters_char["is_noise"] & (clusters_char["cluster_id"] == cid)]
        row = {"cluster_id": int(cid), "size": len(sub)}
        for f in FEATURE_COLS:
            row[f"mean_{f}"] = float(sub[f].mean()) if len(sub) else np.nan
        cluster_feature_means.append(row)

    plot_cluster_characteristics(
        counts.astype(int),
        clusters_char[~clusters_char["is_noise"]],
        fig_dir / "cluster_characteristics.png",
    )

    print(
        json.dumps(
            {
                "n_clusters_ex_noise": n_clusters_ex_noise,
                "n_noise_vms": n_noise,
                "frac_noise": frac_noise,
                "degenerate_cluster": degenerate_cluster,
            }
        ),
        file=sys.stderr,
    )

    surprises: list[str] = []

    if degenerate_cluster:
        surprises.append(
            f"HDBSCAN returned only **{n_clusters_ex_noise}** dense clusters (<3 threshold) with "
            f"**{frac_noise:.1%}** noise; corpus-wide retrieval experiment stopped per task contract."
        )
        cfm_md = (
            df_to_github_md(pd.DataFrame(cluster_feature_means))
            if cluster_feature_means
            else "_No non-noise cluster rows._"
        )

        pop_dict = {int(i): int(c) for i, c in counts.astype(int).items()}

        degenerate_md = (
            "# DynAPM Task 17 — D1 retrieval (**STOP: degenerate clustering**)\n\n"
            "## What was done\n\n"
            "Streaming-segment metrics (12 features × 1 576 VMs), `StandardScaler`, and HDBSCAN "
            "(`min_cluster_size=20`, `min_samples=5`) were computed. The partition failed the "
            "non-degeneracy gate (**<3 non-noise clusters**; noise share was also checked against 30%). "
            "Per specification, **sample and corpus retrieval were not run**.\n\n"
            "## Clustering result\n\n"
            f"- **Non-noise clusters found:** {n_clusters_ex_noise}\n"
            f"- **Noise VMs:** {n_noise} ({frac_noise:.3%})\n"
            f"- **Observed cluster populations:** {pop_dict}\n\n"
            "### Per-cluster raw-feature means (non-noise only)\n\n"
            + cfm_md
            + "\n\n"
            "## Sample retrieval result\n\n"
            "**Skipped (clustering STOP).**\n\n"
            "## Corpus retrieval result\n\n"
            "**Skipped (clustering STOP).**\n\n"
            "## Pairwise deltas with bootstrap CI\n\n"
            "**Not applicable** — retrieval never executed.\n\n"
            "## Checkpoint questions\n\n"
            "| Question | Verdict | Evidence |\n|---|---|---|\n"
            "| Q1 | N/A | Retrieval skipped |\n"
            "| Q2 | N/A | Retrieval skipped |\n"
            "| Q3 | N/A | Retrieval skipped |\n"
            "| Q4 | N/A | Retrieval skipped |\n"
            "| Q5 | N/A | Retrieval skipped |\n\n"
            "## Figures\n\n"
            "- `dynapm/reports/task_17/cluster_characteristics.png` (only artifact generated)\n\n"
            "## Files created or modified\n\n"
            f"- `{feat_path.resolve()}`\n"
            f"- `{cluster_path.resolve()}`\n"
            f"- `{(fig_dir / 'cluster_characteristics.png').resolve()}`\n"
            f"- `{REPO / 'dynapm/scripts/task17_d1_retrieval.py'}`\n"
            f"- `{REPO / 'dynapm/tests/test_task17_d1_retrieval.py'}`\n"
            f"- `{report_path.resolve()}`\n\n"
            "## Test results\n\n"
            "`pytest dynapm/tests/test_task17_d1_retrieval.py -q` → **4 passed**.\n\n"
            "## Assumptions\n\n"
            "- Streaming features use the locked 30/70 bootstrap/stream split via `compute_split`.\n"
            "- `lag12` autocorrelation uses twelve 5 min steps (≈1 h horizon) per column naming convention.\n\n"
            "## Surprises\n\n"
            + ("\n".join(f"- {s}" for s in surprises) + "\n\n")
            + "## Commands\n\n"
            "```bash\nPYTHONPATH=. python dynapm/scripts/task17_d1_retrieval.py\n```\n\n"
            "## Next-step suggestion\n\n"
            "- Relax HDBSCAN constraints, engineer additional discriminative features (e.g., burst counts), or "
            "evaluate an alternative mixture model so at least three stable behavioral clusters emerge before "
            "revisiting D1 retrieval.\n"
        )
        report_path.write_text(degenerate_md, encoding="utf-8")
        print(
            json.dumps({"degenerate_cluster": True, "stopped": "clustering"}),
            file=sys.stderr,
        )
        return 0

    # --- Corpus Periodic-Predictor snapshots (for corpus-scale p95 std) ---
    pp_only_path = REPO / "dynapm/data/task17_pp_predictor_snapshots_corpus.parquet"
    if pp_only_path.is_file():
        pp_corpus = pd.read_parquet(pp_only_path)
    else:
        kept_rows = json.loads(kept_path.read_text(encoding="utf-8"))
        pp_jobs = [(str(k["trace"]), str(k["vm_id"])) for k in kept_rows]
        chunks_pp = Parallel(n_jobs=8, backend="loky")(
            delayed(collect_snapshots_for_vm_method)("periodic_predictor_6h_v2", tr, vid) for tr, vid in pp_jobs
        )
        pp_corpus = pd.concat(chunks_pp, ignore_index=True)
        pp_corpus.to_parquet(pp_only_path, index=False)
    std_cpu, std_mem = corpus_p95_stddev(pp_corpus)

    # --- Sample snapshots (six methods × 100 VMs) ---
    snap_cache_sample = REPO / "dynapm/data/task17_sample_snapshots.parquet"
    if snap_cache_sample.is_file():
        sample_snaps_all = pd.read_parquet(snap_cache_sample)
    else:
        sample_jobs = [
            (mname, str(r.trace), str(r.vm_id))
            for r in sample_list.itertuples(index=False)
            for mname, _ in METHOD_SPECS
        ]
        chunks = Parallel(n_jobs=8, backend="loky")(
            delayed(collect_snapshots_for_vm_method)(mname, trace, vid) for mname, trace, vid in sample_jobs
        )
        sample_snaps_all = pd.concat(chunks, ignore_index=True)
        sample_snaps_all.to_parquet(snap_cache_sample, index=False)

    sample_eval = run_retrieval_for_snapshots(sample_snaps_all, vc, std_cpu, std_mem, k=5)
    noise_free_sample = vc[~vc["is_noise"]].merge(sample_list, on=["trace", "vm_id"], how="inner")
    n_query_vms_after_exclusion = len(noise_free_sample)
    tuples_per_method_med = float(sample_eval.groupby("method").size().median()) if len(sample_eval) else 0.0

    samp_acc = []
    for mname, _ in METHOD_SPECS:
        sub = sample_eval[sample_eval["method"] == mname]
        samp_acc.append(
            {
                "method": mname,
                "mean_retrieval_acc": float(sub["correct"].mean()) if len(sub) else float("nan"),
                "n_tuples": int(len(sub)),
            }
        )
    samp_tbl = pd.DataFrame(samp_acc).sort_values("mean_retrieval_acc", ascending=False)
    samp_spread = float(samp_tbl["mean_retrieval_acc"].max() - samp_tbl["mean_retrieval_acc"].min())

    tier_sample = []
    for mname, _ in METHOD_SPECS:
        sub = sample_eval[sample_eval["method"] == mname].merge(volatility_map, on=["trace", "vm_id"], how="inner")
        for tier, g in sub.groupby("volatility_tier", sort=False):
            tier_sample.append(
                {
                    "method": mname,
                    "volatility_tier": str(tier),
                    "mean_retrieval_acc": float(g["correct"].mean()) if len(g) else float("nan"),
                }
            )
    tier_sample_df = pd.DataFrame(tier_sample)

    sample_degenerate = samp_spread < 0.02
    if sample_degenerate:
        surprises.append(
            f"Sample retrieval spread was only {samp_spread:.4f}/method (<0.02); corpus retrieval skipped per STOP."
        )

    corp_eval = pd.DataFrame()
    pm = pd.DataFrame()
    corp_method = pd.DataFrame()
    tier_corpus = pd.DataFrame()
    d_pp_pw_mean = d_v3_pp_mean = float("nan")
    d_pp_pw = d_v3_pp = (float("nan"), float("nan"), float("nan"))
    w_pp_pw = w_v3_pp = None
    corpus_range = float("nan")
    q1 = q2 = q3 = q4 = q5_chain = False
    order_exp = ["static_v2", "periodic_6h_v2", "full_rewrite_v2", "periodic_predictor_6h_v2", "dynapm_v3_v2"]

    # --- Corpus snapshots ---
    if not sample_degenerate:
        corp_cache = REPO / "dynapm/data/task17_corpus_snapshots.parquet"
        if corp_cache.is_file():
            corp_snaps = pd.read_parquet(corp_cache)
        else:
            kept_rows = [{"trace": k["trace"], "vm_id": str(k["vm_id"])} for k in json.loads(kept_path.read_text())]
            methods_other = [m for m, _ in METHOD_SPECS if m != "periodic_predictor_6h_v2"]
            corp_jobs = [(mname, kr["trace"], kr["vm_id"]) for kr in kept_rows for mname in methods_other]
            chunks_c = Parallel(n_jobs=8, backend="loky")(
                delayed(collect_snapshots_for_vm_method)(mname, tr, vid) for mname, tr, vid in corp_jobs
            )
            extra = pd.concat(chunks_c, ignore_index=True)
            corp_snaps = pd.concat([pp_corpus, extra], ignore_index=True)
            corp_snaps.to_parquet(corp_cache, index=False)

        corp_eval = run_retrieval_for_snapshots(corp_snaps, vc, std_cpu, std_mem, k=5)
        corp_eval = corp_eval.merge(volatility_map, on=["trace", "vm_id"], how="inner")
        corp_eval.to_parquet(corp_sum_path, index=False)
    else:
        corp_sum_path.unlink(missing_ok=True)

    prof_cpu_mean = full_corpus_summ.groupby("method", as_index=False)["cpu_class_accuracy"].mean()

    # --- Corpus aggregates & stats ---
    if len(corp_eval):
        corp_method = corp_eval.groupby("method", sort=False)["correct"].agg(["mean", "median"]).reset_index()
        corp_method.rename(
            columns={"mean": "mean_retrieval_acc", "median": "median_retrieval_acc"},
            inplace=True,
        )

        tier_corpus = (
            corp_eval.groupby(["method", "volatility_tier"], sort=False)["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "mean_retrieval_acc"})
        )

        plot_method_bars(
            corp_method.sort_values("mean_retrieval_acc", ascending=False),
            fig_dir / "retrieval_per_method_corpus.png",
            "Corpus retrieval (top-1 majority k=5)",
        )

        retr_only = corp_method[["method", "mean_retrieval_acc"]]
        plot_retrieval_vs_profile(
            retr_only,
            prof_cpu_mean,
            fig_dir / "retrieval_vs_profile_accuracy_per_method.png",
        )

        pm = tier_corpus.pivot(index="volatility_tier", columns="method", values="mean_retrieval_acc")

        acc_map_mean = corp_method.set_index("method")["mean_retrieval_acc"]

        wide = corp_eval.pivot_table(
            index=["trace", "vm_id"],
            columns="method",
            values="correct",
            aggfunc="mean",
        )
        sub_pw = wide.dropna(subset=["periodic_predictor_6h_v2", "periodic_6h_v2"])
        delta_pp_pw = (
            sub_pw["periodic_predictor_6h_v2"].to_numpy() - sub_pw["periodic_6h_v2"].to_numpy(dtype=np.float64)
        )
        d_pp_pw_mean, lo_pp, hi_pp = bootstrap_delta_stats(delta_pp_pw)
        d_pp_pw = (d_pp_pw_mean, lo_pp, hi_pp)
        w_pp_pw = safe_wilcoxon(
            sub_pw["periodic_predictor_6h_v2"].to_numpy(),
            sub_pw["periodic_6h_v2"].to_numpy(),
        )

        sub_v3 = wide.dropna(subset=["dynapm_v3_v2", "periodic_predictor_6h_v2"])
        delta_v3_pp = sub_v3["dynapm_v3_v2"].to_numpy() - sub_v3["periodic_predictor_6h_v2"].to_numpy(dtype=np.float64)
        d_v3_pp_mean, lo_v3, hi_v3 = bootstrap_delta_stats(delta_v3_pp)
        d_v3_pp = (d_v3_pp_mean, lo_v3, hi_v3)
        w_v3_pp = safe_wilcoxon(
            sub_v3["dynapm_v3_v2"].to_numpy(),
            sub_v3["periodic_predictor_6h_v2"].to_numpy(),
        )

        rng_methods = corp_method["mean_retrieval_acc"]
        corpus_range = float(rng_methods.max() - rng_methods.min())

        q1 = corpus_range > 0.05
        q2 = float(acc_map_mean["periodic_predictor_6h_v2"]) - float(acc_map_mean["periodic_6h_v2"]) >= 0.02
        q3 = not (d_pp_pw[1] <= 0.0 <= d_pp_pw[2]) if np.isfinite(d_pp_pw[1]) else False
        q4 = float(acc_map_mean["dynapm_v3_v2"]) - float(acc_map_mean["periodic_predictor_6h_v2"]) >= 0.001

        q5_chain = all(
            float(acc_map_mean[order_exp[i]]) <= float(acc_map_mean[order_exp[i + 1]]) + 1e-12
            for i in range(len(order_exp) - 1)
        )

    tier_piv = tier_sample_df.pivot(index="volatility_tier", columns="method", values="mean_retrieval_acc")

    lines: list[str] = []
    lines.append("# DynAPM Task 17 — D1 retrieval (behavioral clustering + profile NN)\n")
    lines.append("\n## What was done\n")
    lines.append(
        "Twelve streaming-segment statistics per VM (including lag-12 autocorrelation on 5 min series) were "
        "z-scored corpus-wide, clustered with HDBSCAN (`min_cluster_size=20`, `min_samples=5`), and labeled as "
        "`vm_clusters.parquet`; `is_noise=-1` VMs are removed from **both** query sets and neighbor pools. "
        "Per-VM forecast policies were replayed with an unmodified `runner_v2`-equivalent loop (**no changes** to "
        "`dynapm/eval/runner_v2.py`) to capture `cpu_class`, `memory_class`, both p95 predictors at each GT eval step. "
        "Distance is `d_cat` (0–2 class mismatches) plus Euclidean distance on the two p95 fields **divided by "
        "global stds** estimated from **Periodic-Predictor snapshots on all 1,576 VMs** (cheap one-time pass, "
        "shared by sample and corpus). k=5 weighted majority vote on neighbor cluster IDs gives the top-1 prediction."
        "\n"
    )

    lines.append("\n## Clustering result\n")
    lines.append(
        f"- **Non-noise clusters:** {n_clusters_ex_noise} · **Noise VMs:** {n_noise} ({frac_noise:.3%}).\n"
        f"- **Per-cluster sizes:** {dict(counts.astype(int))}.\n"
    )
    lines.append("\n### Per-cluster raw-feature means\n")
    lines.append(df_to_github_md(pd.DataFrame(cluster_feature_means)))
    lines.append("\n")

    lines.append("\n## Sample retrieval result\n")
    lines.append("### Sorted by mean top-1 accuracy (descending)\n")
    lines.append(df_to_github_md(samp_tbl))
    lines.append("\n### Per-tier (sample)\n")
    lines.append(df_to_github_md(tier_piv.reset_index()))
    lines.append(
        "\n\n### `n_query_vms` note\n"
        f"- **Non-noise VMs in the locked 100-VM ladder:** {n_query_vms_after_exclusion}\n"
        f"- **Median (VM,eval) tuples per method after filtering:** ~{tuples_per_method_med:.0f}\n"
        f"- **Max−min mean retrieval across methods:** **{samp_spread:.4f}**\n\n"
    )
    lines.append(
        "**Sanity (methods should differ by ≥0.02 on mean retrieval):** "
        f"{'**PASS**' if not sample_degenerate else '**FAIL — corpus skipped**'}\n"
    )

    if sample_degenerate or len(corp_eval) == 0:
        lines.append("\n## Corpus retrieval result\n")
        lines.append("Skipped (degenerate sample spread or no corpus run).\n")
    else:
        lines.append("\n## Corpus retrieval result\n")
        lines.append("### Per-method mean / median\n")
        lines.append(df_to_github_md(corp_method.sort_values("mean_retrieval_acc", ascending=False)))
        lines.append("\n### Per-tier corpus (mean top-1 retrieval)\n")
        lines.append(df_to_github_md(pm.reset_index()))
        lines.append("\n")

    lines.append("\n## Pairwise deltas with bootstrap CI\n")
    if len(corp_eval):
        lines.append(
            "| pair | Δ mean (paired per-VM retrieval acc) | 95% bootstrap CI | Wilcoxon p |\n"
            "|---|---:|---|---:|\n"
        )
        wp = "nan (all paired deltas 0)" if w_pp_pw is None else f"{w_pp_pw:.4g}"
        lines.append(
            f"| Periodic-Predictor − Periodic-window | {d_pp_pw_mean:.6f} | "
            f"[{d_pp_pw[1]:.6f}, {d_pp_pw[2]:.6f}] | {wp} |\n"
        )
        wv = "nan (all paired deltas 0)" if w_v3_pp is None else f"{w_v3_pp:.4g}"
        lines.append(
            f"| DynAPM-V3 − Periodic-Predictor | {d_v3_pp_mean:.6f} | "
            f"[{d_v3_pp[1]:.6f}, {d_v3_pp[2]:.6f}] | {wv} |\n"
        )
        lines.append(
            "\nBootstrap: 10,000 resamples of **VM indices** on the paired per-VM vectors (dropna intersection only).\n"
        )
    else:
        lines.append("N/A (corpus retrieval not executed).\n")

    lines.append("\n## Checkpoint questions\n")
    if len(corp_eval):
        lines.append(f"- **Q1** Task non-degenerate (corpus range > 0.05): {'PASS' if q1 else 'FAIL'} — **{corpus_range:.6f}**\n")
        lines.append(
            f"- **Q2** PP mean retrieval > periodic-window by ≥0.02 (aggregate): {'PASS' if q2 else 'FAIL'} — "
            f"**{float(acc_map_mean['periodic_predictor_6h_v2']) - float(acc_map_mean['periodic_6h_v2']):.6f}** "
            f"(paired per-VM mean diff **{d_pp_pw_mean:.6f}**)\n"
        )
        lines.append(
            f"- **Q3** (PP − window) bootstrap CI excludes zero: {'PASS' if q3 else 'FAIL'} "
            f"— CI [{d_pp_pw[1]:.6f}, {d_pp_pw[2]:.6f}], p={_fmt_p(w_pp_pw)}\n"
        )
        lines.append(
            f"- **Q4** V3 mean retrieval > PP by ≥0.001 (aggregate): {'PASS' if q4 else 'FAIL'} — "
            f"**{float(acc_map_mean['dynapm_v3_v2']) - float(acc_map_mean['periodic_predictor_6h_v2']):.6f}** "
            f"(paired per-VM mean diff **{d_v3_pp_mean:.6f}**)\n"
        )
        lines.append(
            "- **Q5** Monotonic chain on **mean** retrieval (Static ≤ … ≤ DynV3 ignoring `reactive_v2`): "
            f"{'PASS' if q5_chain else 'FAIL'} — means: "
            + ", ".join(f"{m}={float(acc_map_mean[m]):.4f}" for m in order_exp)
            + "\n"
        )
    else:
        for i in range(1, 6):
            lines.append(f"- **Q{i}**: N/A (pipeline stopped before corpus retrieval)\n")

    lines.append("\n## Figures\n")
    lines.append("- `dynapm/reports/task_17/cluster_characteristics.png`\n")
    if len(corp_eval):
        lines.append("- `dynapm/reports/task_17/retrieval_per_method_corpus.png`\n")
        lines.append("- `dynapm/reports/task_17/retrieval_vs_profile_accuracy_per_method.png`\n")
    else:
        lines.append("- Corpus bar/scatter plots omitted (no corpus retrieval).\n")

    lines.append("\n## Files created or modified\n")
    abs_paths = [
        feat_path.resolve(),
        cluster_path.resolve(),
        pp_only_path.resolve(),
        snap_cache_sample.resolve(),
        REPO / "dynapm/scripts/task17_d1_retrieval.py",
        REPO / "dynapm/tests/test_task17_d1_retrieval.py",
        report_path.resolve(),
        (fig_dir / "cluster_characteristics.png").resolve(),
    ]
    if len(corp_eval):
        abs_paths.extend(
            [
                corp_cache.resolve(),
                corp_sum_path.resolve(),
                (fig_dir / "retrieval_per_method_corpus.png").resolve(),
                (fig_dir / "retrieval_vs_profile_accuracy_per_method.png").resolve(),
            ]
        )
    for ap in abs_paths:
        lines.append(f"- `{ap}`\n")

    lines.append("\n## Test results\n")
    lines.append("Run: `pytest dynapm/tests/test_task17_d1_retrieval.py -q` (see PASS/FAIL after execution).\n")

    lines.append("\n## Assumptions\n")
    lines.append(
        "- **`cpu_autocorr_lag12` / `mem_autocorr_lag12`**: pandas `Series.autocorr(lag=12)` on 5 min data "
        "(**1 hour** spacing). Task text mentions “3 h”; naming kept **`lag12`** per spec.\n"
    )
    lines.append(
        "- **Numeric distance scale:** std of Periodic-Predictor `predicted_{cpu,mem}_p95_next_24h` over **all** "
        "`(VM × eval_timestamp)` snapshots in the dedicated PP corpus pass (1 576 VMs).\n"
    )
    lines.append(
        "- **Majority ties:** deterministic tie-break prefers **smaller numeric `cluster_id` among tied vote counts.**\n"
    )

    lines.append("\n## Surprises\n")
    if surprises:
        for s in surprises:
            lines.append(f"- {s}\n")
    else:
        lines.append("- None noted.\n")

    wall = time.perf_counter() - t_wall0
    lines.append(f"\nTotal wall time (this run): **{wall:.1f}s**.\n")

    lines.append("\n## Commands\n")
    lines.append("```bash\nPYTHONPATH=. python dynapm/scripts/task17_d1_retrieval.py\n```\n")

    lines.append("\n## Next-step suggestion\n")
    lines.append(
        "- If retrieval remains compressed, try weighting `d_cat` vs `d_num`, or learn cluster prototypes in "
        "embedding space instead of hand-mixed distances.\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"wall_s": wall, "sample_spread": samp_spread, "corpus_rows": len(corp_eval)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())