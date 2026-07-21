#!/usr/bin/env python3
"""
Baseline Comparison for the FGCS manuscript
============================================
One-step-ahead, per-VM VM-resource prediction on the GWA-T-13 Materna trace.

This script is SELF-CONTAINED and LEAK-FREE by construction. It deliberately does
*not* reuse ``paper_pipeline.py`` / ``hybrid_pipeline.py`` because those scripts
contain a target-leakage bug (see the module docstring section "INTEGRITY NOTES"
below and the accompanying report). Every model here is evaluated under one common
protocol so the numbers are directly comparable.

--------------------------------------------------------------------------------
PROTOCOL (identical for every model)
--------------------------------------------------------------------------------
* Task:            one-step-ahead forecast, predict y(t+1) from information <= t.
* Split:           chronological 70% train / 15% validation / 15% test (no shuffle).
* Window:          W = 96 samples (8 h at 5-min sampling) of history.
* Target vars:     CPU %, Memory %, Disk write KB/s, Net Rx KB/s, Net Tx KB/s.
* Metrics:         R^2, RMSE, MAE, MAPE(%).
* Scaling:         RobustScaler fit on TRAIN ONLY; feature clip bounds from TRAIN ONLY.
* Feature usage:
    - Classical ML (Linear, Ridge, SVR, RF, XGBoost, LightGBM, Hybrid):
      same engineered feature set built strictly from y(<=t) plus calendar
      features (lags, rolling stats, differences, quantiles for bursty metrics,
      cyclical time encodings). The current value y(t) is included as a feature
      (this is legitimate for predicting y(t+1) and is exactly what Persistence uses).
    - Persistence:  y_hat(t+1) = y(t) (last observed value).
    - LSTM:         windowed raw target sequence, sequence length = W = 96,
      min-max normalised with TRAIN statistics.
    - ARIMA:        SARIMAX on the univariate target history only, rolling
      one-step-ahead (fit on train, then append observed values without refit).

--------------------------------------------------------------------------------
INTEGRITY NOTES (why these numbers differ from the manuscript's Table 1)
--------------------------------------------------------------------------------
The manuscript reports R^2 ~ 0.90-0.96. Those values are reproducible ONLY when the
target is left UNSHIFTED (predicting y(t)) while the feature set contains
``diff_1 = y(t) - y(t-1)`` together with ``lag_1 = y(t-1)``. Their sum equals y(t)
exactly, so the model can reconstruct the target: this is target leakage, not
forecasting skill. Rolling statistics computed over a window ending at t leak the
target in the same way. Under the leak-free protocol above the honest scores are
much lower. This script reports only leak-free numbers.

Usage:
    python baseline_comparison.py
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import Any, Callable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

import xgboost as xgb

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping

    tf.get_logger().setLevel("ERROR")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG: dict[str, Any] = {
    "input_file": "dataset.csv",
    "output_dir": "baseline_comparison_results",
    "random_state": 42,
    # Chronological split fractions (match the manuscript: 70/15/15).
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "window_W": 96,  # history window in samples (8 h @ 5-min); LSTM sequence length.
    "target_variables": [
        "CPU usage [%]",
        "Memory usage [%]",
        "Disk write throughput [KB/s]",
        "Network received throughput [KB/s]",
        "Network transmitted throughput [KB/s]",
    ],
    # Proposed hybrid: metric -> component model family.
    "hybrid_assignments": {
        "CPU usage [%]": "random_forest",
        "Memory usage [%]": "random_forest",
        "Disk write throughput [KB/s]": "xgboost",
        "Network received throughput [KB/s]": "xgboost",
        "Network transmitted throughput [KB/s]": "xgboost",
    },
    "lag_features": [1, 2, 3, 6, 12, 24, 48, 96],
    "rolling_windows": [12, 24, 48, 96],
    "diff_lags": [1, 12, 24],
    # Metrics whose distributions are bursty/heavy-tailed get extra quantile features.
    "bursty_metrics": [
        "Disk write throughput [KB/s]",
        "Network received throughput [KB/s]",
        "Network transmitted throughput [KB/s]",
    ],
    "quantile_windows": [24, 48, 96],
}

OUTPUT_DIR = CONFIG["output_dir"]
RANDOM_STATE = CONFIG["random_state"]
TARGET_VARIABLES = CONFIG["target_variables"]
HYBRID_ASSIGNMENTS = CONFIG["hybrid_assignments"]
W = CONFIG["window_W"]

# Row order for the output tables.
MODEL_ORDER = [
    "Persistence",
    "Linear Regression",
    "Ridge Regression",
    "SVR",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "ARIMA",
    "Hybrid RF-XGBoost",
]

PAPER_CLAIMS = {
    "window_size_W": 96,
    "train_val_test_split": "70% / 15% / 15%",
    "approx_feature_count": 247,
    "dataset_scope_claim": ">1500 VMs across 3 Materna traces",
    "hybrid_R2_claim": {
        "CPU usage [%]": 0.9416,
        "Memory usage [%]": 0.8955,
        "Disk write throughput [KB/s]": 0.9568,
        "Network received throughput [KB/s]": 0.9221,
        "Network transmitted throughput [KB/s]": 0.9071,
    },
}

HYPERPARAMETERS = {
    "Persistence": {"rule": "y_hat(t+1) = y(t)"},
    "Linear Regression": {"fit_intercept": True},
    "Ridge Regression": {"alpha": 1.0},
    "SVR": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "XGBoost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
    },
    "LightGBM": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
    },
    "LSTM": {
        "sequence_length": W,
        "architecture": "LSTM(64,return_seq)->Dropout(0.2)->LSTM(32)->Dropout(0.2)->Dense(16,relu)->Dense(1)",
        "optimizer": "Adam(lr=1e-3)",
        "loss": "mse",
        "epochs": 50,
        "batch_size": 64,
        "early_stopping_patience": 10,
        "normalisation": "min-max with train statistics",
    },
    "ARIMA": {
        "model": "SARIMAX",
        "order": [2, 0, 2],
        "mode": "rolling one-step-ahead, fit on train then append observed (no refit)",
    },
    "Hybrid RF-XGBoost": HYBRID_ASSIGNMENTS,
}


# ============================================================================
# LOGGING
# ============================================================================

def log_line(message: str, log_lines: list[str]) -> None:
    print(message)
    log_lines.append(message)


def get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "pandas", "sklearn", "scipy", "xgboost", "lightgbm", "statsmodels", "tensorflow"]:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load the semicolon-separated, decimal-comma Materna VM trace."""
    df = None
    for sep in [";", ";\t", "\t", ","]:
        try:
            candidate = pd.read_csv(CONFIG["input_file"], sep=sep, decimal=",", engine="python")
            if candidate.shape[1] > 5:
                df = candidate
                break
        except Exception:
            continue
    if df is None:
        raise RuntimeError("Could not parse the input CSV file")

    df.columns = df.columns.str.strip().str.replace('"', "")

    timestamp_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    try:
        df["Timestamp"] = pd.to_datetime(df[timestamp_col], format="%d.%m.%Y %H:%M:%S")
    except Exception:
        df["Timestamp"] = pd.to_datetime(df[timestamp_col])
    if timestamp_col != "Timestamp":
        df = df.drop(columns=[timestamp_col])

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df.set_index("Timestamp", inplace=True)
    df = df.ffill().bfill()
    return df


# ============================================================================
# LEAK-FREE FEATURE ENGINEERING (all features depend only on y(<=t))
# ============================================================================

def build_supervised(df: pd.DataFrame, target_var: str) -> dict[str, Any]:
    """Construct a leak-free one-step-ahead supervised dataset for ``target_var``.

    Target is y(t+1). Every feature is a function of values observed up to and
    including time t, plus calendar features (which are known ahead of time).
    """
    y = df[target_var].astype(float)
    feat = pd.DataFrame(index=df.index)

    # Current observed value (t): the persistence predictor and strongest feature.
    feat[f"{target_var}__cur"] = y

    # Lag features y(t-l).
    for lag in CONFIG["lag_features"]:
        feat[f"{target_var}__lag_{lag}"] = y.shift(lag)

    # Rolling statistics over windows ending at t (use y(<=t): leak-free for y(t+1)).
    for win in CONFIG["rolling_windows"]:
        roll = y.rolling(window=win)
        feat[f"{target_var}__roll_mean_{win}"] = roll.mean()
        feat[f"{target_var}__roll_std_{win}"] = roll.std()
        feat[f"{target_var}__roll_min_{win}"] = roll.min()
        feat[f"{target_var}__roll_max_{win}"] = roll.max()
        feat[f"{target_var}__roll_range_{win}"] = roll.max() - roll.min()
        feat[f"{target_var}__roll_median_{win}"] = roll.median()
        q75 = roll.quantile(0.75)
        q25 = roll.quantile(0.25)
        feat[f"{target_var}__roll_iqr_{win}"] = q75 - q25

    # Difference features (momentum), all from values <= t.
    for d in CONFIG["diff_lags"]:
        feat[f"{target_var}__diff_{d}"] = y.diff(d)

    # Quantile / tail features for bursty metrics.
    if target_var in CONFIG["bursty_metrics"]:
        for win in CONFIG["quantile_windows"]:
            roll = y.rolling(window=win)
            feat[f"{target_var}__q90_{win}"] = roll.quantile(0.90)
            feat[f"{target_var}__q95_{win}"] = roll.quantile(0.95)

    # Calendar features (cyclical + binary indicators). Known in advance.
    idx = df.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feat["is_business_hours"] = ((idx.hour >= 9) & (idx.hour < 18)).astype(int)
    feat["is_night"] = ((idx.hour >= 22) | (idx.hour < 6)).astype(int)

    # One-step-ahead target.
    target_next = y.shift(-1)
    persist = y  # y(t) is the persistence forecast of y(t+1).

    combined = feat.copy()
    combined["__target__"] = target_next
    combined["__persist__"] = persist
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # Drop warm-up rows (NaNs from lags/rolling) and the final row (no y(t+1)).
    combined = combined.dropna()

    feature_cols = [c for c in feat.columns]
    X = combined[feature_cols].copy()
    y_next = combined["__target__"].to_numpy(dtype=float)
    persist_all = combined["__persist__"].to_numpy(dtype=float)
    raw_target_hist = combined["__persist__"].to_numpy(dtype=float)  # y(t) aligned to rows

    return {
        "X": X,
        "y": y_next,
        "persist": persist_all,
        "raw_hist": raw_target_hist,
        "feature_cols": feature_cols,
        "index": combined.index,
    }


def chronological_split(n: int) -> tuple[int, int]:
    train_end = int(n * CONFIG["train_frac"])
    val_end = int(n * (CONFIG["train_frac"] + CONFIG["val_frac"]))
    return train_end, val_end


def prepare_classical(sup: dict[str, Any]) -> dict[str, Any]:
    """Split, clip (train-only bounds) and scale (train-only fit) the feature matrix."""
    X = sup["X"].copy()
    y = sup["y"]
    persist = sup["persist"]
    n = len(X)
    train_end, val_end = chronological_split(n)

    # Clip extreme feature values using TRAIN quantiles only (leak-free).
    X_train_raw = X.iloc[:train_end]
    lo = X_train_raw.quantile(0.001)
    hi = X_train_raw.quantile(0.999)
    X = X.clip(lower=lo, upper=hi, axis=1)
    X = X.fillna(X_train_raw.median())

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X.iloc[:train_end])
    X_val = scaler.transform(X.iloc[train_end:val_end])
    X_test = scaler.transform(X.iloc[val_end:])

    return {
        "X_train": X_train,
        "y_train": y[:train_end],
        "X_val": X_val,
        "y_val": y[train_end:val_end],
        "X_test": X_test,
        "y_test": y[val_end:],
        "persist_test": persist[val_end:],
        "train_end": train_end,
        "val_end": val_end,
        "n": n,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


# ============================================================================
# MODELS
# ============================================================================

def m_persistence(data: dict[str, Any]) -> np.ndarray:
    return data["persist_test"]


def m_linear(data: dict[str, Any]) -> np.ndarray:
    model = LinearRegression()
    model.fit(data["X_train"], data["y_train"])
    return model.predict(data["X_test"])


def m_ridge(data: dict[str, Any]) -> np.ndarray:
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(data["X_train"], data["y_train"])
    return model.predict(data["X_test"])


def m_svr(data: dict[str, Any]) -> np.ndarray:
    model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    model.fit(data["X_train"], data["y_train"])
    return model.predict(data["X_test"])


def m_random_forest(data: dict[str, Any]) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])
    return model.predict(data["X_test"])


def m_xgboost(data: dict[str, Any]) -> np.ndarray:
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(data["X_train"], data["y_train"], eval_set=[(data["X_val"], data["y_val"])], verbose=False)
    return model.predict(data["X_test"])


def m_lightgbm(data: dict[str, Any]) -> np.ndarray:
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        data["X_train"], data["y_train"], eval_set=[(data["X_val"], data["y_val"])],
        eval_metric="l2", callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.predict(data["X_test"])


def m_hybrid(data: dict[str, Any], target_var: str) -> np.ndarray:
    if HYBRID_ASSIGNMENTS[target_var] == "random_forest":
        return m_random_forest(data)
    return m_xgboost(data)


def m_lstm(sup: dict[str, Any], data: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    """Windowed raw-sequence LSTM; predicts y(t+1) from the last W raw values."""
    if not TENSORFLOW_AVAILABLE:
        return None, "not run (TensorFlow not installed)"

    raw = sup["raw_hist"]  # y(t) aligned to supervised rows
    y_next = sup["y"]      # y(t+1)
    n = len(raw)
    seq_len = W
    if n <= seq_len + 10:
        return None, "not run (series too short for sequence length)"

    seqs, tgts, row_idx = [], [], []
    for i in range(seq_len - 1, n):
        seqs.append(raw[i - seq_len + 1: i + 1])
        tgts.append(y_next[i])
        row_idx.append(i)
    Xs = np.asarray(seqs, dtype=float)[..., np.newaxis]
    ys = np.asarray(tgts, dtype=float)
    row_idx = np.asarray(row_idx)

    train_end = data["train_end"]
    val_end = data["val_end"]
    tr = row_idx < train_end
    va = (row_idx >= train_end) & (row_idx < val_end)
    te = row_idx >= val_end
    if te.sum() != len(data["y_test"]):
        # keep alignment strict; report if mismatch
        pass
    if te.sum() == 0:
        return None, "not run (no test samples after windowing)"

    ymin = float(ys[tr].min())
    ymax = float(ys[tr].max())
    denom = max(ymax - ymin, 1e-8)
    scale = lambda a: (a - ymin) / denom
    unscale = lambda a: a * denom + ymin

    keras.utils.set_random_seed(RANDOM_STATE)
    model = keras.Sequential([
        layers.Input(shape=(seq_len, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    model.fit(
        scale(Xs[tr]), scale(ys[tr]),
        validation_data=(scale(Xs[va]), scale(ys[va])),
        epochs=50, batch_size=64, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )
    pred = unscale(model.predict(scale(Xs[te]), verbose=0).flatten())
    return pred, None


def m_arima(sup: dict[str, Any], data: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    """SARIMAX on univariate target history, rolling one-step-ahead over the test span."""
    if not STATSMODELS_AVAILABLE:
        return None, "not run (statsmodels not installed)"

    raw = sup["raw_hist"]
    n = len(raw)
    train_end = data["train_end"]
    val_end = data["val_end"]
    if n < 60 or val_end >= n:
        return None, "not run (insufficient data for ARIMA)"
    try:
        # Fit on train+val history, then roll through test appending observed values.
        history_end = val_end
        model = SARIMAX(raw[:history_end], order=(2, 0, 2),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=200)
        preds = []
        for i in range(val_end, n):
            fc = float(np.asarray(fit.forecast(steps=1)).ravel()[0])
            preds.append(fc)
            fit = fit.append([raw[i]], refit=False)
        return np.asarray(preds, dtype=float), None
    except Exception as exc:  # noqa: BLE001
        return None, f"failed ({exc})"


# ============================================================================
# TABLE WRITERS
# ============================================================================

def save_markdown_table(df: pd.DataFrame, path: str, title: str) -> None:
    lines = [f"# {title}", ""]
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[h]) else str(row[h]) for h in headers) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fmt(val: float, digits: int) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return f"{val:.{digits}f}"


def save_baseline_latex(df: pd.DataFrame, path: str) -> None:
    lines = [
        "% Auto-generated by baseline_comparison.py -- leak-free one-step-ahead results.",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Leak-free baseline comparison for one-step-ahead, per-VM resource",
        "prediction on the GWA-T-13 Materna trace (chronological 70/15/15 split, $W=96$).}",
        "\\label{tab:baseline_comparison}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\footnotesize",
        "\\begin{tabular}{l l r r r r}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Model} & \\textbf{$R^2$} & \\textbf{RMSE} & "
        "\\textbf{MAE} & \\textbf{MAPE (\\%)} \\\\",
        "\\hline",
    ]
    current_metric = None
    for _, row in df.iterrows():
        metric = str(row["Metric"])
        if current_metric is not None and metric != current_metric:
            lines.append("\\hline")
        metric_tex = metric.replace("%", "\\%").replace("_", "\\_") if metric != current_metric else ""
        current_metric = metric
        model = str(row["Model"])
        if row["Status"] != "ok":
            lines.append(f"{metric_tex} & {model} & \\multicolumn{{4}}{{l}}{{{row['Status']}}} \\\\")
        else:
            lines.append(
                f"{metric_tex} & {model} & {row['R2']:.4f} & {row['RMSE']:.2f} & "
                f"{row['MAE']:.2f} & {row['MAPE (%)']:.1f} \\\\"
            )
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_best_model_latex(df: pd.DataFrame, path: str) -> None:
    lines = [
        "% Auto-generated by baseline_comparison.py -- best baseline vs proposed hybrid.",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Best baseline vs.\\ proposed hybrid RF--XGBoost per metric (leak-free $R^2$).}",
        "\\label{tab:best_model_summary}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\footnotesize",
        "\\begin{tabular}{l l r l r r}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Best baseline} & \\textbf{$R^2$} & "
        "\\textbf{Proposed} & \\textbf{$R^2$} & \\textbf{Gain} \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        metric_tex = str(row["Metric"]).replace("%", "\\%")
        lines.append(
            f"{metric_tex} & {row['Best baseline']} & {row['Best baseline R2']:.4f} & "
            f"{row['Proposed model']} & {row['Proposed R2']:.4f} & {row['Gain']:+.4f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_best_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = results_df[results_df["Status"] == "ok"]
    baselines = ok[ok["Model"] != "Hybrid RF-XGBoost"]
    hybrid = ok[ok["Model"] == "Hybrid RF-XGBoost"]
    for metric in TARGET_VARIABLES:
        mb = baselines[baselines["Metric"] == metric]
        mh = hybrid[hybrid["Metric"] == metric]
        if mb.empty or mh.empty:
            continue
        best = mb.loc[mb["R2"].idxmax()]
        hyb = mh.iloc[0]
        best_overall = ok[ok["Metric"] == metric]["R2"].max()
        rows.append({
            "Metric": metric,
            "Best baseline": best["Model"],
            "Best baseline R2": round(float(best["R2"]), 4),
            "Proposed model": "Hybrid RF-XGBoost",
            "Proposed R2": round(float(hyb["R2"]), 4),
            "Gain": round(float(hyb["R2"] - best["R2"]), 4),
            "Hybrid is best overall": bool(hyb["R2"] >= best_overall - 1e-9),
        })
    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines: list[str] = []
    runtimes: dict[str, dict[str, float]] = {}
    feature_counts: dict[str, int] = {}
    split_sizes: dict[str, dict[str, int]] = {}

    np.random.seed(RANDOM_STATE)

    log_line("=" * 80, log_lines)
    log_line("LEAK-FREE BASELINE COMPARISON EXPERIMENT", log_lines)
    log_line(f"Started: {datetime.now().isoformat(timespec='seconds')}", log_lines)
    log_line("=" * 80, log_lines)

    dataset_path = os.path.abspath(CONFIG["input_file"])
    log_line(f"Dataset path: {dataset_path}", log_lines)
    log_line(f"Random seed: {RANDOM_STATE}", log_lines)
    log_line(f"Split: {int(CONFIG['train_frac']*100)}/{int(CONFIG['val_frac']*100)}/"
             f"{int(CONFIG['test_frac']*100)} (chronological), window W={W}", log_lines)

    df = load_data()
    log_line(f"Raw samples: {len(df)}  (single VM series)", log_lines)
    log_line(f"Time range: {df.index.min()} -> {df.index.max()}", log_lines)
    log_line("NOTE: only one VM CSV (dataset.csv) is available; the manuscript's "
             "claim of >1500 VMs / 3 traces is NOT reflected in the runnable data.", log_lines)

    all_rows: list[dict[str, Any]] = []

    for target_var in TARGET_VARIABLES:
        log_line("\n" + "-" * 66, log_lines)
        log_line(f"Target: {target_var}", log_lines)
        log_line("-" * 66, log_lines)

        sup = build_supervised(df, target_var)
        data = prepare_classical(sup)

        feature_counts[target_var] = len(sup["feature_cols"])
        split_sizes[target_var] = {
            "supervised_rows": data["n"],
            "train": data["train_end"],
            "validation": data["val_end"] - data["train_end"],
            "test": data["n"] - data["val_end"],
        }
        log_line(f"Supervised rows: {data['n']}  "
                 f"(train {split_sizes[target_var]['train']}, "
                 f"val {split_sizes[target_var]['validation']}, "
                 f"test {split_sizes[target_var]['test']})", log_lines)
        log_line(f"Engineered features (classical ML): {feature_counts[target_var]}", log_lines)
        log_line("Target alignment: predict y(t+1) from information <= t (leak-free)", log_lines)

        runtimes[target_var] = {}
        y_test = data["y_test"]

        runners: list[tuple[str, Callable[[], Any], bool]] = [
            ("Persistence", lambda: m_persistence(data), False),
            ("Linear Regression", lambda: m_linear(data), False),
            ("Ridge Regression", lambda: m_ridge(data), False),
            ("SVR", lambda: m_svr(data), False),
            ("Random Forest", lambda: m_random_forest(data), False),
            ("XGBoost", lambda: m_xgboost(data), False),
        ]
        if LIGHTGBM_AVAILABLE:
            runners.append(("LightGBM", lambda: m_lightgbm(data), False))
        else:
            all_rows.append(_skip_row(target_var, "LightGBM", "not run (LightGBM not installed)"))
        runners.append(("LSTM", lambda: m_lstm(sup, data), True))
        runners.append(("ARIMA", lambda: m_arima(sup, data), True))
        runners.append(("Hybrid RF-XGBoost", lambda: m_hybrid(data, target_var), False))

        for model_name, runner, returns_tuple in runners:
            log_line(f"  Running {model_name} ...", log_lines)
            start = time.perf_counter()
            status = "ok"
            y_pred: np.ndarray | None = None
            try:
                out = runner()
                if returns_tuple:
                    y_pred, err = out
                    if err is not None:
                        status = err
                else:
                    y_pred = out
                elapsed = time.perf_counter() - start
                runtimes[target_var][model_name] = round(elapsed, 3)

                if y_pred is None or status != "ok":
                    log_line(f"    -> {status}", log_lines)
                    all_rows.append(_skip_row(target_var, model_name, status))
                    continue

                y_pred = np.asarray(y_pred, dtype=float)
                if len(y_pred) != len(y_test):
                    status = f"failed (pred len {len(y_pred)} != test len {len(y_test)})"
                    log_line(f"    -> {status}", log_lines)
                    all_rows.append(_skip_row(target_var, model_name, status))
                    continue

                metrics = evaluate(y_test, y_pred)
                log_line(f"    R2={metrics['r2']:+.4f}  RMSE={metrics['rmse']:.2f}  "
                         f"MAE={metrics['mae']:.2f}  MAPE={metrics['mape']:.1f}%  ({elapsed:.1f}s)", log_lines)
                all_rows.append({
                    "Metric": target_var, "Model": model_name,
                    "R2": metrics["r2"], "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"], "MAPE (%)": metrics["mape"], "Status": "ok",
                })
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - start
                runtimes[target_var][model_name] = round(elapsed, 3)
                status = f"failed ({exc})"
                log_line(f"    -> {status}", log_lines)
                log_lines.append(traceback.format_exc())
                all_rows.append(_skip_row(target_var, model_name, status))

    results_df = pd.DataFrame(all_rows)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    metric_order = {m: i for i, m in enumerate(TARGET_VARIABLES)}
    results_df["_mo"] = results_df["Model"].map(order)
    results_df["_to"] = results_df["Metric"].map(metric_order)
    results_df = results_df.sort_values(["_to", "_mo"]).drop(columns=["_mo", "_to"]).reset_index(drop=True)

    best_df = build_best_summary(results_df)

    # ---- write CSVs ----
    display_df = results_df.copy()
    for col, dg in [("R2", 4), ("RMSE", 2), ("MAE", 2), ("MAPE (%)", 1)]:
        display_df[col] = display_df[col].map(lambda v, d=dg: fmt(v, d))
    display_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_metrics.csv"), index=False)
    best_df.to_csv(os.path.join(OUTPUT_DIR, "best_model_summary.csv"), index=False)

    # ---- write markdown ----
    save_markdown_table(display_df, os.path.join(OUTPUT_DIR, "baseline_metrics.md"),
                        "Baseline Comparison (leak-free one-step-ahead)")
    save_markdown_table(best_df, os.path.join(OUTPUT_DIR, "best_model_summary.md"),
                        "Best Baseline vs Proposed Hybrid per Metric")

    # ---- write latex ----
    save_baseline_latex(results_df, os.path.join(OUTPUT_DIR, "baseline_metrics_latex.tex"))
    save_best_model_latex(best_df, os.path.join(OUTPUT_DIR, "best_model_summary_latex.tex"))

    # ---- experiment config ----
    experiment_config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": dataset_path,
        "dataset_scope": "single VM CSV (dataset.csv); no multi-VM / 3-trace data present",
        "raw_samples": int(len(df)),
        "random_seed": RANDOM_STATE,
        "protocol": {
            "task": "one-step-ahead, predict y(t+1) from information <= t",
            "split": f"{int(CONFIG['train_frac']*100)}% train / "
                     f"{int(CONFIG['val_frac']*100)}% val / {int(CONFIG['test_frac']*100)}% test (chronological)",
            "window_W": W,
            "scaler": "RobustScaler (fit on train only)",
            "feature_clip": "0.1%/99.9% bounds from train only",
            "leakage_control": "all classical features are functions of y(<=t); "
                               "no feature encodes y(t+1); calendar features are known ahead.",
            "mape_definition": "100 * mean(|(y - y_hat)/y|) over y != 0",
        },
        "evaluation_metrics": ["R2", "RMSE", "MAE", "MAPE(%)"],
        "target_variables": TARGET_VARIABLES,
        "hybrid_model_assignments": HYBRID_ASSIGNMENTS,
        "hyperparameters": HYPERPARAMETERS,
        "feature_counts_per_metric": feature_counts,
        "split_sizes_per_metric": split_sizes,
        "runtime_seconds_per_model": runtimes,
        "models_attempted": MODEL_ORDER,
        "availability": {
            "lightgbm": LIGHTGBM_AVAILABLE,
            "statsmodels_arima": STATSMODELS_AVAILABLE,
            "tensorflow_lstm": TENSORFLOW_AVAILABLE,
        },
        "package_versions": get_package_versions(),
        "paper_claims": PAPER_CLAIMS,
        "paper_vs_code_inconsistencies": {
            "split": {"paper": "70/15/15", "old_readme": "70/10/20",
                      "this_script": "70/15/15 (matched to paper)"},
            "window_W": {"paper": 96, "old_readme_lstm_seq_len": 24,
                         "this_script": 96},
            "feature_count": {"paper": "~247", "this_script_actual": feature_counts,
                              "note": "actual engineered feature count is far below 247"},
            "dataset_scope": {"paper": ">1500 VMs / 3 traces",
                              "this_script": "1 VM (dataset.csv) only"},
            "headline_R2": {"paper": "~0.90-0.96",
                            "cause": "reproducible only with target leakage "
                                     "(diff_1 + lag_1 == y(t) for an unshifted target)",
                            "this_script": "leak-free numbers, see baseline_metrics.csv"},
        },
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)

    log_line("\n" + "=" * 80, log_lines)
    log_line("EXPERIMENT COMPLETE", log_lines)
    log_line(f"Finished: {datetime.now().isoformat(timespec='seconds')}", log_lines)
    log_line("=" * 80, log_lines)
    with open(os.path.join(OUTPUT_DIR, "baseline_run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nResults written to {OUTPUT_DIR}/")


def _skip_row(metric: str, model: str, status: str) -> dict[str, Any]:
    return {"Metric": metric, "Model": model, "R2": np.nan, "RMSE": np.nan,
            "MAE": np.nan, "MAPE (%)": np.nan, "Status": status}


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
