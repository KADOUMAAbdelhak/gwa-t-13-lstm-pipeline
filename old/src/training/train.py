import argparse
import csv
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.data.load_data import list_vm_files, load_vm_to_df
from src.data.preprocessing import normalize_df, select_and_prepare
from src.data.windowing import create_windows
from src.evaluation.evaluate import compute_metrics, plot_predictions, plot_residuals
from src.models.baseline import (
    flatten_windows,
    last_value_predict,
    train_gradient_boosting,
    train_linear_regression,
    train_random_forest,
)
from src.models.lstm_model import build_lstm, get_early_stopper
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger("gwa_t13.train")

def split_indices(n, train_ratio=0.7, val_ratio=0.15):
    i1 = int(n * train_ratio)
    i2 = i1 + int(n * val_ratio)
    return i1, i2

def ensure_dirs(cfg):
    cfg["output"]["results_dir"].mkdir(parents=True, exist_ok=True)
    cfg["output"]["figures_dir"].mkdir(parents=True, exist_ok=True)
    cfg["output"]["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    cfg["output"]["logs_dir"].mkdir(parents=True, exist_ok=True)

def filter_vm_files(vm_files, trace_name, whitelist):
    if not whitelist:
        return vm_files
    allowed = set(whitelist)
    filtered = []
    for vm_path in vm_files:
        rel = f"{trace_name}/{vm_path.name}"
        if rel in allowed or vm_path.name in allowed:
            filtered.append(vm_path)
    return filtered

def run(cfg):
    set_seed(cfg["training"]["seed"])
    # device handling
    if cfg["training"].get("device","gpu") == "gpu":
        # allow growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    ensure_dirs(cfg)

    window_sizes = cfg["training"].get("window_sizes") or [cfg["training"]["window_size"]]
    for ws in window_sizes:
        LOGGER.info("=== Starting run for window size %s ===", ws)
        _run_single_window(cfg, ws)


def _run_single_window(cfg, window_size):
    root = Path(cfg["data"]["raw_dir"])
    traces = cfg["data"]["traces"]
    vm_whitelist = cfg["training"].get("vm_whitelist") or []
    results_rows = []
    per_vm_rows = []

    for trace in traces:
        trace_dir = root / trace
        if not trace_dir.exists():
            LOGGER.warning("Trace directory missing: %s", trace_dir)
            continue
        vm_files = filter_vm_files(list_vm_files(trace_dir), trace, vm_whitelist)
        if not vm_files:
            LOGGER.info("No VM files selected for trace %s (window=%s)", trace, window_size)
            continue
        LOGGER.info("Processing trace %s (%d VM CSV files)", trace, len(vm_files))
        trace_metrics = {"cpu":[], "mem":[], "net":[], "disk":[]}
        for vmfile in vm_files:
            try:
                raw = load_vm_to_df(vmfile)
                df = select_and_prepare(raw)
                if df.shape[0] <= window_size + 5:
                    # skip too short series
                    LOGGER.debug("Skipping %s (trace %s) due to insufficient length", vmfile.name, trace)
                    continue
                scaled_df, scalers = normalize_df(df, scaler_type="standard")
                # per metric loop
                for col, metric_name in [("cpu_pct","cpu"), ("mem_kb","mem"), ("net_kbps","net"), ("disk_write_kbps","disk")]:
                    series = scaled_df[col].values.ravel()
                    X, y = create_windows(series, window_size)
                    if X.shape[0] < 10:
                        LOGGER.debug("Skipping %s %s due to small window count", vmfile.name, metric_name)
                        continue
                    # chronological split
                    i1, i2 = split_indices(X.shape[0], 0.7, 0.15)
                    X_train, y_train = X[:i1], y[:i1]
                    X_val, y_val = X[i1:i2], y[i1:i2]
                    X_test, y_test = X[i2:], y[i2:]
                    if X_test.shape[0] == 0:
                        LOGGER.debug("Skipping %s %s due to empty test split", vmfile.name, metric_name)
                        continue
                    # Baselines
                    # Last value
                    y_pred_last = last_value_predict(X_test)
                    # Linear Regression
                    lr = train_linear_regression(X_train, y_train)
                    y_pred_lr = lr.predict(flatten_windows(X_test)).reshape(-1,1)
                    # Random Forest
                    rf = train_random_forest(X_train, y_train, n_estimators=100, random_state=cfg["training"]["seed"])
                    y_pred_rf = rf.predict(flatten_windows(X_test)).reshape(-1,1)
                    # Gradient Boosting
                    gb = train_gradient_boosting(X_train, y_train, random_state=cfg["training"]["seed"])
                    y_pred_gb = gb.predict(flatten_windows(X_test)).reshape(-1,1)

                    # LSTM
                    tf.keras.backend.clear_session()
                    model = build_lstm(input_shape=(window_size,1), cfg=cfg)
                    es = get_early_stopper(cfg["training"]["patience"])
                    model.fit(X_train, y_train, validation_data=(X_val,y_val),
                              epochs=cfg["training"]["epochs"],
                              batch_size=cfg["training"]["batch_size"],
                              callbacks=[es], verbose=0)
                    y_pred_lstm = model.predict(X_test, batch_size=cfg["training"]["batch_size"])

                    scaler = scalers[col]
                    scaler_dir = cfg["output"]["checkpoints_dir"] / f"ws_{window_size}" / "scalers" / trace
                    scaler_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(scaler, scaler_dir / f"{vmfile.stem}_{metric_name}_scaler.joblib")

                    def invert(arr):
                        return scaler.inverse_transform(arr)

                    y_test_orig = invert(y_test)
                    y_pred_last_orig = invert(y_pred_last)
                    y_pred_lr_orig = invert(y_pred_lr)
                    y_pred_rf_orig = invert(y_pred_rf)
                    y_pred_gb_orig = invert(y_pred_gb)
                    y_pred_lstm_orig = invert(y_pred_lstm)

                    m_last = compute_metrics(y_test_orig, y_pred_last_orig)
                    m_lr = compute_metrics(y_test_orig, y_pred_lr_orig)
                    m_rf = compute_metrics(y_test_orig, y_pred_rf_orig)
                    m_gb = compute_metrics(y_test_orig, y_pred_gb_orig)
                    m_lstm = compute_metrics(y_test_orig, y_pred_lstm_orig)

                    if cfg["training"].get("save_lstm_models"):
                        model_dir = cfg["output"]["checkpoints_dir"] / f"ws_{window_size}" / "models" / trace / metric_name
                        model_dir.mkdir(parents=True, exist_ok=True)
                        model.save(model_dir / f"{vmfile.stem}_lstm.keras")

                    # Save per-vm metric row (values are now in original units)
                    row = {
                        "trace": trace,
                        "vm": vmfile.name,
                        "metric": metric_name,
                        "window_size": window_size,
                        "last_mae": m_last["mae"], "last_rmse": m_last["rmse"], "last_r2": m_last["r2"],
                        "lr_mae": m_lr["mae"], "lr_rmse": m_lr["rmse"], "lr_r2": m_lr["r2"],
                        "rf_mae": m_rf["mae"], "rf_rmse": m_rf["rmse"], "rf_r2": m_rf["r2"],
                        "gb_mae": m_gb["mae"], "gb_rmse": m_gb["rmse"], "gb_r2": m_gb["r2"],
                        "lstm_mae": m_lstm["mae"], "lstm_rmse": m_lstm["rmse"], "lstm_r2": m_lstm["r2"],
                    }
                    per_vm_rows.append(row)
                    # accumulate for trace-level
                    trace_metrics[metric_name].append({
                        "last": m_last, "lr": m_lr, "rf": m_rf, "gb": m_gb, "lstm": m_lstm
                    })

                    # Save a sample figure for this VM/metric (first metric only to save time)
                    figbase = cfg["output"]["figures_dir"] / f"ws_{window_size}" / trace / metric_name
                    figbase.parent.mkdir(parents=True, exist_ok=True)
                    sample_pred_path = str(figbase / f"{vmfile.stem}_pred.png")
                    resid_path = str(figbase / f"{vmfile.stem}_resid.png")
                    # plot first 500 points
                    plot_predictions(y_test_orig, y_pred_lstm_orig, sample_pred_path, title=f"{trace} {vmfile.name} {metric_name} LSTM")
                    plot_residuals(y_test_orig, y_pred_lstm_orig, resid_path, title=f"{trace} {vmfile.name} {metric_name} resid")

                # end per-metric
            except Exception as e:
                LOGGER.exception("Error processing %s: %s", vmfile, e)
                continue
        # compute aggregated trace metrics
        for metric_name, list_metrics in trace_metrics.items():
            if not list_metrics:
                continue
            # compute mean MAE/RMSE/R2 for each model across VMs for this metric
            agg = {"trace": trace, "metric": metric_name, "window_size": window_size}
            models_list = ["last","lr","rf","gb","lstm"]
            for m in models_list:
                maes = [x[m]["mae"] for x in list_metrics]
                rmses = [x[m]["rmse"] for x in list_metrics]
                r2s = [x[m]["r2"] for x in list_metrics if not np.isnan(x[m]["r2"])]
                agg[f"{m}_mae_mean"] = float(np.mean(maes)) if maes else None
                agg[f"{m}_rmse_mean"] = float(np.mean(rmses)) if rmses else None
                agg[f"{m}_r2_mean"] = float(np.mean(r2s)) if r2s else None
            results_rows.append(agg)

    # Save CSVs
    suffix = f"_ws{window_size}"
    res_path = cfg["output"]["results_dir"] / f"metrics_summary{suffix}.csv"
    per_vm_path = cfg["output"]["results_dir"] / f"per_vm_metrics{suffix}.csv"
    # write metrics_summary
    if results_rows:
        keys = sorted(results_rows[0].keys())
        with open(res_path, "w", newline="") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            for r in results_rows:
                writer.writerow(r)
    # write per_vm_metrics
    if per_vm_rows:
        keys = sorted(per_vm_rows[0].keys())
        with open(per_vm_path, "w", newline="") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            for r in per_vm_rows:
                writer.writerow(r)

    LOGGER.info("Done (window=%s). Results in %s", window_size, cfg["output"]["results_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("config_path", "configs/logging.yaml"))
    run(cfg)