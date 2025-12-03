import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_vm_to_df
from src.data.preprocessing import select_and_prepare
from src.evaluation.evaluate import compute_metrics, plot_predictions, plot_residuals
from src.utils.seed import set_seed


def create_sequences(features, target, window_size):
    X, y = [], []
    for idx in range(window_size, len(features)):
        X.append(features[idx - window_size: idx])
        y.append(target[idx])
    if not X:
        return np.zeros((0, window_size, features.shape[1])), np.zeros((0, 1))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    i1 = int(n * train_ratio)
    i2 = i1 + int(n * val_ratio)
    return X[:i1], X[i1:i2], X[i2:], y[:i1], y[i1:i2], y[i2:]


def build_sequence_model(model_type, input_shape, params):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(name=model_type)
    if model_type == "stacked_lstm" or model_type == "hybrid_lstm_rf":
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.LSTM(params["units1"], return_sequences=True))
        model.add(tf.keras.layers.Dropout(params["dropout"]))
        model.add(tf.keras.layers.LSTM(params["units2"]))
    elif model_type == "gru":
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.GRU(params["units1"], return_sequences=True))
        model.add(tf.keras.layers.Dropout(params["dropout"]))
        model.add(tf.keras.layers.GRU(params["units2"]))
    elif model_type == "bilstm":
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params["units1"], return_sequences=True)))
        model.add(tf.keras.layers.Dropout(params["dropout"]))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params["units2"])))
    else:
        raise ValueError(f"Unknown model type {model_type}")
    for units in params.get("dense_units", [64, 32]):
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dropout(params["dropout"]))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def evaluate_predictions(y_true_scaled, y_pred_scaled, target_scaler):
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    metrics = compute_metrics(y_true, y_pred)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    nrmse = float(metrics["rmse"] / (np.max(y_true) - np.min(y_true) + eps))
    metrics.update({"mape": mape, "nrmse": nrmse})
    return metrics, y_true, y_pred


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def random_configs(space, k):
    if len(space) <= k:
        return space
    return random.sample(space, k)


def train_with_config(model_type, params, data, patience):
    X_train, X_val, y_train, y_val = data
    model = build_sequence_model(model_type, X_train.shape[1:], params)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )
    val_pred = model.predict(X_val, batch_size=params["batch_size"], verbose=0)
    return model, val_pred, history


def retrain_best(model_type, params, X_train, y_train):
    model = build_sequence_model(model_type, X_train.shape[1:], params)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ]
    model.fit(
        X_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )
    return model


def run_experiments(args):
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    vm_path = raw_dir / args.vm
    if not vm_path.exists():
        raise FileNotFoundError(f"VM file not found: {vm_path}")

    df = select_and_prepare(load_vm_to_df(vm_path))
    feature_cols = ["cpu_pct", "mem_kb", "disk_write_kbps", "net_rx_kbps", "net_tx_kbps", "net_kbps"]
    df = df.dropna(subset=feature_cols)

    features = df[feature_cols].values
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    target_metrics = ["cpu_pct", "mem_kb", "net_kbps", "disk_write_kbps"]

    model_spaces = {
        "stacked_lstm": [
            {"units1": 128, "units2": 64, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 50, "dense_units": [64]},
            {"units1": 192, "units2": 96, "dropout": 0.25, "lr": 6e-4, "batch_size": 64, "epochs": 60, "dense_units": [128, 64]},
        ],
        "gru": [
            {"units1": 128, "units2": 64, "dropout": 0.25, "lr": 1e-3, "batch_size": 64, "epochs": 45, "dense_units": [64]},
            {"units1": 192, "units2": 96, "dropout": 0.3, "lr": 7e-4, "batch_size": 64, "epochs": 55, "dense_units": [96, 48]},
        ],
        "bilstm": [
            {"units1": 128, "units2": 64, "dropout": 0.25, "lr": 1e-3, "batch_size": 64, "epochs": 60, "dense_units": [64]},
            {"units1": 160, "units2": 80, "dropout": 0.3, "lr": 6e-4, "batch_size": 64, "epochs": 70, "dense_units": [128, 64]},
        ],
        "hybrid_lstm_rf": [
            {"units1": 200, "units2": 100, "dropout": 0.25, "lr": 8e-4, "batch_size": 64, "epochs": 60, "dense_units": [128, 64]},
            {"units1": 256, "units2": 128, "dropout": 0.3, "lr": 5e-4, "batch_size": 64, "epochs": 80, "dense_units": [128, 64]},
        ],
    }

    results = []
    plots_dir = ensure_dir(Path(args.output_dir) / "figures")
    ensure_dir(Path(args.output_dir) / "checkpoints")

    for window_size in args.window_sizes:
        for metric in target_metrics:
            target = df[metric].values.reshape(-1, 1)
            target_scaler = StandardScaler()
            target_scaled = target_scaler.fit_transform(target)

            X, y = create_sequences(features_scaled, target_scaled, window_size)
            if len(X) < 10:
                print(f"Skipping metric {metric} window {window_size} due to insufficient samples")
                continue
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
            if len(X_val) == 0 or len(X_test) == 0:
                print(f"Skipping metric {metric} window {window_size}: invalid split sizes.")
                continue

            trainval_data = (
                np.concatenate([X_train, X_val], axis=0),
                np.concatenate([y_train, y_val], axis=0),
            )

            for model_name, param_space in model_spaces.items():
                best = None
                for params in random_configs(param_space, args.max_configs):
                    model, val_pred, _ = train_with_config(
                        model_name,
                        params,
                        (X_train, X_val, y_train, y_val),
                        patience=args.patience,
                    )
                    metrics_val, _, _ = evaluate_predictions(y_val, val_pred, target_scaler)
                    score = metrics_val["r2"]
                    if best is None or score > best["score"]:
                        best = {
                            "params": params,
                            "score": score,
                            "metrics_val": metrics_val,
                        }

                if best is None:
                    continue

                combined_model = retrain_best(
                    model_name,
                    best["params"],
                    trainval_data[0],
                    trainval_data[1],
                )

                rf_model = None
                if model_name == "hybrid_lstm_rf":
                    combined_preds = combined_model.predict(trainval_data[0], verbose=0)
                    residuals = trainval_data[1] - combined_preds
                    rf_model = RandomForestRegressor(n_estimators=300, random_state=args.seed, n_jobs=-1)
                    rf_model.fit(trainval_data[0].reshape(len(trainval_data[0]), -1), residuals.ravel())

                test_pred = combined_model.predict(X_test, verbose=0)
                if rf_model is not None:
                    rf_adjustment = rf_model.predict(X_test.reshape(len(X_test), -1)).reshape(-1, 1)
                    test_pred = test_pred + rf_adjustment

                metrics_test, y_true, y_pred = evaluate_predictions(y_test, test_pred, target_scaler)

                model_dir = ensure_dir(Path(args.output_dir) / "figures" / f"{metric}" / model_name / f"ws_{window_size}")
                plot_predictions(
                    y_true,
                    y_pred,
                    str(model_dir / "pred_vs_actual.png"),
                    title=f"{metric.upper()} {model_name} ws={window_size}",
                )
                plot_residuals(
                    y_true,
                    y_pred,
                    str(model_dir / "residuals.png"),
                    title=f"{metric.upper()} Residuals {model_name} ws={window_size}",
                )

                result_row = {
                    "vm": args.vm,
                    "metric": metric,
                    "window_size": window_size,
                    "model": model_name,
                    "config": json.dumps(best["params"]),
                    "val_r2": best["metrics_val"]["r2"],
                    "mae": metrics_test["mae"],
                    "rmse": metrics_test["rmse"],
                    "r2": metrics_test["r2"],
                    "mape": metrics_test["mape"],
                    "nrmse": metrics_test["nrmse"],
                }
                results.append(result_row)
                print(f"[RESULT] {result_row}")

    if not results:
        print("No experiments completed. Check data availability and parameters.")
        return

    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / "improve_results.csv"
    summary_path = Path(args.output_dir) / "improve_summary.csv"
    report_path = Path(args.output_dir) / "lstm_improve_report.md"
    results_df.to_csv(results_path, index=False)

    best_rows = []
    for metric in results_df["metric"].unique():
        metric_df = results_df[results_df["metric"] == metric]
        best_row = metric_df.loc[metric_df["r2"].idxmax()]
        best_rows.append(best_row)
    pd.DataFrame(best_rows).to_csv(summary_path, index=False)

    with open(report_path, "w") as f:
        f.write("# LSTM Improvement Experiments\n\n")
        f.write(f"- VM: `{args.vm}`\n")
        f.write(f"- Window sizes: {args.window_sizes}\n")
        f.write(f"- Models evaluated: {', '.join(model_spaces.keys())}\n")
        f.write("\n## Top Configurations per Metric\n\n")
        f.write(pd.DataFrame(best_rows).to_markdown(index=False))
        f.write("\n\n## All Results\n\n")
        f.write(results_df.to_markdown(index=False))

    print(f"Saved detailed results to {results_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved report to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Improved multivariate forecasting experiments.")
    parser.add_argument("--raw-dir", default="data/raw", type=str)
    parser.add_argument("--vm", default="Materna-Trace-1/01.csv", type=str)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[20, 60, 120, 240])
    parser.add_argument("--output-dir", default="outputs/improve", type=str)
    parser.add_argument("--max-configs", default=2, type=int, help="Max hyperparameter configs per model.")
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())

