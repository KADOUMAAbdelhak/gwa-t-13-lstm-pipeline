import numpy as np
from typing import Tuple

def create_windows(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) arrays for a single univariate series using a stride of 1."""
    n = len(series)
    if n <= window_size:
        return np.empty((0, window_size, 1)), np.empty((0,1))
    X = []
    y = []
    for i in range(window_size, n):
        x = series[i-window_size:i]
        X.append(x.reshape(window_size, 1))
        y.append(series[i])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1,1)

def create_sliding_windows(data, window_size, step_size: int = 1):
    """
    Generic sliding-window helper used by notebooks. `step_size` defaults to 1 so
    existing calls that omit this argument remain valid.
    """
    if step_size <= 0:
        raise ValueError("step_size must be a positive integer")

    windows = []
    labels = []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        windows.append(data[start:end])
        labels.append(data[end])

    return np.array(windows), np.array(labels)

def preprocess_windows(data, window_size, step_size):
    data = data.values  # Convert DataFrame to numpy array
    windows, labels = create_sliding_windows(data, window_size, step_size)
    return windows, labels

def split_data(windows, labels, train_size=0.8):
    split_index = int(len(windows) * train_size)
    X_train, X_test = windows[:split_index], windows[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    return X_train, X_test, y_train, y_test