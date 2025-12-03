import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, inverse_fn=None):
    """
    Predict with `model` and compute MAE/RMSE/R².
    Optionally supply `inverse_fn` to convert scaled values back to original units before scoring.
    """
    predictions = model.predict(X_test)
    if inverse_fn:
        y_true_eval = inverse_fn(y_test)
        y_pred_eval = inverse_fn(predictions)
    else:
        y_true_eval = y_test
        y_pred_eval = predictions
    metrics = compute_metrics(y_true_eval, y_pred_eval)
    return {"predictions": predictions, "metrics": metrics}

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

def save_evaluation_results(results, output_path):
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

def plot_results(y_true, y_pred, output_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.legend()
    plt.title('Model Predictions vs True Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.savefig(output_path)
    plt.close()

def plot_predictions(y_true, y_pred, outpath, title=None, n_points=500):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), n_points)
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:n], label="true")
    plt.plot(y_pred[:n], label="pred", alpha=0.7)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_residuals(y_true, y_pred, outpath, title=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    res = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50, density=True, alpha=0.7)
    plt.xlabel("Residual")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()