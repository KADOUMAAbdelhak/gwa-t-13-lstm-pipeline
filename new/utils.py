"""
Shared Utilities for VM Performance Prediction Pipelines
========================================================
This module contains common functions for data loading, preprocessing,
feature engineering, and evaluation that are shared across all model pipelines.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'sequence_length': 24,  # For LSTM/GRU
    'target_variables': [
        'CPU usage [%]',
        'Memory usage [%]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]'
    ]
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def log_message(message, log_file_path):
    """Write message to both console and log file"""
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def save_plot(filename, output_dir):
    """Save current plot to output directory"""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(input_file=None):
    """Load and preprocess the VM trace dataset"""
    if input_file is None:
        input_file = CONFIG['input_file']
    
    try:
        # Load CSV with semicolon delimiter
        df = pd.read_csv(input_file, sep=';', engine='python', quotechar='"')
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        sys.exit(1)
    
    # Clean column names (remove quotes and strip whitespace)
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Clean all string values (remove quotes)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()
    
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    
    # Convert numeric columns (handle comma as decimal separator)
    numeric_cols = [col for col in df.columns if col != 'Timestamp']
    for col in numeric_cols:
        # Convert to string, replace comma with dot, then to float
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float, errors='ignore')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.set_index('Timestamp', inplace=True)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    print(f"✓ Missing values handled: {missing_before} filled")
    
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_time_features(df):
    """Create time-based features"""
    df_feat = df.copy()
    
    # Time components
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['week_of_year'] = df_feat.index.isocalendar().week
    df_feat['month'] = df_feat.index.month
    
    # Cyclical encoding
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    
    # Weekend indicator
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    return df_feat

def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12]):
    """Create lag features"""
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    return df_lag

def create_rolling_features(df, target_col, windows=[6, 12, 24]):
    """Create rolling window statistics"""
    df_roll = df.copy()
    for window in windows:
        df_roll[f'{target_col}_rolling_mean_{window}'] = df_roll[target_col].rolling(window=window).mean()
        df_roll[f'{target_col}_rolling_std_{window}'] = df_roll[target_col].rolling(window=window).std()
    return df_roll

def engineer_features(df):
    """Complete feature engineering pipeline"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_engineered = create_time_features(df)
    print("✓ Time features created")
    
    # Domain-specific features
    df_engineered['cpu_utilization_rate'] = df_engineered['CPU usage [MHZ]'] / (df_engineered['CPU capacity provisioned [MHZ]'] + 1)
    df_engineered['memory_utilization_rate'] = df_engineered['Memory usage [KB]'] / df_engineered['Memory capacity provisioned [KB]']
    df_engineered['network_total'] = df_engineered['Network received throughput [KB/s]'] + df_engineered['Network transmitted throughput [KB/s]']
    print("✓ Domain features created")
    
    # Lag and rolling features for each target
    for target in CONFIG['target_variables']:
        df_engineered = create_lag_features(df_engineered, target, lags=[1, 2, 3, 6, 12])
        df_engineered = create_rolling_features(df_engineered, target, windows=[6, 12, 24])
    print("✓ Lag and rolling features created")
    
    # Remove NaN rows
    df_clean = df_engineered.dropna()
    print(f"✓ Dataset shape after feature engineering: {df_clean.shape}")
    print(f"  Features: {df_clean.shape[1]}")
    print(f"  Samples: {df_clean.shape[0]}")
    
    return df_clean

# ============================================================================
# DATA PREPARATION
# ============================================================================

def get_leakage_features(target_var):
    """Get list of leakage features to exclude for a given target variable"""
    leakage_features = []
    if target_var == 'Memory usage [%]':
        leakage_features = [
            'Memory usage [KB]',
            'Memory capacity provisioned [KB]',
            'memory_utilization_rate'
        ]
    elif target_var == 'CPU usage [%]':
        leakage_features = [
            'CPU usage [MHZ]',
            'CPU capacity provisioned [MHZ]',
            'cpu_utilization_rate'
        ]
    elif target_var in ['Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]']:
        leakage_features = ['network_total']
    return leakage_features

def prepare_ml_data(df, target_var):
    """Prepare data for traditional ML models"""
    # Exclude other target variables from features
    exclude_cols = [col for col in CONFIG['target_variables'] if col != target_var]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove leakage features
    leakage_features = get_leakage_features(target_var)
    for leak_feat in leakage_features:
        if leak_feat in feature_cols:
            feature_cols.remove(leak_feat)
    
    X = df[feature_cols].drop(columns=[target_var])
    y = df[target_var]
    
    # Chronological split
    n = len(df)
    train_end = int(n * (1 - CONFIG['test_size'] - CONFIG['val_size']))
    val_end = int(n * (1 - CONFIG['test_size']))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train_scaled,
        'X_val': X_val_scaled, 'y_val': y_val_scaled,
        'X_test': X_test_scaled, 'y_test': y_test_scaled,
        'y_test_original': y_test.values,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'feature_names': X.columns.tolist(),
        'test_index': y_test.index
    }

def prepare_sequences(df, target_var):
    """Prepare sequences for LSTM/GRU"""
    exclude_cols = [col for col in CONFIG['target_variables'] if col != target_var]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove leakage features
    leakage_features = get_leakage_features(target_var)
    for leak_feat in leakage_features:
        if leak_feat in feature_cols:
            feature_cols.remove(leak_feat)
    
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_var)
    
    seq_length = CONFIG['sequence_length']
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Scale
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_val': X_val_scaled, 'y_val': y_val,
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler
    }

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def plot_predictions(y_true, y_pred, timestamps, target_var, model_name, output_dir):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, y_true, label='Actual', alpha=0.7, linewidth=1)
    plt.plot(timestamps, y_pred, label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'{model_name}: {target_var}', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
    filename = f'predicted_vs_actual_{target_clean}.png'
    save_plot(filename, output_dir)

def save_metrics(metrics, target_var, output_dir, model_name, is_first=False):
    """Save metrics to a text file"""
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    mode = 'w' if is_first else 'a'
    with open(metrics_file, mode, encoding='utf-8') as f:
        if is_first:
            f.write(f"{'='*80}\n")
            f.write(f"METRICS SUMMARY - {model_name.upper()}\n")
            f.write(f"{'='*80}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Target Variable: {target_var}\n")
        f.write(f"{'='*80}\n")
        f.write(f"R² Score: {metrics['r2']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"{'='*80}\n")

