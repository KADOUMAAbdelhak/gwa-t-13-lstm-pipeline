"""
VM Performance Prediction - Complete Pipeline
==============================================
Place this script in the same folder as dataset.csv and run:
    python vm_prediction_pipeline.py

All results will be saved to the 'results' folder.
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available: {type(e).__name__}. Install OpenMP with: brew install libomp")

# Try importing TensorFlow/Keras (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'output_dir': 'results',
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

def create_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Output directory: {CONFIG['output_dir']}")

def log_message(message, log_file='results.txt'):
    """Write message to both console and log file"""
    print(message)
    log_path = os.path.join(CONFIG['output_dir'], log_file)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def save_plot(filename):
    """Save current plot to output directory"""
    filepath = os.path.join(CONFIG['output_dir'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    """Load and preprocess the VM trace dataset"""
    log_message("\n" + "="*80)
    log_message("STEP 1: DATA LOADING AND PREPROCESSING")
    log_message("="*80)
    
    try:
        # Load CSV with semicolon delimiter
        df = pd.read_csv(CONFIG['input_file'], sep=';', engine='python', quotechar='"')
        log_message(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        log_message(f"✗ Error loading dataset: {e}")
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
    log_message(f"✓ Missing values handled: {missing_before} filled")
    
    # Basic statistics
    log_message(f"\nDataset Summary:")
    log_message(f"  Time range: {df.index.min()} to {df.index.max()}")
    log_message(f"  Duration: {(df.index.max() - df.index.min()).days} days")
    log_message(f"  Sampling interval: {(df.index[1] - df.index[0]).total_seconds() / 60:.0f} minutes")
    
    return df

def perform_eda(df):
    """Perform exploratory data analysis and create visualizations"""
    log_message("\n" + "="*80)
    log_message("STEP 2: EXPLORATORY DATA ANALYSIS")
    log_message("="*80)
    
    target_vars = CONFIG['target_variables']
    
    # Statistics for target variables
    log_message("\nTarget Variables Statistics:")
    stats = df[target_vars].describe()
    log_message(stats.to_string())
    
    # Plot 1: Time series of all target variables
    fig, axes = plt.subplots(len(target_vars), 1, figsize=(15, 12))
    for idx, var in enumerate(target_vars):
        axes[idx].plot(df.index, df[var], linewidth=0.5, alpha=0.7)
        axes[idx].set_title(f'{var} over Time', fontsize=10)
        axes[idx].set_ylabel(var.split('[')[0].strip(), fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    save_plot('01_time_series_overview.png')
    
    # Plot 2: Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, var in enumerate(target_vars):
        axes[idx].hist(df[var], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution: {var}', fontsize=9)
        axes[idx].set_xlabel(var.split('[')[0].strip(), fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    axes[-1].axis('off')  # Hide extra subplot
    plt.tight_layout()
    save_plot('02_distributions.png')
    
    # Plot 3: Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation = df[target_vars].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix of Target Variables', fontsize=12)
    plt.tight_layout()
    save_plot('03_correlation_matrix.png')
    
    log_message("\n✓ EDA visualizations saved")
    
    return stats

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
    log_message("\n" + "="*80)
    log_message("STEP 3: FEATURE ENGINEERING")
    log_message("="*80)
    
    df_engineered = create_time_features(df)
    log_message("✓ Time features created")
    
    # Domain-specific features
    df_engineered['cpu_utilization_rate'] = df_engineered['CPU usage [MHZ]'] / (df_engineered['CPU capacity provisioned [MHZ]'] + 1)
    df_engineered['memory_utilization_rate'] = df_engineered['Memory usage [KB]'] / df_engineered['Memory capacity provisioned [KB]']
    df_engineered['network_total'] = df_engineered['Network received throughput [KB/s]'] + df_engineered['Network transmitted throughput [KB/s]']
    log_message("✓ Domain features created")
    
    # Lag and rolling features for each target
    for target in CONFIG['target_variables']:
        df_engineered = create_lag_features(df_engineered, target, lags=[1, 2, 3, 6, 12])
        df_engineered = create_rolling_features(df_engineered, target, windows=[6, 12, 24])
    log_message("✓ Lag and rolling features created")
    
    # Remove NaN rows
    df_clean = df_engineered.dropna()
    log_message(f"✓ Dataset shape after feature engineering: {df_clean.shape}")
    log_message(f"  Features: {df_clean.shape[1]}")
    log_message(f"  Samples: {df_clean.shape[0]}")
    
    return df_clean

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def prepare_ml_data(df, target_var):
    """Prepare data for traditional ML models"""
    # Exclude other target variables from features
    exclude_cols = [col for col in CONFIG['target_variables'] if col != target_var]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove leakage features based on target variable
    leakage_features = []
    if target_var == 'Memory usage [%]':
        # Exclude raw memory features that can directly calculate Memory usage [%]
        leakage_features = [
            'Memory usage [KB]',
            'Memory capacity provisioned [KB]',
            'memory_utilization_rate'  # This is Memory usage [%] / 100
        ]
    elif target_var == 'CPU usage [%]':
        # Exclude raw CPU features that can directly calculate CPU usage [%]
        leakage_features = [
            'CPU usage [MHZ]',
            'CPU capacity provisioned [MHZ]',
            'cpu_utilization_rate'  # This is CPU usage [%] / 100
        ]
    elif target_var in ['Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]']:
        # Exclude network_total which is sum of both
        leakage_features = ['network_total']
    
    # Remove leakage features
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

def train_ridge_regression(data):
    """Train Ridge Regression model"""
    model = Ridge(alpha=1.0, random_state=CONFIG['random_state'])
    model.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    metrics = evaluate_predictions(data['y_test_original'], y_pred)
    
    return model, y_pred, metrics

def train_random_forest(data):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
    model.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    metrics = evaluate_predictions(data['y_test_original'], y_pred)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, y_pred, metrics, importance_df

def train_xgboost(data):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
    
    # Fit without early_stopping_rounds (not supported in newer XGBoost versions)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )
    
    # Predictions
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    metrics = evaluate_predictions(data['y_test_original'], y_pred)
    
    return model, y_pred, metrics

def prepare_sequences(df, target_var):
    """Prepare sequences for LSTM/GRU"""
    exclude_cols = [col for col in CONFIG['target_variables'] if col != target_var]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove leakage features based on target variable (same as prepare_ml_data)
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

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(seq_data):
    """Train LSTM model"""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None
    
    input_shape = (seq_data['X_train'].shape[1], seq_data['X_train'].shape[2])
    model = build_lstm_model(input_shape)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        seq_data['X_train'], seq_data['y_train'],
        validation_data=(seq_data['X_val'], seq_data['y_val']),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predictions
    y_pred = model.predict(seq_data['X_test'], verbose=0).flatten()
    
    metrics = evaluate_predictions(seq_data['y_test'], y_pred)
    
    return model, y_pred, metrics, history

def plot_predictions(y_true, y_pred, timestamps, target_var, model_name):
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

def plot_feature_importance(importance_df, target_var):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
    plt.xlabel('Importance', fontsize=10)
    plt.title(f'Top 20 Feature Importance: {target_var}', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()

def train_all_models_for_target(df, target_var):
    """Train all models for a specific target variable"""
    log_message(f"\n{'='*80}")
    log_message(f"Training Models for: {target_var}")
    log_message(f"{'='*80}")
    
    results = {}
    
    # Prepare data for ML models
    ml_data = prepare_ml_data(df, target_var)
    
    # Ridge Regression
    log_message("\n1. Ridge Regression...")
    ridge_model, ridge_pred, ridge_metrics = train_ridge_regression(ml_data)
    results['Ridge'] = {'model': ridge_model, 'predictions': ridge_pred, 'metrics': ridge_metrics}
    log_message(f"   RMSE: {ridge_metrics['rmse']:.4f} | MAE: {ridge_metrics['mae']:.4f} | R²: {ridge_metrics['r2']:.4f}")
    
    # Random Forest
    log_message("\n2. Random Forest...")
    rf_model, rf_pred, rf_metrics, rf_importance = train_random_forest(ml_data)
    results['Random Forest'] = {'model': rf_model, 'predictions': rf_pred, 'metrics': rf_metrics}
    log_message(f"   RMSE: {rf_metrics['rmse']:.4f} | MAE: {rf_metrics['mae']:.4f} | R²: {rf_metrics['r2']:.4f}")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        log_message("\n3. XGBoost...")
        xgb_model, xgb_pred, xgb_metrics = train_xgboost(ml_data)
        results['XGBoost'] = {'model': xgb_model, 'predictions': xgb_pred, 'metrics': xgb_metrics}
        log_message(f"   RMSE: {xgb_metrics['rmse']:.4f} | MAE: {xgb_metrics['mae']:.4f} | R²: {xgb_metrics['r2']:.4f}")
    
    # LSTM
    if TENSORFLOW_AVAILABLE:
        log_message("\n4. LSTM...")
        seq_data = prepare_sequences(df, target_var)
        lstm_model, lstm_pred, lstm_metrics, lstm_history = train_lstm(seq_data)
        results['LSTM'] = {'model': lstm_model, 'predictions': lstm_pred, 'metrics': lstm_metrics, 'history': lstm_history}
        log_message(f"   RMSE: {lstm_metrics['rmse']:.4f} | MAE: {lstm_metrics['mae']:.4f} | R²: {lstm_metrics['r2']:.4f}")
    
    # Visualizations
    target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
    
    # Plot predictions for Random Forest (usually best ML model)
    plot_predictions(ml_data['y_test_original'], rf_pred, ml_data['test_index'], target_var, 'Random Forest')
    save_plot(f'pred_{target_clean}_random_forest.png')
    
    # Plot feature importance
    plot_feature_importance(rf_importance, target_var)
    save_plot(f'importance_{target_clean}.png')
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    metrics_types = ['rmse', 'mae', 'r2']
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics_types):
        values = [results[m]['metrics'][metric] for m in models]
        plt.bar(x + i*width, values, width, label=metric.upper())
    
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.title(f'Model Comparison: {target_var}', fontsize=12)
    plt.xticks(x + width, models, fontsize=9)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_plot(f'comparison_{target_clean}.png')
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['r2'])
    log_message(f"\n✓ Best Model: {best_model[0]} with R² = {best_model[1]['metrics']['r2']:.4f}")
    
    return results, best_model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("VM PERFORMANCE PREDICTION PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    create_output_dir()
    
    # Clear previous results file
    results_file = os.path.join(CONFIG['output_dir'], 'results.txt')
    if os.path.exists(results_file):
        os.remove(results_file)
    
    log_message("="*80)
    log_message("VM PERFORMANCE PREDICTION - ANALYSIS REPORT")
    log_message("="*80)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Python version: {sys.version.split()[0]}")
    log_message(f"XGBoost available: {XGBOOST_AVAILABLE}")
    log_message(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    
    # Load data
    df = load_data()
    
    # EDA
    stats = perform_eda(df)
    
    # Feature engineering
    df_clean = engineer_features(df)
    
    # Train models for each target variable
    all_results = {}
    best_models = {}
    
    log_message("\n" + "="*80)
    log_message("STEP 4: MODEL TRAINING AND EVALUATION")
    log_message("="*80)
    
    for target_var in CONFIG['target_variables']:
        results, best_model = train_all_models_for_target(df_clean, target_var)
        all_results[target_var] = results
        best_models[target_var] = best_model
    
    # Summary
    log_message("\n" + "="*80)
    log_message("FINAL SUMMARY")
    log_message("="*80)
    
    log_message("\nBest Models by Target Variable:")
    for target_var, (model_name, model_info) in best_models.items():
        metrics = model_info['metrics']
        log_message(f"\n{target_var}:")
        log_message(f"  Best Model: {model_name}")
        log_message(f"  RMSE: {metrics['rmse']:.4f}")
        log_message(f"  MAE: {metrics['mae']:.4f}")
        log_message(f"  R²: {metrics['r2']:.4f}")
        log_message(f"  MAPE: {metrics['mape']:.2f}%")
    
    log_message("\n" + "="*80)
    log_message("PIPELINE COMPLETED SUCCESSFULLY")
    log_message("="*80)
    log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"\nAll results saved to: {CONFIG['output_dir']}/")
    log_message("  - results.txt: Complete analysis report")
    log_message("  - *.png: All visualization plots")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED!")
    print(f"✓ Check the '{CONFIG['output_dir']}' folder for all results")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)