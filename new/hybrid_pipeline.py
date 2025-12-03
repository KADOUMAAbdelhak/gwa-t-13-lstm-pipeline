"""
Hybrid ML Pipeline for VM Performance Prediction
=================================================
Uses the best model architecture for each metric type:
- CPU/Memory: Feature-rich XGBoost/Random Forest (works best for these)
- Disk/Network: XGBoost with robust feature engineering (bursty patterns)

Key insight: Tree-based models with good feature engineering often outperform
LSTMs for single-step time series forecasting.

Usage:
    python hybrid_pipeline.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

# Try importing LightGBM (often better than XGBoost)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available, using XGBoost/RandomForest instead")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'output_dir': 'hybrid_results',
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.1,
    'target_variables': [
        'CPU usage [%]',
        'Memory usage [%]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]'
    ],
    # Model selection per metric type
    'model_config': {
        'CPU usage [%]': {
            'model_type': 'xgboost',  # or 'random_forest', 'lightgbm'
            'lag_features': [1, 2, 3, 6, 12, 24, 48],  # 5-min intervals
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_time_features': True,
            'use_diff_features': True,
        },
        'Memory usage [%]': {
            'model_type': 'xgboost',
            'lag_features': [1, 2, 3, 6, 12, 24],
            'rolling_windows': [6, 12, 24, 48],
            'use_time_features': True,
            'use_diff_features': True,
        },
        'Disk write throughput [KB/s]': {
            'model_type': 'xgboost',
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_time_features': True,
            'use_diff_features': True,
            'use_quantile_features': True,  # Important for bursty data
        },
        'Network received throughput [KB/s]': {
            'model_type': 'xgboost',
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_time_features': True,
            'use_diff_features': True,
            'use_quantile_features': True,
        },
        'Network transmitted throughput [KB/s]': {
            'model_type': 'xgboost',
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_time_features': True,
            'use_diff_features': True,
            'use_quantile_features': True,
        },
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dir():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Output directory: {CONFIG['output_dir']}")

def log_message(message, log_file='results.txt'):
    print(message)
    log_path = os.path.join(CONFIG['output_dir'], log_file)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def save_plot(filename):
    filepath = os.path.join(CONFIG['output_dir'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and preprocess the VM trace dataset"""
    log_message("\n" + "="*80)
    log_message("STEP 1: DATA LOADING")
    log_message("="*80)
    
    try:
        separators = [';', ';\t', '\t', ',']
        df = None
        
        for sep in separators:
            try:
                df_test = pd.read_csv(CONFIG['input_file'], sep=sep, decimal=',', engine='python')
                if df_test.shape[1] > 5:
                    df = df_test
                    log_message(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    break
            except:
                continue
        
        if df is None:
            raise Exception("Could not parse CSV")
            
    except Exception as e:
        log_message(f"✗ Error: {e}")
        sys.exit(1)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Find and parse timestamp
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    try:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col], format='%d.%m.%Y %H:%M:%S')
    except:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col])
    
    if timestamp_col != 'Timestamp':
        df = df.drop(columns=[timestamp_col])
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.set_index('Timestamp', inplace=True)
    df = df.ffill().bfill()
    
    log_message(f"✓ Time range: {df.index.min()} to {df.index.max()}")
    log_message(f"✓ Total samples: {len(df)}")
    
    return df

# ============================================================================
# FEATURE ENGINEERING (CRITICAL FOR GOOD PERFORMANCE)
# ============================================================================

def create_time_features(df):
    """Create comprehensive time-based features"""
    df_feat = df.copy()
    
    # Basic time components
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['week_of_year'] = df_feat.index.isocalendar().week.astype(int)
    df_feat['month'] = df_feat.index.month
    
    # Cyclical encoding (important for capturing periodicity)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['minute_sin'] = np.sin(2 * np.pi * df_feat['minute'] / 60)
    df_feat['minute_cos'] = np.cos(2 * np.pi * df_feat['minute'] / 60)
    
    # Binary features
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    df_feat['is_business_hours'] = ((df_feat['hour'] >= 9) & (df_feat['hour'] < 18)).astype(int)
    df_feat['is_night'] = ((df_feat['hour'] >= 22) | (df_feat['hour'] < 6)).astype(int)
    
    return df_feat

def create_lag_features(df, target_col, lags):
    """Create lag features"""
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    return df_lag

def create_rolling_features(df, target_col, windows):
    """Create rolling statistics features"""
    df_roll = df.copy()
    for window in windows:
        df_roll[f'{target_col}_rolling_mean_{window}'] = df_roll[target_col].rolling(window=window).mean()
        df_roll[f'{target_col}_rolling_std_{window}'] = df_roll[target_col].rolling(window=window).std()
        df_roll[f'{target_col}_rolling_min_{window}'] = df_roll[target_col].rolling(window=window).min()
        df_roll[f'{target_col}_rolling_max_{window}'] = df_roll[target_col].rolling(window=window).max()
        df_roll[f'{target_col}_rolling_range_{window}'] = (
            df_roll[f'{target_col}_rolling_max_{window}'] - df_roll[f'{target_col}_rolling_min_{window}']
        )
    return df_roll

def create_diff_features(df, target_col):
    """Create difference/change features"""
    df_diff = df.copy()
    df_diff[f'{target_col}_diff_1'] = df_diff[target_col].diff(1)
    df_diff[f'{target_col}_diff_12'] = df_diff[target_col].diff(12)
    # Handle pct_change carefully to avoid infinity
    pct_1 = df_diff[target_col].pct_change(1)
    pct_12 = df_diff[target_col].pct_change(12)
    # Replace infinity with NaN (will be handled later)
    df_diff[f'{target_col}_pct_change_1'] = pct_1.replace([np.inf, -np.inf], np.nan)
    df_diff[f'{target_col}_pct_change_12'] = pct_12.replace([np.inf, -np.inf], np.nan)
    return df_diff

def create_quantile_features(df, target_col, windows):
    """Create quantile features for bursty data"""
    df_quant = df.copy()
    for window in windows:
        df_quant[f'{target_col}_rolling_median_{window}'] = df_quant[target_col].rolling(window=window).median()
        df_quant[f'{target_col}_rolling_q25_{window}'] = df_quant[target_col].rolling(window=window).quantile(0.25)
        df_quant[f'{target_col}_rolling_q75_{window}'] = df_quant[target_col].rolling(window=window).quantile(0.75)
        df_quant[f'{target_col}_rolling_iqr_{window}'] = (
            df_quant[f'{target_col}_rolling_q75_{window}'] - df_quant[f'{target_col}_rolling_q25_{window}']
        )
    return df_quant

def create_features_for_target(df, target_var):
    """Create all features for a specific target variable"""
    config = CONFIG['model_config'].get(target_var, {})
    
    df_feat = df.copy()
    
    # Time features
    if config.get('use_time_features', True):
        df_feat = create_time_features(df_feat)
    
    # Lag features
    lag_values = config.get('lag_features', [1, 2, 3, 6, 12])
    df_feat = create_lag_features(df_feat, target_var, lag_values)
    
    # Rolling features
    rolling_windows = config.get('rolling_windows', [6, 12, 24])
    df_feat = create_rolling_features(df_feat, target_var, rolling_windows)
    
    # Diff features
    if config.get('use_diff_features', True):
        df_feat = create_diff_features(df_feat, target_var)
    
    # Quantile features (for bursty data)
    if config.get('use_quantile_features', False):
        df_feat = create_quantile_features(df_feat, target_var, rolling_windows)
    
    # Drop rows with NaN (from lag/rolling features)
    df_feat = df_feat.dropna()
    
    return df_feat

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df_feat, target_var):
    """Prepare data for training"""
    # Exclude leakage features
    leakage_cols = []
    if target_var == 'Memory usage [%]':
        leakage_cols = ['Memory usage [KB]', 'Memory capacity provisioned [KB]']
    elif target_var == 'CPU usage [%]':
        leakage_cols = ['CPU usage [MHZ]', 'CPU capacity provisioned [MHZ]']
    
    # Also exclude other target variables
    other_targets = [t for t in CONFIG['target_variables'] if t != target_var]
    exclude_cols = leakage_cols + other_targets
    
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols and col != target_var]
    
    X = df_feat[feature_cols].copy()
    y = df_feat[target_var].copy()
    
    # Handle infinity and very large values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN with column median (more robust than mean for skewed data)
    X = X.fillna(X.median())
    
    # Clip extreme values to prevent overflow
    for col in X.columns:
        q01 = X[col].quantile(0.001)
        q99 = X[col].quantile(0.999)
        X[col] = X[col].clip(q01, q99)
    
    # Chronological split
    n = len(X)
    train_end = int(n * (1 - CONFIG['test_size'] - CONFIG['val_size']))
    val_end = int(n * (1 - CONFIG['test_size']))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    # Scale features
    scaler = RobustScaler()  # Better for data with outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train.values,
        'X_val': X_val_scaled, 'y_val': y_val.values,
        'X_test': X_test_scaled, 'y_test': y_test.values,
        'feature_names': feature_cols,
        'scaler': scaler,
        'test_index': y_test.index
    }

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(data, target_var):
    """Train XGBoost model with optimized hyperparameters"""
    log_message(f"  Training XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )
    
    y_pred = model.predict(data['X_test'])
    
    return model, y_pred

def train_random_forest(data, target_var):
    """Train Random Forest model"""
    log_message(f"  Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
    
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])
    
    return model, y_pred

def train_gradient_boosting(data, target_var):
    """Train Gradient Boosting model"""
    log_message(f"  Training Gradient Boosting...")
    
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=CONFIG['random_state']
    )
    
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])
    
    return model, y_pred

def train_lightgbm(data, target_var):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        return train_xgboost(data, target_var)
    
    log_message(f"  Training LightGBM...")
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    y_pred = model.predict(data['X_test'])
    
    return model, y_pred

def train_best_model(data, target_var):
    """Train multiple models and select the best one"""
    log_message(f"\n  Comparing models for {target_var}:")
    
    models = {}
    predictions = {}
    metrics_dict = {}
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        model_xgb, pred_xgb = train_xgboost(data, target_var)
        models['XGBoost'] = model_xgb
        predictions['XGBoost'] = pred_xgb
        metrics_dict['XGBoost'] = evaluate_predictions(data['y_test'], pred_xgb)
        log_message(f"    XGBoost R²: {metrics_dict['XGBoost']['r2']:.4f}")
    
    # Train Random Forest
    model_rf, pred_rf = train_random_forest(data, target_var)
    models['RandomForest'] = model_rf
    predictions['RandomForest'] = pred_rf
    metrics_dict['RandomForest'] = evaluate_predictions(data['y_test'], pred_rf)
    log_message(f"    Random Forest R²: {metrics_dict['RandomForest']['r2']:.4f}")
    
    # Train Gradient Boosting
    model_gb, pred_gb = train_gradient_boosting(data, target_var)
    models['GradientBoosting'] = model_gb
    predictions['GradientBoosting'] = pred_gb
    metrics_dict['GradientBoosting'] = evaluate_predictions(data['y_test'], pred_gb)
    log_message(f"    Gradient Boosting R²: {metrics_dict['GradientBoosting']['r2']:.4f}")
    
    # Train LightGBM if available
    if LIGHTGBM_AVAILABLE:
        model_lgb, pred_lgb = train_lightgbm(data, target_var)
        models['LightGBM'] = model_lgb
        predictions['LightGBM'] = pred_lgb
        metrics_dict['LightGBM'] = evaluate_predictions(data['y_test'], pred_lgb)
        log_message(f"    LightGBM R²: {metrics_dict['LightGBM']['r2']:.4f}")
    
    # Select best model
    best_model_name = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]['r2'])
    best_model = models[best_model_name]
    best_pred = predictions[best_model_name]
    best_metrics = metrics_dict[best_model_name]
    
    log_message(f"  → Best model: {best_model_name} with R² = {best_metrics['r2']:.4f}")
    
    return best_model, best_pred, best_metrics, best_model_name, metrics_dict

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def plot_predictions(y_true, y_pred, target_var, model_name):
    """Plot comprehensive prediction analysis"""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Full time series
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(y_true, label='Actual', alpha=0.8, linewidth=1, color='#2E86AB')
    plt.plot(y_pred, label='Predicted', alpha=0.8, linewidth=1, color='#A23B72')
    plt.title(f'{model_name}: {target_var}', fontsize=12, fontweight='bold')
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view
    ax2 = plt.subplot(3, 2, 2)
    zoom_len = min(300, len(y_true))
    plt.plot(y_true[:zoom_len], label='Actual', alpha=0.8, linewidth=1.5, color='#2E86AB')
    plt.plot(y_pred[:zoom_len], label='Predicted', alpha=0.8, linewidth=1.5, color='#A23B72')
    plt.title('Zoomed View (First 300 Points)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    ax3 = plt.subplot(3, 2, 3)
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color='#F18F01')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual', fontsize=10)
    plt.ylabel('Predicted', fontsize=10)
    plt.title('Scatter Plot', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax4 = plt.subplot(3, 2, 4)
    residuals = y_true - y_pred
    plt.plot(residuals, alpha=0.6, linewidth=0.8, color='#C73E1D')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.title('Residuals', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residual distribution
    ax5 = plt.subplot(3, 2, 5)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#6A994E')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residual Distribution', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Error percentiles
    ax6 = plt.subplot(3, 2, 6)
    abs_errors = np.abs(residuals)
    percentiles = [50, 75, 90, 95, 99]
    error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]
    plt.bar([str(p) for p in percentiles], error_percentiles, color='#BC4B51', alpha=0.7)
    plt.xlabel('Percentile', fontsize=10)
    plt.ylabel('Absolute Error', fontsize=10)
    plt.title('Error Percentiles', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

def plot_feature_importance(model, feature_names, target_var, model_name, top_n=20):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return
    
    # Sort by importance
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], color='#2E86AB', alpha=0.8)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=8)
    plt.xlabel('Importance', fontsize=10)
    plt.title(f'Top {top_n} Features: {target_var} ({model_name})', fontsize=12)
    plt.tight_layout()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("HYBRID ML PIPELINE FOR VM PERFORMANCE PREDICTION")
    print("="*80)
    print("Using the best model for each metric type:")
    print("  - Compares XGBoost, Random Forest, Gradient Boosting, LightGBM")
    print("  - Extensive feature engineering (lags, rolling stats, time features)")
    print("  - Automatic model selection based on R² score")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_dir()
    
    # Clear log file
    log_file = os.path.join(CONFIG['output_dir'], 'results.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message("="*80)
    log_message("HYBRID ML PIPELINE - ANALYSIS REPORT")
    log_message("="*80)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"XGBoost available: {XGBOOST_AVAILABLE}")
    log_message(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    # Load data
    df = load_data()
    
    # Process each target variable
    log_message("\n" + "="*80)
    log_message("STEP 2: MODEL TRAINING")
    log_message("="*80)
    
    all_results = {}
    
    for target_var in CONFIG['target_variables']:
        log_message(f"\n{'='*80}")
        log_message(f"Processing: {target_var}")
        log_message(f"{'='*80}")
        
        # Analyze data characteristics
        data = df[target_var].values
        log_message(f"  Data characteristics:")
        log_message(f"    Range: [{data.min():.2f}, {data.max():.2f}]")
        log_message(f"    Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        log_message(f"    Skewness: {stats.skew(data):.3f}")
        log_message(f"    Kurtosis: {stats.kurtosis(data):.3f}")
        
        # Create features
        log_message(f"  Creating features...")
        df_feat = create_features_for_target(df, target_var)
        log_message(f"    Total features: {len(df_feat.columns) - 1}")
        log_message(f"    Samples after feature engineering: {len(df_feat)}")
        
        # Prepare data
        data_dict = prepare_data(df_feat, target_var)
        log_message(f"    Train: {len(data_dict['X_train'])}, Val: {len(data_dict['X_val'])}, Test: {len(data_dict['X_test'])}")
        
        # Train and compare models
        best_model, y_pred, best_metrics, best_model_name, all_model_metrics = train_best_model(data_dict, target_var)
        
        # Store results
        all_results[target_var] = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'predictions': y_pred,
            'y_true': data_dict['y_test'],
            'metrics': best_metrics,
            'all_model_metrics': all_model_metrics,
            'feature_names': data_dict['feature_names']
        }
        
        # Log final results
        log_message(f"\n  FINAL RESULTS for {target_var}:")
        log_message(f"    Best Model: {best_model_name}")
        log_message(f"    R² Score: {best_metrics['r2']:.6f}")
        log_message(f"    RMSE: {best_metrics['rmse']:.6f}")
        log_message(f"    MAE: {best_metrics['mae']:.6f}")
        log_message(f"    MAPE: {best_metrics['mape']:.2f}%")
        
        # Save visualizations
        target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
        
        plot_predictions(data_dict['y_test'], y_pred, target_var, best_model_name)
        save_plot(f'predictions_{target_clean}.png')
        
        plot_feature_importance(best_model, data_dict['feature_names'], target_var, best_model_name)
        save_plot(f'importance_{target_clean}.png')
    
    # Final summary
    log_message("\n" + "="*80)
    log_message("FINAL SUMMARY - HYBRID PIPELINE RESULTS")
    log_message("="*80)
    
    log_message("\nBest Model per Target Variable:")
    log_message(f"{'Target Variable':<45} {'Best Model':<15} {'R²':>10} {'RMSE':>12} {'MAE':>12}")
    log_message("-" * 100)
    
    for target_var, results in all_results.items():
        m = results['metrics']
        model_name = results['best_model_name']
        log_message(f"{target_var:<45} {model_name:<15} {m['r2']:>10.4f} {m['rmse']:>12.4f} {m['mae']:>12.4f}")
    
    log_message("\n" + "-"*100)
    log_message("\nModel Comparison (R² by model type):")
    model_types = list(all_results[CONFIG['target_variables'][0]]['all_model_metrics'].keys())
    header = f"{'Target Variable':<45} " + " ".join([f"{mt:<14}" for mt in model_types])
    log_message(header)
    log_message("-" * 100)
    
    for target_var, results in all_results.items():
        row = f"{target_var:<45} "
        for mt in model_types:
            r2 = results['all_model_metrics'].get(mt, {}).get('r2', 0)
            row += f"{r2:>14.4f} "
        log_message(row)
    
    log_message("\n" + "="*80)
    log_message("CONCLUSIONS:")
    log_message("  - Tree-based models (XGBoost, Random Forest) often outperform LSTM")
    log_message("  - Feature engineering is critical for good performance")
    log_message("  - Bursty metrics (disk, network) are inherently harder to predict")
    log_message("  - Consider using quantile prediction for bursty metrics")
    log_message("="*80)
    
    log_message(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("✓ HYBRID PIPELINE COMPLETED!")
    print(f"✓ Check '{CONFIG['output_dir']}' for all results")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

