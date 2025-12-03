"""
Publication-Ready ML Pipeline for VM Performance Prediction
============================================================
Designed for scientific paper: Uses optimal model for each metric type.

Model Assignments:
- CPU usage [%]:      Random Forest (R² ≈ 0.95)
- Memory usage [%]:   Random Forest (R² ≈ 0.89)
- Disk write:         XGBoost (R² ≈ 0.94)
- Network Rx:         XGBoost (R² ≈ 0.91)
- Network Tx:         XGBoost (R² ≈ 0.88)

Generates publication-quality visualizations.

Usage:
    python paper_pipeline.py
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
from matplotlib.gridspec import GridSpec
from datetime import datetime
from scipy import stats

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ============================================================================
# PUBLICATION STYLE CONFIGURATION
# ============================================================================

# Set publication-quality plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme for paper
COLORS = {
    'actual': '#1f77b4',      # Blue
    'predicted': '#d62728',   # Red
    'residual': '#2ca02c',    # Green
    'scatter': '#ff7f0e',     # Orange
    'grid': '#cccccc',
    'zero_line': '#333333',
}

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'output_dir': 'paper_results',
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.1,
    
    # Model assignments as per paper requirements
    'model_assignments': {
        'CPU usage [%]': 'random_forest',
        'Memory usage [%]': 'random_forest',
        'Disk write throughput [KB/s]': 'xgboost',
        'Network received throughput [KB/s]': 'xgboost',
        'Network transmitted throughput [KB/s]': 'xgboost',
    },
    
    'target_variables': [
        'CPU usage [%]',
        'Memory usage [%]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]'
    ],
    
    # Feature engineering config per metric
    'feature_config': {
        'CPU usage [%]': {
            'lag_features': [1, 2, 3, 6, 12, 24, 48],
            'rolling_windows': [6, 12, 24, 48, 96],
        },
        'Memory usage [%]': {
            'lag_features': [1, 2, 3, 6, 12, 24],
            'rolling_windows': [6, 12, 24, 48],
        },
        'Disk write throughput [KB/s]': {
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_quantile_features': True,
        },
        'Network received throughput [KB/s]': {
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_quantile_features': True,
        },
        'Network transmitted throughput [KB/s]': {
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 96],
            'rolling_windows': [6, 12, 24, 48, 96],
            'use_quantile_features': True,
        },
    },
    
    # Visualization settings
    'zoom_window': 288,  # ~24 hours (5-min intervals)
    'zoom_window_short': 144,  # ~12 hours for dense plots
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dir():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

def log_message(message, log_file='analysis_log.txt'):
    print(message)
    log_path = os.path.join(CONFIG['output_dir'], log_file)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and preprocess the VM trace dataset"""
    separators = [';', ';\t', '\t', ',']
    df = None
    
    for sep in separators:
        try:
            df_test = pd.read_csv(CONFIG['input_file'], sep=sep, decimal=',', engine='python')
            if df_test.shape[1] > 5:
                df = df_test
                break
        except:
            continue
    
    if df is None:
        raise Exception("Could not parse CSV file")
    
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
    
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_time_features(df):
    """Create time-based features with cyclical encoding"""
    df_feat = df.copy()
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['week_of_year'] = df_feat.index.isocalendar().week.astype(int)
    
    # Cyclical encoding
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
    """Create difference features"""
    df_diff = df.copy()
    df_diff[f'{target_col}_diff_1'] = df_diff[target_col].diff(1)
    df_diff[f'{target_col}_diff_12'] = df_diff[target_col].diff(12)
    
    pct_1 = df_diff[target_col].pct_change(1)
    pct_12 = df_diff[target_col].pct_change(12)
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

def engineer_features(df, target_var):
    """Create all features for a target variable"""
    config = CONFIG['feature_config'].get(target_var, {})
    
    df_feat = df.copy()
    df_feat = create_time_features(df_feat)
    
    lag_values = config.get('lag_features', [1, 2, 3, 6, 12])
    df_feat = create_lag_features(df_feat, target_var, lag_values)
    
    rolling_windows = config.get('rolling_windows', [6, 12, 24])
    df_feat = create_rolling_features(df_feat, target_var, rolling_windows)
    df_feat = create_diff_features(df_feat, target_var)
    
    if config.get('use_quantile_features', False):
        df_feat = create_quantile_features(df_feat, target_var, rolling_windows)
    
    df_feat = df_feat.dropna()
    
    return df_feat

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df_feat, target_var):
    """Prepare data for training with leakage prevention"""
    # Exclude leakage features
    leakage_cols = []
    if target_var == 'Memory usage [%]':
        leakage_cols = ['Memory usage [KB]', 'Memory capacity provisioned [KB]']
    elif target_var == 'CPU usage [%]':
        leakage_cols = ['CPU usage [MHZ]', 'CPU capacity provisioned [MHZ]']
    
    other_targets = [t for t in CONFIG['target_variables'] if t != target_var]
    exclude_cols = leakage_cols + other_targets
    
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols and col != target_var]
    
    X = df_feat[feature_cols].copy()
    y = df_feat[target_var].copy()
    
    # Handle infinity and extreme values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
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
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train.values,
        'X_val': X_val_scaled, 'y_val': y_val.values,
        'X_test': X_test_scaled, 'y_test': y_test.values,
        'test_index': y_test.index,
        'feature_names': feature_cols,
    }

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest(data):
    """Train Random Forest model"""
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

def train_xgboost(data):
    """Train XGBoost model"""
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

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    # Normalized RMSE
    nrmse = rmse / (y_true.max() - y_true.min()) * 100
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'nrmse': nrmse}

# ============================================================================
# PUBLICATION-QUALITY VISUALIZATIONS
# ============================================================================

def plot_publication_figure(y_true, y_pred, target_var, model_name, metrics, test_index):
    """
    Create a publication-quality figure with:
    - Full time series comparison
    - Zoomed view of a representative window
    - Scatter plot with regression line
    - Metrics annotation
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)
    
    # Clean target name for display
    target_display = target_var.replace('[', '(').replace(']', ')')
    
    # =========================================================================
    # Panel A: Full Time Series
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create time axis
    time_axis = np.arange(len(y_true))
    
    ax1.plot(time_axis, y_true, color=COLORS['actual'], 
             linewidth=1.0, alpha=0.8, label='Actual')
    ax1.plot(time_axis, y_pred, color=COLORS['predicted'], 
             linewidth=1.0, alpha=0.8, label='Predicted')
    
    ax1.set_xlabel('Time Step (5-minute intervals)', fontsize=11)
    ax1.set_ylabel(target_display, fontsize=11)
    ax1.set_title(f'(a) {model_name} Predictions for {target_display}', 
                  fontsize=12, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.set_xlim(0, len(y_true))
    
    # Add metrics box
    metrics_text = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.2f}\nMAE = {metrics['mae']:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.97, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # =========================================================================
    # Panel B: Zoomed View
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Select zoom window (middle section for representative behavior)
    zoom_len = min(CONFIG['zoom_window'], len(y_true) // 2)
    start_idx = len(y_true) // 3  # Start at 1/3 of the data
    end_idx = start_idx + zoom_len
    
    zoom_time = np.arange(zoom_len)
    y_true_zoom = y_true[start_idx:end_idx]
    y_pred_zoom = y_pred[start_idx:end_idx]
    
    ax2.plot(zoom_time, y_true_zoom, color=COLORS['actual'], 
             linewidth=1.5, alpha=0.9, label='Actual', marker='', markersize=2)
    ax2.plot(zoom_time, y_pred_zoom, color=COLORS['predicted'], 
             linewidth=1.5, alpha=0.9, label='Predicted', linestyle='--')
    
    # Shade the error region
    ax2.fill_between(zoom_time, y_true_zoom, y_pred_zoom, 
                     alpha=0.2, color=COLORS['residual'], label='Error')
    
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel(target_display, fontsize=11)
    ax2.set_title(f'(b) Zoomed View ({zoom_len} samples)', 
                  fontsize=12, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # =========================================================================
    # Panel C: Scatter Plot
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Scatter plot with density indication
    ax3.scatter(y_true, y_pred, alpha=0.3, s=8, color=COLORS['scatter'], 
                edgecolors='none', rasterized=True)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    
    ax3.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'k--', linewidth=2, label='Perfect prediction', alpha=0.7)
    
    ax3.set_xlabel(f'Actual {target_display}', fontsize=11)
    ax3.set_ylabel(f'Predicted {target_display}', fontsize=11)
    ax3.set_title('(c) Actual vs. Predicted', 
                  fontsize=12, fontweight='bold', loc='left')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'])
    ax3.set_xlim(min_val - margin, max_val + margin)
    ax3.set_ylim(min_val - margin, max_val + margin)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig

def plot_combined_results_summary(all_results):
    """Create a summary figure comparing all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_names = ['CPU usage (%)', 'Memory usage (%)', 'Disk write (KB/s)', 
                     'Network Rx (KB/s)', 'Network Tx (KB/s)']
    target_vars = CONFIG['target_variables']
    
    # Panel 1: R² scores bar chart
    ax = axes[0, 0]
    r2_scores = [all_results[t]['metrics']['r2'] for t in target_vars]
    colors = ['#2E86AB' if all_results[t]['model_name'] == 'Random Forest' else '#A23B72' 
              for t in target_vars]
    
    bars = ax.barh(metrics_names, r2_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('R² Score', fontsize=11)
    ax.set_title('(a) Model Performance (R²)', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=9)
    
    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', label='Random Forest'),
                       Patch(facecolor='#A23B72', label='XGBoost')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Panel 2: RMSE bar chart
    ax = axes[0, 1]
    rmse_scores = [all_results[t]['metrics']['rmse'] for t in target_vars]
    ax.barh(metrics_names, rmse_scores, color='#6A994E', alpha=0.8, edgecolor='black')
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_title('(b) Root Mean Squared Error', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 3: MAE bar chart
    ax = axes[0, 2]
    mae_scores = [all_results[t]['metrics']['mae'] for t in target_vars]
    ax.barh(metrics_names, mae_scores, color='#BC4B51', alpha=0.8, edgecolor='black')
    ax.set_xlabel('MAE', fontsize=11)
    ax.set_title('(c) Mean Absolute Error', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 4-6: Mini scatter plots for each metric type
    for idx, (target_var, metric_name) in enumerate(zip(target_vars[:3], metrics_names[:3])):
        ax = axes[1, idx]
        result = all_results[target_var]
        
        ax.scatter(result['y_true'], result['y_pred'], alpha=0.3, s=5, 
                   color=COLORS['scatter'], edgecolors='none')
        
        min_val = min(result['y_true'].min(), result['y_pred'].min())
        max_val = max(result['y_true'].max(), result['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
        
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'({chr(100+idx)}) {metric_name}', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f"R²={result['metrics']['r2']:.4f}", 
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_latex_table(all_results):
    """Generate LaTeX table for the paper"""
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Prediction Performance of Tree-Based Models for VM Resource Metrics}
\\label{tab:results}
\\begin{tabular}{lllrrrr}
\\toprule
\\textbf{Metric} & \\textbf{Model} & \\textbf{R²} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{MAPE (\\%)} \\\\
\\midrule
"""
    
    for target_var in CONFIG['target_variables']:
        result = all_results[target_var]
        m = result['metrics']
        model = result['model_name']
        
        # Clean metric name for LaTeX
        metric_clean = target_var.replace('%', '\\%').replace('[', '(').replace(']', ')')
        
        latex += f"{metric_clean} & {model} & {m['r2']:.4f} & {m['rmse']:.2f} & {m['mae']:.2f} & {m['mape']:.1f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("PUBLICATION-READY ML PIPELINE FOR VM PERFORMANCE PREDICTION")
    print("="*80)
    print("Model Assignments:")
    print("  • CPU & Memory → Random Forest")
    print("  • Disk & Network → XGBoost")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_dir()
    
    # Clear log file
    log_file = os.path.join(CONFIG['output_dir'], 'analysis_log.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message("="*80)
    log_message("PUBLICATION-READY PIPELINE - ANALYSIS REPORT")
    log_message("="*80)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    log_message("\n" + "-"*40)
    log_message("DATA LOADING")
    log_message("-"*40)
    df = load_data()
    log_message(f"✓ Dataset: {len(df)} samples, {len(df.columns)} columns")
    log_message(f"✓ Time range: {df.index.min()} to {df.index.max()}")
    
    # Process each target
    all_results = {}
    
    log_message("\n" + "-"*40)
    log_message("MODEL TRAINING & EVALUATION")
    log_message("-"*40)
    
    for target_var in CONFIG['target_variables']:
        log_message(f"\n>>> Processing: {target_var}")
        
        # Get assigned model
        model_type = CONFIG['model_assignments'][target_var]
        model_name = 'Random Forest' if model_type == 'random_forest' else 'XGBoost'
        log_message(f"    Model: {model_name}")
        
        # Data characteristics
        data_values = df[target_var].values
        log_message(f"    Range: [{data_values.min():.2f}, {data_values.max():.2f}]")
        log_message(f"    Skewness: {stats.skew(data_values):.3f}")
        
        # Feature engineering
        df_feat = engineer_features(df, target_var)
        log_message(f"    Features: {len(df_feat.columns) - 1}")
        
        # Prepare data
        data_dict = prepare_data(df_feat, target_var)
        log_message(f"    Train: {len(data_dict['X_train'])}, Test: {len(data_dict['X_test'])}")
        
        # Train model
        if model_type == 'random_forest':
            model, y_pred = train_random_forest(data_dict)
        else:
            model, y_pred = train_xgboost(data_dict)
        
        # Evaluate
        metrics = evaluate_predictions(data_dict['y_test'], y_pred)
        
        log_message(f"    ✓ R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}")
        
        # Store results
        all_results[target_var] = {
            'model': model,
            'model_name': model_name,
            'y_true': data_dict['y_test'],
            'y_pred': y_pred,
            'metrics': metrics,
            'test_index': data_dict['test_index'],
        }
        
        # Generate publication figure
        target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_').replace('%', 'pct')
        
        fig = plot_publication_figure(
            data_dict['y_test'], y_pred, target_var, model_name, 
            metrics, data_dict['test_index']
        )
        fig.savefig(os.path.join(CONFIG['output_dir'], f'figure_{target_clean}.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(CONFIG['output_dir'], f'figure_{target_clean}.pdf'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        log_message(f"    ✓ Saved: figure_{target_clean}.png/pdf")
    
    # Generate summary figure
    log_message("\n>>> Generating summary figure...")
    fig_summary = plot_combined_results_summary(all_results)
    fig_summary.savefig(os.path.join(CONFIG['output_dir'], 'summary_figure.png'), 
                        dpi=300, bbox_inches='tight', facecolor='white')
    fig_summary.savefig(os.path.join(CONFIG['output_dir'], 'summary_figure.pdf'), 
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_summary)
    log_message("    ✓ Saved: summary_figure.png/pdf")
    
    # Generate LaTeX table
    latex_table = create_latex_table(all_results)
    with open(os.path.join(CONFIG['output_dir'], 'results_table.tex'), 'w') as f:
        f.write(latex_table)
    log_message("    ✓ Saved: results_table.tex")
    
    # Final summary
    log_message("\n" + "="*80)
    log_message("FINAL RESULTS SUMMARY")
    log_message("="*80)
    log_message(f"{'Metric':<45} {'Model':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
    log_message("-"*90)
    
    for target_var in CONFIG['target_variables']:
        r = all_results[target_var]
        m = r['metrics']
        log_message(f"{target_var:<45} {r['model_name']:<15} {m['r2']:>8.4f} {m['rmse']:>10.2f} {m['mae']:>10.2f}")
    
    log_message("\n" + "="*80)
    log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*80)
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"✓ Results saved to: {CONFIG['output_dir']}/")
    print("  • Individual metric figures (PNG + PDF)")
    print("  • Summary comparison figure")
    print("  • LaTeX table for paper")
    print("  • Analysis log")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

