"""
IMPROVED Univariate LSTM Pipeline for VM Performance Prediction
================================================================
This version includes advanced techniques to handle:
- Highly skewed data (extreme outliers)
- Zero-inflated time series (network/disk)
- Bursty behavior (rare spikes)

Key Improvements:
1. Log transformation for skewed data
2. Robust scaling (handle outliers better)
3. Custom loss function (Huber loss)
4. Bidirectional LSTM
5. Better early stopping strategy
6. Handling of different data types (smooth vs bursty)

Usage:
    python improved_lstm_pipeline.py
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available.")
    sys.exit(1)

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'output_dir': 'improved_lstm_results',
    'sequence_length': 48,  # Increased from 24 to capture more patterns
    'forecast_horizon': 1,
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
    
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        log_message("✗ No timestamp column found")
        sys.exit(1)
    
    try:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col], format='%d.%m.%Y %H:%M:%S')
    except:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col])
    
    if timestamp_col != 'Timestamp':
        df = df.drop(columns=[timestamp_col])
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.set_index('Timestamp', inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    log_message(f"✓ Time range: {df.index.min()} to {df.index.max()}")
    log_message(f"✓ Total samples: {len(df)}")
    
    return df

# ============================================================================
# ADVANCED DATA PREPROCESSING
# ============================================================================

def analyze_data_characteristics(data, var_name):
    """Analyze data to determine preprocessing strategy"""
    log_message(f"\n  Analyzing '{var_name}' characteristics:")
    
    # Basic stats
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Zero-inflation check
    zero_pct = (data == 0).sum() / len(data) * 100
    
    # Sparsity (values near zero)
    near_zero_pct = (data < mean * 0.1).sum() / len(data) * 100
    
    # Outliers
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outlier_pct = ((data < q1 - 3*iqr) | (data > q3 + 3*iqr)).sum() / len(data) * 100
    
    log_message(f"    Mean: {mean:.2f}, Std: {std:.2f}, Median: {median:.2f}")
    log_message(f"    Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")
    log_message(f"    Zeros: {zero_pct:.2f}%, Near-zero: {near_zero_pct:.2f}%")
    log_message(f"    Outliers (3×IQR): {outlier_pct:.2f}%")
    
    # Determine data type
    is_smooth = skewness < 2 and kurtosis < 5 and zero_pct < 5
    is_bursty = skewness > 10 or kurtosis > 50 or zero_pct > 10
    is_zero_inflated = zero_pct > 20
    
    if is_smooth:
        data_type = "SMOOTH"
        log_message(f"    → Type: SMOOTH (well-behaved, normally distributed)")
    elif is_zero_inflated:
        data_type = "ZERO_INFLATED"
        log_message(f"    → Type: ZERO-INFLATED (many zeros, occasional activity)")
    elif is_bursty:
        data_type = "BURSTY"
        log_message(f"    → Type: BURSTY (rare extreme spikes)")
    else:
        data_type = "MODERATE"
        log_message(f"    → Type: MODERATE (somewhat skewed)")
    
    return {
        'type': data_type,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'zero_pct': zero_pct,
        'use_log': skewness > 2 and data.min() >= 0
    }

def preprocess_data(data, characteristics):
    """
    Apply appropriate preprocessing based on data characteristics
    """
    data_processed = data.copy()
    transform_info = {'method': 'none', 'params': {}}
    
    if characteristics['type'] in ['BURSTY', 'ZERO_INFLATED']:
        # Log transform for highly skewed data (add 1 to handle zeros)
        if characteristics['use_log']:
            data_processed = np.log1p(data_processed)
            transform_info['method'] = 'log1p'
            log_message(f"    Applied: log1p transformation")
        
        # Clip extreme outliers
        p99 = np.percentile(data_processed, 99)
        p01 = np.percentile(data_processed, 1)
        data_clipped = np.clip(data_processed, p01, p99)
        outliers_clipped = (data_clipped != data_processed).sum()
        if outliers_clipped > 0:
            data_processed = data_clipped
            log_message(f"    Applied: Clipping ({outliers_clipped} outliers)")
    
    return data_processed, transform_info

def create_sequences_advanced(data, seq_length, forecast_horizon=1):
    """Create sequences with the processed data"""
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

def prepare_advanced_data(df, target_var):
    """Advanced data preparation with automatic preprocessing"""
    log_message(f"\n{'='*80}")
    log_message(f"Preparing data for: {target_var}")
    log_message(f"{'='*80}")
    
    # Extract data
    data = df[target_var].values
    log_message(f"  Total samples: {len(data)}")
    log_message(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Analyze characteristics
    characteristics = analyze_data_characteristics(data, target_var)
    
    # Preprocess
    data_processed, transform_info = preprocess_data(data, characteristics)
    log_message(f"  After preprocessing: [{data_processed.min():.2f}, {data_processed.max():.2f}]")
    
    # Create sequences
    seq_length = CONFIG['sequence_length']
    X, y = create_sequences_advanced(data_processed, seq_length, CONFIG['forecast_horizon'])
    
    log_message(f"  Sequences created: {len(X)}")
    log_message(f"  Sequence length: {seq_length}, Forecast: {CONFIG['forecast_horizon']}")
    
    # Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Choose scaler based on data type
    if characteristics['type'] in ['BURSTY', 'ZERO_INFLATED']:
        # RobustScaler is better for outlier-heavy data
        scaler = RobustScaler()
        log_message(f"  Using: RobustScaler (resistant to outliers)")
    else:
        scaler = StandardScaler()
        log_message(f"  Using: StandardScaler")
    
    # Scale
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    log_message(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Store original y_test BEFORE any transformation for proper evaluation
    original_indices = np.arange(train_size + val_size, len(data) - seq_length - CONFIG['forecast_horizon'] + 1)
    y_test_original = data[original_indices + seq_length + CONFIG['forecast_horizon'] - 1]
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'X_val': X_val_scaled,
        'y_val': y_val_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test_scaled,
        'y_test_original': y_test_original,
        'scaler': scaler,
        'characteristics': characteristics,
        'transform_info': transform_info
    }

# ============================================================================
# IMPROVED LSTM MODEL
# ============================================================================

def build_improved_lstm(seq_length, data_type):
    """
    Build LSTM with architecture adapted to data type
    """
    if data_type in ['BURSTY', 'ZERO_INFLATED']:
        # Simpler model for bursty data (prevent overfitting to noise)
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(32, return_sequences=True), input_shape=(seq_length, 1)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(16)),
            layers.Dropout(0.3),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        loss = 'huber'  # More robust to outliers than MSE
        log_message(f"    Architecture: Simplified Bidirectional LSTM (bursty data)")
        log_message(f"    Loss function: Huber (robust to outliers)")
    else:
        # Standard model for smooth data
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(seq_length, 1)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        loss = 'mse'
        log_message(f"    Architecture: Standard Bidirectional LSTM")
        log_message(f"    Loss function: MSE")
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    return model

def train_improved_lstm(seq_data, epochs=150):
    """Train with improved strategy"""
    seq_length = seq_data['X_train'].shape[1]
    data_type = seq_data['characteristics']['type']
    
    log_message(f"\n  Building model...")
    model = build_improved_lstm(seq_length, data_type)
    
    # Better callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,  # More patience
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=0
    )
    
    # Save best model
    checkpoint_path = os.path.join(CONFIG['output_dir'], 'best_model_temp.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    log_message(f"  Training (max {epochs} epochs)...")
    
    history = model.fit(
        seq_data['X_train'], seq_data['y_train'],
        validation_data=(seq_data['X_val'], seq_data['y_val']),
        epochs=epochs,
        batch_size=64,  # Larger batch for stability
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=0
    )
    
    # Load best model
    if os.path.exists(checkpoint_path):
        model = keras.models.load_model(checkpoint_path)
        os.remove(checkpoint_path)
    
    log_message(f"  Completed in {len(history.history['loss'])} epochs")
    log_message(f"  Best val loss: {min(history.history['val_loss']):.6f}")
    
    # Predict
    y_pred_scaled = model.predict(seq_data['X_test'], verbose=0).flatten()
    
    # Inverse transform
    y_pred_transformed = seq_data['scaler'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Reverse log transform if applied
    if seq_data['transform_info']['method'] == 'log1p':
        y_pred = np.expm1(y_pred_transformed)
        log_message(f"  Applied inverse: expm1 (reverse log1p)")
    else:
        y_pred = y_pred_transformed
    
    # Ensure non-negative for physical quantities
    y_pred = np.maximum(y_pred, 0)
    
    return model, y_pred, history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """Comprehensive evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (careful with zeros)
    mask = y_true > 0.01  # Avoid division by near-zero
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    # Explained variance
    explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'max_error': max_error,
        'median_ae': median_ae,
        'explained_var': explained_var
    }

def plot_results(y_true, y_pred, history, target_var):
    """Comprehensive visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # Training history
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='Val', linewidth=1.5)
    plt.title('Training History', fontsize=11, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Full predictions
    ax2 = plt.subplot(3, 2, 2)
    plt.plot(y_true, label='Actual', alpha=0.7, linewidth=1)
    plt.plot(y_pred, label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'{target_var}', fontsize=11, fontweight='bold')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed view
    ax3 = plt.subplot(3, 2, 3)
    zoom = min(300, len(y_true))
    plt.plot(y_true[:zoom], label='Actual', linewidth=1.5)
    plt.plot(y_pred[:zoom], label='Predicted', linewidth=1.5)
    plt.title('Zoomed View (First 300 points)', fontsize=10)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter
    ax4 = plt.subplot(3, 2, 4)
    plt.scatter(y_true, y_pred, alpha=0.4, s=8)
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Residuals
    ax5 = plt.subplot(3, 2, 5)
    residuals = y_true - y_pred
    plt.plot(residuals, alpha=0.6, linewidth=0.8)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residuals', fontsize=10)
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    ax6 = plt.subplot(3, 2, 6)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.title('Error Distribution', fontsize=10)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow required.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("IMPROVED LSTM PIPELINE - VM PERFORMANCE PREDICTION")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_dir()
    
    log_file = os.path.join(CONFIG['output_dir'], 'results.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message("="*80)
    log_message("IMPROVED LSTM PIPELINE - ANALYSIS REPORT")
    log_message("="*80)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"TensorFlow: {tf.__version__}")
    log_message("\nKey Improvements:")
    log_message("  1. Automatic data type detection (smooth/bursty/zero-inflated)")
    log_message("  2. Adaptive preprocessing (log transform, clipping)")
    log_message("  3. Robust scaling for outlier-heavy data")
    log_message("  4. Bidirectional LSTM for better pattern capture")
    log_message("  5. Huber loss for bursty data (robust to outliers)")
    log_message("  6. Longer sequence length (48 vs 24)")
    log_message("  7. Better early stopping (patience=25)")
    
    df = load_data()
    
    log_message("\n" + "="*80)
    log_message("STEP 2: MODEL TRAINING")
    log_message("="*80)
    
    all_results = {}
    
    for target_var in CONFIG['target_variables']:
        # Prepare data
        seq_data = prepare_advanced_data(df, target_var)
        
        # Train
        model, y_pred, history = train_improved_lstm(seq_data, epochs=150)
        
        # Evaluate
        metrics = evaluate_predictions(seq_data['y_test_original'], y_pred)
        
        all_results[target_var] = {
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': seq_data['y_test_original'],
            'history': history,
            'characteristics': seq_data['characteristics']
        }
        
        # Log
        log_message(f"\n  Results:")
        log_message(f"    R²: {metrics['r2']:.6f}")
        log_message(f"    RMSE: {metrics['rmse']:.6f}")
        log_message(f"    MAE: {metrics['mae']:.6f}")
        log_message(f"    MAPE: {metrics['mape']:.2f}%")
        log_message(f"    Explained Variance: {metrics['explained_var']:.6f}")
        
        # Plot
        target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
        plot_results(seq_data['y_test_original'], y_pred, history, target_var)
        save_plot(f'results_{target_clean}.png')
    
    # Summary
    log_message("\n" + "="*80)
    log_message("FINAL SUMMARY")
    log_message("="*80)
    log_message(f"\n{'Target':<45} {'Type':<15} {'R²':>10} {'RMSE':>12} {'MAE':>12}")
    log_message("-" * 95)
    
    for target_var, results in all_results.items():
        m = results['metrics']
        t = results['characteristics']['type']
        log_message(f"{target_var:<45} {t:<15} {m['r2']:>10.4f} {m['rmse']:>12.4f} {m['mae']:>12.4f}")
    
    log_message(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("✓ IMPROVED LSTM COMPLETED!")
    print(f"✓ Results in '{CONFIG['output_dir']}'")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)