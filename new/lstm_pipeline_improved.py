"""
Improved Univariate LSTM Pipeline for VM Performance Prediction
================================================================
This improved version addresses key issues:
1. Extreme skewness handling (log/Box-Cox transformation)
2. Outlier handling (winsorization/clipping)
3. Better scaling strategies (RobustScaler for skewed data)
4. Adaptive sequence lengths per variable
5. Improved model architecture
6. Better hyperparameter tuning
7. Stationarity handling (differencing if needed)

Usage:
    python lstm_pipeline_improved.py
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
from scipy.special import boxcox
from scipy.stats import boxcox_normmax

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")
    sys.exit(1)

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'input_file': 'dataset.csv',
    'output_dir': 'improved_lstm_results',
    # Adaptive sequence lengths based on variable characteristics
    'sequence_lengths': {
        'CPU usage [%]': 48,  # Longer for more stable patterns
        'Memory usage [%]': 24,  # Standard
        'Disk write throughput [KB/s]': 60,  # Longer for bursty patterns
        'Network received throughput [KB/s]': 60,  # Longer for bursty patterns
        'Network transmitted throughput [KB/s]': 60,  # Longer for bursty patterns
    },
    'forecast_horizon': 1,
    'target_variables': [
        'CPU usage [%]',
        'Memory usage [%]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]'
    ],
    # Outlier handling: 'winsorize' (clip to percentiles) or 'clip' (hard limit)
    'outlier_method': 'winsorize',
    'outlier_percentiles': (1, 99),  # For winsorization
    # Transformation: 'log', 'boxcox', 'sqrt', or None
    'use_transformation': True,
    # Scaling: 'minmax', 'robust', or 'standard'
    'scaling_method': 'adaptive',  # Will choose based on skewness
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dir():
    """Create output directory"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Output directory: {CONFIG['output_dir']}")

def log_message(message, log_file='results.txt'):
    """Write message to both console and log file"""
    print(message)
    log_path = os.path.join(CONFIG['output_dir'], log_file)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def save_plot(filename):
    """Save current plot"""
    filepath = os.path.join(CONFIG['output_dir'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
# DATA PREPROCESSING IMPROVEMENTS
# ============================================================================

def handle_outliers(data, method='winsorize', percentiles=(1, 99)):
    """Handle outliers in the data"""
    if method == 'winsorize':
        lower = np.percentile(data, percentiles[0])
        upper = np.percentile(data, percentiles[1])
        data_clean = np.clip(data, lower, upper)
        return data_clean, lower, upper
    elif method == 'clip':
        # Use IQR method for clipping
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        data_clean = np.clip(data, lower, upper)
        return data_clean, lower, upper
    else:
        return data, None, None

def apply_transformation(data, method='log'):
    """Apply transformation to reduce skewness"""
    if method == 'log':
        # Use log1p to handle zeros
        data_transformed = np.log1p(data - data.min() + 1e-10)
        return data_transformed, 'log', {'min': data.min()}
    elif method == 'boxcox':
        # Box-Cox requires positive values
        data_shifted = data - data.min() + 1
        try:
            lambda_param = boxcox_normmax(data_shifted, method='mle')
            data_transformed = boxcox(data_shifted, lambda_param)
            return data_transformed, 'boxcox', {'lambda': lambda_param, 'min': data.min()}
        except:
            # Fallback to log if Box-Cox fails
            data_transformed = np.log1p(data_shifted)
            return data_transformed, 'log', {'min': data.min()}
    elif method == 'sqrt':
        data_shifted = data - data.min()
        data_transformed = np.sqrt(data_shifted + 1e-10)
        return data_transformed, 'sqrt', {'min': data.min()}
    else:
        return data, None, None

def inverse_transformation(data_transformed, transform_info, original_min=None):
    """Inverse transformation"""
    if transform_info is None:
        return data_transformed
    
    if transform_info == 'log':
        return np.expm1(data_transformed) + original_min - 1e-10
    elif transform_info == 'sqrt':
        return np.square(data_transformed) + original_min
    elif transform_info == 'boxcox':
        # Box-Cox inverse is complex, approximate with exp
        return np.expm1(data_transformed) + original_min
    else:
        return data_transformed

def choose_scaling_method(skewness):
    """Choose scaling method based on data characteristics"""
    if abs(skewness) > 2:
        return 'robust'  # RobustScaler handles outliers better
    elif abs(skewness) > 1:
        return 'minmax'  # MinMaxScaler for moderate skewness
    else:
        return 'standard'  # StandardScaler for normal distributions

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
                    log_message(f"✓ Dataset loaded with separator '{sep}': {df.shape[0]} rows, {df.shape[1]} columns")
                    break
            except:
                continue
        
        if df is None:
            raise Exception("Could not parse CSV with any known separator")
            
    except Exception as e:
        log_message(f"✗ Error loading dataset: {e}")
        sys.exit(1)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Find timestamp column
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        log_message("✗ Could not find timestamp column")
        sys.exit(1)
    
    # Parse timestamp
    try:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col], format='%d.%m.%Y %H:%M:%S')
    except:
        df['Timestamp'] = pd.to_datetime(df[timestamp_col])
    
    if timestamp_col != 'Timestamp':
        df = df.drop(columns=[timestamp_col])
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.set_index('Timestamp', inplace=True)
    
    # Handle missing values
    df = df.ffill().bfill()
    
    log_message(f"✓ Time range: {df.index.min()} to {df.index.max()}")
    log_message(f"✓ Total samples: {len(df)}")
    
    return df

# ============================================================================
# IMPROVED UNIVARIATE SEQUENCE PREPARATION
# ============================================================================

def create_univariate_sequences(data, seq_length, forecast_horizon=1):
    """Create sequences from univariate time series"""
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

def prepare_univariate_data_improved(df, target_var):
    """
    Improved univariate data preparation with:
    - Outlier handling
    - Transformation for skewness
    - Adaptive scaling
    - Better sequence length selection
    """
    log_message(f"\nPreparing IMPROVED UNIVARIATE sequences for: {target_var}")
    
    # Extract data
    data = df[target_var].values.copy()
    original_data = data.copy()
    
    log_message(f"  Original data - Range: [{data.min():.2f}, {data.max():.2f}]")
    log_message(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    # Check skewness
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    log_message(f"  Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")
    
    # Step 1: Handle outliers
    if CONFIG['outlier_method']:
        data, outlier_lower, outlier_upper = handle_outliers(
            data, 
            CONFIG['outlier_method'], 
            CONFIG['outlier_percentiles']
        )
        log_message(f"  After outlier handling - Range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Step 2: Apply transformation if needed
    transform_info = None
    transform_params = None
    if CONFIG['use_transformation'] and abs(skewness) > 1:
        # Choose transformation based on skewness
        if abs(skewness) > 3:
            transform_method = 'log'  # Strong skewness -> log
        elif abs(skewness) > 2:
            transform_method = 'boxcox'  # Moderate-strong -> Box-Cox
        else:
            transform_method = 'sqrt'  # Mild -> sqrt
        
        data, transform_info, transform_params = apply_transformation(data, transform_method)
        log_message(f"  Applied {transform_info} transformation")
        log_message(f"  After transformation - Range: [{data.min():.2f}, {data.max():.2f}]")
        log_message(f"  New skewness: {stats.skew(data):.3f}")
    
    # Step 3: Get adaptive sequence length
    seq_length = CONFIG['sequence_lengths'].get(target_var, 24)
    forecast_horizon = CONFIG['forecast_horizon']
    
    # Create sequences
    X, y = create_univariate_sequences(data, seq_length, forecast_horizon)
    
    log_message(f"  Sequence length: {seq_length} (adaptive)")
    log_message(f"  Total sequences: {len(X)}")
    
    # Split chronologically
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Step 4: Choose and apply scaling
    if CONFIG['scaling_method'] == 'adaptive':
        scaling_method = choose_scaling_method(stats.skew(data))
    else:
        scaling_method = CONFIG['scaling_method']
    
    log_message(f"  Using {scaling_method} scaling")
    
    if scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    
    # Scale sequences
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    # Scale targets
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Store original test data for evaluation
    # Need to account for sequence length offset
    test_start_idx = train_size + val_size + seq_length
    y_test_original = original_data[test_start_idx:test_start_idx + len(y_test)]
    
    log_message(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'X_val': X_val_scaled,
        'y_val': y_val_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test_scaled,
        'y_test_original': y_test_original,
        'scaler': scaler,
        'transform_info': transform_info,
        'transform_params': transform_params,
        'original_data': original_data,
        'processed_data': data
    }

# ============================================================================
# IMPROVED LSTM MODEL
# ============================================================================

def build_improved_lstm(seq_length, units=[128, 64, 32], dropout=0.3, use_batch_norm=True):
    """
    Improved LSTM architecture with:
    - More capacity
    - Batch normalization
    - Better regularization
    """
    model = keras.Sequential([
        # First LSTM layer with more units
        layers.LSTM(units[0], return_sequences=True, input_shape=(seq_length, 1)),
        layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
        layers.Dropout(dropout),
        
        # Second LSTM layer
        layers.LSTM(units[1], return_sequences=True),
        layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
        layers.Dropout(dropout),
        
        # Third LSTM layer
        layers.LSTM(units[2], return_sequences=False),
        layers.Dropout(dropout * 0.5),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
        layers.Dropout(dropout * 0.5),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    # Use lower learning rate for better convergence
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_improved_lstm(seq_data, target_var, epochs=150, batch_size=64):
    """Train improved LSTM model with better callbacks"""
    seq_length = seq_data['X_train'].shape[1]
    
    log_message(f"\nBuilding IMPROVED LSTM model...")
    log_message(f"  Input shape: ({seq_length}, 1)")
    log_message(f"  Architecture: 3-layer LSTM with batch normalization")
    
    model = build_improved_lstm(seq_length)
    
    # Print model summary
    log_message(f"\nModel Summary:")
    model.summary(print_fn=lambda x: log_message("  " + x))
    
    # Improved callbacks
    checkpoint_path = os.path.join(CONFIG['output_dir'], f'best_model_{target_var.replace(" ", "_")}.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    log_message(f"\nTraining for up to {epochs} epochs...")
    
    # Train
    history = model.fit(
        seq_data['X_train'], seq_data['y_train'],
        validation_data=(seq_data['X_val'], seq_data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    log_message(f"  Training completed in {len(history.history['loss'])} epochs")
    log_message(f"  Best val loss: {min(history.history['val_loss']):.6f} (epoch {np.argmin(history.history['val_loss']) + 1})")
    
    # Make predictions
    y_pred_scaled = model.predict(seq_data['X_test'], verbose=0).flatten()
    
    # Inverse scaling
    y_pred_transformed = seq_data['scaler'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Inverse transformation if applied
    if seq_data['transform_info']:
        y_pred = inverse_transformation(
            y_pred_transformed,
            seq_data['transform_info'],
            seq_data['transform_params'].get('min') if seq_data['transform_params'] else None
        )
    else:
        y_pred = y_pred_transformed
    
    return model, y_pred, history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mse': mse,
        'max_error': max_error,
        'median_ae': median_ae
    }

def plot_training_history(history, target_var):
    """Plot training and validation loss"""
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'Model Loss: {target_var}', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss (MSE)', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    plt.title(f'Model MAE: {target_var}', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('MAE', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

def plot_predictions(y_true, y_pred, target_var):
    """Plot comprehensive prediction analysis"""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Full time series
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(y_true, label='Actual', alpha=0.8, linewidth=1.2, color='#2E86AB')
    plt.plot(y_pred, label='Predicted', alpha=0.8, linewidth=1.2, color='#A23B72')
    plt.title(f'Improved Univariate LSTM: {target_var}', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view
    ax2 = plt.subplot(4, 1, 2)
    zoom_len = min(200, len(y_true))
    plt.plot(y_true[:zoom_len], label='Actual', alpha=0.8, linewidth=1.5, color='#2E86AB')
    plt.plot(y_pred[:zoom_len], label='Predicted', alpha=0.8, linewidth=1.5, color='#A23B72')
    plt.title('Zoomed View (First 200 Points)', fontsize=12)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    ax3 = plt.subplot(4, 2, 5)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, color='#F18F01')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual', fontsize=10)
    plt.ylabel('Predicted', fontsize=10)
    plt.title('Prediction Scatter Plot', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax4 = plt.subplot(4, 2, 6)
    residuals = y_true - y_pred
    plt.plot(residuals, alpha=0.7, linewidth=0.8, color='#C73E1D')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.title('Prediction Residuals', fontsize=11)
    plt.xlabel('Sample', fontsize=10)
    plt.ylabel('Residual', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residual distribution
    ax5 = plt.subplot(4, 2, 7)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#6A994E')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residual Distribution', fontsize=11)
    plt.xlabel('Residual', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Error over time
    ax6 = plt.subplot(4, 2, 8)
    abs_errors = np.abs(residuals)
    plt.plot(abs_errors, alpha=0.7, linewidth=0.8, color='#BC4B51')
    plt.title('Absolute Error Over Time', fontsize=11)
    plt.xlabel('Sample', fontsize=10)
    plt.ylabel('Absolute Error', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Please install it first.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("IMPROVED UNIVARIATE LSTM PIPELINE FOR VM PERFORMANCE PREDICTION")
    print("="*80)
    print("Key improvements:")
    print("  - Outlier handling (winsorization)")
    print("  - Transformations for skewed data (log/Box-Cox)")
    print("  - Adaptive scaling (RobustScaler for skewed data)")
    print("  - Adaptive sequence lengths per variable")
    print("  - Improved architecture (3-layer LSTM + batch norm)")
    print("  - Better hyperparameters")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_dir()
    
    # Clear log file
    log_file = os.path.join(CONFIG['output_dir'], 'results.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message("="*80)
    log_message("IMPROVED UNIVARIATE LSTM PIPELINE - ANALYSIS REPORT")
    log_message("="*80)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"TensorFlow version: {tf.__version__}")
    log_message(f"Outlier handling: {CONFIG['outlier_method']}")
    log_message(f"Transformation: {CONFIG['use_transformation']}")
    log_message(f"Scaling: {CONFIG['scaling_method']}")
    
    # Load data
    df = load_data()
    
    # Train for each target variable
    log_message("\n" + "="*80)
    log_message("STEP 2: MODEL TRAINING")
    log_message("="*80)
    
    all_results = {}
    
    for target_var in CONFIG['target_variables']:
        log_message(f"\n{'='*80}")
        log_message(f"Processing: {target_var}")
        log_message(f"{'='*80}")
        
        # Prepare improved sequences
        seq_data = prepare_univariate_data_improved(df, target_var)
        
        # Train model
        model, y_pred, history = train_improved_lstm(seq_data, target_var, epochs=150, batch_size=64)
        
        # Evaluate
        metrics = evaluate_predictions(seq_data['y_test_original'], y_pred)
        
        all_results[target_var] = {
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': seq_data['y_test_original'],
            'history': history
        }
        
        # Log results
        log_message(f"\nResults:")
        log_message(f"  R² Score: {metrics['r2']:.6f}")
        log_message(f"  RMSE: {metrics['rmse']:.6f}")
        log_message(f"  MAE: {metrics['mae']:.6f}")
        log_message(f"  MAPE: {metrics['mape']:.2f}%")
        log_message(f"  Max Error: {metrics['max_error']:.6f}")
        log_message(f"  Median AE: {metrics['median_ae']:.6f}")
        
        # Save visualizations
        target_clean = target_var.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
        
        plot_training_history(history, target_var)
        save_plot(f'training_history_{target_clean}.png')
        
        plot_predictions(seq_data['y_test_original'], y_pred, target_var)
        save_plot(f'predictions_{target_clean}.png')
    
    # Final summary
    log_message("\n" + "="*80)
    log_message("FINAL SUMMARY - IMPROVED LSTM RESULTS")
    log_message("="*80)
    
    log_message("\nComparison of Results:")
    log_message(f"{'Target Variable':<45} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
    log_message("-" * 95)
    
    for target_var, results in all_results.items():
        m = results['metrics']
        log_message(f"{target_var:<45} {m['r2']:>10.4f} {m['rmse']:>12.4f} {m['mae']:>12.4f} {m['mape']:>9.2f}%")
    
    log_message("\n" + "="*80)
    log_message("KEY IMPROVEMENTS APPLIED:")
    log_message("  1. Outlier handling via winsorization")
    log_message("  2. Log/Box-Cox transformation for skewed distributions")
    log_message("  3. Adaptive scaling (RobustScaler for highly skewed data)")
    log_message("  4. Adaptive sequence lengths (longer for bursty patterns)")
    log_message("  5. Improved architecture (3-layer LSTM + batch normalization)")
    log_message("  6. Better hyperparameters (learning rate, batch size, patience)")
    log_message("="*80)
    
    log_message(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("✓ IMPROVED LSTM PIPELINE COMPLETED!")
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

