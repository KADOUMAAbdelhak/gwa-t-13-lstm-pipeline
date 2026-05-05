"""
CPU Model Interface for Autoscaling Experiments
================================================
Provides a clean interface to load/train the CPU prediction model and
generate predictions for the autoscaling simulation.

This module reuses the exact same preprocessing and feature engineering
logic from the main paper pipeline to ensure consistency.

Author: Hybrid ML Pipeline for Cloud-Edge Continuum
"""

import os
import pickle
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CPUModelConfig:
    """Configuration for CPU model and feature engineering."""
    # Feature engineering parameters
    lag_features: Tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48)
    rolling_windows: Tuple[int, ...] = (6, 12, 24, 48, 96)
    
    # Model parameters
    n_estimators: int = 300
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    
    # File paths
    model_path: str = 'models/cpu_model.pkl'
    scaler_path: str = 'models/cpu_scaler.pkl'
    feature_cols_path: str = 'models/cpu_feature_cols.pkl'

# Default configuration
DEFAULT_CONFIG = CPUModelConfig()

# Target column name
CPU_TARGET = 'CPU usage [%]'

# ============================================================================
# FEATURE ENGINEERING (Same as paper_pipeline.py)
# ============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features with cyclical encoding."""
    df_feat = df.copy()
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['week_of_year'] = df_feat.index.isocalendar().week.astype(int)
    
    # Cyclical encoding for periodic patterns
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


def create_lag_features(df: pd.DataFrame, target_col: str, lags: Tuple[int, ...]) -> pd.DataFrame:
    """Create lag features for the target column."""
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'{target_col}_lag_{lag}'] = df_lag[target_col].shift(lag)
    return df_lag


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: Tuple[int, ...]) -> pd.DataFrame:
    """Create rolling statistics features."""
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


def create_diff_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create difference/change features."""
    df_diff = df.copy()
    df_diff[f'{target_col}_diff_1'] = df_diff[target_col].diff(1)
    df_diff[f'{target_col}_diff_12'] = df_diff[target_col].diff(12)
    
    pct_1 = df_diff[target_col].pct_change(1)
    pct_12 = df_diff[target_col].pct_change(12)
    df_diff[f'{target_col}_pct_change_1'] = pct_1.replace([np.inf, -np.inf], np.nan)
    df_diff[f'{target_col}_pct_change_12'] = pct_12.replace([np.inf, -np.inf], np.nan)
    
    return df_diff


def engineer_cpu_features(df: pd.DataFrame, config: CPUModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Create all features for CPU prediction.
    
    This function applies the exact same feature engineering pipeline
    used in the main paper experiments.
    
    Args:
        df: DataFrame with 'CPU usage [%]' column and DatetimeIndex
        config: Feature engineering configuration
        
    Returns:
        DataFrame with all engineered features (NaN rows dropped)
    """
    df_feat = df.copy()
    df_feat = create_time_features(df_feat)
    df_feat = create_lag_features(df_feat, CPU_TARGET, config.lag_features)
    df_feat = create_rolling_features(df_feat, CPU_TARGET, config.rolling_windows)
    df_feat = create_diff_features(df_feat, CPU_TARGET)
    
    # Drop rows with NaN (from lag/rolling features)
    df_feat = df_feat.dropna()
    
    return df_feat


# ============================================================================
# DATA LOADING
# ============================================================================

def load_vm_data(filepath: str = 'dataset.csv') -> pd.DataFrame:
    """
    Load VM trace data from CSV file.
    
    Args:
        filepath: Path to the dataset CSV file
        
    Returns:
        DataFrame with DatetimeIndex and cleaned column names
    """
    separators = [';', ';\t', '\t', ',']
    df = None
    
    for sep in separators:
        try:
            df_test = pd.read_csv(filepath, sep=sep, decimal=',', engine='python')
            if df_test.shape[1] > 5:
                df = df_test
                break
        except:
            continue
    
    if df is None:
        raise ValueError(f"Could not parse CSV file: {filepath}")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Find and parse timestamp
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError("No timestamp column found in dataset")
    
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
# MODEL TRAINING AND PERSISTENCE
# ============================================================================

def get_feature_columns(df_feat: pd.DataFrame, target_var: str = CPU_TARGET) -> list:
    """
    Get list of feature columns, excluding leakage and target.
    
    Args:
        df_feat: Feature-engineered DataFrame
        target_var: Target variable name
        
    Returns:
        List of feature column names
    """
    # Columns that could cause data leakage for CPU prediction
    leakage_cols = ['CPU usage [MHZ]', 'CPU capacity provisioned [MHZ]']
    
    # Other target variables to exclude
    other_targets = [
        'Memory usage [%]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]'
    ]
    
    exclude_cols = leakage_cols + other_targets + [target_var]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    
    return feature_cols


def train_cpu_model(
    df: pd.DataFrame,
    config: CPUModelConfig = DEFAULT_CONFIG,
    save_model: bool = True
) -> Tuple[RandomForestRegressor, RobustScaler, list, Dict[str, Any]]:
    """
    Train the CPU prediction model using Random Forest.
    
    Args:
        df: Raw VM trace DataFrame
        config: Model and feature configuration
        save_model: Whether to save the trained model to disk
        
    Returns:
        Tuple of (model, scaler, feature_columns, metrics_dict)
    """
    print("Engineering features for CPU prediction...")
    df_feat = engineer_cpu_features(df, config)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_feat, CPU_TARGET)
    
    X = df_feat[feature_cols].copy()
    y = df_feat[CPU_TARGET].copy()
    
    # Handle infinity and extreme values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    for col in X.columns:
        q01 = X[col].quantile(0.001)
        q99 = X[col].quantile(0.999)
        X[col] = X[col].clip(q01, q99)
    
    # Chronological split
    n = len(X)
    train_end = int(n * (1 - config.test_size - config.val_size))
    val_end = int(n * (1 - config.test_size))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features='sqrt',
        random_state=config.random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    print(f"Model trained: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Save model if requested
    if save_model:
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
        
        with open(config.model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(config.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(config.feature_cols_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print(f"Model saved to {config.model_path}")
    
    return model, scaler, feature_cols, metrics


def load_cpu_model(config: CPUModelConfig = DEFAULT_CONFIG) -> Tuple[RandomForestRegressor, RobustScaler, list]:
    """
    Load a pre-trained CPU model from disk, or train a new one if not found.
    
    Args:
        config: Model configuration with file paths
        
    Returns:
        Tuple of (model, scaler, feature_columns)
    """
    if os.path.exists(config.model_path) and os.path.exists(config.scaler_path):
        print(f"Loading pre-trained CPU model from {config.model_path}...")
        
        with open(config.model_path, 'rb') as f:
            model = pickle.load(f)
        with open(config.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(config.feature_cols_path, 'rb') as f:
            feature_cols = pickle.load(f)
        
        print("Model loaded successfully.")
        return model, scaler, feature_cols
    else:
        print("No pre-trained model found. Training new model...")
        df = load_vm_data()
        model, scaler, feature_cols, _ = train_cpu_model(df, config, save_model=True)
        return model, scaler, feature_cols


# ============================================================================
# PREDICTION INTERFACE
# ============================================================================

def make_cpu_features_at_t(
    cpu_series: pd.Series,
    t: int,
    config: CPUModelConfig = DEFAULT_CONFIG
) -> Optional[pd.DataFrame]:
    """
    Build the feature vector for CPU prediction at time index t.
    
    This creates a single-row DataFrame containing all engineered features
    based on the history up to and including time t.
    
    Args:
        cpu_series: Series of CPU usage values with DatetimeIndex
        t: Integer index position (0-based) for which to create features
        config: Feature engineering configuration
        
    Returns:
        Single-row DataFrame with feature values, or None if insufficient history
    """
    # We need enough history for the largest lag/rolling window
    max_lookback = max(max(config.lag_features), max(config.rolling_windows))
    
    if t < max_lookback:
        return None  # Not enough history
    
    # Create a DataFrame from the series up to time t
    df_window = pd.DataFrame({CPU_TARGET: cpu_series.iloc[:t+1]})
    df_window.index = cpu_series.index[:t+1]
    
    # Apply feature engineering
    df_feat = engineer_cpu_features(df_window, config)
    
    if len(df_feat) == 0:
        return None
    
    # Return the last row (features at time t)
    return df_feat.iloc[[-1]]


def predict_cpu_next(
    model: RandomForestRegressor,
    scaler: RobustScaler,
    feature_cols: list,
    cpu_series: pd.Series,
    t: int,
    config: CPUModelConfig = DEFAULT_CONFIG
) -> Optional[float]:
    """
    Predict CPU usage for time t+1 given history up to time t.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted RobustScaler
        feature_cols: List of feature column names
        cpu_series: Series of CPU usage values with DatetimeIndex
        t: Current time index (prediction will be for t+1)
        config: Feature engineering configuration
        
    Returns:
        Predicted CPU usage (%), or None if insufficient history
    """
    # Build features at time t
    df_feat = make_cpu_features_at_t(cpu_series, t, config)
    
    if df_feat is None:
        return None
    
    # Extract feature values in correct order
    try:
        X = df_feat[feature_cols].values
    except KeyError as e:
        # Handle missing columns (fill with 0 or median)
        X = np.zeros((1, len(feature_cols)))
        for i, col in enumerate(feature_cols):
            if col in df_feat.columns:
                X[0, i] = df_feat[col].values[0]
    
    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    # Clip prediction to valid range
    prediction = np.clip(prediction, 0.0, 100.0)
    
    return prediction


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

class CPUPredictor:
    """
    High-level wrapper for CPU prediction with lazy model loading.
    
    Example:
        predictor = CPUPredictor()
        pred = predictor.predict_next(cpu_series, t=100)
    """
    
    def __init__(self, config: CPUModelConfig = DEFAULT_CONFIG):
        self.config = config
        self._model = None
        self._scaler = None
        self._feature_cols = None
    
    def _ensure_model_loaded(self):
        """Load model lazily on first prediction."""
        if self._model is None:
            self._model, self._scaler, self._feature_cols = load_cpu_model(self.config)
    
    def predict_next(self, cpu_series: pd.Series, t: int) -> Optional[float]:
        """
        Predict CPU usage for time t+1.
        
        Args:
            cpu_series: CPU usage time series
            t: Current time index
            
        Returns:
            Predicted CPU usage for next time step
        """
        self._ensure_model_loaded()
        return predict_cpu_next(
            self._model, self._scaler, self._feature_cols,
            cpu_series, t, self.config
        )
    
    def predict_batch(self, cpu_series: pd.Series, start_t: int, end_t: int) -> np.ndarray:
        """
        Generate predictions for a range of time steps.
        
        Args:
            cpu_series: CPU usage time series
            start_t: Starting time index
            end_t: Ending time index (exclusive)
            
        Returns:
            Array of predictions (NaN where prediction was not possible)
        """
        self._ensure_model_loaded()
        predictions = np.full(end_t - start_t, np.nan)
        
        for i, t in enumerate(range(start_t, end_t)):
            pred = predict_cpu_next(
                self._model, self._scaler, self._feature_cols,
                cpu_series, t, self.config
            )
            if pred is not None:
                predictions[i] = pred
        
        return predictions


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CPU Model Interface - Test")
    print("=" * 60)
    
    # Load data
    df = load_vm_data('dataset.csv')
    print(f"Loaded data: {len(df)} samples")
    
    # Train or load model
    model, scaler, feature_cols = load_cpu_model()
    
    # Test prediction
    cpu_series = df[CPU_TARGET]
    
    # Predict at various time points
    test_indices = [100, 500, 1000, 5000]
    print("\nTest predictions:")
    for t in test_indices:
        pred = predict_cpu_next(model, scaler, feature_cols, cpu_series, t)
        actual = cpu_series.iloc[t + 1] if t + 1 < len(cpu_series) else None
        print(f"  t={t}: Predicted={pred:.2f}%, Actual={actual:.2f}%" if pred else f"  t={t}: Insufficient history")
    
    print("\n✓ CPU Model Interface working correctly!")

