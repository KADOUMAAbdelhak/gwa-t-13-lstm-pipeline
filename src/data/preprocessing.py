import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict

# column positions (0-indexed) according to description:
# 0 Timestamp, 1 CPU cores, 2 cpu_capacity, 3 cpu_usage_MHz, 4 cpu_usage_pct,
# 5 mem_provisioned_KB, 6 mem_usage_KB, 7 mem_usage_pct, 8 disk_write_KBps,
# 9 disk_size_GB, 10 net_recv_KBps, 11 net_tx_KBps

EXPECTED_COLS = list(range(12))

def df_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure at least 12 columns (pad if needed)
    df = raw_df.copy()
    # if columns are strings (headers), try to detect header row (skip if numeric)
    if df.shape[1] < 12:
        # pad
        for i in range(12 - df.shape[1]):
            df[i + df.shape[1]] = np.nan
    df = df.iloc[:, :12]
    df.columns = list(range(12))
    return df

def select_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_from_raw(df_raw)
    # Skip header row if first row looks like text (not numeric timestamp)
    if df.shape[0] > 0:
        first_val = str(df.iloc[0, 0])
        # If first value looks like a header (contains "Timestamp" or is not parseable as date/number)
        if "timestamp" in first_val.lower() or not any(c.isdigit() for c in first_val):
            df = df.iloc[1:].copy()
    
    # Parse timestamp - try European format first (DD.MM.YYYY HH:MM:SS), then Unix ms
    ts_col = df[0].astype(str)
    idx = None
    # Try European date format
    try:
        idx = pd.to_datetime(ts_col, format='%d.%m.%Y %H:%M:%S', errors='coerce')
        if idx.notna().sum() == 0:
            # Fallback: try Unix milliseconds
            ts_numeric = pd.to_numeric(ts_col, errors='coerce')
            idx = pd.to_datetime(ts_numeric.dropna().astype(int), unit='ms', origin='unix', errors='coerce')
    except:
        # Fallback: try Unix milliseconds
        ts_numeric = pd.to_numeric(ts_col, errors='coerce')
        idx = pd.to_datetime(ts_numeric.dropna().astype(int), unit='ms', origin='unix', errors='coerce')
    
    # Retain rows where timestamp is valid
    valid_mask = idx.notna()
    if valid_mask.sum() == 0:
        return pd.DataFrame(columns=["cpu_pct", "mem_kb", "disk_write_kbps", "net_kbps"])
    
    df = df.loc[valid_mask].copy()
    df.index = idx[valid_mask]
    
    # Helper to convert comma decimals to float
    def to_numeric_eu(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
    
    # select metrics (handle comma decimal separators)
    cpu_pct = to_numeric_eu(df[4])
    mem_kb = to_numeric_eu(df[6])
    disk_write = to_numeric_eu(df[8])
    net_rx = to_numeric_eu(df[10])
    net_tx = to_numeric_eu(df[11])
    net = net_rx.fillna(0) + net_tx.fillna(0)
    out = pd.DataFrame({
        "cpu_pct": cpu_pct,
        "mem_kb": mem_kb,
        "disk_write_kbps": disk_write,
        "net_kbps": net,
    }, index=df.index)
    # sort index
    out = out.sort_index()
    # forward fill only if needed, but don't interpolate aggressively:
    # fill small gaps with forward-fill, then backfill for leading NaNs
    out = out.ffill().bfill()
    # Drop any rows that still contain NaNs to avoid downstream sklearn errors
    out = out.dropna()
    return out

def normalize_df(df: pd.DataFrame, scaler_type="standard") -> Tuple[pd.DataFrame, Dict[str, object]]:
    scalers = {}
    scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        arr = df[col].values.reshape(-1, 1).astype(float)
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        # handle constant series
        if np.nanstd(arr) == 0.0:
            scaler.mean_ = np.nanmean(arr)
            scaler.scale_ = 1.0
            scaled_col = (arr - scaler.mean_) / scaler.scale_
        else:
            scaler.fit(arr)
            scaled_col = scaler.transform(arr)
        scaled[col] = scaled_col.ravel()
        scalers[col] = scaler
    return scaled, scalers

def normalize_data(df):
    """Normalize the DataFrame."""
    return (df - df.mean()) / df.std()

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    return df.fillna(method='ffill').fillna(method='bfill')

def preprocess_data(file_path):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(file_path)
    df = handle_missing_values(df)
    df = normalize_data(df)
    return df

def preprocess_multiple_files(file_paths):
    """Preprocess multiple CSV files and return a dictionary of DataFrames."""
    processed_data = {}
    for file_path in file_paths:
        vm_name = file_path.split('/')[-1].replace('.csv', '')
        processed_data[vm_name] = preprocess_data(file_path)
    return processed_data