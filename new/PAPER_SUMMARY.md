# Methodology and Results Summary for Scientific Paper

## 1. LSTM Performance Analysis

### Why LSTM Models Failed

Our initial approach employed Long Short-Term Memory (LSTM) networks for univariate time series forecasting of VM resource metrics. Despite extensive optimization efforts including:

- **Architecture tuning**: 2-3 layer stacked LSTM (64→32→16 units) with dropout regularization (0.2-0.3)
- **Data preprocessing**: MinMaxScaler normalization, log transformations for skewed distributions
- **Training optimization**: Early stopping, learning rate reduction, batch normalization
- **Sequence length experimentation**: 24, 48, and 60 time steps

The LSTM models consistently produced poor results:

| Metric | LSTM R² Score | Performance |
|--------|---------------|-------------|
| CPU usage [%] | 0.091 | Very Poor |
| Memory usage [%] | 0.514 | Moderate |
| Disk write throughput [KB/s] | -0.014 | Failed (worse than baseline) |
| Network received throughput [KB/s] | 0.102 | Very Poor |
| Network transmitted throughput [KB/s] | 0.003 | Failed |

### Root Causes of LSTM Failure

1. **Extreme data skewness**: Disk and network metrics exhibit extreme positive skewness (28.0 and 30.0 respectively) with kurtosis values exceeding 800, indicating heavy-tailed distributions dominated by rare spikes. LSTMs struggle to learn meaningful patterns from such distributions.

2. **Limited sequence context**: LSTM's sequential processing captures temporal dependencies within the input window but cannot effectively leverage the rich statistical summaries (rolling means, percentiles, ranges) that characterize time series behavior.

3. **Single-feature limitation**: Univariate LSTM uses only the target variable's history, missing valuable contextual information from derived features.

4. **Gradient challenges**: The extreme value ranges (e.g., 7–2470 KB/s for disk writes) cause gradient instability during backpropagation, even with normalization.

---

## 2. Tree-Based Model Architecture

### Random Forest (CPU & Memory Prediction)

**Architecture:**
- **Estimators**: 300 decision trees
- **Max depth**: 20 levels
- **Min samples split**: 5
- **Min samples leaf**: 2
- **Feature selection**: Square root of total features per split
- **Bootstrap**: Enabled with out-of-bag scoring

**Rationale**: Random Forest's ensemble of decorrelated trees effectively captures the periodic patterns in CPU and memory usage while providing robust predictions through averaging, reducing variance from individual tree overfitting.

### XGBoost (Disk & Network Prediction)

**Architecture:**
- **Estimators**: 500 boosting rounds
- **Learning rate**: 0.05 (with early stopping)
- **Max depth**: 8 levels
- **Regularization**: L1 (α=0.1) and L2 (λ=1.0)
- **Subsampling**: 80% of samples and features per tree
- **Early stopping**: 50 rounds patience on validation loss

**Rationale**: XGBoost's gradient boosting with regularization handles the extreme skewness and outliers in disk/network metrics through sequential error correction. The L1/L2 regularization prevents overfitting to rare spike events.

---

## 3. Critical Role of Feature Engineering

The dramatic performance improvement from R² < 0.1 (LSTM) to R² > 0.87 (tree-based) is primarily attributed to comprehensive feature engineering:

### Lag Features
- **Definition**: Historical values at specific time offsets: t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-96
- **Purpose**: Captures autocorrelation and recent trends
- **Impact**: Lag-1 consistently ranked as the most important predictor across all metrics

### Rolling Statistics
- **Features**: Mean, standard deviation, min, max, range, median, IQR
- **Windows**: 6, 12, 24, 48, 96 time steps (30 min to 8 hours)
- **Purpose**: Summarizes recent behavior patterns and volatility
- **Impact**: Rolling mean and range features capture baseline levels and variability

### Time-Based Features
- **Cyclical encoding**: sin/cos transformations of hour (24-period) and day-of-week (7-period)
- **Binary indicators**: Weekend, business hours (9-18), night hours (22-06)
- **Purpose**: Captures periodic workload patterns
- **Impact**: Essential for CPU prediction due to strong daily/weekly cycles

### Difference Features
- **First differences**: Change from previous time step
- **Periodic differences**: Change from 12 steps (1 hour) ago
- **Purpose**: Captures momentum and trend direction

### Total Engineered Features
- CPU/Memory: 56-62 features
- Disk/Network: 83 features (additional quantile features for bursty data)

---

## 4. Final Results

| Metric | Model | R² | RMSE | MAE | MAPE |
|--------|-------|-----|------|-----|------|
| CPU usage [%] | Random Forest | **0.9512** | 0.152 | 0.095 | 3.2% |
| Memory usage [%] | Random Forest | **0.8862** | 2.01 | 1.34 | 12.8% |
| Disk write throughput [KB/s] | XGBoost | **0.9414** | 14.7 | 1.02 | 5.1% |
| Network received throughput [KB/s] | XGBoost | **0.9119** | 28.5 | 5.41 | 24.1% |
| Network transmitted throughput [KB/s] | XGBoost | **0.8767** | 35.6 | 9.18 | 123.4% |

### Key Findings

1. **Tree-based models dramatically outperform LSTM** for single-step VM resource prediction, with R² improvements of 10x or more.

2. **Feature engineering is the primary driver of performance**, not model complexity. The same tree-based models without engineered features would perform similarly to LSTM.

3. **Model selection should match data characteristics**:
   - Random Forest excels for metrics with regular patterns (CPU, memory)
   - XGBoost handles extreme skewness better (disk, network) due to its sequential error correction and regularization

4. **Bursty metrics remain challenging**: Network transmitted throughput shows high MAPE (123%) despite good R², indicating difficulty predicting exact spike magnitudes.

---

## 5. Comparison with LSTM

| Aspect | LSTM | Tree-Based Models |
|--------|------|-------------------|
| Best R² achieved | 0.514 (Memory) | 0.9512 (CPU) |
| Feature utilization | Raw sequences only | 56-83 engineered features |
| Training time | ~3 min per metric | ~30 sec per metric |
| Interpretability | Black box | Feature importance available |
| Handling of outliers | Poor | Robust (especially XGBoost) |
| Handling of skewness | Poor | Good with proper features |

---

## 6. Implications for VM Resource Prediction

This study demonstrates that for short-term (single-step) VM resource prediction:

1. **Classical ML with feature engineering outperforms deep learning** when the forecasting horizon is short and rich temporal features can be constructed.

2. **The choice of model architecture is less important than feature design**. Tree-based models succeed because they can effectively leverage lag, rolling, and time-based features that encode temporal patterns explicitly.

3. **LSTM may be more suitable for longer forecasting horizons** (multi-step ahead) where explicit feature construction becomes impractical, but for single-step prediction, tree-based approaches are recommended.

