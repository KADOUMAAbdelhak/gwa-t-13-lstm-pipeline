# VM Performance Prediction - Modular Pipeline

This directory contains a modular pipeline for VM performance prediction, with separate scripts for each machine learning model.

## Structure

```
new/
├── utils.py                    # Shared utilities (data loading, preprocessing, feature engineering)
├── ridge_pipeline.py           # Ridge Regression pipeline
├── random_forest_pipeline.py  # Random Forest pipeline
├── xgboost_pipeline.py         # XGBoost pipeline
├── lstm_pipeline.py            # LSTM pipeline
├── run_all.py                  # Script to run all models
├── dataset.csv                 # Input dataset
└── README.md                   # This file
```

## Model-Specific Results Folders

Each model pipeline creates its own results folder:

- `ridge_results/` - Ridge Regression results
- `random_forest_results/` - Random Forest results
- `xgboost_results/` - XGBoost results
- `lstm_results/` - LSTM results

Each results folder contains:
- `results.txt` - Complete analysis report with logs
- `metrics.txt` - Metrics summary (R², RMSE, MAE, MAPE) for all target variables
- `predicted_vs_actual_*.png` - Prediction plots for each target variable

## Usage

### Run Individual Models

```bash
# Ridge Regression
python ridge_pipeline.py

# Random Forest
python random_forest_pipeline.py

# XGBoost
python xgboost_pipeline.py

# LSTM
python lstm_pipeline.py
```

### Run All Models

```bash
python run_all.py
```

## Target Variables

All models are trained on the following target variables:
- CPU usage [%]
- Memory usage [%]
- Disk write throughput [KB/s]
- Network received throughput [KB/s]
- Network transmitted throughput [KB/s]

## Output Format

For each model and target variable, the following metrics are saved:
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error

Additionally, a `predicted_vs_actual_*.png` plot is generated showing the actual vs predicted values over time.

## Customization

Each model pipeline is independent and can be customized:

1. **Hyperparameters**: Modify the model initialization in each pipeline script
2. **Feature Engineering**: Edit `utils.py` to add/remove features
3. **Data Preprocessing**: Modify data loading/preprocessing functions in `utils.py`
4. **Evaluation Metrics**: Add custom metrics in `utils.py`'s `evaluate_predictions()` function

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost (optional, for XGBoost pipeline)
- tensorflow (optional, for LSTM pipeline)

## Notes

- All models use chronological train/validation/test splits (70%/10%/20%)
- Data leakage is prevented by excluding features that directly calculate target variables
- All features are standardized using StandardScaler
- LSTM models use sequence length of 24 time steps

