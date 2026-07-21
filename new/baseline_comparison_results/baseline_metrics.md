# Baseline Comparison (leak-free one-step-ahead)

| Metric | Model | R2 | RMSE | MAE | MAPE (%) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CPU usage [%] | Persistence | 0.0312 | 0.69 | 0.29 | 9.4 | ok |
| CPU usage [%] | Linear Regression | -0.0277 | 0.71 | 0.33 | 10.9 | ok |
| CPU usage [%] | Ridge Regression | -0.0117 | 0.70 | 0.32 | 10.6 | ok |
| CPU usage [%] | SVR | 0.2384 | 0.61 | 0.25 | 7.6 | ok |
| CPU usage [%] | Random Forest | -0.4305 | 0.83 | 0.54 | 19.4 | ok |
| CPU usage [%] | XGBoost | -0.4035 | 0.83 | 0.52 | 18.6 | ok |
| CPU usage [%] | LightGBM | -0.3174 | 0.80 | 0.48 | 17.0 | ok |
| CPU usage [%] | LSTM | 0.1653 | 0.64 | 0.26 | 8.0 | ok |
| CPU usage [%] | ARIMA | -0.4257 | 0.83 | 0.33 | 10.3 | ok |
| CPU usage [%] | Hybrid RF-XGBoost | -0.4305 | 0.83 | 0.54 | 19.4 | ok |
| Memory usage [%] | Persistence | 0.3872 | 4.92 | 3.87 | 41.1 | ok |
| Memory usage [%] | Linear Regression | 0.3357 | 5.12 | 4.06 | 46.0 | ok |
| Memory usage [%] | Ridge Regression | 0.4756 | 4.55 | 3.72 | 42.4 | ok |
| Memory usage [%] | SVR | 0.1635 | 5.75 | 4.36 | 40.9 | ok |
| Memory usage [%] | Random Forest | 0.4772 | 4.54 | 3.68 | 42.4 | ok |
| Memory usage [%] | XGBoost | 0.4089 | 4.83 | 3.85 | 44.9 | ok |
| Memory usage [%] | LightGBM | 0.4588 | 4.62 | 3.63 | 40.6 | ok |
| Memory usage [%] | LSTM | 0.4924 | 4.48 | 3.68 | 44.6 | ok |
| Memory usage [%] | ARIMA | 0.3460 | 5.08 | 3.98 | 45.4 | ok |
| Memory usage [%] | Hybrid RF-XGBoost | 0.4772 | 4.54 | 3.68 | 42.4 | ok |
| Disk write throughput [KB/s] | Persistence | -0.9918 | 98.92 | 6.31 | 27.7 | ok |
| Disk write throughput [KB/s] | Linear Regression | -1.3737 | 107.99 | 23.43 | 172.6 | ok |
| Disk write throughput [KB/s] | Ridge Regression | -1.7994 | 117.27 | 25.67 | 188.6 | ok |
| Disk write throughput [KB/s] | SVR | -0.0002 | 70.10 | 3.49 | 9.3 | ok |
| Disk write throughput [KB/s] | Random Forest | -0.1006 | 73.53 | 10.46 | 66.5 | ok |
| Disk write throughput [KB/s] | XGBoost | -0.0018 | 70.15 | 7.32 | 43.5 | ok |
| Disk write throughput [KB/s] | LightGBM | -0.0410 | 71.51 | 9.24 | 57.1 | ok |
| Disk write throughput [KB/s] | LSTM | -0.0080 | 70.37 | 5.75 | 29.0 | ok |
| Disk write throughput [KB/s] | ARIMA | -0.0146 | 70.60 | 5.46 | 24.7 | ok |
| Disk write throughput [KB/s] | Hybrid RF-XGBoost | -0.0018 | 70.15 | 7.32 | 43.5 | ok |
| Network received throughput [KB/s] | Persistence | -0.1140 | 109.40 | 41.03 | 555.9 | ok |
| Network received throughput [KB/s] | Linear Regression | 0.1292 | 96.73 | 40.36 | 685.5 | ok |
| Network received throughput [KB/s] | Ridge Regression | 0.1392 | 96.17 | 40.48 | 706.1 | ok |
| Network received throughput [KB/s] | SVR | -0.0277 | 105.08 | 34.80 | 230.7 | ok |
| Network received throughput [KB/s] | Random Forest | 0.1852 | 93.57 | 43.37 | 995.9 | ok |
| Network received throughput [KB/s] | XGBoost | 0.0706 | 99.93 | 38.16 | 659.1 | ok |
| Network received throughput [KB/s] | LightGBM | 0.1908 | 93.25 | 37.85 | 697.1 | ok |
| Network received throughput [KB/s] | LSTM | 0.1328 | 96.53 | 39.99 | 848.4 | ok |
| Network received throughput [KB/s] | ARIMA | 0.0499 | 101.03 | 40.75 | 777.1 | ok |
| Network received throughput [KB/s] | Hybrid RF-XGBoost | 0.0706 | 99.93 | 38.16 | 659.1 | ok |
| Network transmitted throughput [KB/s] | Persistence | -0.4806 | 131.15 | 39.17 | 809.4 | ok |
| Network transmitted throughput [KB/s] | Linear Regression | -0.1408 | 115.13 | 46.49 | 931.2 | ok |
| Network transmitted throughput [KB/s] | Ridge Regression | -0.1494 | 115.56 | 46.72 | 944.9 | ok |
| Network transmitted throughput [KB/s] | SVR | -0.0086 | 108.25 | 29.03 | 233.4 | ok |
| Network transmitted throughput [KB/s] | Random Forest | -0.3691 | 126.12 | 41.58 | 1332.0 | ok |
| Network transmitted throughput [KB/s] | XGBoost | 0.0031 | 107.62 | 34.55 | 911.5 | ok |
| Network transmitted throughput [KB/s] | LightGBM | -0.1679 | 116.49 | 36.83 | 921.3 | ok |
| Network transmitted throughput [KB/s] | LSTM | 0.0350 | 105.88 | 37.12 | 1136.8 | ok |
| Network transmitted throughput [KB/s] | ARIMA | -0.2621 | 121.09 | 35.15 | 469.4 | ok |
| Network transmitted throughput [KB/s] | Hybrid RF-XGBoost | 0.0031 | 107.62 | 34.55 | 911.5 | ok |
