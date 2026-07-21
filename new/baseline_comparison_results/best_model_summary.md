# Best Baseline vs Proposed Hybrid per Metric

| Metric | Best baseline | Best baseline R2 | Proposed model | Proposed R2 | Gain | Hybrid is best overall |
| --- | --- | --- | --- | --- | --- | --- |
| CPU usage [%] | SVR | 0.2384 | Hybrid RF-XGBoost | -0.4305 | -0.6689 | False |
| Memory usage [%] | LSTM | 0.4924 | Hybrid RF-XGBoost | 0.4772 | -0.0152 | False |
| Disk write throughput [KB/s] | SVR | -0.0002 | Hybrid RF-XGBoost | -0.0018 | -0.0016 | False |
| Network received throughput [KB/s] | LightGBM | 0.1908 | Hybrid RF-XGBoost | 0.0706 | -0.1201 | False |
| Network transmitted throughput [KB/s] | LSTM | 0.035 | Hybrid RF-XGBoost | 0.0031 | -0.0319 | False |
