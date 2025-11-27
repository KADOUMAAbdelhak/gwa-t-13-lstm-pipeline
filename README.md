# GWA-T-13 LSTM Pipeline

This project implements a training pipeline for the GWA-T-13 Materna dataset using Long Short-Term Memory (LSTM) networks. The pipeline includes data loading, preprocessing, model training, evaluation, and visualization.

## Project Structure

```
gwa-t-13-lstm-pipeline
├── data
│   ├── raw
│   │   ├── Materna-Trace-1
│   │   ├── Materna-Trace-2
│   │   └── Materna-Trace-3
│   ├── processed
│   └── external
├── src
│   ├── data
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   └── windowing.py
│   ├── models
│   │   ├── baseline.py
│   │   ├── lstm_model.py
│   │   └── model_utils.py
│   ├── training
│   │   ├── train.py
│   │   ├── callbacks.py
│   │   └── hyperparams.py
│   ├── evaluation
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── utils
│   │   ├── io.py
│   │   ├── config.py
│   │   └── seed.py
│   └── experiments
│       └── run_experiment.py
├── notebooks
│   ├── 01-eda.ipynb
│   └── 02-training.ipynb
├── configs
│   ├── config.yaml
│   └── logging.yaml
├── scripts
│   ├── download_dataset.sh
│   └── run_training.sh
├── outputs
│   ├── checkpoints
│   ├── logs
│   ├── results
│   └── figures
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gwa-t-13-lstm-pipeline
   ```

2. Install the required packages:
   - Using pip:
     ```
     pip install -r requirements.txt
     ```
   - Or using conda:
     ```
     conda env create -f environment.yml
     ```

## Dataset

Place the official GWA-T-13 Materna traces under `data/raw/Materna-Trace-{1,2,3}`. Each trace directory must contain the `00-Trace-Info.txt` metadata file and hundreds of per-VM CSVs (one file per VM). The helper script `scripts/download_dataset.sh` now serves as a verifier: it checks whether those directories exist locally instead of trying to download anything.

## Usage

1. Install dependencies (pip or conda, see above).
2. Ensure the dataset is available locally (run `./scripts/download_dataset.sh` to verify).
3. Train/evaluate the full pipeline from the project root:
   ```
   ./scripts/run_training.sh
   ```
   The script exports `PYTHONPATH` and calls `python3 src/training/train.py --config configs/config.yaml`. You may edit `configs/config.yaml` to change device (CPU/GPU), window size, batch size, or to enable model checkpoint saving.
4. Inspect outputs under `outputs/`:
   - `outputs/results/per_vm_metrics.csv` – MAE/RMSE/R² per VM/metric/model (in original units).
   - `outputs/results/metrics_summary.csv` – trace-level averages.
   - `outputs/figures/<trace>/<metric>/` – prediction/residual plots for representative VMs.
   - `outputs/checkpoints/scalers/` – joblib dumps of the fitted scalers (per VM/metric).
   - `outputs/checkpoints/models/` – optional LSTM checkpoints when `training.save_lstm_models` is true.
   - `outputs/logs/training.log` – structured log output configured via `configs/logging.yaml`.

## Notebooks

The notebooks in `notebooks/` illustrate exploratory analysis and small-scale experiments. They rely on the same data loaders/windowing utilities used by the production pipeline but are not part of the automated training loop.

## Outputs

- Model checkpoints will be saved in `outputs/checkpoints`.
- Logs of the training process will be found in `outputs/logs`.
- Evaluation results will be stored in `outputs/results`.
- Generated figures will be located in `outputs/figures`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.