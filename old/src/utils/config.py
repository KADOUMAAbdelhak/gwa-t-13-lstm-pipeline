# File: /gwa-t-13-lstm-pipeline/gwa-t-13-lstm-pipeline/src/utils/config.py

import yaml
from pathlib import Path

def load_config(path="configs/config.yaml"):
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # ensure output dirs are Path objects
    cfg["output"]["results_dir"] = Path(cfg["output"]["results_dir"])
    cfg["output"]["figures_dir"] = Path(cfg["output"]["figures_dir"])
    cfg["output"]["checkpoints_dir"] = Path(cfg["output"]["checkpoints_dir"])
    cfg["output"]["logs_dir"] = Path(cfg["output"]["logs_dir"])
    return cfg

class Config:
    def __init__(self, config_file='configs/config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        return load_config(self.config_file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def save(self):
        with open(self.config_file, 'w') as file:
            yaml.dump(self.config, file)