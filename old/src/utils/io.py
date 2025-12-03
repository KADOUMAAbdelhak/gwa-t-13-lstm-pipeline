from pathlib import Path
import json

import pandas as pd
from tensorflow.keras.models import load_model as keras_load_model


def save_to_csv(dataframe, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(file_path, index=False)

def load_from_csv(file_path):
    return pd.read_csv(file_path)

def save_model(model, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(file_path)

def load_model(file_path):
    return keras_load_model(file_path)

def save_results(results, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)