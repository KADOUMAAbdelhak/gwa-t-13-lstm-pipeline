import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

def build_lstm(input_shape, cfg):
    units1 = cfg["models"]["lstm"]["units1"]
    units2 = cfg["models"]["lstm"]["units2"]
    dense1 = cfg["models"]["lstm"]["dense1"]
    dense2 = cfg["models"]["lstm"]["dense2"]

    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(units1, return_sequences=True)(inp)
    x = layers.LSTM(units2)(x)
    x = layers.Dense(dense1, activation="relu")(x)
    x = layers.Dense(dense2, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])
    return model

def get_early_stopper(patience):
    return callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)