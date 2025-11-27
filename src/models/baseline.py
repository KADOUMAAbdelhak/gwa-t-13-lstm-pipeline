import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

class BaselineModels:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor()
        }

    def fit(self, X_train, y_train, model_name):
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        return model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {'mse': mse, 'rmse': rmse}

def last_value_model(y_train, y_test):
    last_value = y_train.iloc[-1]
    predictions = np.full(y_test.shape, last_value)
    return predictions

def evaluate_last_value(y_true, y_pred):
    return {'mse': mean_squared_error(y_true, y_pred), 'rmse': np.sqrt(mean_squared_error(y_true, y_pred))}

def last_value_predict(X):
    # X shape: (n, window, 1) -> predict last value in each window
    return X[:, -1, 0].reshape(-1,1)

def flatten_windows(X):
    # (n, window, 1) -> (n, window)
    return X.reshape(X.shape[0], X.shape[1])

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(flatten_windows(X_train), y_train.ravel())
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=0):
    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    model.fit(flatten_windows(X_train), y_train.ravel())
    return model

def train_gradient_boosting(X_train, y_train, random_state=0):
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(flatten_windows(X_train), y_train.ravel())
    return model