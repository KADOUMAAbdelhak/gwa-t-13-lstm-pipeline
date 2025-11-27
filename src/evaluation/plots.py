import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title='Model Predictions', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_residuals(y_true, y_pred, title='Residuals', xlabel='Time', ylabel='Residual'):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals', color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', lw=2, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparison(y_true, y_pred_list, labels, title='Model Comparison', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    for y_pred, label in zip(y_pred_list, labels):
        plt.plot(y_pred, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()