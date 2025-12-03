# Hyperparameter configurations for the LSTM model

class Hyperparams:
    def __init__(self):
        # Model hyperparameters
        self.input_shape = (30, 1)  # Example input shape for LSTM (timesteps, features)
        self.output_shape = 1  # Number of output features
        self.lstm_units = 50  # Number of LSTM units
        self.dense_units = 25  # Number of units in the dense layer
        self.dropout_rate = 0.2  # Dropout rate for regularization
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.batch_size = 32  # Batch size for training
        self.epochs = 100  # Number of epochs for training
        self.validation_split = 0.2  # Fraction of data to be used for validation

        # Early stopping parameters
        self.early_stopping_patience = 10  # Patience for early stopping
        self.early_stopping_min_delta = 0.001  # Minimum change to qualify as an improvement

        # Checkpoint parameters
        self.checkpoint_filepath = 'outputs/checkpoints/best_model.h5'  # Path to save the best model

    def display(self):
        print("Hyperparameters:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")