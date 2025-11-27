from src.data.load_data import load_data
from src.data.preprocessing import preprocess_data
from src.data.windowing import create_windows
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.utils.config import load_config
import os

def run_experiment():
    # Load configuration
    config = load_config()

    # Load data
    raw_data = load_data(config['data']['raw_data_path'])
    
    # Preprocess data
    processed_data = preprocess_data(raw_data)
    
    # Create sliding windows
    windows = create_windows(processed_data, config['data']['window_size'])
    
    # Train the model
    model = train_model(windows, config['training'])
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, windows, config['evaluation'])
    
    # Save results
    results_path = os.path.join(config['outputs']['results_path'], 'evaluation_results.csv')
    evaluation_results.to_csv(results_path, index=False)
    
    print("Experiment completed. Results saved to:", results_path)

if __name__ == "__main__":
    run_experiment()