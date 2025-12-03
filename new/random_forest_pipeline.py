"""
Random Forest Pipeline for VM Performance Prediction
=====================================================
This script trains Random Forest models for all target variables
and saves results to 'random_forest_results' folder.

Usage:
    python random_forest_pipeline.py
"""

import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Import shared utilities
from utils import (
    CONFIG, create_output_dir, log_message, save_plot, save_metrics,
    load_data, engineer_features, prepare_ml_data, evaluate_predictions,
    plot_predictions
)

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest_model(data):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
    model.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred_scaled = model.predict(data['X_test'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    metrics = evaluate_predictions(data['y_test_original'], y_pred)
    
    return model, y_pred, metrics

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline for Random Forest"""
    model_name = "Random Forest"
    output_dir = create_output_dir('random_forest_results')
    log_file = os.path.join(output_dir, 'results.txt')
    
    # Clear previous log file
    if os.path.exists(log_file):
        os.remove(log_file)
    
    print("\n" + "="*80)
    print(f"{model_name.upper()} PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    
    log_message("="*80, log_file)
    log_message(f"{model_name.upper()} - ANALYSIS REPORT", log_file)
    log_message("="*80, log_file)
    log_message(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message(f"Python version: {sys.version.split()[0]}", log_file)
    
    # Load and preprocess data
    log_message("\n" + "="*80, log_file)
    log_message("STEP 1: DATA LOADING AND PREPROCESSING", log_file)
    log_message("="*80, log_file)
    df = load_data()
    
    # Feature engineering
    log_message("\n" + "="*80, log_file)
    log_message("STEP 2: FEATURE ENGINEERING", log_file)
    log_message("="*80, log_file)
    df_clean = engineer_features(df)
    
    # Train models for each target variable
    log_message("\n" + "="*80, log_file)
    log_message("STEP 3: MODEL TRAINING AND EVALUATION", log_file)
    log_message("="*80, log_file)
    
    all_results = {}
    is_first_target = True
    
    for target_var in CONFIG['target_variables']:
        log_message(f"\n{'='*80}", log_file)
        log_message(f"Training {model_name} for: {target_var}", log_file)
        log_message(f"{'='*80}", log_file)
        
        # Prepare data
        ml_data = prepare_ml_data(df_clean, target_var)
        
        # Train model
        model, y_pred, metrics = train_random_forest_model(ml_data)
        
        # Store results
        all_results[target_var] = {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'y_true': ml_data['y_test_original'],
            'test_index': ml_data['test_index']
        }
        
        # Log metrics
        log_message(f"\nResults for {target_var}:", log_file)
        log_message(f"  R² Score: {metrics['r2']:.6f}", log_file)
        log_message(f"  RMSE: {metrics['rmse']:.6f}", log_file)
        log_message(f"  MAE: {metrics['mae']:.6f}", log_file)
        log_message(f"  MAPE: {metrics['mape']:.2f}%", log_file)
        
        # Save metrics to file
        save_metrics(metrics, target_var, output_dir, model_name, is_first=is_first_target)
        is_first_target = False
        
        # Plot predictions
        plot_predictions(
            ml_data['y_test_original'],
            y_pred,
            ml_data['test_index'],
            target_var,
            model_name,
            output_dir
        )
    
    # Summary
    log_message("\n" + "="*80, log_file)
    log_message("FINAL SUMMARY", log_file)
    log_message("="*80, log_file)
    
    log_message("\nResults Summary:", log_file)
    for target_var, results in all_results.items():
        metrics = results['metrics']
        log_message(f"\n{target_var}:", log_file)
        log_message(f"  R²: {metrics['r2']:.6f}", log_file)
        log_message(f"  RMSE: {metrics['rmse']:.6f}", log_file)
        log_message(f"  MAE: {metrics['mae']:.6f}", log_file)
    
    log_message("\n" + "="*80, log_file)
    log_message("PIPELINE COMPLETED SUCCESSFULLY", log_file)
    log_message("="*80, log_file)
    log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message(f"\nAll results saved to: {output_dir}/", log_file)
    log_message("  - results.txt: Complete analysis report", log_file)
    log_message("  - metrics.txt: Metrics summary", log_file)
    log_message("  - *.png: Prediction plots", log_file)
    
    print("\n" + "="*80)
    print(f"✓ {model_name.upper()} PIPELINE COMPLETED!")
    print(f"✓ Check the '{output_dir}' folder for all results")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

