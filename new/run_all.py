"""
Run All Model Pipelines
=======================
This script runs all model pipelines sequentially.

Usage:
    python run_all.py
"""

import subprocess
import sys
from datetime import datetime

def run_pipeline(script_name):
    """Run a pipeline script and return success status"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ {script_name} not found")
        return False

def main():
    """Run all model pipelines"""
    print("\n" + "="*80)
    print("RUNNING ALL MODEL PIPELINES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipelines = [
        'ridge_pipeline.py',
        'random_forest_pipeline.py',
        'xgboost_pipeline.py',
        'lstm_pipeline.py'
    ]
    
    results = {}
    
    for pipeline in pipelines:
        success = run_pipeline(pipeline)
        results[pipeline] = success
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPipeline Results:")
    for pipeline, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {pipeline}: {status}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✓ All pipelines completed successfully!")
    else:
        print("\n⚠ Some pipelines failed. Check the output above for details.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

