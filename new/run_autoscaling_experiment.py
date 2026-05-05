"""
Autoscaling Experiment Driver
=============================
Main script to run the predictive vs reactive autoscaling comparison experiment.

This script:
1. Loads VM trace data and the trained CPU prediction model
2. Runs both reactive and predictive scaling policies
3. Computes comparative metrics
4. Generates publication-quality figures and tables

Usage:
    python run_autoscaling_experiment.py [--base-capacity 50] [--config config.yaml]

Author: Hybrid ML Pipeline for Cloud-Edge Continuum
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Local imports
from cpu_model_interface import (
    CPUPredictor, CPUModelConfig, load_vm_data, CPU_TARGET, load_cpu_model
)
from autoscaling_simulation import (
    AutoscalingConfig, simulate_policy_reactive, simulate_policy_predictive,
    run_comparison_experiment, aggregate_multi_vm_results, extract_time_series_data,
    SimulationMetrics
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = 'autoscaling_results'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Publication-quality plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme
COLORS = {
    'reactive': '#1f77b4',     # Blue
    'predictive': '#d62728',   # Red
    'actual_cpu': '#2ca02c',   # Green
    'capacity': '#ff7f0e',     # Orange
    'violation': '#e377c2',    # Pink
    'grid': '#cccccc',
}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_policy_comparison_bars(
    metrics_reactive: SimulationMetrics,
    metrics_predictive: SimulationMetrics,
    save_path: str
) -> None:
    """
    Create bar charts comparing reactive vs predictive policies.
    
    Generates two side-by-side bar charts:
    1. SLA Violation Fraction
    2. Mean Replicas Used
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left panel: SLA Violation Fraction
    ax1 = axes[0]
    policies = ['Reactive', 'Predictive']
    violations = [
        metrics_reactive.sla_violation_fraction * 100,
        metrics_predictive.sla_violation_fraction * 100
    ]
    colors = [COLORS['reactive'], COLORS['predictive']]
    
    bars1 = ax1.bar(policies, violations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('SLA Violation Rate (%)', fontsize=12)
    ax1.set_title('(a) SLA Violations', fontsize=13, fontweight='bold', loc='left')
    ax1.set_ylim(0, max(violations) * 1.3 if max(violations) > 0 else 10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, violations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Right panel: Mean Replicas
    ax2 = axes[1]
    replicas = [
        metrics_reactive.mean_replicas,
        metrics_predictive.mean_replicas
    ]
    
    bars2 = ax2.bar(policies, replicas, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Mean Replicas', fontsize=12)
    ax2.set_title('(b) Resource Usage', fontsize=13, fontweight='bold', loc='left')
    ax2.set_ylim(0, max(replicas) * 1.3 if max(replicas) > 0 else 3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, replicas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_detailed_metrics_comparison(
    metrics_reactive: SimulationMetrics,
    metrics_predictive: SimulationMetrics,
    save_path: str
) -> None:
    """
    Create detailed comparison of all metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    policies = ['Reactive', 'Predictive']
    colors = [COLORS['reactive'], COLORS['predictive']]
    
    # Panel A: SLA Violations
    ax = axes[0, 0]
    violations = [
        metrics_reactive.sla_violation_fraction * 100,
        metrics_predictive.sla_violation_fraction * 100
    ]
    bars = ax.bar(policies, violations, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('(a) SLA Violation Rate', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, violations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Violation Episodes
    ax = axes[0, 1]
    episodes = [
        metrics_reactive.violation_episodes,
        metrics_predictive.violation_episodes
    ]
    bars = ax.bar(policies, episodes, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Episodes')
    ax.set_title('(b) Violation Episodes', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, episodes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    # Panel C: Mean Replicas
    ax = axes[1, 0]
    replicas = [
        metrics_reactive.mean_replicas,
        metrics_predictive.mean_replicas
    ]
    bars = ax.bar(policies, replicas, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Mean Replicas')
    ax.set_title('(c) Average Resource Usage', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, replicas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Panel D: Scaling Events
    ax = axes[1, 1]
    scale_events = [
        metrics_reactive.total_scale_up_events + metrics_reactive.total_scale_down_events,
        metrics_predictive.total_scale_up_events + metrics_predictive.total_scale_down_events
    ]
    bars = ax.bar(policies, scale_events, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Total Scaling Events')
    ax.set_title('(d) Scaling Activity', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, scale_events):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_time_series_comparison(
    records_reactive: List,
    records_predictive: List,
    save_path: str,
    zoom_window: int = 288
) -> None:
    """
    Create time series comparison showing CPU, capacity, and violations.
    """
    # Extract data
    data_reactive = extract_time_series_data(records_reactive)
    data_predictive = extract_time_series_data(records_predictive)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Determine zoom range (middle of the data)
    total_len = len(data_reactive['t'])
    start_idx = max(0, total_len // 3)
    end_idx = min(total_len, start_idx + zoom_window)
    
    # Panel A: Full time series - Reactive
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data_reactive['t'], data_reactive['cpu_actual'], 
             color=COLORS['actual_cpu'], linewidth=0.8, alpha=0.7, label='CPU Demand')
    ax1.plot(data_reactive['t'], data_reactive['capacity'], 
             color=COLORS['capacity'], linewidth=1.2, label='Capacity')
    
    # Shade violations
    violations = data_reactive['sla_violated']
    ax1.fill_between(data_reactive['t'], 0, data_reactive['cpu_actual'].max() * 1.1,
                     where=violations, alpha=0.2, color=COLORS['violation'], label='SLA Violation')
    
    ax1.set_ylabel('CPU (%)')
    ax1.set_title('(a) Reactive Policy - Full Timeline', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(data_reactive['t'][0], data_reactive['t'][-1])
    
    # Panel B: Full time series - Predictive
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(data_predictive['t'], data_predictive['cpu_actual'], 
             color=COLORS['actual_cpu'], linewidth=0.8, alpha=0.7, label='CPU Demand')
    ax2.plot(data_predictive['t'], data_predictive['capacity'], 
             color=COLORS['capacity'], linewidth=1.2, label='Capacity')
    
    violations = data_predictive['sla_violated']
    ax2.fill_between(data_predictive['t'], 0, data_predictive['cpu_actual'].max() * 1.1,
                     where=violations, alpha=0.2, color=COLORS['violation'], label='SLA Violation')
    
    ax2.set_ylabel('CPU (%)')
    ax2.set_title('(b) Predictive Policy - Full Timeline', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(data_predictive['t'][0], data_predictive['t'][-1])
    
    # Panel C: Zoomed - Reactive
    ax3 = plt.subplot(3, 2, 3)
    zoom_t = data_reactive['t'][start_idx:end_idx]
    zoom_cpu = data_reactive['cpu_actual'][start_idx:end_idx]
    zoom_cap = data_reactive['capacity'][start_idx:end_idx]
    zoom_viol = data_reactive['sla_violated'][start_idx:end_idx]
    
    ax3.plot(zoom_t, zoom_cpu, color=COLORS['actual_cpu'], linewidth=1.2, label='CPU Demand')
    ax3.plot(zoom_t, zoom_cap, color=COLORS['capacity'], linewidth=1.5, label='Capacity')
    ax3.fill_between(zoom_t, 0, zoom_cpu.max() * 1.1,
                     where=zoom_viol, alpha=0.3, color=COLORS['violation'])
    
    ax3.set_ylabel('CPU (%)')
    ax3.set_title(f'(c) Reactive Policy - Zoomed ({zoom_window} steps)', fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Zoomed - Predictive
    ax4 = plt.subplot(3, 2, 4)
    zoom_t = data_predictive['t'][start_idx:end_idx]
    zoom_cpu = data_predictive['cpu_actual'][start_idx:end_idx]
    zoom_cap = data_predictive['capacity'][start_idx:end_idx]
    zoom_viol = data_predictive['sla_violated'][start_idx:end_idx]
    
    ax4.plot(zoom_t, zoom_cpu, color=COLORS['actual_cpu'], linewidth=1.2, label='CPU Demand')
    ax4.plot(zoom_t, zoom_cap, color=COLORS['capacity'], linewidth=1.5, label='Capacity')
    ax4.fill_between(zoom_t, 0, zoom_cpu.max() * 1.1,
                     where=zoom_viol, alpha=0.3, color=COLORS['violation'])
    
    ax4.set_ylabel('CPU (%)')
    ax4.set_title(f'(d) Predictive Policy - Zoomed ({zoom_window} steps)', fontweight='bold', loc='left')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel E: Replica comparison over time
    ax5 = plt.subplot(3, 2, (5, 6))
    ax5.step(data_reactive['t'], data_reactive['replicas'], 
             color=COLORS['reactive'], linewidth=1.5, label='Reactive', where='post')
    ax5.step(data_predictive['t'], data_predictive['replicas'], 
             color=COLORS['predictive'], linewidth=1.5, label='Predictive', where='post', linestyle='--')
    
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Number of Replicas')
    ax5.set_title('(e) Replica Count Comparison', fontweight='bold', loc='left')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(data_reactive['t'][0], data_reactive['t'][-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_prediction_accuracy(
    records_predictive: List,
    save_path: str
) -> None:
    """
    Plot prediction accuracy analysis for the predictive policy.
    """
    data = extract_time_series_data(records_predictive)
    
    # Filter out NaN predictions
    mask = ~np.isnan(data['cpu_predicted'])
    actual = data['cpu_actual'][mask]
    predicted = data['cpu_predicted'][mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Actual vs Predicted scatter
    ax1 = axes[0]
    ax1.scatter(actual, predicted, alpha=0.3, s=10, color=COLORS['predictive'])
    
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)
    
    ax1.set_xlabel('Actual CPU (%)')
    ax1.set_ylabel('Predicted CPU (%)')
    ax1.set_title('(a) Prediction Accuracy', fontweight='bold', loc='left')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, fontsize=11,
             va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: Prediction error distribution
    ax2 = axes[1]
    errors = predicted - actual
    ax2.hist(errors, bins=50, color=COLORS['predictive'], alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(errors), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    
    ax2.set_xlabel('Prediction Error (Predicted - Actual)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Prediction Error Distribution', fontweight='bold', loc='left')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_table(
    metrics_reactive: SimulationMetrics,
    metrics_predictive: SimulationMetrics
) -> str:
    """Generate LaTeX table for paper."""
    
    # Calculate improvements
    if metrics_reactive.sla_violation_fraction > 0:
        violation_reduction = ((metrics_reactive.sla_violation_fraction - metrics_predictive.sla_violation_fraction) 
                               / metrics_reactive.sla_violation_fraction) * 100
    else:
        violation_reduction = 0
    
    replica_increase = ((metrics_predictive.mean_replicas - metrics_reactive.mean_replicas) 
                        / metrics_reactive.mean_replicas) * 100
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Reactive vs. Predictive Autoscaling Policies}
\label{tab:autoscaling_results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Reactive} & \textbf{Predictive} & \textbf{Improvement} \\
\midrule
"""
    
    latex += f"SLA Violation Rate (\\%) & {metrics_reactive.sla_violation_fraction*100:.2f} & {metrics_predictive.sla_violation_fraction*100:.2f} & {violation_reduction:.1f}\\% reduction \\\\\n"
    latex += f"Violation Episodes & {metrics_reactive.violation_episodes} & {metrics_predictive.violation_episodes} & - \\\\\n"
    latex += f"Mean Violation Length & {metrics_reactive.mean_violation_episode_length:.1f} & {metrics_predictive.mean_violation_episode_length:.1f} & - \\\\\n"
    latex += f"Mean Replicas & {metrics_reactive.mean_replicas:.2f} & {metrics_predictive.mean_replicas:.2f} & +{replica_increase:.1f}\\% \\\\\n"
    latex += f"Max Replicas Used & {metrics_reactive.max_replicas_used} & {metrics_predictive.max_replicas_used} & - \\\\\n"
    latex += f"Scale-Up Events & {metrics_reactive.total_scale_up_events} & {metrics_predictive.total_scale_up_events} & - \\\\\n"
    latex += f"Scale-Down Events & {metrics_reactive.total_scale_down_events} & {metrics_predictive.total_scale_down_events} & - \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(
    base_capacity: float = 50.0,
    th_up: float = 0.7,
    th_down: float = 0.3,
    k_up: int = 2,
    k_down: int = 4
) -> Dict[str, Any]:
    """
    Run the complete autoscaling comparison experiment.
    
    Args:
        base_capacity: Base CPU capacity per replica
        th_up: Scale-up threshold
        th_down: Scale-down threshold
        k_up: Consecutive steps above threshold for scale-up
        k_down: Consecutive steps below threshold for scale-down
        
    Returns:
        Dictionary with all experiment results
    """
    print("\n" + "=" * 70)
    print("PREDICTIVE AUTOSCALING EXPERIMENT")
    print("=" * 70)
    print(f"Comparing Reactive vs. Predictive Autoscaling Policies")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading VM trace data...")
    df = load_vm_data('dataset.csv')
    cpu_series = df[CPU_TARGET]
    print(f"  Loaded {len(cpu_series)} samples ({len(cpu_series)/12:.1f} hours at 5-min intervals)")
    
    # Calculate data statistics for context
    cpu_stats = {
        'mean': cpu_series.mean(),
        'std': cpu_series.std(),
        'p50': cpu_series.quantile(0.5),
        'p95': cpu_series.quantile(0.95),
        'p99': cpu_series.quantile(0.99),
        'max': cpu_series.max()
    }
    print(f"  CPU stats: mean={cpu_stats['mean']:.1f}%, P95={cpu_stats['p95']:.1f}%, max={cpu_stats['max']:.1f}%")
    
    # Initialize predictor (loads or trains model)
    print("\n[2/5] Loading CPU prediction model...")
    predictor = CPUPredictor()
    predictor._ensure_model_loaded()
    
    # Configure autoscaling
    config = AutoscalingConfig(
        th_up=th_up,
        th_down=th_down,
        k_up=k_up,
        k_down=k_down,
        base_capacity=base_capacity,
        min_replicas=1,
        max_replicas=10,
        cooldown_up=3,
        cooldown_down=6,
        window_size=96
    )
    
    print(f"\n  Autoscaling Configuration:")
    print(f"    Base capacity: {config.base_capacity}% per replica")
    print(f"    Thresholds: scale-up > {config.th_up*100:.0f}%, scale-down < {config.th_down*100:.0f}%")
    print(f"    Consecutive steps: k_up={config.k_up}, k_down={config.k_down}")
    print(f"    Replica range: [{config.min_replicas}, {config.max_replicas}]")
    
    # Run simulations
    print("\n[3/5] Running simulations...")
    
    print("  Running reactive policy simulation...")
    metrics_reactive, records_reactive = simulate_policy_reactive(cpu_series, config)
    print(f"    → {metrics_reactive.total_steps} steps simulated")
    print(f"    → SLA violations: {metrics_reactive.sla_violations} ({metrics_reactive.sla_violation_fraction*100:.2f}%)")
    
    print("  Running predictive policy simulation...")
    metrics_predictive, records_predictive = simulate_policy_predictive(cpu_series, config, predictor)
    print(f"    → {metrics_predictive.total_steps} steps simulated")
    print(f"    → SLA violations: {metrics_predictive.sla_violations} ({metrics_predictive.sla_violation_fraction*100:.2f}%)")
    
    # Generate visualizations
    print("\n[4/5] Generating visualizations...")
    
    plot_policy_comparison_bars(
        metrics_reactive, metrics_predictive,
        os.path.join(FIGURES_DIR, 'comparison_bars.png')
    )
    
    plot_detailed_metrics_comparison(
        metrics_reactive, metrics_predictive,
        os.path.join(FIGURES_DIR, 'detailed_comparison.png')
    )
    
    plot_time_series_comparison(
        records_reactive, records_predictive,
        os.path.join(FIGURES_DIR, 'time_series_comparison.png')
    )
    
    plot_prediction_accuracy(
        records_predictive,
        os.path.join(FIGURES_DIR, 'prediction_accuracy.png')
    )
    
    # Generate outputs
    print("\n[5/5] Saving results...")
    
    # Save LaTeX table
    latex_table = generate_latex_table(metrics_reactive, metrics_predictive)
    latex_path = os.path.join(OUTPUT_DIR, 'autoscaling_results_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"  ✓ LaTeX table: {latex_path}")
    
    # Save metrics to CSV
    results_df = pd.DataFrame({
        'Metric': [
            'SLA Violation Rate (%)',
            'Violation Episodes',
            'Mean Violation Length',
            'Mean Replicas',
            'Max Replicas',
            'Scale-Up Events',
            'Scale-Down Events',
            'Mean Overprovision'
        ],
        'Reactive': [
            metrics_reactive.sla_violation_fraction * 100,
            metrics_reactive.violation_episodes,
            metrics_reactive.mean_violation_episode_length,
            metrics_reactive.mean_replicas,
            metrics_reactive.max_replicas_used,
            metrics_reactive.total_scale_up_events,
            metrics_reactive.total_scale_down_events,
            metrics_reactive.mean_overprovision
        ],
        'Predictive': [
            metrics_predictive.sla_violation_fraction * 100,
            metrics_predictive.violation_episodes,
            metrics_predictive.mean_violation_episode_length,
            metrics_predictive.mean_replicas,
            metrics_predictive.max_replicas_used,
            metrics_predictive.total_scale_up_events,
            metrics_predictive.total_scale_down_events,
            metrics_predictive.mean_overprovision
        ]
    })
    
    csv_path = os.path.join(OUTPUT_DIR, 'autoscaling_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Results CSV: {csv_path}")
    
    # Save experiment configuration
    config_dict = {
        'base_capacity': config.base_capacity,
        'th_up': config.th_up,
        'th_down': config.th_down,
        'k_up': config.k_up,
        'k_down': config.k_down,
        'min_replicas': config.min_replicas,
        'max_replicas': config.max_replicas,
        'window_size': config.window_size,
        'total_samples': len(cpu_series),
        'simulated_steps': metrics_reactive.total_steps
    }
    config_df = pd.DataFrame([config_dict])
    config_df.to_csv(os.path.join(OUTPUT_DIR, 'experiment_config.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    POLICY COMPARISON                            │")
    print("├─────────────────────────┬───────────────┬───────────────────────┤")
    print("│ Metric                  │    Reactive   │      Predictive       │")
    print("├─────────────────────────┼───────────────┼───────────────────────┤")
    print(f"│ SLA Violation Rate      │ {metrics_reactive.sla_violation_fraction*100:>10.2f}%  │ {metrics_predictive.sla_violation_fraction*100:>18.2f}%  │")
    print(f"│ Violation Episodes      │ {metrics_reactive.violation_episodes:>12}  │ {metrics_predictive.violation_episodes:>20}  │")
    print(f"│ Mean Replicas           │ {metrics_reactive.mean_replicas:>12.2f}  │ {metrics_predictive.mean_replicas:>20.2f}  │")
    print(f"│ Scale Events            │ {metrics_reactive.total_scale_up_events + metrics_reactive.total_scale_down_events:>12}  │ {metrics_predictive.total_scale_up_events + metrics_predictive.total_scale_down_events:>20}  │")
    print("└─────────────────────────┴───────────────┴───────────────────────┘")
    
    # Calculate improvement
    if metrics_reactive.sla_violation_fraction > 0:
        improvement = ((metrics_reactive.sla_violation_fraction - metrics_predictive.sla_violation_fraction) 
                       / metrics_reactive.sla_violation_fraction) * 100
        print(f"\n✓ Predictive policy reduces SLA violations by {improvement:.1f}%")
    
    replica_cost = ((metrics_predictive.mean_replicas - metrics_reactive.mean_replicas) 
                    / metrics_reactive.mean_replicas) * 100
    print(f"✓ At a cost of {replica_cost:+.1f}% more replicas on average")
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/")
    print(f"✓ Figures saved to: {FIGURES_DIR}/")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    return {
        'metrics_reactive': metrics_reactive,
        'metrics_predictive': metrics_predictive,
        'records_reactive': records_reactive,
        'records_predictive': records_predictive,
        'config': config,
        'cpu_stats': cpu_stats
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Predictive vs Reactive Autoscaling Comparison Experiment'
    )
    parser.add_argument(
        '--base-capacity', type=float, default=50.0,
        help='Base CPU capacity per replica (default: 50.0)'
    )
    parser.add_argument(
        '--th-up', type=float, default=0.7,
        help='Scale-up threshold as fraction (default: 0.7)'
    )
    parser.add_argument(
        '--th-down', type=float, default=0.3,
        help='Scale-down threshold as fraction (default: 0.3)'
    )
    parser.add_argument(
        '--k-up', type=int, default=2,
        help='Consecutive steps above threshold before scale-up (default: 2)'
    )
    parser.add_argument(
        '--k-down', type=int, default=4,
        help='Consecutive steps below threshold before scale-down (default: 4)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        run_experiment(
            base_capacity=args.base_capacity,
            th_up=args.th_up,
            th_down=args.th_down,
            k_up=args.k_up,
            k_down=args.k_down
        )
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

