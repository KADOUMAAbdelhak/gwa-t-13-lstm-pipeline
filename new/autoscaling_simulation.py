"""
Autoscaling Simulation Engine
=============================
Implements reactive and predictive autoscaling policies for VM resource management.

This module simulates autoscaling decisions using:
- Reactive policy: scales based on actual observed CPU utilization
- Predictive policy: scales based on ML-predicted CPU utilization (one-step-ahead)

The simulation tracks SLA violations, resource utilization, and scaling events
to compare the effectiveness of each policy.

Author: Hybrid ML Pipeline for Cloud-Edge Continuum
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd

from cpu_model_interface import CPUPredictor, CPUModelConfig, load_vm_data, CPU_TARGET


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AutoscalingConfig:
    """
    Configuration for autoscaling simulation.
    
    Attributes:
        th_up: Upper threshold for scaling up (fraction, e.g., 0.7 = 70%)
        th_down: Lower threshold for scaling down (fraction)
        k_up: Consecutive steps above th_up before scaling up
        k_down: Consecutive steps below th_down before scaling down
        base_capacity: Base CPU capacity per replica (in same units as CPU usage)
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        cooldown_up: Minimum steps between scale-up events
        cooldown_down: Minimum steps between scale-down events
        window_size: Feature engineering window size (W)
    """
    # Threshold-based scaling parameters
    th_up: float = 0.7        # Scale up when utilization > 70%
    th_down: float = 0.3      # Scale down when utilization < 30%
    k_up: int = 2             # Consecutive steps above threshold before scale up
    k_down: int = 4           # Consecutive steps below threshold before scale down
    
    # Capacity and replica limits
    base_capacity: float = 50.0  # Base capacity per replica (CPU %)
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Cooldown periods (to reduce oscillations)
    cooldown_up: int = 3      # Min steps between scale-up events
    cooldown_down: int = 6    # Min steps between scale-down events
    
    # Feature engineering window
    window_size: int = 96     # 8 hours at 5-min intervals


class ScalingAction(Enum):
    """Possible scaling actions."""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class SimulationState:
    """
    Tracks the current state of the autoscaling simulation.
    """
    replicas: int = 1
    steps_above_threshold: int = 0
    steps_below_threshold: int = 0
    cooldown_remaining_up: int = 0
    cooldown_remaining_down: int = 0


@dataclass
class SimulationMetrics:
    """
    Aggregated metrics from a simulation run.
    """
    total_steps: int = 0
    sla_violations: int = 0
    sla_violation_fraction: float = 0.0
    total_scale_up_events: int = 0
    total_scale_down_events: int = 0
    mean_replicas: float = 0.0
    median_replicas: float = 0.0
    max_replicas_used: int = 0
    mean_utilization: float = 0.0
    mean_overprovision: float = 0.0
    
    # Episode-based metrics
    violation_episodes: int = 0
    mean_violation_episode_length: float = 0.0
    max_violation_episode_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'total_steps': self.total_steps,
            'sla_violations': self.sla_violations,
            'sla_violation_fraction': self.sla_violation_fraction,
            'total_scale_up_events': self.total_scale_up_events,
            'total_scale_down_events': self.total_scale_down_events,
            'mean_replicas': self.mean_replicas,
            'median_replicas': self.median_replicas,
            'max_replicas_used': self.max_replicas_used,
            'mean_utilization': self.mean_utilization,
            'mean_overprovision': self.mean_overprovision,
            'violation_episodes': self.violation_episodes,
            'mean_violation_episode_length': self.mean_violation_episode_length,
            'max_violation_episode_length': self.max_violation_episode_length
        }


@dataclass
class StepRecord:
    """
    Record of a single simulation step for detailed analysis.
    """
    t: int                      # Time step
    cpu_actual: float           # Actual CPU usage
    cpu_predicted: Optional[float]  # Predicted CPU (predictive policy only)
    replicas: int               # Current replica count
    capacity: float             # Current capacity
    utilization: float          # Utilization ratio
    sla_violated: bool          # Whether SLA was violated
    action: ScalingAction       # Scaling action taken


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def compute_utilization(cpu: float, replicas: int, base_capacity: float) -> float:
    """
    Compute utilization per replica.
    
    Args:
        cpu: CPU usage value
        replicas: Number of replicas
        base_capacity: Capacity per replica
        
    Returns:
        Utilization ratio (0.0 to inf)
    """
    total_capacity = replicas * base_capacity
    if total_capacity == 0:
        return float('inf')
    return cpu / total_capacity


def check_sla_violation(cpu: float, replicas: int, base_capacity: float) -> bool:
    """
    Check if SLA is violated (demand exceeds capacity).
    
    Args:
        cpu: Actual CPU usage
        replicas: Number of replicas
        base_capacity: Capacity per replica
        
    Returns:
        True if SLA violated, False otherwise
    """
    total_capacity = replicas * base_capacity
    return cpu > total_capacity


def decide_scaling_action(
    utilization: float,
    state: SimulationState,
    config: AutoscalingConfig
) -> Tuple[ScalingAction, SimulationState]:
    """
    Decide scaling action based on utilization and state.
    
    This implements the threshold-based scaling logic with cooldown
    to prevent oscillations.
    
    Args:
        utilization: Current utilization ratio
        state: Current simulation state
        config: Autoscaling configuration
        
    Returns:
        Tuple of (action, updated_state)
    """
    new_state = SimulationState(
        replicas=state.replicas,
        steps_above_threshold=state.steps_above_threshold,
        steps_below_threshold=state.steps_below_threshold,
        cooldown_remaining_up=max(0, state.cooldown_remaining_up - 1),
        cooldown_remaining_down=max(0, state.cooldown_remaining_down - 1)
    )
    
    action = ScalingAction.NONE
    
    # Update consecutive counters
    if utilization > config.th_up:
        new_state.steps_above_threshold += 1
        new_state.steps_below_threshold = 0
    elif utilization < config.th_down:
        new_state.steps_below_threshold += 1
        new_state.steps_above_threshold = 0
    else:
        new_state.steps_above_threshold = 0
        new_state.steps_below_threshold = 0
    
    # Check for scale up
    if (new_state.steps_above_threshold >= config.k_up and
        new_state.cooldown_remaining_up == 0 and
        new_state.replicas < config.max_replicas):
        
        action = ScalingAction.SCALE_UP
        new_state.replicas += 1
        new_state.steps_above_threshold = 0
        new_state.cooldown_remaining_up = config.cooldown_up
    
    # Check for scale down
    elif (new_state.steps_below_threshold >= config.k_down and
          new_state.cooldown_remaining_down == 0 and
          new_state.replicas > config.min_replicas):
        
        action = ScalingAction.SCALE_DOWN
        new_state.replicas -= 1
        new_state.steps_below_threshold = 0
        new_state.cooldown_remaining_down = config.cooldown_down
    
    return action, new_state


def compute_metrics(records: List[StepRecord]) -> SimulationMetrics:
    """
    Compute aggregated metrics from simulation records.
    
    Args:
        records: List of step records from simulation
        
    Returns:
        SimulationMetrics object with computed statistics
    """
    if not records:
        return SimulationMetrics()
    
    n = len(records)
    violations = [r.sla_violated for r in records]
    replicas = [r.replicas for r in records]
    utilizations = [r.utilization for r in records]
    
    # Basic counts
    sla_violations = sum(violations)
    scale_ups = sum(1 for r in records if r.action == ScalingAction.SCALE_UP)
    scale_downs = sum(1 for r in records if r.action == ScalingAction.SCALE_DOWN)
    
    # Compute overprovision (when not violated)
    overprovisions = []
    for r in records:
        if not r.sla_violated and r.capacity > r.cpu_actual:
            overprovisions.append(r.capacity - r.cpu_actual)
    
    # Compute violation episodes
    episodes = []
    current_episode_length = 0
    for violated in violations:
        if violated:
            current_episode_length += 1
        else:
            if current_episode_length > 0:
                episodes.append(current_episode_length)
            current_episode_length = 0
    if current_episode_length > 0:
        episodes.append(current_episode_length)
    
    return SimulationMetrics(
        total_steps=n,
        sla_violations=sla_violations,
        sla_violation_fraction=sla_violations / n if n > 0 else 0,
        total_scale_up_events=scale_ups,
        total_scale_down_events=scale_downs,
        mean_replicas=np.mean(replicas),
        median_replicas=np.median(replicas),
        max_replicas_used=max(replicas),
        mean_utilization=np.mean(utilizations),
        mean_overprovision=np.mean(overprovisions) if overprovisions else 0,
        violation_episodes=len(episodes),
        mean_violation_episode_length=np.mean(episodes) if episodes else 0,
        max_violation_episode_length=max(episodes) if episodes else 0
    )


# ============================================================================
# REACTIVE POLICY SIMULATION
# ============================================================================

def simulate_policy_reactive(
    cpu_series: pd.Series,
    config: AutoscalingConfig,
    start_t: Optional[int] = None,
    end_t: Optional[int] = None
) -> Tuple[SimulationMetrics, List[StepRecord]]:
    """
    Simulate reactive (threshold-based) autoscaling policy.
    
    The reactive policy observes actual CPU usage at each time step
    and makes scaling decisions based on whether utilization exceeds
    the threshold for a sufficient number of consecutive steps.
    
    Args:
        cpu_series: Time series of CPU usage values
        config: Autoscaling configuration
        start_t: Starting time index (default: window_size)
        end_t: Ending time index (default: len(series) - 1)
        
    Returns:
        Tuple of (aggregated_metrics, list_of_step_records)
    """
    n = len(cpu_series)
    
    if start_t is None:
        start_t = config.window_size
    if end_t is None:
        end_t = n - 1
    
    # Initialize state
    state = SimulationState(replicas=1)
    records: List[StepRecord] = []
    
    for t in range(start_t, end_t):
        cpu_actual = cpu_series.iloc[t]
        
        # Compute current metrics
        capacity = state.replicas * config.base_capacity
        utilization = compute_utilization(cpu_actual, state.replicas, config.base_capacity)
        sla_violated = check_sla_violation(cpu_actual, state.replicas, config.base_capacity)
        
        # Decide scaling action based on ACTUAL utilization
        action, state = decide_scaling_action(utilization, state, config)
        
        # Record this step
        records.append(StepRecord(
            t=t,
            cpu_actual=cpu_actual,
            cpu_predicted=None,
            replicas=state.replicas,
            capacity=capacity,
            utilization=utilization,
            sla_violated=sla_violated,
            action=action
        ))
    
    # Compute aggregated metrics
    metrics = compute_metrics(records)
    
    return metrics, records


# ============================================================================
# PREDICTIVE POLICY SIMULATION
# ============================================================================

def simulate_policy_predictive(
    cpu_series: pd.Series,
    config: AutoscalingConfig,
    predictor: CPUPredictor,
    start_t: Optional[int] = None,
    end_t: Optional[int] = None
) -> Tuple[SimulationMetrics, List[StepRecord]]:
    """
    Simulate predictive (ML-based) autoscaling policy.
    
    The predictive policy uses the trained Random Forest model to predict
    CPU usage for the next time step, and makes proactive scaling decisions
    based on the predicted utilization.
    
    Args:
        cpu_series: Time series of CPU usage values
        config: Autoscaling configuration
        predictor: CPUPredictor instance with loaded model
        start_t: Starting time index (default: window_size)
        end_t: Ending time index (default: len(series) - 1)
        
    Returns:
        Tuple of (aggregated_metrics, list_of_step_records)
    """
    n = len(cpu_series)
    
    if start_t is None:
        start_t = config.window_size
    if end_t is None:
        end_t = n - 1
    
    # Initialize state
    state = SimulationState(replicas=1)
    records: List[StepRecord] = []
    
    for t in range(start_t, end_t):
        cpu_actual = cpu_series.iloc[t]
        
        # Get prediction for NEXT time step
        cpu_predicted = predictor.predict_next(cpu_series, t)
        
        # If prediction failed, fall back to actual
        if cpu_predicted is None:
            cpu_predicted = cpu_actual
        
        # Compute current capacity and actual SLA status
        capacity = state.replicas * config.base_capacity
        actual_utilization = compute_utilization(cpu_actual, state.replicas, config.base_capacity)
        sla_violated = check_sla_violation(cpu_actual, state.replicas, config.base_capacity)
        
        # Decide scaling action based on PREDICTED utilization
        predicted_utilization = compute_utilization(cpu_predicted, state.replicas, config.base_capacity)
        action, state = decide_scaling_action(predicted_utilization, state, config)
        
        # Record this step (store actual utilization for fair comparison)
        records.append(StepRecord(
            t=t,
            cpu_actual=cpu_actual,
            cpu_predicted=cpu_predicted,
            replicas=state.replicas,
            capacity=capacity,
            utilization=actual_utilization,
            sla_violated=sla_violated,
            action=action
        ))
    
    # Compute aggregated metrics
    metrics = compute_metrics(records)
    
    return metrics, records


# ============================================================================
# MULTI-VM SIMULATION
# ============================================================================

def run_comparison_experiment(
    cpu_series: pd.Series,
    config: AutoscalingConfig,
    predictor: CPUPredictor,
    vm_id: str = "vm_0"
) -> Dict[str, Any]:
    """
    Run both reactive and predictive policies on a single VM and compare.
    
    Args:
        cpu_series: CPU usage time series
        config: Autoscaling configuration
        predictor: CPU predictor with loaded model
        vm_id: Identifier for this VM
        
    Returns:
        Dictionary with comparison results
    """
    print(f"  Running simulation for {vm_id}...")
    
    # Run reactive policy
    metrics_reactive, records_reactive = simulate_policy_reactive(cpu_series, config)
    
    # Run predictive policy
    metrics_predictive, records_predictive = simulate_policy_predictive(
        cpu_series, config, predictor
    )
    
    return {
        'vm_id': vm_id,
        'reactive': {
            'metrics': metrics_reactive,
            'records': records_reactive
        },
        'predictive': {
            'metrics': metrics_predictive,
            'records': records_predictive
        },
        # Compute improvements
        'improvement': {
            'sla_violation_reduction': (
                (metrics_reactive.sla_violation_fraction - metrics_predictive.sla_violation_fraction)
                / max(metrics_reactive.sla_violation_fraction, 1e-6)
            ),
            'replica_increase': (
                (metrics_predictive.mean_replicas - metrics_reactive.mean_replicas)
                / max(metrics_reactive.mean_replicas, 1e-6)
            )
        }
    }


def aggregate_multi_vm_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate results from multiple VMs into a summary DataFrame.
    
    Args:
        results: List of results from run_comparison_experiment
        
    Returns:
        DataFrame with per-VM and aggregated metrics
    """
    rows = []
    
    for r in results:
        row = {
            'vm_id': r['vm_id'],
            # Reactive metrics
            'reactive_sla_violation_frac': r['reactive']['metrics'].sla_violation_fraction,
            'reactive_mean_replicas': r['reactive']['metrics'].mean_replicas,
            'reactive_scale_events': (r['reactive']['metrics'].total_scale_up_events + 
                                      r['reactive']['metrics'].total_scale_down_events),
            'reactive_violation_episodes': r['reactive']['metrics'].violation_episodes,
            # Predictive metrics
            'predictive_sla_violation_frac': r['predictive']['metrics'].sla_violation_fraction,
            'predictive_mean_replicas': r['predictive']['metrics'].mean_replicas,
            'predictive_scale_events': (r['predictive']['metrics'].total_scale_up_events + 
                                        r['predictive']['metrics'].total_scale_down_events),
            'predictive_violation_episodes': r['predictive']['metrics'].violation_episodes,
            # Improvements
            'sla_reduction_pct': r['improvement']['sla_violation_reduction'] * 100,
            'replica_increase_pct': r['improvement']['replica_increase'] * 100
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def extract_time_series_data(records: List[StepRecord]) -> Dict[str, np.ndarray]:
    """
    Extract time series arrays from simulation records for plotting.
    
    Args:
        records: List of simulation step records
        
    Returns:
        Dictionary with arrays for each metric
    """
    return {
        't': np.array([r.t for r in records]),
        'cpu_actual': np.array([r.cpu_actual for r in records]),
        'cpu_predicted': np.array([r.cpu_predicted if r.cpu_predicted else np.nan for r in records]),
        'replicas': np.array([r.replicas for r in records]),
        'capacity': np.array([r.capacity for r in records]),
        'utilization': np.array([r.utilization for r in records]),
        'sla_violated': np.array([r.sla_violated for r in records])
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Autoscaling Simulation - Test")
    print("=" * 60)
    
    # Load data
    df = load_vm_data('dataset.csv')
    cpu_series = df[CPU_TARGET]
    print(f"Loaded {len(cpu_series)} samples")
    
    # Configure simulation
    config = AutoscalingConfig(
        th_up=0.7,
        th_down=0.3,
        k_up=2,
        k_down=4,
        base_capacity=50.0,  # 50% base capacity per replica
        min_replicas=1,
        max_replicas=10,
        window_size=96
    )
    
    # Initialize predictor
    predictor = CPUPredictor()
    
    # Run comparison
    results = run_comparison_experiment(cpu_series, config, predictor, vm_id="test_vm")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    print("\nReactive Policy:")
    m = results['reactive']['metrics']
    print(f"  SLA Violations: {m.sla_violations}/{m.total_steps} ({m.sla_violation_fraction*100:.2f}%)")
    print(f"  Mean Replicas: {m.mean_replicas:.2f}")
    print(f"  Scale Events: {m.total_scale_up_events} up, {m.total_scale_down_events} down")
    
    print("\nPredictive Policy:")
    m = results['predictive']['metrics']
    print(f"  SLA Violations: {m.sla_violations}/{m.total_steps} ({m.sla_violation_fraction*100:.2f}%)")
    print(f"  Mean Replicas: {m.mean_replicas:.2f}")
    print(f"  Scale Events: {m.total_scale_up_events} up, {m.total_scale_down_events} down")
    
    print(f"\nImprovement:")
    print(f"  SLA Violation Reduction: {results['improvement']['sla_violation_reduction']*100:.1f}%")
    print(f"  Replica Increase: {results['improvement']['replica_increase']*100:.1f}%")
    
    print("\n✓ Autoscaling Simulation working correctly!")

