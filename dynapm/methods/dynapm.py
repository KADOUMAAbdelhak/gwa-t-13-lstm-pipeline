"""DynAPM method factories and ablations."""
from __future__ import annotations

from dynapm.core.update_engine import UpdateEngine


def dynapm_factory(alpha: float, **kwargs):
    def factory(profile, cpu_thresholds, mem_thresholds):
        return UpdateEngine(
            profile,
            cpu_thresholds,
            mem_thresholds,
            alpha=alpha,
            use_decision_head=True,
            use_churn_control=True,
        )

    return factory


def dynapm_no_learner_factory(alpha: float, **kwargs):
    def factory(profile, cpu_thresholds, mem_thresholds):
        return UpdateEngine(
            profile,
            cpu_thresholds,
            mem_thresholds,
            alpha=alpha,
            use_decision_head=False,
            use_churn_control=True,
        )

    return factory


def dynapm_no_churn_factory(alpha: float, **kwargs):
    def factory(profile, cpu_thresholds, mem_thresholds):
        return UpdateEngine(
            profile,
            cpu_thresholds,
            mem_thresholds,
            alpha=alpha,
            use_decision_head=True,
            use_churn_control=False,
        )

    return factory
