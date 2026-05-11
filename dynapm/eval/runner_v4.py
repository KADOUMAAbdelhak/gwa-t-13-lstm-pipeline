"""Compatibility shim for DynAPM runners.

Deprecated: ``dynapm.eval.runner_v4`` previously hosted a duplicate evaluator with
``verify_pending_commits`` hooks. ``runner_v2.run_method_on_vm`` is now the canonical
implementation and invokes ``verify_pending_commits`` when a policy exposes it.
Prefer importing ``run_method_on_vm`` from ``dynapm.eval.runner_v2``.
"""
from __future__ import annotations

from dynapm.eval.runner_v2 import run_method_on_vm

__all__ = ["run_method_on_vm"]
