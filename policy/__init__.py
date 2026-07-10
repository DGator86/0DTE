"""
policy
======
Prediction Engine V2 policy layer
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17, PR 10).

Separates forecast (PredictionBundle) from trade policy. Two implementations
run in parallel during the transition:

  LegacyMatrixPolicy  — wraps regime classifier + 27-cell matrix intent
  PredictionPolicy    — consumes PredictionBundle (+ structural / risk state)

PolicyRouter modes:
  legacy    — matrix only (pre-V2 path)
  shadow    — both run; legacy is authoritative; disagreement journaled
  champion  — V2 authoritative; explicit fallback_legacy when V2 unavailable

Promotion is a single config pointer (mode), not a code change.
"""
from policy.contracts import (
    PolicyDecision, PolicyInput, PolicyMode, StructuralState,
)
from policy.legacy_matrix import LegacyMatrixPolicy
from policy.prediction_policy import PredictionPolicy, PredictionPolicyConfig
from policy.router import PolicyRouteResult, PolicyRouter, PolicyRouterConfig

__all__ = [
    "PolicyDecision", "PolicyInput", "PolicyMode", "StructuralState",
    "LegacyMatrixPolicy",
    "PredictionPolicy", "PredictionPolicyConfig",
    "PolicyRouteResult", "PolicyRouter", "PolicyRouterConfig",
]
