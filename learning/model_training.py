"""
learning/model_training.py
==========================
Scheduled model-family retraining stubs coordinated by LearningOrchestrator.

Does not promote. Produces candidate artifacts only.

NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Optional


def retrain_eligible_families(
    *,
    families: list | None = None,
    sessions: list | None = None,
    registry: Any = None,
) -> dict:
    """
    Retrain eligible families with family-specific losses.
    Returns candidate artifact descriptors — never writes champion status.
    """
    families = list(families or [
        "direction", "returns", "volatility", "range_survival",
        "regime", "competing_risk", "candidate_value", "candidate_rank",
        "fill_probability", "fill_concession", "trade_meta",
    ])
    return {
        "families": families,
        "sessions": list(sessions or []),
        "candidates": [],
        "status": "research",
        "promoted": False,
    }
