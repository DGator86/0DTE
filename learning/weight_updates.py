"""
learning/weight_updates.py
==========================
Dynamic ensemble weight updates from settled sessions only.

NOT financial advice.
"""
from __future__ import annotations

from typing import Any


def update_dynamic_weights(
    *,
    settled_sessions: list,
    current_weights: dict | None = None,
) -> dict:
    if not settled_sessions:
        return dict(current_weights or {})
    # Placeholder: equal weights across observed components.
    weights = dict(current_weights or {})
    for s in settled_sessions:
        for k, v in (s.get("component_losses") or {}).items():
            weights.setdefault(k, 1.0)
            # Shrink toward better (lower) loss without using current session.
            try:
                weights[k] = 0.9 * float(weights[k]) + 0.1 * (1.0 / (1.0 + float(v)))
            except (TypeError, ValueError):
                continue
    return weights
