"""
learning/labels.py
==================
Label construction helpers (handoff §17).

NOT financial advice.
"""
from __future__ import annotations

from learning.settlement import LABEL_VERSION, build_market_labels


def meta_decision_labels(decisions: list) -> list:
    """Evaluation labels for TRADE / NO_EDGE / ABSTAIN / HARD_VETO."""
    out = []
    for d in decisions:
        row = dict(d) if isinstance(d, dict) else {}
        action = row.get("final_action") or row.get("v3_final_action")
        pnl = row.get("realized_executable_pnl")
        out.append({
            "snapshot_id": row.get("snapshot_id"),
            "action": action,
            "positive_executable_value": (
                None if pnl is None else bool(float(pnl) > 0)),
            "label_version": LABEL_VERSION,
        })
    return out


__all__ = ["LABEL_VERSION", "build_market_labels", "meta_decision_labels"]
