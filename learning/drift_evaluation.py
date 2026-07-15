"""
learning/drift_evaluation.py
============================
Drift severity evaluation (NORMAL/WATCH/DEGRADED/FREEZE).

NOT financial advice.
"""
from __future__ import annotations

from typing import Any


def evaluate_drift(*, metrics: dict | None = None) -> dict:
    metrics = dict(metrics or {})
    severity = str(metrics.get("severity") or "NORMAL").upper()
    if severity not in ("NORMAL", "WATCH", "DEGRADED", "FREEZE"):
        # Derive from simple thresholds when provided
        cal_err = float(metrics.get("calibration_error") or 0.0)
        if cal_err >= 0.25:
            severity = "FREEZE"
        elif cal_err >= 0.15:
            severity = "DEGRADED"
        elif cal_err >= 0.08:
            severity = "WATCH"
        else:
            severity = "NORMAL"
    return {
        "severity": severity,
        "metrics": metrics,
        "auto_promote_replacement": False,
    }
