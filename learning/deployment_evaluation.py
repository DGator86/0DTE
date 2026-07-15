"""
learning/deployment_evaluation.py
=================================
Complete economic stack evaluation for a deployment candidate.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Optional


def evaluate_deployment_bundle(
    *,
    deployment_id: str,
    comparison_deployment_id: Optional[str] = None,
    session_start: str = "",
    session_end: str = "",
    sessions: list | None = None,
    metrics: dict | None = None,
    slice_metrics: dict | None = None,
    bootstrap_intervals: dict | None = None,
    drift: dict | None = None,
) -> dict:
    sessions = list(sessions or [])
    return {
        "evaluation_id": uuid.uuid4().hex,
        "deployment_id": deployment_id,
        "comparison_deployment_id": comparison_deployment_id,
        "session_start": session_start,
        "session_end": session_end,
        "sessions_count": len(sessions),
        "metrics_json": dict(metrics or {}),
        "slice_metrics_json": dict(slice_metrics or {}),
        "bootstrap_intervals_json": dict(bootstrap_intervals or {}),
        "drift_json": dict(drift or {}),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "promoted": False,
    }
