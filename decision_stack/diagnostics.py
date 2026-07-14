"""
decision_stack/diagnostics.py
=============================
Structured component-failure records (handoff §20).

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional


def component_failure_record(
    *,
    snapshot_id: str,
    component: str,
    stage: str,
    exception: BaseException | None = None,
    message: str = "",
    required_or_optional: str = "optional",
    fallback_action: str = "record_only",
    deployment_mode: str = "shadow",
    model_id: Optional[str] = None,
    configuration_hash: str = "",
) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "component": component,
        "stage": stage,
        "exception_type": (
            type(exception).__name__ if exception is not None else None),
        "message": message or (str(exception) if exception else ""),
        "required_or_optional": required_or_optional,
        "fallback_action": fallback_action,
        "deployment_mode": deployment_mode,
        "model_id": model_id,
        "configuration_hash": configuration_hash,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
