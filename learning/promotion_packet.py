"""
learning/promotion_packet.py
============================
Joint promotion packet for complete deployment bundles.
Requires human reviewer — never auto-promotes.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional


ALLOWED_TRANSITIONS = {
    "research": {"shadow"},
    "shadow": {"advisory"},
    "advisory": {"candidate"},
    "candidate": {"champion"},
}


def build_joint_promotion_packet(
    *,
    deployment_id: str,
    current_status: str,
    proposed_status: str,
    legacy_rule_config_id: Optional[str],
    model_artifact_ids: dict,
    feature_version: str,
    label_version: str,
    configuration_hash: str,
    fold_definitions: dict | None = None,
    oos_metrics: dict | None = None,
    bootstrap_intervals: dict | None = None,
    known_weaknesses: list | None = None,
    unsupported_slices: list | None = None,
    rollback_deployment_id: Optional[str] = None,
    reviewer: Optional[str] = None,
    approval_note: Optional[str] = None,
) -> dict:
    if proposed_status not in ALLOWED_TRANSITIONS.get(current_status, set()):
        raise ValueError(
            f"illegal promotion transition {current_status!r} → "
            f"{proposed_status!r}; shadow cannot skip to champion")
    return {
        "deployment_id": deployment_id,
        "current_status": current_status,
        "proposed_status": proposed_status,
        "legacy_rule_config_id": legacy_rule_config_id,
        "model_artifact_ids": dict(model_artifact_ids or {}),
        "feature_version": feature_version,
        "label_version": label_version,
        "configuration_hash": configuration_hash,
        "fold_definitions": dict(fold_definitions or {}),
        "oos_metrics": dict(oos_metrics or {}),
        "bootstrap_intervals": dict(bootstrap_intervals or {}),
        "known_weaknesses": list(known_weaknesses or []),
        "unsupported_slices": list(unsupported_slices or []),
        "rollback_deployment_id": rollback_deployment_id,
        "reviewer": reviewer,
        "approval_note": approval_note,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "auto_promoted": False,
        "approved": bool(reviewer and approval_note and rollback_deployment_id),
    }


def approve_promotion(packet: dict, *, reviewer: str, approval_note: str) -> dict:
    if not reviewer or not approval_note:
        raise ValueError("promotion requires reviewer and approval_note")
    if not packet.get("rollback_deployment_id"):
        raise ValueError("promotion requires rollback_deployment_id")
    if not packet.get("configuration_hash"):
        raise ValueError("promotion requires configuration_hash")
    if not packet.get("fold_definitions"):
        raise ValueError("promotion requires fold_definitions")
    out = dict(packet)
    out["reviewer"] = reviewer
    out["approval_note"] = approval_note
    out["approved"] = True
    out["approved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out
