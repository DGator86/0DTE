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
    artifacts = packet.get("model_artifact_ids") or {}
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("promotion requires nonempty model_artifact_ids")
    required_roles = (
        "prediction_model_group", "candidate_value", "candidate_rank",
        "fill_probability", "fill_concession", "meta_model",
    )
    # Accept either role keys or any nonempty role→id mapping that covers
    # the required set (also allow short aliases used in tests).
    aliases = {
        "prediction_model_group": ("prediction_model_group", "group", "model_group"),
        "candidate_value": ("candidate_value", "cv"),
        "candidate_rank": ("candidate_rank", "cr"),
        "fill_probability": ("fill_probability", "fp"),
        "fill_concession": ("fill_concession", "fc"),
        "meta_model": ("meta_model", "meta", "mm"),
    }
    missing_roles = []
    for role in required_roles:
        keys = aliases.get(role, (role,))
        if not any(artifacts.get(k) for k in keys):
            missing_roles.append(role)
    if missing_roles:
        raise ValueError(
            f"promotion requires model_artifact_ids roles: {missing_roles}")
    oos = packet.get("oos_metrics") or {}
    if not isinstance(oos, dict) or not oos:
        raise ValueError("promotion requires nonempty oos_metrics")
    boot = packet.get("bootstrap_intervals")
    if not isinstance(boot, dict) or not boot:
        raise ValueError("promotion requires nonempty bootstrap_intervals")
    weaknesses = packet.get("known_weaknesses")
    if not isinstance(weaknesses, list) or not weaknesses:
        raise ValueError("promotion requires nonempty known_weaknesses")
    unsupported = packet.get("unsupported_slices")
    if not isinstance(unsupported, list) or not unsupported:
        raise ValueError("promotion requires nonempty unsupported_slices")
    out = dict(packet)
    out["reviewer"] = reviewer
    out["approval_note"] = approval_note
    out["approved"] = True
    out["approved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out["review_id"] = (
        packet.get("review_id")
        or f"rev-{out['approved_at']}"
    )
    return out


def validate_champion_packet(packet: dict) -> None:
    """Champion mode requires a verified promotion packet with review id."""
    if not packet.get("approved"):
        raise ValueError("champion requires approved promotion packet")
    if not packet.get("review_id"):
        raise ValueError("champion requires approved_review_id / review_id")
    if not packet.get("configuration_hash"):
        raise ValueError("champion requires configuration_hash")
    if not packet.get("rollback_deployment_id"):
        raise ValueError("champion requires rollback_deployment_id")
    artifacts = packet.get("model_artifact_ids") or {}
    if not artifacts:
        raise ValueError("champion requires model_artifact_ids")

