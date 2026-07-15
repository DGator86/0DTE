"""
prediction/promotion.py
=======================
Human-gated model promotion
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §35–§37).

No automatic promotion. Requires review ID, reviewer, approval note,
and rollback target.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from prediction.deployment import DeploymentError, assert_mode_permission


@dataclass
class PromotionReviewPacket:
    review_id: str
    model_group_id: str
    model_ids: list
    artifact_hashes: dict
    dataset_hashes: dict
    configuration_hashes: dict
    git_commit: str
    dependency_versions: dict
    feature_versions: dict
    label_versions: dict
    policy_version: str
    execution_version: str
    training_sessions: list
    calibration_sessions: list
    outer_test_sessions: list
    fold_definitions: dict
    headline_metrics: dict
    slice_metrics: dict
    bootstrap_intervals: dict
    drift_status: dict
    legacy_comparison: dict
    known_weaknesses: list
    unsupported_slices: list
    rollback_target: dict
    reviewer: Optional[str] = None
    review_timestamp: Optional[str] = None
    approval_status: str = "pending"
    approval_note: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def packet_hash(self) -> str:
        payload = dict(self.to_dict())
        payload.pop("approval_note", None)
        payload.pop("approval_status", None)
        payload.pop("review_timestamp", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")).hexdigest()


def validate_promotion_packet(packet: PromotionReviewPacket) -> None:
    if not packet.review_id:
        raise DeploymentError("promotion requires review_id")
    if not packet.rollback_target:
        raise DeploymentError("promotion requires rollback_target")
    if not packet.artifact_hashes:
        raise DeploymentError("promotion requires artifact_hashes")
    if not packet.fold_definitions:
        raise DeploymentError("promotion requires fold_definitions")
    if not packet.dataset_hashes:
        raise DeploymentError("promotion requires dataset_hashes")


def promote_model(
    *,
    packet: PromotionReviewPacket,
    target_status: str,
    reviewer: str,
    approval_note: str,
    current_artifact_status: str,
    registry_set_status,
) -> dict:
    """
    Explicit human promotion. Does not retrain. Does not auto-pick best model.

    registry_set_status: callable(model_id, status, note) -> meta
    """
    validate_promotion_packet(packet)
    if not reviewer:
        raise DeploymentError("promotion requires reviewer identity")
    if not packet.review_id:
        raise DeploymentError("promotion requires review_id")
    if not approval_note:
        raise DeploymentError("promotion requires approval_note")
    if target_status == "champion":
        # Shadow cannot become champion directly
        if current_artifact_status == "shadow":
            raise DeploymentError(
                "shadow artifact cannot become champion directly; "
                "promote to candidate first")
        if current_artifact_status not in (
                "candidate", "pending_review", "champion"):
            raise DeploymentError(
                f"cannot promote status {current_artifact_status!r} "
                f"to champion without candidate/pending_review")
        assert_mode_permission("candidate", "candidate")
    # Apply status updates
    updated = []
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    for mid in packet.model_ids:
        meta = registry_set_status(
            mid, target_status,
            note=f"review={packet.review_id}; {approval_note}")
        updated.append(meta.get("model_id", mid))
    packet.reviewer = reviewer
    packet.approval_note = approval_note
    packet.approval_status = "approved"
    packet.review_timestamp = ts
    return {
        "review_id": packet.review_id,
        "target_status": target_status,
        "reviewer": reviewer,
        "approval_note": approval_note,
        "updated_models": updated,
        "rollback_target": dict(packet.rollback_target),
        "packet_hash": packet.packet_hash(),
        "approved_at": ts,
        "retrained": False,
    }
