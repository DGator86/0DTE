"""
prediction/reports/promotion_packet.py
======================================
Human-readable promotion report builder (Part 3 §37.2).
"""
from __future__ import annotations

from prediction.promotion import PromotionReviewPacket


def render_promotion_report(packet: PromotionReviewPacket) -> str:
    d = packet.to_dict()
    sections = [
        "# Promotion Review Packet",
        "",
        "## Executive summary",
        f"Review ID: {d['review_id']}",
        f"Model group: {d['model_group_id']}",
        f"Approval status: {d['approval_status']}",
        "",
        "## Model architecture",
        f"Models: {', '.join(d['model_ids'])}",
        "",
        "## Training data",
        f"Training sessions: {len(d['training_sessions'])}",
        f"Calibration sessions: {len(d['calibration_sessions'])}",
        f"Outer test sessions: {len(d['outer_test_sessions'])}",
        "",
        "## Validation design",
        f"Fold definitions keys: {sorted((d['fold_definitions'] or {}).keys())}",
        "",
        "## Headline metrics",
        str(d.get("headline_metrics") or {}),
        "",
        "## Drift analysis",
        str(d.get("drift_status") or {}),
        "",
        "## Legacy comparison",
        str(d.get("legacy_comparison") or {}),
        "",
        "## Known limitations",
        "\n".join(f"- {w}" for w in (d.get("known_weaknesses") or ["none"])),
        "",
        "## Rollback plan",
        str(d.get("rollback_target") or {}),
        "",
        "## Promotion recommendation",
        f"Reviewer: {d.get('reviewer') or '(pending)'}",
        f"Note: {d.get('approval_note') or '(pending)'}",
    ]
    return "\n".join(sections) + "\n"
