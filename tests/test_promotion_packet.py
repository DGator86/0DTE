"""
tests/test_promotion_packet.py
==============================
V3 Part 3 PR30 — promotion packet and human approval (§37 / §52).
"""
from __future__ import annotations

import pytest

from prediction.deployment import DeploymentError
from prediction.promotion import (
    PromotionReviewPacket, promote_model, validate_promotion_packet,
)
from prediction.reports.promotion_packet import render_promotion_report


def _packet(**kw) -> PromotionReviewPacket:
    base = dict(
        review_id="rev-1",
        model_group_id="grp-1",
        model_ids=["m1"],
        artifact_hashes={"m1": "abc"},
        dataset_hashes={"train": "def"},
        configuration_hashes={"cfg": "ghi"},
        git_commit="deadbeef",
        dependency_versions={"sklearn": "1.0"},
        feature_versions={"features": "v3"},
        label_versions={"labels": "v3"},
        policy_version="v3",
        execution_version="v3",
        training_sessions=["2026-01-01"],
        calibration_sessions=["2026-02-01"],
        outer_test_sessions=["2026-03-01"],
        fold_definitions={"fold0": {"train": [], "test": []}},
        headline_metrics={"brier_skill": 0.1},
        slice_metrics={},
        bootstrap_intervals={},
        drift_status={"severity": "NORMAL"},
        legacy_comparison={"utility_delta": 0.05},
        known_weaknesses=["late session thin"],
        unsupported_slices=[],
        rollback_target={"prediction_model_group": "legacy"},
    )
    base.update(kw)
    return PromotionReviewPacket(**base)


def test_packet_requires_hashes_and_rollback():
    p = _packet()
    validate_promotion_packet(p)
    with pytest.raises(DeploymentError):
        validate_promotion_packet(_packet(review_id=""))
    with pytest.raises(DeploymentError):
        validate_promotion_packet(_packet(rollback_target={}))
    with pytest.raises(DeploymentError):
        validate_promotion_packet(_packet(fold_definitions={}))


def test_promote_requires_reviewer_and_note():
    calls = []

    def set_status(mid, status, note=""):
        calls.append((mid, status, note))
        return {"model_id": mid, "status": status}

    with pytest.raises(DeploymentError, match="reviewer"):
        promote_model(
            packet=_packet(), target_status="candidate", reviewer="",
            approval_note="ok", current_artifact_status="shadow",
            registry_set_status=set_status,
        )
    with pytest.raises(DeploymentError, match="shadow"):
        promote_model(
            packet=_packet(), target_status="champion", reviewer="alice",
            approval_note="ship it", current_artifact_status="shadow",
            registry_set_status=set_status,
        )
    result = promote_model(
        packet=_packet(), target_status="candidate", reviewer="alice",
        approval_note="looks good", current_artifact_status="shadow",
        registry_set_status=set_status,
    )
    assert result["retrained"] is False
    assert calls and calls[0][1] == "candidate"
    report = render_promotion_report(_packet())
    assert "Rollback plan" in report
