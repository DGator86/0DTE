"""tests/test_joint_promotion.py"""
import pytest

from learning.promotion_packet import (
    approve_promotion, build_joint_promotion_packet,
)


def test_promotion_requires_reviewer_note_rollback_folds():
    pkt = build_joint_promotion_packet(
        deployment_id="d1",
        current_status="candidate",
        proposed_status="champion",
        legacy_rule_config_id="r1",
        model_artifact_ids={"group": "g1"},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="abc",
        fold_definitions={"outer": ["s1"]},
        rollback_deployment_id="d0",
        known_weaknesses=["x"],
        unsupported_slices=["y"],
    )
    with pytest.raises(ValueError):
        approve_promotion(pkt, reviewer="", approval_note="ok")
    approved = approve_promotion(
        pkt, reviewer="bob", approval_note="ship it")
    assert approved["approved"] is True


def test_shadow_cannot_become_champion():
    with pytest.raises(ValueError, match="illegal"):
        build_joint_promotion_packet(
            deployment_id="d1",
            current_status="shadow",
            proposed_status="champion",
            legacy_rule_config_id=None,
            model_artifact_ids={},
            feature_version="v2",
            label_version="v2",
            configuration_hash="x",
            rollback_deployment_id="d0",
        )
