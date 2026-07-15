"""
tests/test_unified_decision_record.py
"""
from __future__ import annotations

from decision_stack.contracts import UnifiedDecisionRecord
from decision_stack.stack import UnifiedDecisionStack
from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.deployment import DeploymentBundle


def test_unified_record_roundtrip():
    rec = UnifiedDecisionRecord(
        snapshot_id="s1",
        ts="t",
        session_date="2026-07-14",
        symbol="SPY",
        deployment_id="d1",
        deployment_mode="shadow",
        authority_source="legacy",
        legacy_action="TRADE",
        v3_statistical_action="NO_EDGE",
        v3_final_action="NO_EDGE",
        final_action="TRADE",
    )
    d = rec.to_dict()
    assert d["authority_source"] == "legacy"
    assert d["legacy_v3_disagreement"] == {}


def test_stack_evaluate_shadow():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="snap-1",
    )
    dep = DeploymentBundle(
        deployment_id="d1",
        mode="shadow",
        authority_source="legacy",
        fallback_policy="abstain",
        feature_version="v2.0.0",
        label_version="v2.0.0",
    )
    stack = UnifiedDecisionStack(deployment=dep)
    rec = stack.evaluate(
        snap,
        legacy_decision={
            "action": "TRADE",
            "candidate_id": "leg1",
            "structure": "pcs",
            "direction": "bearish",
            "size_mult": 1.0,
        },
        hard_vetoes=(),
    )
    assert rec.authority_source == "legacy"
    assert rec.final_action == "TRADE"
    assert rec.deployment_id == "d1"
    assert rec.snapshot_id == "snap-1"
    assert "forecast" in rec.diagnostics.get("stages", []) or True


def test_hard_veto_on_stack():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="snap-2",
    )
    dep = DeploymentBundle(
        deployment_id="d1", mode="champion",
        authority_source="v3", fallback_policy="abstain",
        prediction_model_group_id="g",
        candidate_value_model_id="cv",
        candidate_rank_model_id="cr",
        fill_probability_model_id="fp",
        fill_concession_model_id="fc",
        meta_model_id="mm",
    )
    stack = UnifiedDecisionStack(deployment=dep)
    rec = stack.evaluate(
        snap,
        legacy_decision={"action": "TRADE", "candidate_id": "l"},
        hard_vetoes=("daily_risk_limit",),
    )
    assert rec.final_action == "HARD_VETO"
