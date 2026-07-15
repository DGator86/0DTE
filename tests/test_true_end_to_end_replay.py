"""tests/test_true_end_to_end_replay.py"""
from decision_stack.stack import UnifiedDecisionStack
from learning.deployment_evaluation import evaluate_deployment_bundle
from learning.orchestrator import LearningOrchestrator
from learning.promotion_packet import build_joint_promotion_packet
from learning.settlement import settle_session_counterfactuals
from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.deployment import DeploymentBundle
from prediction.part3_decision import build_v3_decision


def test_repeated_replay_identical_records():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="replay-1",
    )
    universe = build_candidate_universe(
        snapshot_id="replay-1",
        generated_at=snap.ts,
        candidates=[{
            "family": "put_credit",
            "ev": 0.15,
            "prob_profit": 0.65,
            "legs": [
                {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                 "expiration": "2026-07-14"},
                {"right": "P", "side": "buy", "qty": 1, "strike": 485,
                 "expiration": "2026-07-14"},
            ],
        }],
    )
    dep = DeploymentBundle(
        deployment_id="replay-dep",
        mode="shadow",
        authority_source="legacy",
        fallback_policy="abstain",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="h1",
    )
    stack = UnifiedDecisionStack(
        deployment=dep,
        candidate_universe_fn=lambda snapshot, forecast=None: universe,
    )
    legacy = {
        "action": "TRADE",
        "candidate_id": universe.candidate_ids()[0],
        "structure": "put_credit",
        "direction": "bearish",
        "size_mult": 1.0,
    }
    r1 = stack.evaluate(snap, legacy_decision=legacy)
    r2 = stack.evaluate(snap, legacy_decision=legacy)
    assert r1.snapshot_id == r2.snapshot_id
    assert r1.final_action == r2.final_action
    assert r1.authority_source == "legacy"

    forecast = PredictionBundle(
        snapshot_id="replay-1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    v3 = build_v3_decision(
        snapshot=snap, forecast=forecast, universe=universe, mode="shadow")
    settle_session_counterfactuals(
        session_date=snap.session_date,
        journal_rows=[{"snapshot_id": "replay-1"}],
        candidate_evaluations=list(v3.evaluations),
    )
    LearningOrchestrator().run_daily(session_date=snap.session_date)
    evaluate_deployment_bundle(
        deployment_id=dep.deployment_id, sessions=[snap.session_date])
    pkt = build_joint_promotion_packet(
        deployment_id=dep.deployment_id,
        current_status="research",
        proposed_status="shadow",
        legacy_rule_config_id=None,
        model_artifact_ids={},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="h1",
        fold_definitions={"outer": [snap.session_date]},
        rollback_deployment_id="prior",
        known_weaknesses=["cold_start"],
        unsupported_slices=["illiquid"],
    )
    assert pkt["auto_promoted"] is False
    assert "cold_start" in pkt["known_weaknesses"]
