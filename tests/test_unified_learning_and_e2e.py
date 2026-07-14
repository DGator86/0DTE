"""
tests/test_dual_paper_accounts.py
tests/test_counterfactual_settlement.py
tests/test_unified_labels.py
tests/test_learning_orchestrator.py
tests/test_joint_deployment_evaluation.py
tests/test_joint_promotion.py
tests/test_atomic_full_stack_rollback.py
tests/test_true_end_to_end_replay.py
"""
from __future__ import annotations

import pytest

from decision_stack.authority import resolve_authority
from learning.deployment_evaluation import evaluate_deployment_bundle
from learning.orchestrator import LearningOrchestrator
from learning.promotion_packet import (
    approve_promotion, build_joint_promotion_packet,
)
from learning.settlement import settle_session_counterfactuals
from learning.labels import meta_decision_labels
from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.deployment import (
    DeploymentBundle, load_deployment_pointer, rollback_deployment,
    write_deployment_bundle,
)
from prediction.part3_decision import build_v3_decision
from decision_stack.stack import UnifiedDecisionStack


def test_dual_paper_authority_split():
    r = resolve_authority(
        mode="candidate",
        legacy_decision={"action": "TRADE", "candidate_id": "L"},
        v3_decision={"action": "NO_EDGE", "candidate_id": "V"},
    )
    assert r.reference_account == "legacy"
    assert r.candidate_account == "v3"


def test_counterfactual_settlement_idempotent():
    a = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1", "action": "NO_EDGE"}],
        candidate_evaluations=[
            {"candidate_id": "c1", "snapshot_id": "s1"},
        ],
        fill_records=[{"fill_status": "unfilled", "candidate_id": "c1"}],
    )
    b = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1", "action": "NO_EDGE"}],
        candidate_evaluations=[
            {"candidate_id": "c1", "snapshot_id": "s1"},
        ],
        fill_records=[{"fill_status": "unfilled", "candidate_id": "c1"}],
    )
    assert a["complete"] is False and b["complete"] is False
    assert len(a["unfilled_attempts"]) == 1


def test_unified_labels_meta():
    labels = meta_decision_labels([
        {"snapshot_id": "s1", "final_action": "TRADE",
         "realized_executable_pnl": 12.0},
        {"snapshot_id": "s2", "final_action": "NO_EDGE",
         "realized_executable_pnl": -5.0},
    ])
    assert labels[0]["positive_executable_value"] is True
    assert labels[1]["positive_executable_value"] is False


def test_learning_orchestrator_no_auto_promote():
    orch = LearningOrchestrator()
    daily = orch.run_daily(session_date="2026-07-14", journal_rows=[])
    assert daily["promoted"] is False
    weekly = orch.run_weekly(sessions=["2026-07-01", "2026-07-02"])
    assert weekly["promoted"] is False
    with pytest.raises(ValueError, match="holdout"):
        orch.run_weekly(sessions=[])


def test_joint_deployment_evaluation():
    ev = evaluate_deployment_bundle(
        deployment_id="d1",
        comparison_deployment_id="legacy",
        sessions=["a", "b"],
        metrics={"net_pnl": 1.0},
    )
    assert ev["promoted"] is False
    assert ev["sessions_count"] == 2


def test_joint_promotion_requires_human():
    pkt = build_joint_promotion_packet(
        deployment_id="d1",
        current_status="candidate",
        proposed_status="champion",
        legacy_rule_config_id="rule1",
        model_artifact_ids={"group": "g1"},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="abc",
        fold_definitions={"outer": ["s1"]},
        oos_metrics={"net_pnl": 1.0},
        bootstrap_intervals={"net_pnl": [0.0, 2.0]},
        known_weaknesses=["cold_start"],
        unsupported_slices=["illiquid"],
        rollback_deployment_id="d0",
    )
    assert pkt["auto_promoted"] is False
    assert pkt["approved"] is False
    approved = approve_promotion(
        pkt, reviewer="alice", approval_note="looks good")
    assert approved["approved"] is True


def test_shadow_cannot_skip_to_champion():
    with pytest.raises(ValueError, match="illegal promotion"):
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


def test_promotion_requires_reviewer():
    pkt = build_joint_promotion_packet(
        deployment_id="d1",
        current_status="advisory",
        proposed_status="candidate",
        legacy_rule_config_id=None,
        model_artifact_ids={},
        feature_version="v2",
        label_version="v2",
        configuration_hash="x",
        fold_definitions={"outer": ["s"]},
        rollback_deployment_id="d0",
    )
    with pytest.raises(ValueError, match="reviewer"):
        approve_promotion(pkt, reviewer="", approval_note="x")


def test_atomic_full_stack_rollback(tmp_path):
    path = str(tmp_path / "deployment.json")
    prior = DeploymentBundle(
        deployment_id="prior",
        mode="shadow",
        prediction_model_group_id="g0",
        candidate_value_model_id="cv0",
        candidate_rank_model_id="cr0",
        fill_probability_model_id="fp0",
        fill_concession_model_id="fc0",
        meta_model_id="mm0",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="legacy",
        fallback_policy="abstain",
    )
    current = DeploymentBundle(
        deployment_id="current",
        mode="shadow",
        prediction_model_group_id="g1",
        candidate_value_model_id="cv1",
        candidate_rank_model_id="cr1",
        fill_probability_model_id="fp1",
        fill_concession_model_id="fc1",
        meta_model_id="mm1",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="legacy",
        fallback_policy="abstain",
        previous_deployment_id="prior",
        rollback_deployment_id="prior",
    )
    write_deployment_bundle(path, current)
    audit = rollback_deployment(
        path, prior_pointer=prior.to_dict(), reason="regression",
        trigger_source="human")
    restored = load_deployment_pointer(path)
    assert restored["deployment_id"] == "prior"
    assert restored["prediction_model_group"] == "g0"
    assert audit["human_or_automatic"] == "human"


def test_true_end_to_end_replay():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "adx": 20.0},
        snapshot_id="e2e-1",
    )
    universe = build_candidate_universe(
        snapshot_id="e2e-1",
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
        deployment_id="e2e-dep",
        mode="shadow",
        authority_source="legacy",
        fallback_policy="abstain",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="e2ehash",
    )

    def _universe_fn(snapshot, forecast=None):
        return universe

    stack = UnifiedDecisionStack(
        deployment=dep,
        candidate_universe_fn=_universe_fn,
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
    assert r1.snapshot_id == r2.snapshot_id == "e2e-1"
    assert r1.final_action == r2.final_action
    assert r1.authority_source == "legacy"
    assert r1.deployment_id == "e2e-dep"

    forecast = PredictionBundle(
        snapshot_id="e2e-1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    v3 = build_v3_decision(
        snapshot=snap, forecast=forecast, universe=universe, mode="shadow")
    assert v3.candidate_id is not None

    settlement = settle_session_counterfactuals(
        session_date=snap.session_date,
        journal_rows=[{"snapshot_id": "e2e-1", "action": r1.final_action}],
        candidate_evaluations=list(v3.evaluations),
    )
    assert settlement["complete"] is False  # no settlement_fn

    orch = LearningOrchestrator()
    learned = orch.run_daily(
        session_date=snap.session_date,
        journal_rows=[{"snapshot_id": "e2e-1"}],
        candidate_evaluations=list(v3.evaluations),
    )
    assert learned["promoted"] is False

    evaluation = evaluate_deployment_bundle(
        deployment_id=dep.deployment_id,
        comparison_deployment_id="legacy-baseline",
        sessions=[snap.session_date],
        metrics={"net_pnl": 0.0},
    )
    pkt = build_joint_promotion_packet(
        deployment_id=dep.deployment_id,
        current_status="research",
        proposed_status="shadow",
        legacy_rule_config_id=None,
        model_artifact_ids={},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash=dep.configuration_hash,
        fold_definitions={"outer": [snap.session_date]},
        rollback_deployment_id="prior",
        known_weaknesses=["cold_start"],
        unsupported_slices=["low_liquidity"],
    )
    assert evaluation["deployment_id"] == dep.deployment_id
    assert pkt["rollback_deployment_id"] == "prior"
    assert r1.to_dict()["snapshot_id"] == snap.snapshot_id
