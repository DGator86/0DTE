"""
tests/test_recorded_feed_unified_replay.py
=========================================
Recorded-feed-style replay through UnifiedOrchestrator + UnifiedDecisionStack
with persistence assertions (PR #117 merge gate).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from decision_stack.stack import UnifiedDecisionStack
from prediction.deployment import DeploymentBundle
from prediction.storage import PredictionStore
from unified_loop import SyntheticUnifiedFeed, UnifiedOrchestrator

ET = ZoneInfo("America/New_York")


def test_orchestrator_replay_persists_decision_graph(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "pred.sqlite"))
    persisted = {"n": 0}

    def _persist(record, snapshot=None, universe=None, forecast=None,
                 v3_result=None):
        evaluations = None
        if v3_result is not None:
            evaluations = getattr(v3_result, "evaluations", None)
        from decision_stack.persistence import persist_unified_decision
        persist_unified_decision(
            store, record, snapshot=snapshot, universe=universe,
            forecast=forecast, evaluations=evaluations)
        persisted["n"] += 1

    dep = DeploymentBundle(
        deployment_id="replay-shadow",
        mode="shadow",
        authority_source="legacy",
        fallback_policy="abstain",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="replayhash",
    )
    stack = UnifiedDecisionStack(
        deployment=dep,
        persist_fn=_persist,
    )
    feed = SyntheticUnifiedFeed(days=2, seed=11)
    orch = UnifiedOrchestrator(
        feed=feed,
        decision_stack=stack,
        deployment_bundle=dep,
        prediction_store=store,
    )
    start = dt.datetime(2026, 6, 26, 10, 0, tzinfo=ET)
    ticks = [start + dt.timedelta(minutes=i * 5) for i in range(8)]
    results = orch.run_replay(ticks)
    assert len(results) >= 3
    assert persisted["n"] >= 1

    n_dec = store.conn.execute(
        "SELECT COUNT(*) FROM unified_decisions").fetchone()[0]
    n_snap = store.conn.execute(
        "SELECT COUNT(*) FROM canonical_snapshots").fetchone()[0]
    assert n_dec >= 1
    assert n_snap >= 1

    # Paper intents are built after unified evaluation (fields present).
    last = results[-1]
    assert last.authority_source is not None or last.legacy_decision is not None
    assert isinstance(last.paper_intents, list)


def test_paper_intents_use_post_stack_v3(tmp_path):
    """V3 paper intent must read _tick_unified_v3, not a stale pre-stack summary."""
    dep = DeploymentBundle(
        deployment_id="order-test",
        mode="shadow",
        authority_source="legacy",
        fallback_policy="abstain",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        configuration_hash="h",
    )
    stack = UnifiedDecisionStack(deployment=dep)
    orch = UnifiedOrchestrator(
        feed=SyntheticUnifiedFeed(days=1, seed=3),
        decision_stack=stack,
        deployment_bundle=dep,
    )
    # Simulate a stale pre-stack Part 3 summary that would wrongly TRADE,
    # then a post-stack V3 decision of NO_EDGE — paper must follow post-stack.
    orch._tick_part3 = {
        "decision_summary": {
            "action": "TRADE",
            "statistical_action": "TRADE",
            "selected_candidate_id": "stale",
            "family": "put_credit",
        }
    }
    orch._tick_unified_v3 = {
        "final_action": "NO_EDGE",
        "statistical_action": "NO_EDGE",
        "candidate_id": None,
        "structure": None,
        "direction": None,
    }
    orch._tick_authoritative = {
        "final_action": "NO_EDGE",
        "selected_candidate_id": None,
        "structure": None,
        "direction": None,
        "size_mult": 0.0,
    }
    intents = orch._build_paper_intents(
        snap=None, signals={}, intent=None, regime_state=None,
        decision=None, decide_pdf=None, cfg=None, pin_active=False,
        density_mode="vrp", density_moments=None, final_size_mult=1.0,
        matrix_stand_down=True,
    )
    assert all(i.get("track") != "v3" for i in intents)
