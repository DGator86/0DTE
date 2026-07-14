"""
tests/test_trained_artifact_contracts.py
=======================================
Merge-gate tests for PR #117 review round 2:

* Typed CandidateForecastV3 adapter (real CandidateValueModel)
* Shared fill feature schema train/serve
* Fail-closed forecast / unified-stack exceptions
* Atomic persist_decision_graph
* Ticket.from_unified_decision uses authoritative candidate
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from decision_stack.persistence import persist_unified_decision
from decision_stack.stack import UnifiedDecisionStack
from learning.promotion_packet import approve_promotion, build_joint_promotion_packet
from notifier import Ticket
from prediction.adapters import (
    AdapterError,
    adapt_candidate_forecast_v3,
    fill_attempt_features_from_candidate,
    fill_features_from_attempt,
)
from prediction.canonical_snapshot import (
    build_canonical_snapshot,
    compute_source_ages_seconds,
    extract_source_timestamps,
)
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.deployment import DeploymentBundle
from prediction.models.candidate_value import (
    QUANTILES_V3,
    CandidateValueConfig,
    CandidateValueModel,
)
from prediction.models.fill_probability import FillProbabilityModel
from prediction.part3_decision import build_v3_decision
from prediction.storage import PredictionStore
from spread_selector import Leg, SpreadCandidate


ET = ZoneInfo("America/New_York")


def _synth_cv(n=40, seed=7):
    rng = np.random.default_rng(seed)
    rows, y_pnl, y_profit, sessions, groups = [], [], [], [], []
    for i in range(n):
        f1 = float(rng.normal())
        rows.append({
            "family": "put_credit",
            "ev": 0.1 + 0.05 * f1,
            "credit": 0.5,
            "max_loss": 2.0,
            "prob_profit": 0.55,
            "capital": 2.0,
            "score": 0.2,
            "f1": f1,
        })
        pnl = 0.15 * f1 + float(rng.normal(0, 0.25))
        y_pnl.append(pnl)
        y_profit.append(1 if pnl > 0 else 0)
        sessions.append(f"S{i % 6:02d}")
        groups.append(f"snap-{i // 2}")
    return rows, y_pnl, y_profit, sessions, groups


SMALL_CV = CandidateValueConfig(
    expanded_distribution=True,
    quantiles=QUANTILES_V3,
    c_grid=(0.5,),
    l1_ratio_grid=(0.5,),
    alpha_grid=(0.01,),
    quantile_max_iter=40,
    max_iter=200,
    inner_folds=2,
    min_train_sessions=2,
    min_validation_sessions=1,
)


def test_adapt_candidate_forecast_v3_uses_real_model_fields():
    rows, y_pnl, y_profit, sessions, groups = _synth_cv()
    model = CandidateValueModel(config=SMALL_CV).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    preds = model.predict_v3(
        rows[:2], candidate_ids=["c0", "c1"])
    adapted = adapt_candidate_forecast_v3(preds[0])
    assert adapted["p_positive_pnl"] == pytest.approx(preds[0].p_profit)
    assert adapted["absolute_utility"] == pytest.approx(preds[0].utility_score)
    assert "q05" in adapted["pnl_quantiles"]
    assert adapted["pnl_quantiles"]["q95"] == pytest.approx(preds[0].pnl_q95)
    with pytest.raises(AdapterError):
        adapt_candidate_forecast_v3(SimpleNamespace(p_positive_pnl=0.9))


def test_fill_features_train_serve_schema_identical():
    attempt = {
        "n_legs": 2,
        "side": "credit",
        "mid_credit_at_submit": 0.50,
        "natural_credit_at_submit": 0.40,
        "limit_credit": 0.40,
        "relative_spread": 0.2,
        "absolute_spread": 0.10,
        "option_price_scale": 0.50,
        "quote_age_seconds": 3.0,
        "minutes_to_close": 90.0,
        "realized_volatility": 0.12,
        "implied_remaining_move": 0.08,
        "data_quality": 0.9,
        "replacement_count": 0,
        "requested_quantity": 1,
        "family": "put_credit",
    }
    train = fill_features_from_attempt(attempt)
    serve = fill_attempt_features_from_candidate(
        candidate={"family": "put_credit"},
        mid_credit=0.50,
        natural_credit=0.40,
        family="put_credit",
        n_legs=2,
        quote_age_seconds=3.0,
        minutes_to_close=90.0,
        realized_volatility=0.12,
        implied_remaining_move=0.08,
        data_quality=0.9,
    )
    assert set(train.keys()) == set(serve.keys())
    for k in train:
        if train[k] is None:
            assert serve[k] is None
        else:
            assert float(train[k]) == pytest.approx(float(serve[k]))


def test_fill_probability_model_accepts_shared_features():
    from execution.fill_records import FillRecord
    rows = []
    for i in range(16):
        filled = i % 2 == 0
        rows.append(FillRecord(
            fill_record_id=f"r{i}",
            snapshot_id="s",
            candidate_id=f"c{i}",
            session_date="2026-07-01",
            decision_ts="2026-07-01T14:00:00Z",
            submitted_ts="2026-07-01T14:00:01Z",
            resolved_ts="2026-07-01T14:00:20Z",
            symbol="SPY",
            family="put_credit",
            side="credit",
            n_legs=2,
            limit_credit=0.40,
            mid_credit_at_submit=0.50,
            natural_credit_at_submit=0.30,
            relative_spread=0.1,
            absolute_spread=0.2,
            option_price_scale=0.5,
            quote_age_seconds=1.0,
            minutes_to_close=100.0,
            requested_quantity=1,
            source="paper",
            mode="shadow",
            filled=filled,
            fill_credit=0.40 if filled else None,
            seconds_to_first_fill=15.0 if filled else None,
            expired_unfilled=not filled,
        ))
    model = FillProbabilityModel().fit(rows)
    feats = fill_features_from_attempt(rows[0])
    out = model.predict(feats, family="put_credit")
    assert 0.0 <= float(out.p_fill_60s) <= 1.0


def test_candidate_mode_forecast_none_unavailable():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="fc-none",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    universe = build_candidate_universe(
        snapshot_id="fc-none",
        generated_at=snap.ts,
        candidates=[{
            "family": "put_credit", "ev": 0.1, "prob_profit": 0.6,
            "legs": [
                {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                 "expiration": "2026-07-14"},
            ],
        }],
    )
    result = build_v3_decision(
        snapshot=snap, forecast=None, universe=universe, mode="candidate")
    assert result.statistical_action == "UNAVAILABLE"
    assert result.final_action == "UNAVAILABLE"
    assert "forecast_unavailable" in result.reasons


def test_stack_forecast_failure_fail_closed_candidate():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="stack-fail",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    dep = DeploymentBundle(
        deployment_id="cand",
        mode="candidate",
        prediction_model_group_id="g",
        candidate_value_model_id="cv",
        candidate_rank_model_id="cr",
        fill_probability_model_id="fp",
        fill_concession_model_id="fc",
        meta_model_id="mm",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="v3",
        fallback_policy="abstain",
        configuration_hash="h",
    )

    class _BoomRuntime:
        def forecast(self, snapshot):
            raise RuntimeError("forecast exploded")

    stack = UnifiedDecisionStack(
        deployment=dep,
        prediction_runtime=_BoomRuntime(),
        candidate_universe_fn=lambda s, forecast=None: build_candidate_universe(
            snapshot_id=snap.snapshot_id, generated_at=snap.ts, candidates=()),
    )
    rec = stack.evaluate(
        snap,
        legacy_decision={"action": "TRADE", "candidate_id": "L",
                         "structure": "put_credit", "direction": "put",
                         "size_mult": 1.0},
    )
    assert rec.v3_statistical_action == "UNAVAILABLE"
    assert rec.v3_final_action == "UNAVAILABLE"
    assert "forecast_unavailable" in rec.reasons
    # Candidate mode keeps legacy as reference authority — V3 must not TRADE.
    assert rec.v3_final_action != "TRADE"
    assert "forecast_error" in rec.diagnostics


def test_persist_decision_graph_atomic(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "pred.sqlite"))
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="graph-1",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    forecast = PredictionBundle(
        snapshot_id="graph-1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    universe = build_candidate_universe(
        snapshot_id="graph-1", generated_at=snap.ts,
        candidates=[{"family": "put_credit", "ev": 0.1}],
    )
    decision = {
        "snapshot_id": "graph-1",
        "deployment_id": "d1",
        "deployment_mode": "shadow",
        "authority_source": "legacy",
        "legacy_action": "NO_EDGE",
        "v3_statistical_action": "ABSTAIN",
        "v3_final_action": "ABSTAIN",
        "final_action": "NO_EDGE",
        "selected_candidate_id": None,
        "hard_vetoes": [],
        "reasons": ["test"],
        "fallback_used": False,
        "configuration_hash": "x",
    }
    evaluations = [{
        "snapshot_id": "graph-1",
        "candidate_id": universe.candidate_ids()[0],
        "final_rank": 1,
        "absolute_utility": 0.1,
        "expected_net_pnl": 0.05,
        "p_positive_pnl": 0.6,
        "fill_probability": 0.7,
        "pnl_quantiles": {"q50": 0.05},
        "vetoes": [],
        "model_versions": {"candidate_value": "test"},
        "ranking_uncertainty": 0.2,
    }]
    persist_unified_decision(
        store, decision, snapshot=snap, universe=universe,
        forecast=forecast, evaluations=evaluations,
    )
    n_ev = store.conn.execute(
        "SELECT COUNT(*) FROM candidate_evaluations WHERE snapshot_id=?",
        ("graph-1",)).fetchone()[0]
    n_rank = store.conn.execute(
        "SELECT COUNT(*) FROM candidate_ranks WHERE snapshot_id=?",
        ("graph-1",)).fetchone()[0]
    n_dec = store.conn.execute(
        "SELECT COUNT(*) FROM unified_decisions WHERE snapshot_id=?",
        ("graph-1",)).fetchone()[0]
    assert n_ev == 1 and n_rank == 1 and n_dec == 1


def test_source_timestamps_not_fabricated_now():
    now = dt.datetime(2026, 7, 14, 10, 0, tzinfo=ET)
    bars = SimpleNamespace(ts=np.array(
        ["2026-07-14T09:58:00"], dtype="datetime64[ns]"))
    src = extract_source_timestamps(
        now_iso=now.isoformat(), bars=bars, chain=None, market=None)
    assert "bars" in src
    assert "2026-07-14T10:00:00" not in src["bars"]
    ages = compute_source_ages_seconds(now.isoformat(), src)
    assert ages["bars"] >= 60.0


def test_ticket_from_unified_decision_not_legacy():
    cand = SpreadCandidate(
        family="call_credit",
        short_strikes=(505.0,),
        long_strikes=(510.0,),
        legs=(
            Leg(strike=505.0, kind="C", qty=-1),
            Leg(strike=510.0, kind="C", qty=1),
        ),
        credit=0.40,
        max_loss=4.60,
        ev=0.12,
        ev_per_risk=0.03,
        theta=0.05,
        gamma=-0.01,
        prob_profit=0.55,
        prob_touch_short=0.3,
        distance_to_wall=2.0,
        liquidity_score=0.8,
        wall_safety=0.7,
        gamma_safety=0.7,
        touch_safety=0.7,
        score=0.5,
        passes_vetoes=True,
        veto_reasons=(),
    )
    cand.candidate_id = "auth-cand"

    legacy_cand = SpreadCandidate(
        family="put_credit",
        short_strikes=(490.0,),
        long_strikes=(485.0,),
        legs=(
            Leg(strike=490.0, kind="P", qty=-1),
            Leg(strike=485.0, kind="P", qty=1),
        ),
        credit=0.35,
        max_loss=4.65,
        ev=0.10,
        ev_per_risk=0.02,
        theta=0.04,
        gamma=-0.01,
        prob_profit=0.50,
        prob_touch_short=0.3,
        distance_to_wall=2.0,
        liquidity_score=0.8,
        wall_safety=0.7,
        gamma_safety=0.7,
        touch_safety=0.7,
        score=0.4,
        passes_vetoes=True,
        veto_reasons=(),
    )

    @dataclass
    class _Dec:
        decision: str = "NO_TRADE"
        gate_pass: bool = False
        candidate: object = None
        session_date: str = "2026-07-14"
        gate_score: float = 0.0

    result = SimpleNamespace(
        ts=dt.datetime(2026, 7, 14, 10, 0, tzinfo=ET),
        decision=_Dec(candidate=legacy_cand),
        intent=SimpleNamespace(
            exec_regime="trend", context_regime="calm",
            direction_bias="call",
            decision=SimpleNamespace(direction="call", structure="NT"),
        ),
        regime=SimpleNamespace(dominant_regime="trend"),
        final_size_mult=0.0,
        authoritative_decision={
            "final_action": "TRADE",
            "selected_candidate_id": "auth-cand",
            "structure": "call_credit",
            "direction": "call",
            "size_mult": 0.75,
        },
        authority_source="v3",
        paper_intents=[{
            "track": "v3",
            "candidate": cand,
            "candidate_id": "auth-cand",
            "size_mult": 0.75,
        }],
    )
    ticket = Ticket.from_unified_decision(result, "SPY")
    assert ticket is not None
    assert ticket.family == "call_credit"
    assert 505.0 in ticket.short_calls
    assert ticket.size_mult == pytest.approx(0.75)
    assert Ticket.from_tick_result(result, "SPY") is None


def test_promotion_rejects_empty_bootstrap_and_incomplete_roles():
    pkt = build_joint_promotion_packet(
        deployment_id="d1",
        current_status="candidate",
        proposed_status="champion",
        legacy_rule_config_id="r",
        model_artifact_ids={"group": "g1"},
        feature_version="v2",
        label_version="v2",
        configuration_hash="h",
        fold_definitions={"outer": ["s"]},
        oos_metrics={"net_pnl": 1.0},
        bootstrap_intervals={},
        known_weaknesses=["x"],
        unsupported_slices=["y"],
        rollback_deployment_id="d0",
    )
    with pytest.raises(ValueError, match="model_artifact"):
        approve_promotion(pkt, reviewer="a", approval_note="ok")
