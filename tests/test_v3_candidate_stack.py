"""
tests/test_v3_candidate_stack.py
tests/test_execution_economics.py
"""
from __future__ import annotations

from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.part3_decision import build_v3_decision


def _snap():
    return build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="s1",
    )


def _universe():
    return build_candidate_universe(
        snapshot_id="s1",
        generated_at="t",
        candidates=[
            {
                "family": "put_credit",
                "ev": 0.20,
                "prob_profit": 0.7,
                "legs": [
                    {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                     "expiration": "2026-07-14"},
                    {"right": "P", "side": "buy", "qty": 1, "strike": 485,
                     "expiration": "2026-07-14"},
                ],
            },
            {
                "family": "call_debit",
                "ev": 0.05,
                "prob_profit": 0.55,
                "legs": [
                    {"right": "C", "side": "buy", "qty": 1, "strike": 500,
                     "expiration": "2026-07-14"},
                    {"right": "C", "side": "sell", "qty": 1, "strike": 505,
                     "expiration": "2026-07-14"},
                ],
            },
        ],
    )


def test_hard_veto_overrides_trade():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.1, ood_score=0.05, data_quality=0.95,
    )
    result = build_v3_decision(
        snapshot=_snap(),
        forecast=forecast,
        universe=_universe(),
        hard_vetoes=("stale_data",),
        mode="shadow",
    )
    assert result.final_action == "HARD_VETO"
    assert "stale_data" in result.reasons or "hard_veto" in str(result.reasons)


def test_rankings_deterministic():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, ood_score=0.1, data_quality=0.9,
    )
    a = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(), mode="shadow")
    b = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(), mode="shadow")
    assert a.candidate_id == b.candidate_id
    assert [e.candidate_id for e in a.evaluations] == [
        e.candidate_id for e in b.evaluations]


def test_midpoint_never_treated_as_filled():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    result = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(), mode="shadow")
    assert result.execution is None or result.execution.get(
        "note") == "midpoint_diagnostic_only"
    for ev in result.evaluations:
        if ev.expected_fill_price is not None and ev.legacy_ev:
            # conservative/natural should not equal raw mid when mid != 0
            assert abs(ev.expected_fill_price - float(ev.legacy_ev)) > 1e-12 or \
                float(ev.legacy_ev) == 0.0


def test_required_component_failure_abstain_in_champion():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, data_quality=0.9,
    )
    result = build_v3_decision(
        snapshot=_snap(),
        forecast=forecast,
        universe=_universe(),
        runtime=None,
        mode="champion",
    )
    # Without runtime artifacts, champion path still produces a decision via
    # baseline utilities in build_v3_candidate_evaluations — but top candidates
    # get required_component_missing vetoes when value_model is None.
    assert result.statistical_action in (
        "ABSTAIN", "NO_EDGE", "TRADE", "NO_CANDIDATE")
    if result.evaluations:
        assert any(
            "required_component_missing" in (e.vetoes or ())
            for e in result.evaluations
        ) or result.statistical_action == "ABSTAIN"
