"""
tests/test_part3_live_wiring.py
===============================
Part 3 live_state wiring — payload from ranking + TickResult serialization.
"""
from __future__ import annotations

from types import SimpleNamespace
import datetime as dt

from dashboard.state import serialize_tick_result
from prediction.models.candidate_value import CandidateForecast
from prediction.part3_shadow import build_part3_live_payload
from prediction.storage import PredictionStore


def test_build_part3_live_payload_from_forecasts(tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    fc_a = CandidateForecast(
        candidate_id="a", expected_net_pnl=0.4, p_profit=0.6,
        pnl_q10=-0.5, pnl_q50=0.2, pnl_q90=0.8, expected_shortfall=0.5,
        fill_uncertainty=0.2, model_uncertainty=0.2, utility_score=0.25,
    )
    fc_b = CandidateForecast(
        candidate_id="b", expected_net_pnl=-0.1, p_profit=0.4,
        pnl_q10=-0.8, pnl_q50=-0.1, pnl_q90=0.3, expected_shortfall=0.7,
        fill_uncertainty=0.3, model_uncertainty=0.3, utility_score=-0.05,
    )
    cands = [
        SimpleNamespace(
            family="put_credit", legs=[1, 2], credit=0.45, max_loss=1.0,
            execution={"mid_credit": 0.45, "natural_credit": 0.28},
            _v2_candidate_id="a", score=1.0,
        ),
        SimpleNamespace(
            family="call_credit", legs=[1, 2], credit=0.40, max_loss=1.0,
            execution={"mid_credit": 0.40, "natural_credit": 0.25},
            _v2_candidate_id="b", score=0.5,
        ),
    ]
    payload = build_part3_live_payload(
        snapshot_id="2026-07-14|t0",
        ts="2026-07-14T15:00:00Z",
        symbol="SPY",
        candidates=cands,
        forecasts={"a": fc_a, "b": fc_b},
        signals={
            "v2_top_candidate_id": "a",
            "legacy_top_candidate_id": "b",
            "v2_top_family": "put_credit",
            "v2_policy_confidence": 0.7,
            "v2_policy_uncertainty": 0.2,
            "data_quality": 0.9,
        },
        mode="shadow",
        store=store,
    )
    assert payload["mode"] == "shadow"
    assert "SHADOW" in payload["shadow_label"]
    assert payload["decision_summary"]["action"] in (
        "TRADE", "NO_EDGE", "ABSTAIN", "HARD_VETO")
    assert payload["ranking"]["top_candidate_id"] in ("a", "b")
    assert payload["execution"]
    assert store.fetch_meta_decisions("2026-07-14|t0")
    store.close()


def test_serialize_tick_result_includes_part3():
    from decision_matrix import Decision, TradeIntent
    from gate_scorer import MarketSnapshot
    from regime_classifier import RegimeState
    from unified_loop import TickResult, TickSnapshot
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
    part3 = {
        "mode": "shadow",
        "shadow_label": "SHADOW — not an executed order",
        "generated_at": "2026-07-14T15:00:00Z",
        "model_versions": {"part3": "v3.0.0"},
        "decision_summary": {
            "action": "NO_EDGE", "statistical_action": "NO_EDGE",
        },
    }
    regime = RegimeState(
        confidences={"compression": 72}, reliabilities={"compression": 0.8},
        dominant_regime="compression", permitted_engine="premium_selling",
        vetoes=[], global_information_gain=12.0, standardized={},
        stand_down=False,
    )
    intent = TradeIntent(
        exec_regime="compression", context_regime="compression",
        direction_bias="neutral", bias_value=0.0,
        decision=Decision("NT", "none", "NONE", "", "", ""),
        size_mult=0.0, vetoes=[], note="",
    )
    market = MarketSnapshot(
        spot=600.0, net_gex=1e9, gamma_flip=595.0, call_wall=605.0,
        put_wall=595.0, gex_pct_rank=0.5, vix9d=12.0, vix=13.0, vix3m=15.0,
        vvix=92.0, vvix_baseline=95.0, straddle_breakeven=4.0,
        expected_range=3.0, adx=14.0, rsi=50.0, bb_width=1.4,
        bb_width_baseline=2.0, vwap=600.0, vwap_reversion_count=0,
        tick_abs_mean=400.0, cvd_slope=0.0,
        now=dt.datetime(2026, 7, 14, 15, 0, tzinfo=ET), has_catalyst=False,
    )
    result = TickResult(
        ts=dt.datetime(2026, 7, 14, 15, 0, tzinfo=ET),
        regime=regime, intent=intent, decision=None, final_size_mult=0.0,
        vetoes=[], snapshot=TickSnapshot(market=market, bars=None, chain=None),
        part3=part3,
    )
    payload = serialize_tick_result(result)
    assert payload["part3"]["decision_summary"]["action"] == "NO_EDGE"
    assert payload["part3"]["mode"] == "shadow"
