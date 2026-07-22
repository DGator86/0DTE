"""SPY-DER parallel track + live_state panel wiring."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from dashboard.state import serialize_tick_result
from decision_matrix import Decision, TradeIntent
from gate_scorer import MarketSnapshot
from paper_broker import PAPER_TRACKS, PAPER_TRACK_LABELS
from regime_classifier import RegimeState
from unified_loop import TickResult, TickSnapshot

ET = ZoneInfo("America/New_York")


def test_paper_tracks_include_spy_der():
    assert "spy_der" in PAPER_TRACKS
    assert PAPER_TRACK_LABELS["spy_der"] == "SPY-DER"


def test_live_state_parallel_tracks_include_spy_der():
    market = MarketSnapshot(
        spot=600.0, net_gex=4e9, gamma_flip=595.0,
        call_wall=605.0, put_wall=595.0, gex_pct_rank=0.85,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=14.0, rsi=50.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=600.0, vwap_reversion_count=2,
        tick_abs_mean=400.0, cvd_slope=0.01,
        now=dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET),
        has_catalyst=False,
    )
    regime = RegimeState(
        confidences={"compression": 72},
        reliabilities={"compression": 0.8},
        dominant_regime="compression",
        permitted_engine="premium_selling",
        vetoes=[],
        global_information_gain=12.0,
        standardized={},
        stand_down=False,
    )
    intent = TradeIntent(
        exec_regime="compression",
        context_regime="compression",
        direction_bias="neutral",
        bias_value=0.0,
        decision=Decision("IC", "both", "HIGH", "theta", "note", "15m"),
        size_mult=1.0,
        vetoes=[],
        note="",
    )
    result = TickResult(
        ts=dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET),
        regime=regime,
        intent=intent,
        decision=None,
        final_size_mult=0.0,
        vetoes=[],
        snapshot=TickSnapshot(market=market, bars=None, chain=None),
        signals={},
        spy_der={
            "track": "spy_der",
            "label": "SPY-DER",
            "action": "TRADE",
            "structure": "put_credit",
            "direction": "bearish",
            "confidence": 0.7,
            "available": True,
            "source": "deterministic",
            "mode": "shadow",
        },
    )
    payload = serialize_tick_result(result, feed_source="Tradier")
    tracks = payload["forecast"]["parallel_tracks"]
    assert set(tracks) >= {"legacy", "v2", "v3", "spy_der"}
    assert tracks["spy_der"]["action"] == "TRADE"
    assert tracks["spy_der"]["label"] == "SPY-DER"
