"""
tests/test_regime_alignment.py
================================
Unit tests for position-relative Regime Alignment Score (RAS).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from decision_matrix import Decision, TradeIntent
from gate_scorer import MarketSnapshot
from regime_classifier import RegimeState
from regime_alignment import (
    EntrySnapshot,
    PositionContext,
    RASConfig,
    build_entry_snapshot,
    compute_ras,
    derive_position_bias,
    entry_snapshot_from_dict,
    entry_snapshot_to_dict,
    position_context_from_entry_ctx,
    structure_class_from_family,
)

ET = ZoneInfo("America/New_York")


def _market(**kw) -> MarketSnapshot:
    spot = kw.get("spot", 600.0)
    flip = kw.get("gamma_flip", spot - 5.0)
    return MarketSnapshot(
        spot=spot,
        net_gex=kw.get("net_gex", 4e9),
        gamma_flip=flip,
        call_wall=kw.get("call_wall", spot + 5),
        put_wall=kw.get("put_wall", spot - 5),
        gex_pct_rank=kw.get("gex_pct_rank", 0.85),
        vix9d=12.0, vix=13.0, vix3m=15.0,
        vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=kw.get("adx", 12.0), rsi=51.0,
        bb_width=1.4, bb_width_baseline=2.0,
        vwap=spot, vwap_reversion_count=5,
        tick_abs_mean=450.0, cvd_slope=0.05,
        now=dt.datetime(2026, 6, 25, 11, 30, tzinfo=ET),
        has_catalyst=False,
    )


def _regime(**kw) -> RegimeState:
    confidences = kw.get("confidences", {
        "trend": 70.0, "directional_confidence": 68.0,
        "compression": 30.0, "expansion": 20.0,
    })
    return RegimeState(
        confidences=confidences,
        reliabilities={k: 0.8 for k in confidences},
        dominant_regime=kw.get("dominant_regime", "trend"),
        permitted_engine=kw.get("permitted_engine", "directional"),
        vetoes=list(kw.get("vetoes", [])),
        global_information_gain=kw.get("ig", 20.0),
        standardized=kw.get("standardized", {
            "flip_cushion": (60.0, 1.0),
            "flip_proximity": (30.0, 1.0),
            "gamma_sign": (65.0, 1.0),
        }),
        stand_down=False,
    )


def _intent(**kw) -> TradeIntent:
    structure = kw.get("structure", "LCS")
    direction = kw.get("direction", "call")
    exec_r = kw.get("exec_regime", "trend")
    ctx_r = kw.get("context_regime", "trend")
    bias = kw.get("direction_bias", "bull")
    return TradeIntent(
        exec_regime=exec_r,
        context_regime=ctx_r,
        direction_bias=bias,
        bias_value=kw.get("bias_value", 65.0),
        decision=Decision(structure, direction, "HIGH", "test", "rule", "15m"),
        size_mult=1.0,
        vetoes=list(kw.get("vetoes", [])),
        note="",
    )


def _entry(**kw) -> EntrySnapshot:
    return EntrySnapshot(
        dominant_regime=kw.get("dominant_regime", "trend"),
        permitted_engine=kw.get("permitted_engine", "directional"),
        exec_regime=kw.get("exec_regime", "trend"),
        context_regime=kw.get("context_regime", "trend"),
        direction_bias=kw.get("direction_bias", "bull"),
        bias_value=kw.get("bias_value", 65.0),
        vetoes=list(kw.get("vetoes", [])),
        net_gex=kw.get("net_gex", 4e9),
        gamma_flip=kw.get("gamma_flip", 595.0),
        flip_cushion=kw.get("flip_cushion", 0.008),
        spot=kw.get("spot", 600.0),
        structure=kw.get("structure", "LPS"),
        structure_class=kw.get("structure_class", "directional"),
        dominant_confidence=kw.get("dominant_confidence", 68.0),
    )


def _ctx(**kw) -> PositionContext:
    return PositionContext(
        position_id=kw.get("position_id", "pos1"),
        direction=kw.get("direction", "put"),
        position_bias=kw.get("position_bias", "bear"),
        entry=kw.get("entry", _entry()),
    )


def test_derive_position_bias():
    assert derive_position_bias("call", "LCS", "directional") == "bull"
    assert derive_position_bias("put", "LPS", "directional") == "bear"
    assert derive_position_bias("both", "STG", "directional") == "vol"
    assert derive_position_bias("put", "PCS", "premium") == "neutral"


def test_structure_class_from_family():
    assert structure_class_from_family("long_put_spread") == "directional"
    assert structure_class_from_family("put_credit") == "premium"


def test_bull_lcs_aligned_positive_ras():
    regime = _regime()
    intent = _intent(structure="LCS", direction="call")
    market = _market()
    entry = build_entry_snapshot(regime, intent, market, "directional", "LCS")
    ctx = PositionContext("p1", "call", "bull", entry)
    ras = compute_ras(regime, intent, market, ctx)
    assert ras.score > 0
    assert ras.action == "ok"


def test_bear_put_spread_hostile_flip_negative_ras():
    entry = _entry(structure="LPS", structure_class="directional",
                   direction_bias="bear", bias_value=40.0,
                   flip_cushion=-0.01, spot=598.0, gamma_flip=600.0)
    regime = _regime(
        vetoes=["below_gamma_flip"],
        standardized={
            "flip_cushion": (35.0, 1.0),
            "flip_proximity": (85.0, 1.0),
            "gamma_sign": (30.0, 1.0),
        },
    )
    intent = _intent(structure="LPS", direction="put",
                     exec_regime="compression", context_regime="trend",
                     direction_bias="bull", bias_value=62.0)
    market = _market(spot=602.0, gamma_flip=600.0, net_gex=-1e9)
    ctx = PositionContext("p2", "put", "bear", entry)
    ras = compute_ras(regime, intent, market, ctx)
    assert ras.score < -20
    comp_names = {c.name for c in ras.components}
    assert "direction_alignment" in comp_names
    assert "gamma_alignment" in comp_names


def test_premium_pcs_short_gamma_veto_escalation():
    entry = _entry(structure="PCS", structure_class="premium",
                   permitted_engine="premium_selling", vetoes=[])
    regime = _regime(
        dominant_regime="compression",
        permitted_engine="premium_selling",
        vetoes=["short_gamma_regime"],
        confidences={"compression": 55.0, "trend": 40.0},
    )
    intent = _intent(structure="PCS", direction="put",
                     exec_regime="compression", context_regime="compression",
                     direction_bias="neutral", bias_value=50.0)
    market = _market(net_gex=-2e9, spot=599.0, gamma_flip=600.0)
    ctx = PositionContext("p3", "put", "neutral", entry)
    ras = compute_ras(regime, intent, market, ctx)
    veto_comp = next(c for c in ras.components if c.name == "veto_escalation")
    assert veto_comp.raw < 0
    gamma_comp = next(c for c in ras.components if c.name == "gamma_alignment")
    assert gamma_comp.raw <= 0


def test_missing_data_graceful():
    entry = _entry()
    ctx = PositionContext("p4", "put", "bear", entry)
    ras = compute_ras(_regime(), None, None, ctx)
    assert ras.action == "ok"
    assert -5 <= ras.score <= 5


def test_entry_snapshot_round_trip():
    regime = _regime()
    intent = _intent()
    market = _market()
    snap = build_entry_snapshot(regime, intent, market, "directional", "LCS")
    d = entry_snapshot_to_dict(snap)
    restored = entry_snapshot_from_dict(d)
    assert restored.exec_regime == snap.exec_regime
    assert restored.flip_cushion == pytest.approx(snap.flip_cushion)


def test_position_context_from_entry_ctx():
    entry = entry_snapshot_to_dict(_entry())
    ctx = position_context_from_entry_ctx("abc", {
        "direction": "put",
        "position_bias": "bear",
        "entry_snapshot": entry,
        "ras_ema_score": -10.0,
    })
    assert ctx is not None
    assert ctx.position_id == "abc"
    assert ctx.position_bias == "bear"
    assert ctx.prev_ema_score == -10.0


def test_exit_action_suppressed_when_disabled():
    cfg = RASConfig(exit_enabled=False, exit_threshold=-10.0)
    entry = _entry(structure="LPS", direction_bias="bear")
    regime = _regime(vetoes=["catalyst:FOMC"])
    intent = _intent(structure="LPS", direction="put",
                     direction_bias="bull", bias_value=70.0)
    market = _market(net_gex=-5e9, spot=605.0, gamma_flip=600.0)
    ctx = PositionContext("p5", "put", "bear", entry)
    ras = compute_ras(regime, intent, market, ctx, cfg=cfg)
    assert ras.score < cfg.exit_threshold
    assert ras.action == "warning"
