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
    RASComponent,
    RASConfig,
    RASResult,
    build_entry_snapshot,
    compute_ras,
    compute_regime_alignment,
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
        bias_fast=kw.get("bias_fast"),
        bias_slow=kw.get("bias_slow"),
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


def test_exit_action_active_with_default_config():
    """Full activation: with library defaults (exit_enabled=True) a deeply
    negative score must surface as an actual exit action, not a warning."""
    assert RASConfig().exit_enabled is True
    cfg = RASConfig(exit_threshold=-10.0)
    entry = _entry(structure="LPS", direction_bias="bear")
    regime = _regime(vetoes=["catalyst:FOMC"])
    intent = _intent(structure="LPS", direction="put",
                     direction_bias="bull", bias_value=70.0)
    market = _market(net_gex=-5e9, spot=605.0, gamma_flip=600.0)
    ctx = PositionContext("p5b", "put", "bear", entry)
    ras = compute_ras(regime, intent, market, ctx, cfg=cfg)
    assert ras.score < cfg.exit_threshold
    assert ras.action == "exit"


def test_compute_regime_alignment_alias_matches_compute_ras():
    """The handoff-specified public name must produce the same result."""
    regime = _regime()
    intent = _intent(structure="LCS", direction="call")
    market = _market()
    entry = build_entry_snapshot(regime, intent, market, "directional", "LCS")
    ctx1 = PositionContext("p6", "call", "bull", entry)
    ctx2 = PositionContext("p6", "call", "bull", entry)
    a = compute_regime_alignment(regime, intent, market, ctx1)
    b = compute_ras(regime, intent, market, ctx2)
    assert a.score == b.score
    assert a.action == b.action
    assert [c.name for c in a.components] == [c.name for c in b.components]


def test_score_deteriorates_as_regime_turns_hostile():
    """Bull debit spread; the tape flips bear + short gamma across successive
    evaluations. The EMA-fed score must fall monotonically and every
    component must carry a non-empty note."""
    regime0 = _regime()
    intent0 = _intent(structure="LCS", direction="call",
                      direction_bias="bull", bias_value=68.0)
    market0 = _market(spot=600.0, gamma_flip=594.0, net_gex=4e9)
    entry = build_entry_snapshot(regime0, intent0, market0, "directional", "LCS")
    ctx = PositionContext("p7", "call", "bull", entry)

    hostile_std = {"flip_proximity": (85.0, 1.0), "gamma_sign": (25.0, 1.0)}
    stages = [
        (regime0, intent0, market0),
        (_regime(confidences={"trend": 55.0, "directional_confidence": 52.0,
                              "compression": 30.0, "expansion": 25.0}),
         _intent(structure="LCS", direction="call",
                 exec_regime="trend", context_regime="compression",
                 direction_bias="neutral", bias_value=50.0),
         _market(spot=596.0, gamma_flip=595.0, net_gex=1e9)),
        (_regime(vetoes=["below_gamma_flip"], standardized=hostile_std,
                 confidences={"trend": 42.0, "directional_confidence": 40.0,
                              "compression": 30.0, "expansion": 40.0}),
         _intent(structure="LCS", direction="call",
                 exec_regime="compression", context_regime="trend",
                 direction_bias="bear", bias_value=35.0),
         _market(spot=593.0, gamma_flip=596.0, net_gex=-2e9)),
        (_regime(dominant_regime="breakout", permitted_engine="none",
                 vetoes=["below_gamma_flip", "catalyst:CPI"],
                 standardized=hostile_std,
                 confidences={"trend": 28.0, "directional_confidence": 25.0,
                              "compression": 22.0, "expansion": 60.0}),
         _intent(structure="LCS", direction="call",
                 exec_regime="breakout", context_regime="breakout",
                 direction_bias="bear", bias_value=25.0),
         _market(spot=590.0, gamma_flip=597.0, net_gex=-4e9)),
    ]

    scores = []
    for regime, intent, market in stages:
        ras = compute_regime_alignment(regime, intent, market, ctx)
        scores.append(ras.score)
        ctx.prev_ema_score = ras.ema_score
        for c in ras.components:
            assert c.note, f"component {c.name} has an empty note"
    assert scores[0] > 0
    assert all(b < a for a, b in zip(scores, scores[1:])), scores
    assert scores[-1] < -30


# --------------------------------------------------------------------------- #
# fast_momentum component (turn-detection channel)                             #
# --------------------------------------------------------------------------- #
def _comp(ras: RASResult, name: str):
    return next((c for c in ras.components if c.name == name), None)


def test_fast_momentum_skipped_when_intent_predates_bias_fast():
    """Intents without bias_fast (legacy journal rows, old tests) must produce
    the exact pre-existing component set — no dilution of the score."""
    ctx = PositionContext("f0", "call", "bull", _entry())
    ras = compute_ras(_regime(), _intent(), _market(), ctx)
    assert _comp(ras, "fast_momentum") is None
    # the full baseline set (incl. channel_break), just never fast_momentum
    assert len(ras.components) == 7
    assert {c.name for c in ras.components} == {
        "direction_alignment", "matrix_alignment", "gamma_alignment",
        "veto_escalation", "confidence_erosion", "regime_flip",
        "channel_break"}


def test_fast_momentum_warns_before_blend_flips():
    """The V-turn scenario from live trading: blend still bullish (60% slow),
    fast composite already bearish. The component must go hostile NOW."""
    ctx = PositionContext("f1", "call", "bull", _entry())
    intent = _intent(direction_bias="bull", bias_value=61.0, bias_fast=28.0)
    ras = compute_ras(_regime(), intent, _market(), ctx)
    comp = _comp(ras, "fast_momentum")
    assert comp is not None
    assert comp.raw < -0.8                     # (28-50)/25 = -0.88
    assert "fast composite 28" in comp.note

    # sanity: direction_alignment is still fully aligned on the same tick —
    # fast_momentum is the ONLY component that can see the turn this early
    assert _comp(ras, "direction_alignment").raw == 1.0


def test_fast_momentum_asymmetric_upside_cap():
    """Quick to cut, slow to add: a screaming-hot fast read must not be able
    to mask deterioration elsewhere (capped at +0.5), while a hostile read
    keeps the full -1.0 range."""
    ctx = PositionContext("f2", "call", "bull", _entry())
    hot = compute_ras(_regime(), _intent(bias_fast=95.0), _market(), ctx)
    assert _comp(hot, "fast_momentum").raw == 0.5
    cold = compute_ras(_regime(), _intent(bias_fast=5.0), _market(), ctx)
    assert _comp(cold, "fast_momentum").raw == -1.0


def test_fast_momentum_premium_threatened_by_either_direction():
    """Neutral premium position: strong fast momentum in ANY direction
    threatens the range; small drift inside the deadband is ignored."""
    entry = _entry(structure="IC", structure_class="premium",
                   direction_bias="neutral", bias_value=50.0)
    ctx = PositionContext("f3", "both", "neutral", entry)
    calm = compute_ras(_regime(), _intent(bias_fast=55.0), _market(), ctx)
    assert _comp(calm, "fast_momentum").raw == 0.0
    hot = compute_ras(_regime(), _intent(bias_fast=85.0), _market(), ctx)
    assert _comp(hot, "fast_momentum").raw < -0.5


def test_fast_momentum_bear_position_mirrors():
    ctx = PositionContext("f4", "put", "bear", _entry(direction_bias="bear"))
    ras = compute_ras(_regime(), _intent(direction_bias="bear",
                                         bias_fast=75.0), _market(), ctx)
    assert _comp(ras, "fast_momentum").raw == -1.0


# --------------------------------------------------------------------------- #
# channel_break component (Bollinger / Keltner / Donchian deterioration)       #
# --------------------------------------------------------------------------- #
def _chan_std(**kw) -> dict:
    """Standardized dict with channel features (values are 0..100 cells)."""
    base = {
        "flip_cushion": (60.0, 1.0),
        "flip_proximity": (30.0, 1.0),
        "gamma_sign": (65.0, 1.0),
        "donchian_breakout_up": (kw.get("up", 50.0), 1.0),
        "donchian_breakout_down": (kw.get("dn", 50.0), 1.0),
        "keltner_position": (kw.get("kpos", 50.0), 1.0),
        "bb_expansion": (kw.get("exp", 50.0), 1.0),
    }
    return base


def _channel_component(regime, bias, structure="LCS",
                       structure_class="directional"):
    entry = _entry(structure=structure, structure_class=structure_class)
    ctx = PositionContext("pc", "call", bias, entry)
    intent = _intent(structure=structure, direction_bias=bias if bias in ("bull", "bear") else "neutral",
                     bias_value=50.0 if bias not in ("bull", "bear") else (65.0 if bias == "bull" else 35.0))
    ras = compute_ras(regime, intent, _market(), ctx)
    return next(c for c in ras.components if c.name == "channel_break")


def test_channel_break_unavailable_is_neutral():
    comp = _channel_component(_regime(), "bull")
    assert comp.raw == 0.0
    assert "unavailable" in comp.note


def test_channel_break_strong_hostile_breakout():
    # bull position, strong downside Donchian break -> full -1.0
    regime = _regime(standardized=_chan_std(dn=90.0))
    comp = _channel_component(regime, "bull")
    assert comp.raw == -1.0
    assert "down" in comp.note


def test_channel_break_building_breakout_warns():
    regime = _regime(standardized=_chan_std(dn=70.0))
    comp = _channel_component(regime, "bull")
    assert comp.raw == -0.5


def test_channel_break_supportive_breakout_positive():
    regime = _regime(standardized=_chan_std(up=75.0))
    comp = _channel_component(regime, "bull")
    assert comp.raw == pytest.approx(0.3)


def test_channel_break_neutral_premium_hit_either_side():
    regime_up = _regime(standardized=_chan_std(up=90.0))
    comp = _channel_component(regime_up, "neutral", structure="IC",
                              structure_class="premium")
    assert comp.raw == -1.0
    regime_dn = _regime(standardized=_chan_std(dn=90.0))
    comp = _channel_component(regime_dn, "neutral", structure="IC",
                              structure_class="premium")
    assert comp.raw == -1.0


def test_channel_break_keltner_pin_with_expansion():
    # bull position pinned at lower Keltner band while Bollinger expands
    regime = _regime(standardized=_chan_std(kpos=10.0, exp=70.0))
    comp = _channel_component(regime, "bull")
    assert comp.raw == -0.5
    assert "keltner" in comp.note.lower()
    # pin without expansion stays quiet
    regime2 = _regime(standardized=_chan_std(kpos=10.0, exp=50.0))
    comp2 = _channel_component(regime2, "bull")
    assert comp2.raw == 0.0


def test_channel_break_stacks_break_and_pin():
    regime = _regime(standardized=_chan_std(dn=90.0, kpos=5.0, exp=80.0))
    comp = _channel_component(regime, "bull")
    assert comp.raw == -1.0                      # clipped at the unit bound


def test_channel_break_vol_position_likes_breaks():
    regime = _regime(standardized=_chan_std(up=80.0))
    comp = _channel_component(regime, "vol", structure="STG")
    assert comp.raw == pytest.approx(0.5)


def test_channel_break_drives_exit_action():
    """A hostile break plus the other deteriorating components must reach the
    tighten/exit band through the standard scoring path."""
    cfg = RASConfig(exit_threshold=-40.0)
    entry = _entry(structure="LCS", structure_class="directional",
                   direction_bias="bull")
    regime = _regime(vetoes=["catalyst:FOMC"],
                     standardized=_chan_std(dn=95.0, kpos=5.0, exp=80.0))
    intent = _intent(structure="LCS", direction="call",
                     exec_regime="compression", context_regime="compression",
                     direction_bias="bear", bias_value=30.0)
    market = _market(net_gex=-3e9, spot=594.0, gamma_flip=600.0)
    ctx = PositionContext("p8", "call", "bull", entry)
    ras = compute_ras(regime, intent, market, ctx, cfg=cfg)
    chan = next(c for c in ras.components if c.name == "channel_break")
    assert chan.raw == -1.0
    assert ras.action == "exit"


# --------------------------------------------------------------------------- #
# Journal RAS logging                                                          #
# --------------------------------------------------------------------------- #
def _ras_result(position_id="pos1", score=-42.0, action="warning") -> RASResult:
    return RASResult(
        score=score,
        components=[
            RASComponent(name="direction_alignment", raw=-1.0, weight=1.5,
                         contribution=-1.5, note="bias flipped bear"),
            RASComponent(name="gamma_alignment", raw=-0.8, weight=1.5,
                         contribution=-1.2, note="below flip, short gamma"),
        ],
        action=action, position_id=position_id, ema_score=score,
    )


def test_journal_log_ras_round_trip(tmp_path):
    from journal import Journal
    jrn = Journal(str(tmp_path / "j.sqlite"))
    row_id = jrn.log_ras("2026-07-08T10:31:00-04:00", "2026-07-08",
                         _ras_result())
    assert row_id > 0

    rows = jrn.fetch_ras(position_id="pos1")
    assert len(rows) == 1
    r = rows[0]
    assert r["position_id"] == "pos1"
    assert r["score"] == -42.0
    assert r["ema_score"] == -42.0
    assert r["action"] == "warning"
    assert len(r["components"]) == 2
    c = r["components"][0]
    assert c["name"] == "direction_alignment"
    assert c["raw"] == -1.0
    assert c["weight"] == 1.5
    assert c["contribution"] == -1.5
    assert c["note"] == "bias flipped bear"
    jrn.close()


def test_journal_fetch_ras_filters(tmp_path):
    from journal import Journal
    jrn = Journal(str(tmp_path / "j.sqlite"))
    jrn.log_ras("2026-07-08T10:31:00-04:00", "2026-07-08", _ras_result("a"))
    jrn.log_ras("2026-07-08T10:32:00-04:00", "2026-07-08", _ras_result("b"))
    jrn.log_ras("2026-07-09T10:31:00-04:00", "2026-07-09", _ras_result("a"))
    assert len(jrn.fetch_ras()) == 3
    assert len(jrn.fetch_ras(position_id="a")) == 2
    assert len(jrn.fetch_ras(session_date="2026-07-08")) == 2
    assert len(jrn.fetch_ras(position_id="a", session_date="2026-07-09")) == 1
    jrn.close()


def test_ras_table_migration_on_legacy_db(tmp_path):
    """A journal DB created before ras_evaluations existed must gain the
    table on reopen, not crash."""
    import sqlite3
    from journal import Journal
    path = str(tmp_path / "legacy.sqlite")
    Journal(path).close()                       # full pre-existing schema...
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE ras_evaluations")  # ...minus the new table
    conn.commit()
    conn.close()
    jrn = Journal(path)
    jrn.log_ras("2026-07-08T10:31:00-04:00", "2026-07-08", _ras_result())
    assert len(jrn.fetch_ras()) == 1
    jrn.close()


# --------------------------------------------------------------------------- #
# Orchestrator integration: open position -> ras_evaluations rows              #
# --------------------------------------------------------------------------- #
def test_orchestrator_tick_journals_ras_for_open_position():
    from journal import Journal
    from unified_loop import UnifiedOrchestrator, SyntheticUnifiedFeed

    feed = SyntheticUnifiedFeed(days=5)
    jrn = Journal(":memory:")
    orch = UnifiedOrchestrator(feed=feed, journal=jrn)
    ctx = PositionContext("live-pos", "call", "bull", _entry())

    start = dt.datetime(2026, 6, 27, 9, 30, tzinfo=ET)
    n_ras_ticks = 0
    for i in range(10):
        result = orch.tick(start + dt.timedelta(minutes=i),
                           position_contexts=[ctx])
        if result is not None and result.ras_results:
            n_ras_ticks += 1
            ctx.prev_ema_score = result.ras_results[0].ema_score
    assert n_ras_ticks > 0

    rows = jrn.fetch_ras(position_id="live-pos")
    assert len(rows) == n_ras_ticks
    for r in rows:
        assert r["session_date"] == "2026-06-27"
        assert r["action"] in ("ok", "warning", "tighten", "exit")
        assert isinstance(r["score"], float)
        names = {c["name"] for c in r["components"]}
        assert "direction_alignment" in names
        assert "gamma_alignment" in names
        assert all(c["note"] for c in r["components"])


def test_signals_flatten_uses_worst_position():
    """With several open positions, signals_json must carry the minimum
    (worst) score instead of arbitrary key overwrites."""
    import json as _json
    from unified_loop import UnifiedOrchestrator

    healthy = _ras_result("h", score=20.0, action="ok")
    sick = _ras_result("s", score=-60.0, action="tighten")
    merged, sj = UnifiedOrchestrator._signals_with_ras({}, [healthy, sick])
    assert merged["ras_score"] == -60.0
    decoded = _json.loads(sj)
    assert decoded["ras_score"] == -60.0
    assert decoded["ras_action"] == 2   # tighten
