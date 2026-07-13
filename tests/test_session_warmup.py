"""
Session entry warmup — no new trades in the first 30 minutes.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from gate_scorer import (
    Decision, GateConfig, MarketSnapshot, evaluate,
    evaluate_directional_gates, evaluate_gates,
)

ET = ZoneInfo("America/New_York")


def _snap(hour=11, minute=20, **kw) -> MarketSnapshot:
    base = dict(
        spot=602.50, net_gex=4.2e9, gamma_flip=596.0,
        call_wall=603.0, put_wall=598.0, gex_pct_rank=0.88,
        gex_rank_warm=True,
        vix9d=12.1, vix=13.0, vix3m=15.2, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.10, expected_range=3.20,
        adx=12.5, rsi=52.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=601.9, vwap_reversion_count=5,
        tick_abs_mean=480.0, cvd_slope=0.05,
        now=dt.datetime(2026, 7, 8, hour, minute, tzinfo=ET),
        has_catalyst=False,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def test_default_warmup_is_30_minutes_after_open():
    cfg = GateConfig()
    assert cfg.morning_entry_time == dt.time(10, 0)
    assert cfg.morning_resolve_time == dt.time(10, 30)
    assert cfg.late_lockout_time == dt.time(15, 30)


def test_premium_gate_blocks_during_warmup():
    r = evaluate(_snap(hour=9, minute=45), GateConfig())
    assert r.decision is Decision.NO_GO
    assert any(g.startswith("WARMUP") for g in r.failed_gates)


def test_premium_gate_allows_after_warmup():
    # 10:00 is the first legal entry minute; soft timing haircut still applies
    # until morning_resolve_time (10:30), but the hard gate must clear.
    r = evaluate(_snap(hour=10, minute=0), GateConfig())
    assert not any(g.startswith("WARMUP") for g in r.failed_gates)
    assert r.decision is Decision.GO


def test_directional_gate_also_blocks_warmup():
    failed = evaluate_directional_gates(_snap(hour=9, minute=55), GateConfig())
    assert any(g.startswith("WARMUP") for g in failed)
    failed_ok = evaluate_directional_gates(_snap(hour=10, minute=5), GateConfig())
    assert not any(g.startswith("WARMUP") for g in failed_ok)


def test_warmup_gate_is_calibratable():
    cfg = GateConfig(morning_entry_time=dt.time(9, 45))
    assert not any(g.startswith("WARMUP")
                   for g in evaluate_gates(_snap(hour=9, minute=50), cfg))
    assert any(g.startswith("WARMUP")
               for g in evaluate_gates(_snap(hour=9, minute=40), cfg))


def test_unified_loop_journals_session_warmup_flag():
    from unified_loop import SyntheticUnifiedFeed, UnifiedOrchestrator
    from journal import Journal

    feed = SyntheticUnifiedFeed(days=2)
    jrn = Journal(":memory:")
    orch = UnifiedOrchestrator(feed=feed, journal=jrn)
    early = dt.datetime(2026, 7, 6, 9, 40, tzinfo=ET)
    late = dt.datetime(2026, 7, 6, 10, 15, tzinfo=ET)
    r_early = orch.tick(early)
    r_late = orch.tick(late)
    assert r_early is not None and r_late is not None
    assert r_early.signals.get("session_warmup") == 1.0
    assert r_late.signals.get("session_warmup") == 0.0
    # No TRADE ticket may clear the gate during warmup.
    if r_early.decision is not None:
        assert r_early.decision.decision != "TRADE" or not r_early.decision.gate_pass
