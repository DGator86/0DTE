"""
tests/test_paper_broker.py
==========================
Validates the in-house paper broker: entry sizing, mark-to-market P&L, and that
each exit rule (stop / target / trailing / EOD) fires with the right realized
P&L. Uses synthetic chains and a duck-typed TickResult — no network, no feed.

The test instrument is a 740/735 put credit spread (short 740P, long 735P):
collects ~1.00 credit at mid, width 5 -> max loss ~4.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg
from paper_broker import PaperBroker, PaperConfig
from regime_alignment import RASComponent, RASResult, RASConfig
from risk_manager import PositionMonitor, PositionRiskConfig

ET = ZoneInfo("America/New_York")
LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))   # short 740P / long 735P


def _chain(spot, p740, p735):
    """Build a chain where the 740 and 735 puts have the given mid prices
    (±0.05 to make a 0.10 bid-ask spread). Calls are filler."""
    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    return ChainSnapshot([q(740.0, p740), q(735.0, p735)], spot=spot, t_years=2e-4)


def _candidate():
    return SimpleNamespace(legs=LEGS, credit=1.00, family="put_credit",
                           short_strikes=(740.0,), long_strikes=(735.0,), max_loss=4.0)


def _result(chain, trade=True):
    dec = SimpleNamespace(
        decision="TRADE" if trade else "NO_TRADE",
        gate_pass=bool(trade),
        candidate=_candidate() if trade else None,
    )
    return SimpleNamespace(decision=dec, final_size_mult=1.0,
                           snapshot=SimpleNamespace(chain=chain))


def _broker(tmp_path, **cfg):
    return PaperBroker(db_path=str(tmp_path / "paper.sqlite"), cfg=PaperConfig(**cfg))


T0 = dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET)
T1 = dt.datetime(2026, 6, 30, 11, 0, tzinfo=ET)


def test_entry_opens_and_sizes(tmp_path):
    b = _broker(tmp_path)
    # entry chain: 740P mid 1.50, 735P mid 0.50 -> credit 1.00, after slippage 0.95
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    assert len(b.open_positions) == 1
    pos = b.open_positions[0]
    assert pos.contracts == 1                      # risk 500*1 / (~4.05*100=405) -> 1 lot
    assert pos.entry_credit == pytest.approx(0.95, abs=1e-6)
    assert pos.max_profit_ps == pytest.approx(0.95, abs=0.02)
    assert pos.max_loss_ps == pytest.approx(4.05, abs=0.05)


def test_profit_target_exit(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))         # open, entry 0.95
    # spread collapses: 740P 0.50 / 735P 0.15 -> credit_now 0.35; pnl 0.60 >= 0.6*0.95
    ev = b.on_tick(T1, _result(_chain(744, 0.50, 0.15), trade=False))
    assert not b.open_positions
    r = b.report()
    assert r["trades"] == 1 and r["by_exit_reason"] == {"target": 1}
    assert r["total_pnl"] > 0                                # ~ (0.60 - 0.05)*100 = $55


def test_stop_loss_exit(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))         # open, entry 0.95
    # spread blows out: 740P 3.90 / 735P 0.50 -> credit_now 3.40; pnl -2.45 <= -0.6*4.05
    b.on_tick(T1, _result(_chain(736, 3.90, 0.50), trade=False))
    assert not b.open_positions
    r = b.report()
    assert r["by_exit_reason"] == {"stop": 1}
    assert r["total_pnl"] < 0


def test_trailing_stop_exit(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))         # entry 0.95
    # tick up to +0.50 profit (arms trailing): credit_now 0.45 -> pnl 0.50
    b.on_tick(T1, _result(_chain(744, 0.45, 0.00), trade=False))
    assert b.open_positions and b.open_positions[0].trailing_armed
    # give back to +0.10: credit_now 0.85 -> peak(0.50)-0.10 = 0.40 >= 0.4*0.95
    b.on_tick(T1, _result(_chain(743, 0.85, 0.00), trade=False))
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"trail": 1}


def test_eod_exit(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    eod = dt.datetime(2026, 6, 30, 15, 56, tzinfo=ET)
    b.on_tick(eod, _result(_chain(742, 1.50, 0.50), trade=False))   # unchanged price, but EOD
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"eod": 1}


def test_max_one_position(tmp_path):
    b = _broker(tmp_path, max_open_positions=1)
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    b.on_tick(T1, _result(_chain(742, 1.50, 0.50)))         # still TRADE, but slot full
    assert len(b.open_positions) == 1


def test_daily_loss_limit_blocks_entry(tmp_path):
    b = _broker(tmp_path, daily_loss_limit_frac=0.05)       # very tight
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    b.on_tick(T1, _result(_chain(736, 3.90, 0.50), trade=False))   # stop -> realized loss
    assert b._day_realized[T1.astimezone(ET).date().isoformat()] < 0
    # next TRADE on same day should be blocked by the daily loss limit
    b.on_tick(T1, _result(_chain(742, 1.50, 0.50)))
    assert len(b.open_positions) == 0


def test_equity_accounting(tmp_path):
    b = _broker(tmp_path)
    assert b.cash == 1000.0
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    b.on_tick(T1, _result(_chain(744, 0.50, 0.15), trade=False))   # target win
    assert b.cash > 1000.0
    assert b.report()["equity"] == pytest.approx(b.cash, abs=0.01)


def test_equity_survives_restart(tmp_path):
    """Open positions can't survive a process restart (in-memory only), but
    realized equity must resume from the last closed trade instead of
    silently resetting to starting_cash -- that history is already on disk."""
    db_path = str(tmp_path / "paper.sqlite")
    b1 = PaperBroker(db_path=db_path, cfg=PaperConfig())
    b1.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    b1.on_tick(T1, _result(_chain(744, 0.50, 0.15), trade=False))   # target win, closes
    closed_equity = b1.cash
    assert closed_equity > 1000.0

    # Simulate a restart: a fresh PaperBroker instance against the same db_path.
    b2 = PaperBroker(db_path=db_path, cfg=PaperConfig())
    assert b2.cash == pytest.approx(closed_equity, abs=0.01)
    assert b2.open_positions == []   # open positions are still lost -- expected


def test_ras_invalidate_exit(tmp_path):
    cfg = PaperConfig(ras_exit_enabled=True)
    monitor = PositionMonitor(PositionRiskConfig(
        ras=RASConfig(exit_enabled=True, exit_threshold=-70.0)))
    b = PaperBroker(
        db_path=str(tmp_path / "paper.sqlite"),
        cfg=cfg,
        position_monitor=monitor,
    )
    b.on_tick(T0, _result(_chain(742, 1.50, 0.50)))
    pos = b.open_positions[0]
    ras = RASResult(
        score=-85.0,
        components=[RASComponent("gamma_alignment", -1.0, 1.5, -1.5, "hostile")],
        action="exit",
        position_id=pos.id,
        ema_score=-85.0,
    )
    result = _result(_chain(742, 1.50, 0.50), trade=False)
    result.ras_results = [ras]
    b.on_tick(T1, result)
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"ras_invalidate": 1}
