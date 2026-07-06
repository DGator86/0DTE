"""
Peak-relative trailing stop.

The old trail measured giveback as a fraction of MAX PROFIT: a trade that
armed at 35% and peaked at 37% could ride back to a LOSS before "trailing"
out, and far-OTM debit spreads never armed at all. These tests pin the new
guarantees:
  1. an armed winner cannot round-trip into a loser (breakeven lock)
  2. giveback is bounded relative to the PEAK, not max profit
  3. far-OTM debit spreads arm via the R-multiple path
  4. deep-in-profit trades tighten their leash
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg
from paper_broker import PaperBroker, PaperConfig

ET = ZoneInfo("America/New_York")
T = [dt.datetime(2026, 7, 6, 10, 0, tzinfo=ET) + dt.timedelta(minutes=i)
     for i in range(0, 120)]

PUT_LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))     # credit spread, maxL ~4
CALL_LEGS = (Leg(749.0, "C", 1), Leg(750.0, "C", -1))    # far-OTM debit, maxL ~0.12


def _put_chain(spot, p740, p735):
    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    return ChainSnapshot([q(740.0, p740), q(735.0, p735)], spot=spot, t_years=2e-4)


def _call_chain(spot, c749, c750):
    def q(strike, cmid):
        return ChainQuote(strike=strike, call_bid=cmid - 0.01, call_ask=cmid + 0.01,
                          put_bid=0.01, put_ask=0.03)
    return ChainSnapshot([q(749.0, c749), q(750.0, c750)], spot=spot, t_years=2e-4)


def _result(chain, legs=PUT_LEGS, credit=1.00, family="put_credit", trade=True):
    cand = SimpleNamespace(legs=legs, credit=credit, family=family,
                           short_strikes=(), long_strikes=(), max_loss=4.0)
    dec = SimpleNamespace(decision="TRADE" if trade else "NO_TRADE",
                          gate_pass=bool(trade), gate_score=80.0, gate_kelly=1.0,
                          candidate=cand if trade else None)
    return SimpleNamespace(decision=dec, final_size_mult=1.0,
                           snapshot=SimpleNamespace(chain=chain))


def _broker(tmp_path, **cfg):
    return PaperBroker(db_path=str(tmp_path / "paper.sqlite"), cfg=PaperConfig(**cfg))


# --------------------------------------------------------------------------- #
# 1. armed winner cannot become a loser                                        #
# --------------------------------------------------------------------------- #
def test_armed_winner_never_exits_negative(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(_put_chain(742, 1.50, 0.50)))          # entry 0.95
    # peak at +0.43 of ~0.95 maxP: armed (>= 0.3325), but under the OLD rule
    # the giveback threshold (0.4 * 0.95 = 0.38) would have let it ride to +0.05
    b.on_tick(T[1], _result(_put_chain(743, 0.55, 0.03), trade=False))   # pnl ~ +0.43
    pos = b.open_positions[0]
    assert pos.trailing_armed
    # fade to +0.17: below the peak-relative floor (0.43 * 0.6 = 0.258) -> exit
    b.on_tick(T[2], _result(_put_chain(742, 0.78, 0.03), trade=False))   # pnl ~ +0.17
    assert not b.open_positions
    r = b.report()
    assert r["by_exit_reason"] == {"trail": 1}
    assert r["total_pnl"] > 0                                      # kept most of the win


def test_breakeven_lock_floors_collapse_at_breakeven(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(_put_chain(742, 1.50, 0.50)))          # entry 0.95, slip 0.05
    b.on_tick(T[1], _result(_put_chain(743, 0.55, 0.03), trade=False))   # arm at +0.37
    # instant collapse straight through the floor to -0.25: trail fires on this
    # tick (not the hard stop at -2.43), realizing the gap price
    b.on_tick(T[2], _result(_put_chain(741, 1.17, 0.03), trade=False))   # pnl ~ -0.19
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"trail": 1}


# --------------------------------------------------------------------------- #
# 2. giveback bounded relative to peak                                         #
# --------------------------------------------------------------------------- #
def test_holds_inside_peak_relative_leash(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(_put_chain(742, 1.50, 0.50)))          # entry 0.95
    b.on_tick(T[1], _result(_put_chain(743, 0.55, 0.03), trade=False))   # peak +0.37
    # +0.30 is above the floor (0.222): must HOLD, not exit
    b.on_tick(T[2], _result(_put_chain(743, 0.62, 0.03), trade=False))
    assert len(b.open_positions) == 1


# --------------------------------------------------------------------------- #
# 3. far-OTM debit spreads arm via the R path                                  #
# --------------------------------------------------------------------------- #
def test_deep_otm_debit_arms_on_r_multiple(tmp_path):
    b = _broker(tmp_path)
    # buy 749C 0.14 / sell 750C 0.04 -> debit ~0.10, +slip: maxL ~0.12, maxP ~0.88
    b.on_tick(T[0], _result(_call_chain(748, 0.14, 0.04),
                            legs=CALL_LEGS, credit=-0.10, family="long_call_spread"))
    pos = b.open_positions[0]
    assert pos.max_loss_ps < 0.2
    # +0.10 gain = ~0.8R but only ~11% of max profit: OLD rule never arms here
    b.on_tick(T[1], _result(_call_chain(749, 0.26, 0.06), trade=False))
    assert b.open_positions[0].trailing_armed
    # fade below the peak-relative floor -> trail exit, small win kept
    b.on_tick(T[2], _result(_call_chain(748.4, 0.155, 0.045), trade=False))
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"trail": 1}


# --------------------------------------------------------------------------- #
# 4. tightening once deep in profit                                            #
# --------------------------------------------------------------------------- #
def test_leash_tightens_deep_in_profit(tmp_path):
    # raise the target so the tighten band (peak >= 0.5 * maxP) is reachable
    b = _broker(tmp_path, profit_target_frac=0.90, trailing_tighten_at=0.50)
    b.on_tick(T[0], _result(_put_chain(742, 1.50, 0.50)))          # entry 0.95, maxP ~0.95
    # peak +0.62 (> 0.5*0.95 = 0.475): tight leash, giveback only 25% of peak
    b.on_tick(T[1], _result(_put_chain(744, 0.35, 0.02), trade=False))   # pnl ~ +0.62
    pos = b.open_positions[0]
    floor = b._trail_floor(pos)
    assert floor == pytest.approx(0.62 * 0.75, abs=0.02)
    # +0.40 breaches the tight floor (~0.465) but NOT the loose one (~0.372)
    b.on_tick(T[2], _result(_put_chain(743.5, 0.58, 0.03), trade=False))
    assert not b.open_positions
    assert b.report()["by_exit_reason"] == {"trail": 1}


def test_peak_recorded_for_giveback_audit(tmp_path):
    b = _broker(tmp_path)
    b.on_tick(T[0], _result(_put_chain(742, 1.50, 0.50)))
    b.on_tick(T[1], _result(_put_chain(743, 0.55, 0.03), trade=False))
    b.on_tick(T[2], _result(_put_chain(742, 0.78, 0.03), trade=False))
    row = b._db.execute("SELECT peak_pnl_ps, pnl_ps FROM paper_trades").fetchone()
    assert row[0] == pytest.approx(0.43, abs=0.03)
    assert row[0] >= row[1]                                        # peak >= realized
