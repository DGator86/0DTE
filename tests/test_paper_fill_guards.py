"""
tests/test_paper_fill_guards.py
===============================
Broker fill-quality guards: never paper a lottery-ticket candidate, and cap
the contract count so a tiny-max-loss structure can't balloon.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg
from paper_broker import PaperBroker, PaperConfig

ET = ZoneInfo("America/New_York")
T0 = dt.datetime(2026, 6, 30, 11, 0, tzinfo=ET)
LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))


def _chain(spot=742, p740=1.50, p735=0.50):
    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    return ChainSnapshot([q(740.0, p740), q(735.0, p735)], spot=spot, t_years=2e-4)


def _cand(*, prob_profit=0.6, credit=1.00, family="put_credit", legs=LEGS):
    return SimpleNamespace(
        legs=legs, credit=credit, family=family,
        short_strikes=(740.0,), long_strikes=(735.0,), max_loss=4.0,
        ev=0.2, ev_per_risk=0.05, prob_profit=prob_profit)


def _result(chain, cand):
    intents = [{"track": "legacy", "candidate": cand, "size_mult": 1.0,
                "structure": "PCS", "reason": "matrix"}]
    return SimpleNamespace(
        decision=None, final_size_mult=1.0,
        snapshot=SimpleNamespace(chain=chain, market=None),
        signals={"session_warmup": 0.0, "policy_mode": "shadow"},
        intent=None, regime=None, ras_results=[], part3=None,
        paper_intents=intents)


def test_normal_candidate_fills(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0))
    b.on_tick(T0, _result(_chain(), _cand(prob_profit=0.6)))
    assert len(b.open_positions) == 1


def test_low_prob_profit_is_skipped(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0))
    # 0.3% win-prob lottery ticket — below the 5% default floor.
    b.on_tick(T0, _result(_chain(), _cand(prob_profit=0.003)))
    assert b.open_positions == []


def test_cheap_debit_is_skipped(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0))
    # A ~$0.03 long put: debit (negative credit), max_loss below the $0.10 floor.
    chain = _chain(spot=742, p735=0.03)
    cand = _cand(credit=-0.03, family="long_put", legs=(Leg(735.0, "P", 1),),
                 prob_profit=0.6)
    b.on_tick(T0, _result(chain, cand))
    assert b.open_positions == []


def test_prob_floor_disabled_allows_fill(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0, min_prob_profit=0.0))
    b.on_tick(T0, _result(_chain(), _cand(prob_profit=0.003)))
    assert len(b.open_positions) == 1  # guard off => old behaviour


def test_contract_count_is_capped(tmp_path):
    # Big account so sizing would otherwise buy many contracts; cap holds it down.
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0, starting_cash=5000.0,
                                    max_contracts=3))
    b.on_tick(T0, _result(_chain(), _cand(prob_profit=0.6)))
    assert len(b.open_positions) == 1
    assert b.open_positions[0].contracts == 3
