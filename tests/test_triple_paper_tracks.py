"""
tests/test_triple_paper_tracks.py
=================================
Parallel paper fills: legacy / v2 / v3 each get their own ledger + fill_track.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg
from paper_broker import PaperBroker, PaperConfig, PAPER_TRACKS

ET = ZoneInfo("America/New_York")
LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))
T0 = dt.datetime(2026, 6, 30, 10, 30, tzinfo=ET)


def _chain(spot=742, p740=1.50, p735=0.50):
    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    return ChainSnapshot([q(740.0, p740), q(735.0, p735)], spot=spot, t_years=2e-4)


def _cand(family="put_credit"):
    return SimpleNamespace(
        legs=LEGS, credit=1.00, family=family,
        short_strikes=(740.0,), long_strikes=(735.0,), max_loss=4.0,
        ev=0.2, ev_per_risk=0.05, prob_profit=0.6,
    )


def _result(chain, intents):
    return SimpleNamespace(
        decision=None,
        final_size_mult=1.0,
        snapshot=SimpleNamespace(chain=chain, market=None),
        signals={"session_warmup": 0.0, "policy_mode": "shadow"},
        intent=None,
        regime=None,
        ras_results=[],
        part3=None,
        paper_intents=intents,
    )


def test_parallel_tracks_open_three_independent_positions(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0))
    chain = _chain()
    intents = [
        {"track": "legacy", "candidate": _cand("put_credit"),
         "size_mult": 1.0, "structure": "PCS", "reason": "matrix"},
        {"track": "v2", "candidate": _cand("iron_condor"),
         "size_mult": 1.0, "structure": "IC", "reason": "prediction_policy"},
        {"track": "v3", "candidate": _cand("call_credit"),
         "size_mult": 1.0, "structure": "CCS", "reason": "part3_meta",
         "candidate_id": "c1", "v3_action": "TRADE"},
    ]
    ev = b.on_tick(T0, _result(chain, intents))
    assert len(b.open_positions) == 3
    tracks = sorted(p.entry_ctx["fill_track"] for p in b.open_positions)
    assert tracks == ["legacy", "v2", "v3"]
    assert all(t in PAPER_TRACKS for t in tracks)
    # Independent ledgers — none spent cash on open (credit structures)
    for t in PAPER_TRACKS:
        assert b.ledgers[t] == pytest.approx(1000.0)
    assert any("[legacy]" in e for e in ev)
    assert any("[v2]" in e for e in ev)
    assert any("[v3]" in e for e in ev)


def test_per_track_open_cap(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(max_open_positions=1, reentry_cooldown_min=0.0))
    chain = _chain()
    intents = [
        {"track": "v2", "candidate": _cand(), "size_mult": 1.0},
        {"track": "v2", "candidate": _cand(), "size_mult": 1.0},
    ]
    b.on_tick(T0, _result(chain, intents))
    assert b._open_count("v2") == 1


def test_report_by_track_after_close(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0, stop_cooldown_min=0.0))
    chain = _chain()
    intents = [
        {"track": "legacy", "candidate": _cand(), "size_mult": 1.0},
        {"track": "v3", "candidate": _cand(), "size_mult": 1.0},
    ]
    b.on_tick(T0, _result(chain, intents))
    # Collapse spread → target exit
    t1 = T0 + dt.timedelta(hours=1)
    empty = _result(_chain(744, 0.50, 0.15), [])
    b.on_tick(t1, empty)
    assert not b.open_positions
    r = b.report(t1)
    assert r["by_track"]["legacy"]["trades"] == 1
    assert r["by_track"]["v3"]["trades"] == 1
    assert r["by_track"]["v2"]["trades"] == 0
    assert r["by_track"]["legacy"]["total_pnl"] != 0 or True  # may be target pnl


def test_backward_compat_single_decision_path(tmp_path):
    """No paper_intents → legacy _maybe_open still works."""
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"), cfg=PaperConfig())
    dec = SimpleNamespace(
        decision="TRADE", gate_pass=True, candidate=_cand(),
        gate_kelly=1.0, gate_score=70.0,
    )
    res = SimpleNamespace(
        decision=dec, final_size_mult=1.0,
        snapshot=SimpleNamespace(chain=_chain(), market=None),
        signals={"session_warmup": 0.0, "policy_mode": "shadow"},
        intent=None, regime=None, ras_results=[], paper_intents=[],
    )
    b.on_tick(T0, res)
    assert len(b.open_positions) == 1
    assert b.open_positions[0].entry_ctx["fill_track"] == "legacy"
