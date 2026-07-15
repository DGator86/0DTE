"""
tests/test_track_parity.py
==========================
Apples-to-apples helpers: V3 signals flatten, paper track summary,
family→structure mapping, fair-comparison stand-down.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from paper_broker import PaperBroker, PaperConfig
from prediction.track_parity import (
    family_to_structure, flatten_part3_signals, paper_track_summary,
    settle_paper_outcome,
)
from prediction.storage import PredictionStore
from rnd_extractor import ChainQuote, ChainSnapshot
from spread_selector import Leg

ET = ZoneInfo("America/New_York")
LEGS = (Leg(740.0, "P", -1), Leg(735.0, "P", 1))


def test_family_to_structure():
    assert family_to_structure("iron_condor") == "IC"
    assert family_to_structure("put_credit") == "PCS"
    assert family_to_structure(None) is None


def test_flatten_part3_signals():
    signals = {}
    flatten_part3_signals({
        "decision_summary": {
            "action": "HARD_VETO",
            "statistical_action": "TRADE",
            "selected_candidate_id": "c1",
            "family": "iron_condor",
            "expected_order_value": 0.12,
            "hard_vetoes": ["stand_down"],
        },
        "ranking": {"top_candidate_id": "c1", "top_score_margin": 0.2},
    }, signals)
    assert signals["v3_statistical_action"] == "TRADE"
    assert signals["v3_action"] == "HARD_VETO"
    assert signals["v3_selected_candidate_id"] == "c1"
    assert signals["v3_family"] == "iron_condor"
    assert signals["v3_expected_order_value"] == 0.12
    assert signals["v3_hard_veto_count"] == 1.0


def test_paper_track_summary(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "p.sqlite"),
                    cfg=PaperConfig(reentry_cooldown_min=0.0, stop_cooldown_min=0.0))

    def q(strike, pmid):
        return ChainQuote(strike=strike, call_bid=0.01, call_ask=0.03,
                          put_bid=pmid - 0.05, put_ask=pmid + 0.05)
    chain = ChainSnapshot([q(740.0, 1.50), q(735.0, 0.50)], spot=742, t_years=2e-4)
    cand = SimpleNamespace(
        legs=LEGS, credit=1.0, family="put_credit",
        short_strikes=(740.0,), long_strikes=(735.0,), max_loss=4.0,
        ev=0.1, ev_per_risk=0.05, prob_profit=0.55,
    )
    t0 = dt.datetime(2026, 6, 30, 10, 30, tzinfo=ET)
    intents = [
        {"track": "legacy", "candidate": cand, "size_mult": 1.0,
         "snapshot_id": "s1", "candidate_id": "leg1"},
        {"track": "v2", "candidate": cand, "size_mult": 1.0,
         "snapshot_id": "s1", "candidate_id": "v2a"},
    ]
    res = SimpleNamespace(
        decision=None, final_size_mult=1.0,
        snapshot=SimpleNamespace(chain=chain, market=None),
        signals={"session_warmup": 0.0}, intent=None, regime=None,
        ras_results=[], paper_intents=intents, part3=None,
    )
    b.on_tick(t0, res)
    assert b.open_positions[0].entry_ctx["snapshot_id"] == "s1"
    # Close both
    chain2 = ChainSnapshot([q(740.0, 0.50), q(735.0, 0.15)], spot=744, t_years=2e-4)
    b.on_tick(t0 + dt.timedelta(hours=1), SimpleNamespace(
        decision=None, final_size_mult=0.0,
        snapshot=SimpleNamespace(chain=chain2, market=None),
        signals={}, intent=None, regime=None, ras_results=[],
        paper_intents=[], part3=None,
    ))
    summary = paper_track_summary(str(tmp_path / "p.sqlite"))
    assert summary["by_track"]["legacy"]["trades"] == 1
    assert summary["by_track"]["v2"]["trades"] == 1
    assert summary["tracks_with_trades"] == 2


def test_settle_paper_outcome(tmp_path):
    store = PredictionStore(str(tmp_path / "pred.sqlite"))
    pos = SimpleNamespace(
        id="abc", family="put_credit", contracts=1,
        entry_ctx={"candidate_id": "c99", "fill_track": "v3",
                   "snapshot_id": "snap1"},
    )
    settle_paper_outcome(
        store, pos=pos, pnl_dollars=12.5, exit_reason="target",
        closed_at=dt.datetime(2026, 6, 30, 15, 0, tzinfo=ET))
    rows = store.fetch_candidates()
    # outcome is joined; may be empty if no snapshot row — check outcomes table
    cur = store.conn.execute(
        "SELECT candidate_id, pnl_mid, first_event FROM candidate_outcomes")
    got = cur.fetchone()
    assert got[0] == "c99"
    assert got[1] == 12.5
    assert got[2] == "target"
    store.close()


def test_fair_comparison_blocks_v2_v3_on_stand_down():
    from unified_loop import UnifiedOrchestrator
    orch = UnifiedOrchestrator(
        feed=SimpleNamespace(), paper_fair_comparison=True)
    intents = orch._build_paper_intents(
        snap=SimpleNamespace(chain=object(), market=None),
        signals={
            "v2_policy_action": "TRADE",
            "v2_policy_structure": "IC",
            "_snapshot_id": "s",
        },
        intent=SimpleNamespace(decision=SimpleNamespace(
            structure="NT", direction="none"), size_mult=0.0),
        regime_state=SimpleNamespace(stand_down=True, vetoes=[]),
        decision=None, decide_pdf=None, cfg=None,
        pin_active=False, density_mode="vrp", density_moments=None,
        final_size_mult=0.0, matrix_stand_down=True,
    )
    # Stand-down: only legacy could try, but NT → no intents
    assert intents == []
