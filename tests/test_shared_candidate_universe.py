"""
tests/test_shared_candidate_universe.py / candidate_id_determinism
"""
from __future__ import annotations

from prediction.candidate_universe import (
    build_candidate_universe, make_candidate_id,
)


def test_candidate_ids_deterministic():
    legs = [
        {"right": "C", "side": "buy", "qty": 1, "strike": 500.0,
         "expiration": "2026-07-14"},
        {"right": "C", "side": "sell", "qty": 1, "strike": 505.0,
         "expiration": "2026-07-14"},
    ]
    a = make_candidate_id("snap1", family="call_debit", legs=legs)
    b = make_candidate_id("snap1", family="call_debit", legs=legs)
    assert a == b
    c = make_candidate_id("snap2", family="call_debit", legs=legs)
    assert a != c


def test_legacy_and_v3_identical_ids():
    cands = [
        {"family": "put_credit", "ev": 0.1, "legs": [
            {"right": "P", "side": "sell", "qty": 1, "strike": 490,
             "expiration": "2026-07-14"},
            {"right": "P", "side": "buy", "qty": 1, "strike": 485,
             "expiration": "2026-07-14"},
        ]},
        {"family": "call_debit", "ev": 0.05, "legs": [
            {"right": "C", "side": "buy", "qty": 1, "strike": 500,
             "expiration": "2026-07-14"},
            {"right": "C", "side": "sell", "qty": 1, "strike": 505,
             "expiration": "2026-07-14"},
        ]},
    ]
    u1 = build_candidate_universe(
        snapshot_id="s1", generated_at="t1", candidates=cands)
    u2 = build_candidate_universe(
        snapshot_id="s1", generated_at="t1",
        candidates=[dict(c) for c in cands])
    assert u1.candidate_ids() == u2.candidate_ids()
    assert len(u1.candidates) == 2


def test_exclusions_recorded():
    u = build_candidate_universe(
        snapshot_id="s1",
        generated_at="t1",
        candidates=[],
        excluded_at_generation=[{"reason": "illiquid", "strike": 500}],
    )
    assert u.excluded_at_generation[0]["reason"] == "illiquid"
    assert u.to_dict()["excluded_count"] == 1


def test_no_duplicate_economic_candidates():
    legs = [{"right": "C", "side": "buy", "qty": 1, "strike": 500,
             "expiration": "2026-07-14"}]
    cands = [
        {"family": "call", "legs": legs},
        {"family": "call", "legs": legs},
    ]
    u = build_candidate_universe(
        snapshot_id="s1", generated_at="t1", candidates=cands)
    assert len(u.candidates) == 1
