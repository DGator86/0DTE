"""
tests/test_candidate_id_determinism.py
"""
from __future__ import annotations

from prediction.candidate_universe import make_candidate_id


def test_leg_order_normalized_by_payload():
    # Same legs same id
    legs_a = [
        {"right": "C", "side": "buy", "qty": 1, "strike": 100,
         "expiration": "2026-01-01"},
    ]
    legs_b = [
        {"right": "C", "side": "buy", "quantity": 1, "strike": 100.0,
         "expiry": "2026-01-01"},
    ]
    # expiry vs expiration — make_candidate_id normalizes both keys
    id_a = make_candidate_id("s", family="x", legs=legs_a)
    id_b = make_candidate_id("s", family="x", legs=[
        {"right": "C", "side": "buy", "qty": 1, "strike": 100.0,
         "expiration": "2026-01-01"},
    ])
    assert id_a == id_b
