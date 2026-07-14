"""
tests/test_ood.py
=================
V3 Part 1 §7.4 / §12 — OOD detector determinism and monotonicity.
"""
from __future__ import annotations

import numpy as np

from prediction.ood import OODDetector, OODDetectorConfig


def _train_rows(n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "dist_vwap": float(rng.normal(0, 0.01)),
            "dist_gamma_flip": float(rng.normal(0, 0.01)),
            "gex_rank": float(rng.uniform(0, 1)),
            "realized_vol": float(rng.uniform(0.005, 0.02)),
            "adx": float(rng.uniform(10, 30)),
            "minutes_to_close": float(rng.uniform(30, 300)),
        })
    return rows


def test_determinism():
    rows = _train_rows()
    d1 = OODDetector().fit(rows)
    d2 = OODDetector().fit(rows)
    r = rows[0]
    a, b = d1.score_one(r), d2.score_one(r)
    assert a.score == b.score
    assert a.percentile == b.percentile


def test_greater_distance_does_not_reduce_uncertainty():
    rows = _train_rows()
    det = OODDetector().fit(rows)
    in_dist = {
        "dist_vwap": 0.0, "dist_gamma_flip": 0.0, "gex_rank": 0.5,
        "realized_vol": 0.01, "adx": 20.0, "minutes_to_close": 120.0,
    }
    far = {
        "dist_vwap": 5.0, "dist_gamma_flip": -5.0, "gex_rank": 0.5,
        "realized_vol": 0.5, "adx": 80.0, "minutes_to_close": 1.0,
    }
    a = det.score_one(in_dist)
    b = det.score_one(far)
    assert b.score >= a.score - 1e-12
    assert b.percentile >= a.percentile - 1e-12


def test_percentile_buckets():
    rows = _train_rows()
    det = OODDetector(OODDetectorConfig()).fit(rows)
    r = det.score_one(rows[0])
    assert r.state_bucket in (
        "normal_support", "reduced_support", "high_ood", "extreme_ood",
        "unknown")
    assert 0.0 <= r.score <= 1.0
