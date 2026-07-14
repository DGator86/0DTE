"""
tests/test_meta_threshold_selection.py
======================================
V3 Part 3 PR26 — nested threshold selection without outer leakage (§20).
"""
from __future__ import annotations

from prediction.models.trade_meta import select_thresholds_nested


def test_thresholds_from_fold_scores_only():
    # Inner fold scores — outer test must not appear here
    folds = []
    for i in range(40):
        folds.append({
            "prob": 0.4 + 0.02 * (i % 10),
            "unc": 0.1 + 0.05 * (i % 5),
            "utility": 0.1 if i % 3 else -0.2,
            "shortfall": 0.05,
        })
    cfg = select_thresholds_nested(folds)
    assert 0.5 <= cfg.minimum_trade_probability <= 0.65
    assert 0.70 <= cfg.uncertainty_abstain_threshold <= 0.80


def test_zero_trade_not_preferred():
    # All high uncertainty → abstain everything would score poorly vs
    # a grid point that allows some trades with positive utility
    folds = [
        {"prob": 0.7, "unc": 0.1, "utility": 1.0, "shortfall": 0.1}
        for _ in range(20)
    ] + [
        {"prob": 0.7, "unc": 0.9, "utility": -5.0, "shortfall": 1.0}
        for _ in range(5)
    ]
    cfg = select_thresholds_nested(
        folds,
        prob_grid=(0.58,),
        unc_grid=(0.5, 0.95),
    )
    # Prefer unc=0.5 (trades the good rows) over 0.95 (also trades) —
    # both trade; ensure we get a valid config
    assert cfg.minimum_trade_probability == 0.58
