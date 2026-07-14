"""
tests/test_fill_probability_model.py
====================================
V3 Part 3 PR22 — empirical fill probability (§13 / §46).
"""
from __future__ import annotations

import pytest

from execution.fill_records import FillRecord
from prediction.fill_training import (
    blend_with_prior, empirical_weight, stage1_attempts, stage2_fills,
)
from prediction.models.fill_probability import (
    FillProbabilityModel, enforce_horizon_order, fill_horizon_labels,
)


def _rec(**kw) -> FillRecord:
    base = dict(
        fill_record_id="x",
        snapshot_id="s",
        candidate_id="c",
        session_date="2026-07-01",
        decision_ts="2026-07-01T14:00:00Z",
        submitted_ts="2026-07-01T14:00:01Z",
        resolved_ts="2026-07-01T14:00:20Z",
        symbol="SPY",
        family="put_credit",
        side="credit",
        n_legs=2,
        limit_credit=0.40,
        mid_credit_at_submit=0.50,
        natural_credit_at_submit=0.30,
        relative_spread=0.1,
        absolute_spread=0.2,
        option_price_scale=0.5,
        quote_age_seconds=1.0,
        minutes_to_close=100.0,
        requested_quantity=1,
        source="paper",
        mode="shadow",
    )
    base.update(kw)
    return FillRecord(**base)


def _dataset():
    rows = []
    for i in range(20):
        filled = i % 3 != 0
        sec = 10.0 + (i % 5) * 10 if filled else None
        rows.append(_rec(
            fill_record_id=f"r{i}",
            filled=filled,
            fill_credit=0.40 if filled else None,
            seconds_to_first_fill=sec,
            relative_spread=0.05 + 0.02 * (i % 4),
            quote_age_seconds=float(i % 6),
            expired_unfilled=not filled,
        ))
    return rows


def test_unfilled_enter_stage1_not_stage2():
    rows = _dataset()
    assert len(stage1_attempts(rows)) == 20
    assert all(r.filled for r in stage2_fills(rows))
    assert len(stage2_fills(rows)) < 20


def test_horizon_labels_include_negatives():
    rows = _dataset()
    ys = fill_horizon_labels(rows)
    assert len(ys["y_15"]) == 20
    assert ys["y_15"].sum() < ys["y_60"].sum() or ys["y_15"].sum() == ys["y_60"].sum()


def test_probability_ordering():
    assert enforce_horizon_order([0.9, 0.5, 0.7, 0.4]) == [0.9, 0.9, 0.9, 0.9]
    assert enforce_horizon_order([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]


def test_model_fit_predict_ordered():
    rows = _dataset()
    model = FillProbabilityModel().fit(rows)
    from prediction.models.fill_probability import _features_from_record
    fc = model.predict(_features_from_record(rows[0]), family="put_credit")
    assert fc.p_fill_15s <= fc.p_fill_30s <= fc.p_fill_60s <= fc.p_fill_before_cancel
    assert fc.calibration_support == 20


def test_low_support_increases_prior_weight():
    w_low = empirical_weight(10, prior_equivalent_support=100)
    w_high = empirical_weight(500, prior_equivalent_support=100)
    assert w_low < w_high
    blended, w = blend_with_prior(0.9, 0.5, 0, prior_equivalent_support=100)
    assert blended == pytest.approx(0.5)
    assert w == 0.0
