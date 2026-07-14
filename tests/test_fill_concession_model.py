"""
tests/test_fill_concession_model.py
===================================
V3 Part 3 PR23 — fill concession Stage 2 (§14 / §47).
"""
from __future__ import annotations

import pytest

from execution.fill_records import FillRecord
from prediction.fill_training import stage1_attempts, stage2_fills
from prediction.models.fill_concession import FillConcessionModel
from prediction.models.fill_probability import _features_from_record


def _fill(i: int, frac: float) -> FillRecord:
    mid, nat = 0.50, 0.30
    fill_credit = mid - frac * (mid - nat)
    return FillRecord(
        fill_record_id=f"f{i}",
        snapshot_id=f"s{i % 3}",
        candidate_id=f"c{i}",
        session_date=f"2026-07-{(i % 5) + 1:02d}",
        decision_ts="2026-07-01T14:00:00Z",
        submitted_ts="2026-07-01T14:00:01Z",
        resolved_ts="2026-07-01T14:00:05Z",
        symbol="SPY",
        family="put_credit",
        side="credit",
        n_legs=2,
        limit_credit=fill_credit,
        mid_credit_at_submit=mid,
        natural_credit_at_submit=nat,
        relative_spread=0.1,
        absolute_spread=0.2,
        option_price_scale=0.5,
        quote_age_seconds=1.0 + 0.1 * i,
        minutes_to_close=100.0,
        filled=True,
        fill_credit=fill_credit,
        requested_quantity=1,
        source="paper",
        mode="shadow",
    )


def test_only_fills_enter_stage2():
    fills = [_fill(i, 0.3 + 0.05 * (i % 4)) for i in range(12)]
    unfilled = FillRecord(
        **{**fills[0].to_dict(), "fill_record_id": "u1", "filled": False,
           "fill_credit": None})
    all_recs = fills + [unfilled]
    assert len(stage1_attempts(all_recs)) == 13
    assert len(stage2_fills(all_recs)) == 12


def test_quantiles_ordered_and_conservative():
    fills = [_fill(i, 0.2 + 0.05 * (i % 6)) for i in range(15)]
    model = FillConcessionModel().fit(fills)
    fc = model.predict(_features_from_record(fills[0]), family="put_credit")
    assert fc.fill_q10 <= fc.fill_q50 <= fc.fill_q90
    assert fc.conservative_fill_fraction >= fc.expected_fill_fraction - 1e-9
    assert fc.diagnostics["fallback_level"]


def test_deterministic():
    fills = [_fill(i, 0.4) for i in range(10)]
    f1 = FillConcessionModel().fit(fills).predict(
        _features_from_record(fills[0]), family="put_credit")
    f2 = FillConcessionModel().fit(fills).predict(
        _features_from_record(fills[0]), family="put_credit")
    assert f1.to_dict() == f2.to_dict()
