"""
tests/test_fill_records.py
==========================
V3 Part 3 PR21 — fill-attempt recording (§11–§12 / §45).
"""
from __future__ import annotations

import pytest

from execution.fill_records import (
    FillRecord, enrich_fill_fractions, fill_fraction, validate_fill_record,
)
from prediction.storage import PredictionStore


def _base(**overrides) -> dict:
    d = dict(
        fill_record_id="fr1",
        snapshot_id="2026-07-01|t0",
        candidate_id="c1",
        session_date="2026-07-01",
        decision_ts="2026-07-01T14:00:00Z",
        submitted_ts="2026-07-01T14:00:01Z",
        resolved_ts="2026-07-01T14:00:10Z",
        symbol="SPY",
        family="put_credit",
        side="credit",
        n_legs=2,
        limit_credit=0.40,
        mid_credit_at_submit=0.50,
        natural_credit_at_submit=0.30,
        relative_spread=0.10,
        absolute_spread=0.20,
        option_price_scale=0.50,
        quote_age_seconds=1.0,
        minutes_to_close=120.0,
        requested_quantity=1,
        source="paper",
        mode="shadow",
    )
    d.update(overrides)
    return d


class TestFillFraction:
    def test_credit_mid_to_natural(self):
        raw, clipped = fill_fraction(0.50, 0.30, 0.40, side="credit")
        assert raw == pytest.approx(0.5)
        assert clipped == pytest.approx(0.5)

    def test_credit_at_mid(self):
        raw, clipped = fill_fraction(0.50, 0.30, 0.50, side="credit")
        assert raw == pytest.approx(0.0)
        assert clipped == pytest.approx(0.0)

    def test_debit_adverse(self):
        # mid debit -1.0, natural -1.20, fill -1.10 → halfway
        raw, clipped = fill_fraction(-1.0, -1.20, -1.10, side="debit")
        assert raw == pytest.approx(0.5)
        assert clipped == pytest.approx(0.5)

    def test_raw_anomaly_retained(self):
        # Better than mid (higher credit than mid)
        raw, clipped = fill_fraction(0.50, 0.30, 0.55, side="credit")
        assert raw < 0.0
        assert clipped == 0.0
        # Worse than natural
        raw2, clipped2 = fill_fraction(0.50, 0.30, 0.20, side="credit")
        assert raw2 > 1.0
        assert clipped2 == 1.0


class TestProvenance:
    def test_sources_distinguishable(self):
        for src in ("paper", "broker_actual", "hypothetical"):
            rec = FillRecord(**_base(source=src, filled=False,
                                     fill_record_id=f"id-{src}"))
            validate_fill_record(rec)

    def test_hypothetical_cannot_be_filled(self):
        rec = FillRecord(**_base(source="hypothetical", filled=True,
                                 fill_credit=0.4))
        with pytest.raises(ValueError, match="filled=False"):
            validate_fill_record(rec)

    def test_simulated_not_broker_actual(self):
        rec = FillRecord(**_base(
            source="broker_actual", filled=True, fill_credit=0.4,
            diagnostics={"simulated": True}))
        with pytest.raises(ValueError, match="simulated"):
            validate_fill_record(rec)

    def test_midpoint_diagnostic_not_filled(self):
        rec = FillRecord(**_base(
            filled=True, fill_credit=0.5,
            diagnostics={"midpoint_diagnostic": True}))
        with pytest.raises(ValueError, match="midpoint"):
            validate_fill_record(rec)

    def test_unfilled_and_cancelled_stored(self, tmp_path):
        store = PredictionStore(str(tmp_path / "f.sqlite"))
        for i, kw in enumerate((
            dict(filled=False, expired_unfilled=True, fill_record_id="u1",
                 resolved_ts="2026-07-01T14:01:00Z"),
            dict(filled=False, cancelled=True, fill_record_id="c1",
                 source="cancelled"),
            dict(filled=True, partial_fill=True, filled_quantity=1,
                 requested_quantity=2, fill_credit=0.42, fill_record_id="p1"),
        )):
            rec = enrich_fill_fractions(FillRecord(**_base(**kw)))
            validate_fill_record(rec)
            store.log_fill_record(rec)
        rows = store.fetch_fill_records(session_date="2026-07-01")
        assert len(rows) == 3
        store.close()

    def test_invalid_timestamps_rejected(self):
        rec = FillRecord(**_base(
            decision_ts="2026-07-01T15:00:00Z",
            submitted_ts="2026-07-01T14:00:00Z"))
        with pytest.raises(ValueError, match="decision_ts"):
            validate_fill_record(rec)

    def test_duplicate_id_idempotent(self, tmp_path):
        store = PredictionStore(str(tmp_path / "f.sqlite"))
        rec = FillRecord(**_base(filled=True, fill_credit=0.4))
        store.log_fill_record(rec)
        store.log_fill_record(rec)  # replace
        assert len(store.fetch_fill_records()) == 1
        store.close()
