"""
tests/test_full_v3_forecast_runtime.py
"""
from __future__ import annotations

from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.forecast_assembly import build_v3_forecast


def test_deterministic_replay_forecast():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="fixed",
    )
    a = build_v3_forecast(snapshot=snap, mode="shadow")
    b = build_v3_forecast(snapshot=snap, mode="shadow")
    assert a.snapshot_id == b.snapshot_id
    assert a.session_date == b.session_date
