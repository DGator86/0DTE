"""
tests/test_canonical_snapshot_asof.py
=====================================
As-of safety for CanonicalSnapshot.
"""
from __future__ import annotations

import pytest

from prediction.canonical_snapshot import (
    CanonicalSnapshotError, build_canonical_snapshot,
)


def test_equal_timestamp_allowed():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 1.0},
        source_timestamps={"bars": "2026-07-14T10:00:00-04:00"},
        snapshot_id="s1",
    )
    assert snap.source_timestamps["bars"] == "2026-07-14T10:00:00-04:00"


def test_past_timestamp_allowed():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 1.0},
        source_timestamps={"bars": "2026-07-14T09:00:00-04:00"},
        snapshot_id="s1",
    )
    assert snap is not None


def test_gate_result_forbidden():
    with pytest.raises(CanonicalSnapshotError):
        build_canonical_snapshot(
            symbol="SPY",
            ts="2026-07-14T10:00:00-04:00",
            session_date="2026-07-14",
            raw_features={"gate_result": "pass"},
            snapshot_id="s1",
        )
