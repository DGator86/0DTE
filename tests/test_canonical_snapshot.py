"""
tests/test_canonical_snapshot.py
================================
UNIFIED PR2 — CanonicalSnapshot contracts.
"""
from __future__ import annotations

import pytest

from prediction.canonical_snapshot import (
    CanonicalSnapshotError, build_canonical_snapshot,
)


def test_deterministic_snapshot_hash():
    a = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "adx": 20.0},
        source_seq=1,
    )
    b = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "adx": 20.0},
        source_seq=1,
    )
    assert a.snapshot_id == b.snapshot_id
    assert a.snapshot_hash() == b.snapshot_hash()


def test_missing_remains_missing():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "call_wall": None},
        snapshot_id="s1",
    )
    assert snap.raw_features["call_wall"] is None
    assert snap.missingness["call_wall"] is True


def test_no_post_routing_fields():
    with pytest.raises(CanonicalSnapshotError, match="post-routing"):
        build_canonical_snapshot(
            symbol="SPY",
            ts="2026-07-14T10:00:00-04:00",
            session_date="2026-07-14",
            raw_features={"spot": 1.0, "selected_structure": "call_debit"},
            snapshot_id="s1",
        )


def test_future_dated_source_rejected():
    with pytest.raises(CanonicalSnapshotError, match="future-dated"):
        build_canonical_snapshot(
            symbol="SPY",
            ts="2026-07-14T10:00:00-04:00",
            session_date="2026-07-14",
            raw_features={"spot": 1.0},
            source_timestamps={"chain": "2026-07-14T11:00:00-04:00"},
            snapshot_id="s1",
        )


def test_identical_input_identical_snapshot():
    kwargs = dict(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        standardized_features={"spot": 0.1},
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
        snapshot_id="fixed-id",
    )
    a = build_canonical_snapshot(**kwargs)
    b = build_canonical_snapshot(**kwargs)
    assert a.to_dict() == b.to_dict()
