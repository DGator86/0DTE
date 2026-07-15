"""
tests/test_feed_status_and_canonical_snapshot.py
================================================
PR B — FeedStatus + CanonicalSnapshot contracts
(unified integration handoff §6.1 / §6.2 / §18 PR B).

No authority / orchestrator wiring. Offline, seeded, deterministic.
"""
from __future__ import annotations

import copy

import pytest

from prediction.canonical_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    CanonicalSnapshotError,
    build_canonical_snapshot,
    configuration_hash_for,
)
from prediction.dataset import FEATURE_VERSION, make_snapshot_id
from prediction.feed_status import (
    build_feed_status,
    classify_feed_status,
    overall_feed_status,
)


def _feeds(**overrides):
    """Four required sources, all LIVE unless overridden."""
    base = {
        "spot": build_feed_status(
            source="spot", provider="Tradier",
            observed_at="2026-07-15T10:00:00-04:00",
            received_at="2026-07-15T10:00:00.200-04:00",
            age_seconds=0.5, freshness_limit_seconds=5.0, required=True,
        ),
        "bars": build_feed_status(
            source="bars", provider="Massive",
            observed_at="2026-07-15T09:59:00-04:00",
            received_at="2026-07-15T10:00:00-04:00",
            age_seconds=4.0, freshness_limit_seconds=90.0, required=True,
        ),
        "option_chain": build_feed_status(
            source="option_chain", provider="Tradier",
            observed_at="2026-07-15T10:00:00-04:00",
            received_at="2026-07-15T10:00:00.100-04:00",
            age_seconds=1.0, freshness_limit_seconds=15.0, required=True,
        ),
        "settlement": build_feed_status(
            source="settlement", provider="Yahoo",
            observed_at="2026-07-14T16:00:00-04:00",
            received_at="2026-07-15T09:30:00-04:00",
            age_seconds=20.0, freshness_limit_seconds=86_400.0, required=True,
        ),
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# FeedStatus classification                                                    #
# --------------------------------------------------------------------------- #
def test_classify_live_delayed_stale_missing_invalid_fallback():
    assert classify_feed_status(
        age_seconds=0.5, freshness_limit_seconds=5.0) == "LIVE"
    assert classify_feed_status(
        age_seconds=3.0, freshness_limit_seconds=5.0) == "DELAYED"
    assert classify_feed_status(
        age_seconds=6.0, freshness_limit_seconds=5.0) == "STALE"
    assert classify_feed_status(
        age_seconds=None, freshness_limit_seconds=5.0,
        present=False) == "MISSING"
    assert classify_feed_status(
        age_seconds=0.1, freshness_limit_seconds=5.0,
        valid=False) == "INVALID"
    assert classify_feed_status(
        age_seconds=0.1, freshness_limit_seconds=5.0,
        is_fallback=True) == "FALLBACK"
    # Age unknown but present → not LIVE
    assert classify_feed_status(
        age_seconds=None, freshness_limit_seconds=5.0,
        present=True) == "DELAYED"


def test_overall_live_only_when_all_required_live():
    feeds = _feeds()
    assert overall_feed_status(feeds) == "LIVE"

    stale = _feeds(spot=build_feed_status(
        source="spot", age_seconds=30.0, freshness_limit_seconds=5.0,
        required=True, provider="Tradier",
        observed_at="2026-07-15T09:59:30-04:00",
        received_at="2026-07-15T10:00:00-04:00",
    ))
    assert overall_feed_status(stale) == "STALE"

    missing = _feeds(option_chain=build_feed_status(
        source="option_chain", freshness_limit_seconds=15.0,
        present=False, required=True,
    ))
    assert overall_feed_status(missing) == "MISSING"

    invalid = _feeds(bars=build_feed_status(
        source="bars", age_seconds=1.0, freshness_limit_seconds=90.0,
        valid=False, required=True,
    ))
    assert overall_feed_status(invalid) == "INVALID"

    delayed = _feeds(spot=build_feed_status(
        source="spot", age_seconds=3.0, freshness_limit_seconds=5.0,
        required=True,
    ))
    assert overall_feed_status(delayed) == "DEGRADED"

    fallback = _feeds(settlement=build_feed_status(
        source="settlement", age_seconds=10.0,
        freshness_limit_seconds=86_400.0, is_fallback=True, required=True,
    ))
    assert overall_feed_status(fallback) == "DEGRADED"


def test_feed_status_roundtrip_dict():
    fs = build_feed_status(
        source="spot", provider="Tradier", age_seconds=0.8,
        freshness_limit_seconds=5.0, observed_at="t0", received_at="t1",
    )
    d = fs.to_dict()
    assert d["status"] == "LIVE"
    back = type(fs).from_dict(d)
    assert back == fs


# --------------------------------------------------------------------------- #
# CanonicalSnapshot                                                            #
# --------------------------------------------------------------------------- #
def test_deterministic_snapshot_id_and_hash():
    kwargs = dict(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 600.0, "adx": None, "net_gex": 1e9},
        source_timestamps={
            "spot": "2026-07-15T10:00:00-04:00",
            "bars": "2026-07-15T09:59:00-04:00",
        },
        source_ages_seconds={"spot": 0.5, "bars": 60.0},
        feed_statuses=_feeds(),
        quality={"feature_coverage": 0.9},
        source_seq=0,
    )
    a = build_canonical_snapshot(**kwargs)
    b = build_canonical_snapshot(**kwargs)
    assert a.snapshot_id == b.snapshot_id
    assert a.snapshot_hash() == b.snapshot_hash()
    assert a.to_dict() == b.to_dict()
    # Matches dataset identity helper
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ts = datetime.fromisoformat(kwargs["ts"])
    assert a.snapshot_id == make_snapshot_id(
        "SPY", ts, FEATURE_VERSION, 0)


def test_missingness_preserved_never_coerced_to_zero():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 600.0, "gamma_flip": None, "call_wall": None},
        feed_statuses=_feeds(),
    )
    assert snap.raw_features["gamma_flip"] is None
    assert snap.missingness["gamma_flip"] is True
    assert snap.missingness["call_wall"] is True
    assert snap.missingness["spot"] is False
    # Explicit missingness wins over derivation
    snap2 = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": None},
        missingness={"spot": False},  # caller asserted measured-zero path
        feed_statuses=_feeds(),
    )
    assert snap2.raw_features["spot"] is None
    assert snap2.missingness["spot"] is False


def test_stale_source_reflected_in_overall_status():
    feeds = _feeds(option_chain=build_feed_status(
        source="option_chain", provider="Tradier",
        observed_at="2026-07-15T09:59:00-04:00",
        received_at="2026-07-15T10:00:00-04:00",
        age_seconds=90.0, freshness_limit_seconds=15.0, required=True,
    ))
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 600.0},
        feed_statuses=feeds,
    )
    assert snap.feed_statuses["option_chain"].status == "STALE"
    assert snap.overall_feed_status() == "STALE"
    assert snap.to_dict()["overall_feed_status"] == "STALE"


def test_different_source_timestamps_change_snapshot_hash():
    base = dict(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 600.0},
        feed_statuses=_feeds(),
        source_seq=0,
    )
    a = build_canonical_snapshot(
        **base,
        source_timestamps={"bars": "2026-07-15T09:59:00-04:00"},
        source_ages_seconds={"bars": 60.0},
    )
    b = build_canonical_snapshot(
        **base,
        source_timestamps={"bars": "2026-07-15T09:58:00-04:00"},
        source_ages_seconds={"bars": 120.0},
    )
    assert a.snapshot_id == b.snapshot_id  # id from symbol|ts|fv|seq
    assert a.snapshot_hash() != b.snapshot_hash()


def test_different_feed_ages_change_snapshot_hash():
    a = build_canonical_snapshot(
        symbol="SPY", ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15", raw_features={"spot": 1.0},
        feed_statuses=_feeds(),
    )
    feeds_b = _feeds(spot=build_feed_status(
        source="spot", provider="Tradier",
        observed_at="2026-07-15T09:59:59-04:00",
        received_at="2026-07-15T10:00:00-04:00",
        age_seconds=1.5, freshness_limit_seconds=5.0, required=True,
    ))
    b = build_canonical_snapshot(
        symbol="SPY", ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15", raw_features={"spot": 1.0},
        feed_statuses=feeds_b,
    )
    assert a.snapshot_hash() != b.snapshot_hash()


def test_replay_identical_inputs_identical_outputs():
    kwargs = dict(
        symbol="IWM",
        ts="2026-07-15T11:30:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 754.81, "adx": 45.18, "vix": None},
        standardized_features={"adx": 45.0},
        source_timestamps={"spot": "2026-07-15T11:30:00-04:00"},
        source_ages_seconds={"spot": 0.2},
        feed_statuses=_feeds(),
        quality={"feature_coverage": 0.88},
        structural_sources={"gex": {"net_gex": 1.7e9}},
        source_seq=3,
    )
    first = build_canonical_snapshot(**kwargs)
    # Deep-copy inputs to prove no mutation / shared-state dependence
    second = build_canonical_snapshot(**copy.deepcopy(kwargs))
    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_hash() == second.snapshot_hash()
    assert first.configuration_hash == second.configuration_hash
    assert first.to_dict() == second.to_dict()


def test_rejects_future_dated_source_and_policy_fields():
    with pytest.raises(CanonicalSnapshotError, match="future-dated"):
        build_canonical_snapshot(
            symbol="SPY",
            ts="2026-07-15T10:00:00-04:00",
            session_date="2026-07-15",
            raw_features={"spot": 1.0},
            source_timestamps={"bars": "2026-07-15T10:00:01-04:00"},
            feed_statuses=_feeds(),
        )
    with pytest.raises(CanonicalSnapshotError, match="forbidden"):
        build_canonical_snapshot(
            symbol="SPY",
            ts="2026-07-15T10:00:00-04:00",
            session_date="2026-07-15",
            raw_features={"spot": 1.0, "selected_candidate_id": "x"},
            feed_statuses=_feeds(),
        )


def test_schema_version_and_configuration_hash():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={},
        feed_statuses=_feeds(),
        structural_state_version="ss.v1",
    )
    assert snap.schema_version == SNAPSHOT_SCHEMA_VERSION
    assert snap.configuration_hash == configuration_hash_for(
        feature_version=FEATURE_VERSION,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        structural_state_version="ss.v1",
    )
    # Changing structural version changes configuration hash
    other = configuration_hash_for(
        feature_version=FEATURE_VERSION,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        structural_state_version="ss.v2",
    )
    assert snap.configuration_hash != other


def test_frozen_immutable():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-15T10:00:00-04:00",
        session_date="2026-07-15",
        raw_features={"spot": 1.0},
        feed_statuses=_feeds(),
    )
    with pytest.raises(Exception):
        snap.symbol = "QQQ"  # type: ignore[misc]
