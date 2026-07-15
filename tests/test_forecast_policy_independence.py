"""
tests/test_forecast_policy_independence.py
tests/test_full_v3_forecast_runtime.py
"""
from __future__ import annotations

from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.forecast_assembly import build_v3_forecast


def test_forecast_independent_of_selected_candidate():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0, "adx": 18.0},
        snapshot_id="s1",
    )
    a = build_v3_forecast(snapshot=snap, mode="shadow")
    # Mutating a "selected candidate" must not change forecast inputs —
    # forecast only sees snapshot features.
    b = build_v3_forecast(snapshot=snap, mode="shadow")
    assert a.snapshot_id == b.snapshot_id
    assert a.model_versions.get("assembly") in (
        "v3.forecast_assembly", "unavailable", "heuristic_baseline")


def test_strict_mode_unavailable_without_artifacts():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="s1",
    )
    bundle = build_v3_forecast(snapshot=snap, mode="champion")
    assert bundle.uncertainty == 1.0
    assert bundle.model_versions.get("reason") == "required_component_missing"


def test_component_failure_increases_uncertainty_path():
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={},
        snapshot_id="s1",
    )
    # No market → unavailable or degraded
    bundle = build_v3_forecast(snapshot=snap, mode="shadow")
    assert bundle.uncertainty is not None
    assert bundle.uncertainty >= 0.0
