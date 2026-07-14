"""
tests/test_part2_shadow_integration.py
======================================
V3 Part 2 PR16 — shadow sequence failure isolation (§37).
"""
from __future__ import annotations

import pytest

from prediction.contracts import PredictionBundle
from prediction.models.regime_moe import RegimeProbabilities
from prediction.part2_shadow import run_part2_shadow_tick
from prediction.storage import PredictionStore


def _bundle():
    return PredictionBundle(
        snapshot_id="snap-p2", ts="2026-07-14T15:00:00Z",
        session_date="2026-07-14", symbol="SPY",
        p_up_30m=0.55, uncertainty=0.2,
    )


def test_shadow_builds_structural_and_persists(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "p2.sqlite"))
    result = run_part2_shadow_tick(
        base_bundle=_bundle(),
        spot=600.0, symbol="SPY", ts="2026-07-14T15:00:00Z",
        current_sources={
            "oi": {"net_gex": 1e9, "gamma_flip": 598.0,
                   "call_wall": 610.0, "put_wall": 590.0},
            "volume": {"net_gex": 0.8e9, "gamma_flip": 597.0,
                       "call_wall": 609.0, "put_wall": 591.0},
        },
        expected_remaining_move=2.0,
        store=store,
    )
    assert result.structural_state is not None
    assert result.bundle.structural_state_version == "v3.0.0"
    assert store.fetch_structural_state("snap-p2") is not None
    assert result.errors == []


def test_regime_attach_and_persist(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "p2.sqlite"))

    def fake_regime(_row):
        return RegimeProbabilities(
            long_gamma_pin=0.4, short_gamma_trend=0.3,
            flip_transition=0.2, volatility_expansion=0.1,
            uncertainty=0.35, dominant_regime="long_gamma_pin",
            class_support={}, calibrated=True, model_version="test-regime",
        )

    result = run_part2_shadow_tick(
        base_bundle=_bundle(),
        spot=600.0, symbol="SPY", ts="2026-07-14T15:00:00Z",
        current_sources={"oi": {"net_gex": 1e9, "gamma_flip": 598.0,
                                "call_wall": 610.0, "put_wall": 590.0}},
        store=store,
        regime_predict=fake_regime,
        feature_row={"f": 1.0},
    )
    assert result.bundle.dominant_regime == "long_gamma_pin"
    assert abs(sum(result.bundle.regime_probabilities.values()) - 1.0) <= 1e-9
    assert store.fetch_regime_outputs("snap-p2")


def test_failure_isolation_does_not_crash(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "p2.sqlite"))

    def boom(_row):
        raise RuntimeError("regime exploded")

    result = run_part2_shadow_tick(
        base_bundle=_bundle(),
        spot=600.0, symbol="SPY", ts="2026-07-14T15:00:00Z",
        current_sources={"oi": {"net_gex": 1e9, "gamma_flip": 598.0,
                                "call_wall": 610.0, "put_wall": 590.0}},
        store=store,
        regime_predict=boom,
        feature_row={"f": 1.0},
    )
    assert result.bundle.snapshot_id == "snap-p2"
    assert any(e["stage"] == "regime_probabilities" for e in result.errors)
    assert "part2_errors" in result.bundle.diagnostics
    # Legacy fields preserved
    assert result.bundle.p_up_30m == pytest.approx(0.55)
