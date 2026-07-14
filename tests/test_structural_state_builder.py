"""
tests/test_structural_state_builder.py
======================================
V3 Part 2 PR7 — StructuralStateBuilder as-of-safety, velocity, stability,
invalid geometry, determinism (§40).
"""
from __future__ import annotations

import copy

import pytest

from prediction.structural_state import (
    StructuralStateBuilder,
    StructuralStateConfig,
    sources_from_gex_bundle,
)


def _sources(**kwargs):
    base = {
        "oi": {
            "net_gex": 2e9, "gamma_flip": 598.0,
            "call_wall": 610.0, "put_wall": 590.0,
            "abs_gamma_by_strike": {
                590.0: 1.0, 595.0: 2.0, 600.0: 5.0,
                605.0: 3.0, 610.0: 4.0, 615.0: 1.0,
            },
        },
        "volume": {
            "net_gex": 1.5e9, "gamma_flip": 597.5,
            "call_wall": 609.0, "put_wall": 591.0,
        },
        "hybrid": {
            "net_gex": 1.8e9, "gamma_flip": 597.8,
            "call_wall": 609.5, "put_wall": 590.5,
        },
    }
    base.update(kwargs)
    return base


def test_builder_preserves_variants():
    b = StructuralStateBuilder()
    state = b.build(
        ts="2026-07-14T15:00:00Z",
        symbol="SPY",
        spot=600.0,
        expected_remaining_move=2.0,
        current_sources=_sources(),
        historical_states=[],
        source_ages={"oi": 2.0, "volume": 5.0},
        source_versions={"oi": "v1", "volume": "v1"},
        gex_percentile=0.65,
    )
    assert state.net_gex_oi == pytest.approx(2e9)
    assert state.net_gex_volume == pytest.approx(1.5e9)
    assert state.net_gex_hybrid == pytest.approx(1.8e9)
    assert state.gex_disagreement is not None
    assert 0.0 <= state.gex_disagreement <= 1.0
    assert state.gex_sign_agreement is not None
    assert state.gex_concentration is not None
    assert state.gex_hhi is not None
    assert state.quality_score > 0
    assert state.source_ages["oi"] == 2.0


def test_missing_variant_not_imputed():
    src = _sources()
    del src["volume"]
    state = StructuralStateBuilder().build(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0, current_sources=src,
        historical_states=[],
    )
    assert state.net_gex_volume is None
    assert state.gex_disagreement is None  # needs both oi and volume
    assert "source:volume" in state.diagnostics["missing_inputs"]


def test_velocity_uses_only_prior_observations():
    hist = [
        {"ts": "2026-07-14T14:55:00Z", "gamma_flip": 595.0,
         "call_wall": 608.0, "put_wall": 592.0},
        # Future relative to observation — must be ignored
        {"ts": "2026-07-14T15:05:00Z", "gamma_flip": 700.0,
         "call_wall": 720.0, "put_wall": 500.0},
    ]
    state = StructuralStateBuilder().build(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0, current_sources=_sources(),
        historical_states=hist,
    )
    # flip moved 598 (oi) via hybrid fallback 597.8 from 595 → positive vel
    assert state.flip_velocity_5m is not None
    assert state.flip_velocity_5m > 0
    # Future 700 must not dominate
    assert state.flip_velocity_5m < 0.1  # (597.8-595)/600 ≈ 0.0047


def test_future_history_cannot_alter_state():
    base_hist = [
        {"ts": "2026-07-14T14:55:00Z", "gamma_flip": 595.0,
         "call_wall": 608.0, "put_wall": 592.0},
    ]
    b = StructuralStateBuilder()
    kwargs = dict(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0, current_sources=_sources(),
    )
    a = b.build(historical_states=base_hist, **kwargs)
    polluted = base_hist + [
        {"ts": "2026-07-14T15:10:00Z", "gamma_flip": 999.0,
         "call_wall": 999.0, "put_wall": 1.0},
    ]
    c = b.build(historical_states=polluted, **kwargs)
    assert a.flip_velocity_5m == c.flip_velocity_5m
    assert a.call_wall_velocity_5m == c.call_wall_velocity_5m
    assert a.flip_stability == c.flip_stability


def test_stability_uses_prior_only():
    hist = [
        {"ts": "2026-07-14T14:50:00Z", "gamma_flip": 597.0},
        {"ts": "2026-07-14T14:55:00Z", "gamma_flip": 597.5},
    ]
    state = StructuralStateBuilder().build(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0, current_sources=_sources(),
        historical_states=hist,
    )
    assert state.flip_stability is not None
    assert 0.0 <= state.flip_stability <= 1.0


def test_invalid_geometry_recorded():
    src = _sources()
    src["oi"]["call_wall"] = 580.0  # below put wall 590
    src["oi"]["put_wall"] = 590.0
    state = StructuralStateBuilder().build(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=-1.0,  # invalid
        current_sources=src, historical_states=[],
    )
    inv = state.diagnostics["invalid_geometry"]
    assert any("call_wall_below_put_wall" in x for x in inv)
    assert "negative_expected_remaining_move" in inv
    assert state.distance_to_flip_expected_move is None


def test_non_finite_spot_flagged():
    state = StructuralStateBuilder().build(
        ts="t", symbol="SPY", spot=float("nan"),
        expected_remaining_move=2.0, current_sources=_sources(),
        historical_states=[],
    )
    assert "non_finite_spot" in state.diagnostics["invalid_geometry"]


def test_builder_deterministic():
    b = StructuralStateBuilder(StructuralStateConfig())
    kwargs = dict(
        ts="2026-07-14T15:00:00Z", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0,
        current_sources=_sources(),
        historical_states=[
            {"ts": "2026-07-14T14:55:00Z", "gamma_flip": 595.0,
             "call_wall": 608.0, "put_wall": 592.0},
        ],
        source_ages={"oi": 1.0},
        gex_percentile=0.5,
    )
    a = b.build(**kwargs)
    b2 = b.build(**copy.deepcopy(kwargs))
    assert a.to_dict() == b2.to_dict()


def test_expected_move_normalization():
    state = StructuralStateBuilder().build(
        ts="t", symbol="SPY", spot=600.0,
        expected_remaining_move=2.0, current_sources=_sources(),
        historical_states=[],
    )
    # hybrid flip 597.8 → (600-597.8)/2
    assert state.distance_to_flip_expected_move == pytest.approx(
        (600.0 - 597.8) / 2.0)
    assert state.wall_channel_width_expected_move == pytest.approx(
        (609.5 - 590.5) / 2.0)


def test_sources_from_gex_bundle():
    from gex.contracts import (
        GEXSnapshot, GexAssumption, GexVariantBundle, GexVariantId,
    )

    def snap(variant, net):
        return GEXSnapshot(
            net_gex=net, gamma_flip=598.0, call_wall=610.0, put_wall=590.0,
            gex_concentration=0.3, wall_concentration=0.2, quality_score=0.9,
            assumption_set=GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS,
            variant=variant,
        )

    bundle = GexVariantBundle(
        spot=600.0, authoritative=GexVariantId.OI,
        oi=snap(GexVariantId.OI, 2e9),
        volume=snap(GexVariantId.VOLUME, 1e9),
        hybrid=snap(GexVariantId.HYBRID, 1.5e9),
    )
    src = sources_from_gex_bundle(bundle)
    assert src["oi"]["net_gex"] == pytest.approx(2e9)
    assert src["volume"]["net_gex"] == pytest.approx(1e9)
    state = StructuralStateBuilder().build(
        ts="t", symbol="SPY", spot=600.0, expected_remaining_move=1.0,
        current_sources=src, historical_states=[],
    )
    assert state.net_gex_oi is not None
