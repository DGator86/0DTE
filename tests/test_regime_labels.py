"""
tests/test_regime_labels.py
===========================
V3 Part 2 PR8 — future-behavior regime labels (§41).
"""
from __future__ import annotations

import pytest

from prediction.regime_labels import (
    REGIME_CLASSES,
    RegimeLabelConfig,
    compute_path_stats,
    directional_efficiency,
    label_regime,
    pullback_fraction,
    reversion_count,
    total_path_variation,
)


def test_regime_classes():
    assert REGIME_CLASSES == (
        "long_gamma_pin",
        "short_gamma_trend",
        "flip_transition",
        "volatility_expansion",
    )


def test_directional_efficiency_clean_vs_oscillatory():
    clean = [100.0, 101.0, 102.0, 103.0, 104.0]
    assert directional_efficiency(clean) == pytest.approx(1.0)
    osc = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.5]
    assert directional_efficiency(osc) < 0.3
    assert total_path_variation(osc) > abs(osc[-1] - osc[0])


def test_pullback_fraction_shallow_trend():
    # Steady up with tiny pullbacks
    prices = [100, 101, 100.8, 102, 101.7, 103]
    assert pullback_fraction(prices) < 0.2


def test_reversion_count_uses_frozen_refs_only():
    # Oscillate around 100
    prices = [100.0, 101.0, 99.0, 101.0, 99.0, 100.5]
    assert reversion_count(prices, [100.0]) >= 2
    # Future-updated flip must not be used — caller freezes refs
    assert reversion_count(prices, [None, None]) == 0


def test_pin_label_clean_oscillation():
    # Contained oscillatory path around flip=100, walls 97/103
    prices = [100.0, 100.4, 99.7, 100.3, 99.8, 100.2, 99.9, 100.1]
    # expected move large so move fraction stays small
    res = label_regime(
        prices,
        expected_remaining_move=5.0,
        frozen_gamma_flip=100.0,
        frozen_call_wall=103.0,
        frozen_put_wall=97.0,
        frozen_vwap=100.0,
        cfg=RegimeLabelConfig(pin_min_reversion_count=2),
    )
    assert res.is_pin_behavior
    assert res.regime_label == "long_gamma_pin"
    assert not res.is_trend_behavior


def test_trend_label_persistent_path():
    prices = [100.0, 100.5, 101.0, 101.4, 101.8, 102.3, 102.8, 103.2]
    res = label_regime(
        prices,
        expected_remaining_move=2.0,
        frozen_gamma_flip=99.5,
        frozen_call_wall=105.0,
        frozen_put_wall=95.0,
        frozen_vwap=100.0,
    )
    assert res.is_trend_behavior
    assert res.regime_label == "short_gamma_trend"
    assert res.path_stats.directional_efficiency >= 0.6


def test_transition_label_flip_cross():
    # Start below flip, cross and occupy both sides meaningfully
    prices = [99.0, 99.2, 99.5, 100.2, 100.8, 99.4, 100.5, 99.1]
    res = label_regime(
        prices,
        expected_remaining_move=1.5,
        frozen_gamma_flip=100.0,
        frozen_call_wall=105.0,
        frozen_put_wall=95.0,
        frozen_vwap=100.0,
    )
    assert res.path_stats.flip_crossed
    assert res.is_transition_behavior
    assert res.regime_label in ("flip_transition", "volatility_expansion")


def test_vol_expansion_two_sided():
    # Large two-sided swings, low directional efficiency
    prices = [100.0, 102.5, 97.5, 103.0, 96.5, 102.0, 97.0, 100.5]
    res = label_regime(
        prices,
        expected_remaining_move=2.0,
        frozen_gamma_flip=100.0,
        frozen_call_wall=110.0,
        frozen_put_wall=90.0,
    )
    assert res.is_vol_expansion_behavior
    assert res.regime_label == "volatility_expansion"


def test_ambiguous_remains_unlabeled():
    # Tiny drift — meets no class thresholds
    prices = [100.0, 100.05, 100.02, 100.08, 100.04]
    res = label_regime(
        prices,
        expected_remaining_move=3.0,
        frozen_gamma_flip=100.0,
        frozen_call_wall=105.0,
        frozen_put_wall=95.0,
        cfg=RegimeLabelConfig(pin_min_reversion_count=5),
    )
    assert res.regime_label is None
    assert res.diagnostics["unclassified"] is True


def test_precedence_vol_over_transition():
    # Path that could look like transition AND vol expansion
    prices = [100.0, 103.0, 96.0, 104.0, 95.5, 103.5, 96.5, 100.2]
    res = label_regime(
        prices,
        expected_remaining_move=2.0,
        frozen_gamma_flip=100.0,
        frozen_call_wall=110.0,
        frozen_put_wall=90.0,
    )
    assert res.regime_label == "volatility_expansion"
    assert res.is_vol_expansion_behavior


def test_frozen_walls_not_future_updated():
    # If future walls moved, labeling must still use frozen levels
    prices = [100.0, 100.3, 99.8, 100.2, 99.7, 100.1]
    frozen_cw, frozen_pw = 102.0, 98.0
    stats = compute_path_stats(
        prices,
        expected_remaining_move=4.0,
        frozen_gamma_flip=100.0,
        frozen_call_wall=frozen_cw,
        frozen_put_wall=frozen_pw,
    )
    assert stats.diagnostics["frozen_call_wall"] == frozen_cw
    assert stats.diagnostics["frozen_put_wall"] == frozen_pw
    assert not stats.call_wall_breached
    assert not stats.put_wall_breached


def test_no_candidate_fields_in_result():
    res = label_regime(
        [100.0, 100.1, 99.9, 100.05],
        expected_remaining_move=2.0,
        frozen_gamma_flip=100.0,
    )
    d = res.to_dict()
    assert "candidate" not in d
    assert "pnl" not in d
    assert "family" not in d
    assert "legs" not in d
    keys = set(d.keys())
    assert keys == {
        "regime_label", "is_pin_behavior", "is_trend_behavior",
        "is_transition_behavior", "is_vol_expansion_behavior",
        "label_version", "path_stats", "diagnostics",
    }


def test_current_gex_sign_is_feature_only():
    prices = [100.0, 100.4, 99.7, 100.3, 99.8, 100.2, 99.9, 100.1]
    a = label_regime(
        prices, expected_remaining_move=5.0, frozen_gamma_flip=100.0,
        frozen_call_wall=103.0, frozen_put_wall=97.0, frozen_vwap=100.0,
        current_gex_sign=1.0,
    )
    b = label_regime(
        prices, expected_remaining_move=5.0, frozen_gamma_flip=100.0,
        frozen_call_wall=103.0, frozen_put_wall=97.0, frozen_vwap=100.0,
        current_gex_sign=-1.0,
    )
    # Label must not depend on GEX sign
    assert a.regime_label == b.regime_label
    assert a.diagnostics["current_gex_sign"] == 1.0
    assert b.diagnostics["current_gex_sign"] == -1.0


def test_deterministic_precedence():
    prices = [100.0, 102.5, 97.5, 103.0, 96.5, 102.0, 97.0, 100.5]
    results = [
        label_regime(
            prices, expected_remaining_move=2.0, frozen_gamma_flip=100.0,
            frozen_call_wall=110.0, frozen_put_wall=90.0,
        ).regime_label
        for _ in range(5)
    ]
    assert len(set(results)) == 1
