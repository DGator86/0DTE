"""
tests/test_structural_state_v3.py
=================================
V3 Part 2 PR7 — StructuralState contract, disagreement, concentration,
compatibility fallbacks (§40).
"""
from __future__ import annotations

import math

import pytest

from prediction.structural_state import (
    STRUCTURAL_STATE_VERSION,
    StructuralState,
    concentration_metrics,
    gex_disagreement,
    gex_sign_agreement,
    multi_variant_disagreement_stats,
)


def test_version_constant():
    assert STRUCTURAL_STATE_VERSION == "v3.0.0"
    s = StructuralState(ts="t", symbol="SPY", spot=600.0)
    assert s.version == "v3.0.0"


def test_gex_disagreement_bounded():
    d = gex_disagreement(1e9, -1e9)
    assert d is not None
    assert 0.0 <= d <= 1.0
    assert d == pytest.approx(1.0, abs=1e-6)
    assert gex_disagreement(1e9, 1e9) == pytest.approx(0.0)
    assert gex_disagreement(None, 1e9) is None
    assert gex_disagreement(1e9, None) is None
    assert gex_disagreement(float("nan"), 1.0) is None


def test_sign_agreement_bounded():
    a = gex_sign_agreement([1.0, 2.0, -3.0])
    assert a is not None and 0.0 <= a <= 1.0
    assert gex_sign_agreement([1.0]) is None
    assert gex_sign_agreement([None, 1.0]) is None
    assert gex_sign_agreement([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_concentration_metrics_bounded():
    m = concentration_metrics({100.0: 10.0, 101.0: 5.0, 102.0: 1.0,
                               103.0: 1.0, 104.0: 1.0, 105.0: 1.0})
    assert 0.0 <= m["gex_concentration"] <= 1.0
    assert 0.0 <= m["gex_hhi"] <= 1.0
    assert 0.0 <= m["largest_strike_share"] <= 1.0
    assert 0.0 <= m["top_three_strike_share"] <= 1.0
    assert m["largest_strike_share"] == pytest.approx(10.0 / 19.0)
    empty = concentration_metrics({})
    assert empty["gex_concentration"] is None


def test_missing_variants_remain_missing():
    s = StructuralState(
        ts="2026-07-14T14:00:00Z", symbol="SPY", spot=600.0,
        net_gex_oi=1e9, gamma_flip_oi=598.0,
        call_wall_oi=610.0, put_wall_oi=590.0,
        # volume / hybrid intentionally absent
    )
    assert s.net_gex_volume is None
    assert s.gamma_flip_hybrid is None
    assert s.net_gex == pytest.approx(1e9)  # oi fallback after hybrid miss
    assert s.compatibility_provenance()["net_gex"] == "oi"


def test_compatibility_fallback_order_hybrid_first():
    s = StructuralState(
        ts="t", symbol="SPY", spot=600.0,
        net_gex_oi=1.0, net_gex_volume=2.0, net_gex_hybrid=3.0,
        gamma_flip_oi=590.0, gamma_flip_volume=591.0, gamma_flip_hybrid=592.0,
        call_wall_oi=610.0, call_wall_volume=611.0, call_wall_hybrid=612.0,
        put_wall_oi=580.0, put_wall_volume=581.0, put_wall_hybrid=582.0,
        diagnostics={"fallback_order": ["hybrid", "oi", "volume"]},
    )
    assert s.net_gex == 3.0
    assert s.gamma_flip == 592.0
    assert s.call_wall == 612.0
    assert s.put_wall == 582.0
    assert s.compatibility_provenance()["net_gex"] == "hybrid"


def test_compatibility_does_not_use_zero_for_missing():
    s = StructuralState(ts="t", symbol="SPY", spot=600.0)
    assert s.net_gex is None
    assert s.gamma_flip is None
    # Legacy conversion may use 0.0 but notes provenance
    legacy = s.to_legacy_policy_state()
    assert legacy.net_gex == 0.0
    assert "net_gex_src=None" in legacy.notes


def test_multi_variant_stats():
    stats = multi_variant_disagreement_stats(
        {"oi": 1.0, "volume": 1.5, "hybrid": 1.2})
    assert stats["n_variants"] == 3
    assert 0.0 <= stats["max_normalized_pairwise_difference"] <= 1.0
    assert stats["median_variant"] == pytest.approx(1.2)


def test_to_dict_roundtrip():
    s = StructuralState(
        ts="2026-07-14T14:00:00Z", symbol="SPY", spot=600.0,
        net_gex_oi=1e9, gamma_flip_oi=598.0,
        call_wall_oi=610.0, put_wall_oi=590.0,
        gex_disagreement=0.2, quality_score=0.8,
        source_ages={"oi": 1.5},
        diagnostics={"fallback_order": ["hybrid", "oi", "volume"]},
    )
    d = s.to_dict()
    assert d["version"] == "v3.0.0"
    assert d["net_gex"] == pytest.approx(1e9)
    s2 = StructuralState.from_dict(d)
    assert s2.net_gex_oi == s.net_gex_oi
    assert s2.source_ages == {"oi": 1.5}


def test_policy_from_v3_structural():
    from policy.contracts import StructuralState as Legacy

    s = StructuralState(
        ts="t", symbol="SPY", spot=601.0,
        net_gex_hybrid=4e9, gamma_flip_hybrid=599.0,
        call_wall_hybrid=610.0, put_wall_hybrid=590.0,
        gex_percentile=0.7,
        diagnostics={"fallback_order": ["hybrid", "oi", "volume"]},
    )
    legacy = Legacy.from_v3_structural(s)
    assert legacy.spot == 601.0
    assert legacy.net_gex == pytest.approx(4e9)
    assert "hybrid" in legacy.notes
