"""
tests/test_fill_prior_blending.py
=================================
V3 Part 3 PR22/23 — prior/empirical blending (§15).
"""
from __future__ import annotations

import pytest

from prediction.fill_training import blend_with_prior, fallback_level, empirical_weight


def test_empirical_weight_formula():
    assert empirical_weight(100, prior_equivalent_support=100) == pytest.approx(0.5)
    assert empirical_weight(0, prior_equivalent_support=100) == pytest.approx(0.0)


def test_blend():
    b, w = blend_with_prior(0.8, 0.4, 100, prior_equivalent_support=100)
    assert w == pytest.approx(0.5)
    assert b == pytest.approx(0.6)


def test_fallback_hierarchy():
    assert fallback_level(family_support=30) == "exact_family"
    assert fallback_level(family_support=0, broad_family_support=30) == "broad_family"
    assert fallback_level(
        family_support=0, broad_family_support=0, leg_group_support=60
    ) == "leg_count_group"
    assert fallback_level(
        family_support=0, broad_family_support=0, leg_group_support=0,
        global_support=150,
    ) == "global_empirical"
    assert fallback_level(
        family_support=0, broad_family_support=0, leg_group_support=0,
        global_support=10,
    ) == "deterministic_prior"
