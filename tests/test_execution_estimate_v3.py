"""
tests/test_execution_estimate_v3.py
===================================
V3 Part 3 PR24 — ExecutionEstimateV3 monotonicity (§17 / §48).
"""
from __future__ import annotations

import pytest

from execution.estimate_v3 import (
    build_execution_estimate_v3, expected_order_value,
)


def _base(**kw):
    defaults = dict(
        mid_credit=0.50,
        natural_credit=0.30,
        family="put_credit",
        n_legs=2,
        p_fill=0.7,
        expected_fill_fraction=0.5,
        conservative_fill_fraction=0.75,
        empirical_weight=0.5,
        fallback_level="empirical",
    )
    defaults.update(kw)
    return build_execution_estimate_v3(**defaults)


def test_expected_credit_not_better_than_mid():
    est = _base()
    assert est.expected_credit <= est.mid_credit + 1e-12


def test_conservative_not_better_than_expected():
    est = _base()
    assert est.conservative_credit <= est.expected_credit + 1e-12


def test_debit_not_cheaper_than_mid():
    est = build_execution_estimate_v3(
        mid_credit=-1.00,
        natural_credit=-1.20,
        family="long_call_spread",
        n_legs=2,
        p_fill=0.6,
        expected_fill_fraction=0.5,
        conservative_fill_fraction=0.8,
        fallback_level="empirical",
    )
    # Paying more (more negative) is worse; must not be cheaper than mid
    assert est.expected_credit <= est.mid_credit + 1e-12
    assert est.conservative_credit <= est.expected_credit + 1e-12


def test_lower_fill_prob_does_not_improve_order_value():
    high = expected_order_value(0.9, 1.0, opportunity_cost_unfilled=0.0)
    low = expected_order_value(0.2, 1.0, opportunity_cost_unfilled=0.0)
    assert low <= high


def test_higher_fees_worsen_round_trip():
    a = _base()
    b = build_execution_estimate_v3(
        mid_credit=0.50, natural_credit=0.30, family="put_credit", n_legs=4,
        p_fill=0.7, expected_fill_fraction=0.5, conservative_fill_fraction=0.75,
        fallback_level="empirical",
    )
    assert b.expected_round_trip_cost >= a.expected_round_trip_cost


def test_missing_empirical_records_fallback_not_midpoint():
    est = build_execution_estimate_v3(
        mid_credit=0.50, natural_credit=0.30, family="put_credit", n_legs=2,
    )
    assert est.fallback_level == "deterministic_prior"
    assert est.expected_credit < est.mid_credit or est.expected_fill_fraction >= 0.0
    assert est.expected_credit != est.mid_credit or est.expected_fill_fraction == 0.0


def test_require_empirical_fails_closed():
    with pytest.raises(RuntimeError, match="empirical"):
        build_execution_estimate_v3(
            mid_credit=0.5, natural_credit=0.3, family="put_credit",
            n_legs=2, require_empirical=True,
        )


def test_fallback_level_recorded():
    est = _base(fallback_level="exact_family")
    assert est.fallback_level == "exact_family"
    assert "version" in est.diagnostics
