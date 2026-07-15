"""
tests/test_execution_economics.py
"""
from __future__ import annotations

from execution.estimate_v3 import build_execution_estimate_v3, expected_order_value


def test_midpoint_not_assumed_fill():
    est = build_execution_estimate_v3(
        mid_credit=1.0,
        natural_credit=0.85,
        family="put_credit",
        n_legs=2,
    )
    assert float(est.p_fill) < 1.0
    # Expected executable credit must concede toward natural — never mid.
    assert float(est.expected_credit) < float(est.mid_credit)
    assert float(est.expected_credit) >= float(est.natural_credit) - 1e-9
    assert est.fallback_level == "deterministic_prior"


def test_expected_order_value_accounts_for_unfilled():
    eov = expected_order_value(
        p_fill=0.5,
        expected_net_pnl_given_fill=10.0,
        opportunity_cost_unfilled=2.0,
    )
    assert eov == 0.5 * 10.0 - 0.5 * 2.0
