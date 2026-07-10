"""
Session-level bootstrap confidence intervals (validation/bootstrap.py, PR 1
of Prediction Engine V2): deterministic given the seed, resamples complete
sessions, and degrades honestly with tiny samples.
"""
import pytest

from validation.bootstrap import bootstrap_ci, session_bootstrap


def test_deterministic_given_seed():
    vals = [0.5, -0.2, 1.1, 0.0, -0.7, 0.3, 0.9, -0.1]
    a = bootstrap_ci(vals, seed=123)
    b = bootstrap_ci(vals, seed=123)
    assert a == b
    c = bootstrap_ci(vals, seed=124)
    assert (c["ci_low"], c["ci_high"]) != (a["ci_low"], a["ci_high"])


def test_ci_brackets_point_estimate():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = bootstrap_ci(vals)
    assert out["n"] == 6
    assert out["stat"] == pytest.approx(3.5)
    assert out["ci_low"] <= out["stat"] <= out["ci_high"]
    assert out["ci_low"] < out["ci_high"]
    # bootstrap means of a sample can never leave the sample's range
    assert out["ci_low"] >= 1.0 and out["ci_high"] <= 6.0


def test_empty_and_singleton_samples():
    out = bootstrap_ci([])
    assert out["n"] == 0
    assert out["stat"] is None and out["ci_low"] is None

    out = bootstrap_ci([0.42])
    assert out["n"] == 1
    # one session = one observation: the interval collapses to the point
    assert out["stat"] == out["ci_low"] == out["ci_high"] == pytest.approx(0.42)


def test_session_bootstrap_order_independent():
    by_date = {"2026-06-03": 0.3, "2026-06-01": -0.1, "2026-06-02": 0.2}
    reordered = {d: by_date[d] for d in ["2026-06-01", "2026-06-02", "2026-06-03"]}
    a = session_bootstrap(by_date)
    b = session_bootstrap(reordered)
    assert a == b
    assert a["n_sessions"] == 3
    assert a["stat"] == pytest.approx((0.3 - 0.1 + 0.2) / 3, abs=1e-6)


def test_wider_spread_wider_interval():
    tight = bootstrap_ci([1.0, 1.1, 0.9, 1.05, 0.95] * 4, seed=7)
    wide = bootstrap_ci([5.0, -3.0, 4.0, -2.0, 1.0] * 4, seed=7)
    assert (wide["ci_high"] - wide["ci_low"]) > (tight["ci_high"] - tight["ci_low"])
