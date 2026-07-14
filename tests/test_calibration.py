"""
tests/test_calibration.py
=========================
PR 4 acceptance — probability calibration:
  * outputs always within [0, 1];
  * Platt/sigmoid is the default and repairs miscalibrated scores;
  * isotonic is GATED behind sample-size and session-count minimums;
  * calibrators are deterministic and monotone;
  * degenerate (one-class) calibration sets never crash.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.calibration import (IdentityCalibrator, IsotonicCalibrator,
                                    SigmoidCalibrator,
                                    calibration_slope_intercept,
                                    fit_calibrator, reliability_bins,
                                    select_calibrator)
from prediction.models.base import brier_score

RNG = np.random.default_rng(11)


def _overconfident_sample(n=4000):
    """True probabilities are mild; raw scores are pushed toward 0/1."""
    p_true = RNG.uniform(0.35, 0.65, size=n)
    y = (RNG.uniform(size=n) < p_true).astype(int)
    logit = np.log(p_true / (1 - p_true))
    p_raw = 1.0 / (1.0 + np.exp(-4.0 * logit))       # overconfident distortion
    return p_raw, y


class TestSigmoid:
    def test_repairs_overconfidence(self):
        p_raw, y = _overconfident_sample()
        cal = SigmoidCalibrator().fit(p_raw, y)
        assert brier_score(y, cal.transform(p_raw)) < brier_score(y, p_raw)

    def test_bounds(self):
        p_raw, y = _overconfident_sample(500)
        p = SigmoidCalibrator().fit(p_raw, y).transform(
            np.array([0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0]))
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_deterministic(self):
        p_raw, y = _overconfident_sample(800)
        a = SigmoidCalibrator().fit(p_raw, y).transform(p_raw)
        b = SigmoidCalibrator().fit(p_raw, y).transform(p_raw)
        assert np.array_equal(a, b)

    def test_monotone(self):
        p_raw, y = _overconfident_sample(800)
        cal = SigmoidCalibrator().fit(p_raw, y)
        grid = np.linspace(0.01, 0.99, 50)
        out = cal.transform(grid)
        assert np.all(np.diff(out) >= 0)

    def test_one_class_calibration_set_is_identity(self):
        cal = SigmoidCalibrator().fit(np.array([0.6, 0.7]), np.array([1, 1]))
        p = cal.transform(np.array([0.3, 0.9]))
        assert np.all((p >= 0.0) & (p <= 1.0))       # no crash, bounded

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            SigmoidCalibrator().transform([0.5])


class TestIsotonic:
    def test_monotone_and_bounded(self):
        p_raw, y = _overconfident_sample(3000)
        cal = IsotonicCalibrator().fit(p_raw, y)
        grid = np.linspace(0.0, 1.0, 100)
        out = cal.transform(grid)
        assert np.all(np.diff(out) >= -1e-12)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_improves_brier_on_large_sample(self):
        p_raw, y = _overconfident_sample(5000)
        cal = IsotonicCalibrator().fit(p_raw, y)
        assert brier_score(y, cal.transform(p_raw)) < brier_score(y, p_raw)


class TestSelection:
    def test_small_sample_stays_sigmoid(self):
        p_raw, y = _overconfident_sample(200)
        cal, diag = select_calibrator(p_raw, y, n_sessions=5)
        assert isinstance(cal, SigmoidCalibrator)
        assert diag["chosen"] == "sigmoid"
        assert diag["brier_isotonic"] is None        # gate never evaluated it

    def test_few_sessions_stays_sigmoid_even_with_many_rows(self):
        p_raw, y = _overconfident_sample(5000)
        cal, diag = select_calibrator(p_raw, y, n_sessions=10)
        assert diag["chosen"] == "sigmoid"

    def test_isotonic_possible_when_gates_pass(self):
        p_raw, y = _overconfident_sample(5000)
        # Many synthetic "sessions" so nested session holdout works
        sessions = [f"s{i % 60:02d}" for i in range(len(y))]
        cal, diag = select_calibrator(p_raw, y, n_sessions=60,
                                      sessions=sessions,
                                      min_samples=1000, min_sessions=40)
        assert diag["brier_isotonic"] is not None    # gate opened
        assert diag["chosen"] in ("sigmoid", "isotonic")
        assert diag["comparison"] == "nested_holdout"
        # Winner was selected on nested eval, then refit on all rows
        assert np.all((cal.transform(p_raw) >= 0.0)
                      & (cal.transform(p_raw) <= 1.0))

    def test_fit_calibrator_by_name(self):
        p_raw, y = _overconfident_sample(300)
        for name, cls in (("sigmoid", SigmoidCalibrator),
                          ("isotonic", IsotonicCalibrator),
                          ("identity", IdentityCalibrator)):
            assert isinstance(fit_calibrator(p_raw, y, name), cls)
        with pytest.raises(ValueError):
            fit_calibrator(p_raw, y, "quantum")


class TestDiagnostics:
    def test_reliability_bins_sum_to_n(self):
        p_raw, y = _overconfident_sample(1000)
        bins = reliability_bins(p_raw, y, n_bins=10)
        assert sum(b["n"] for b in bins) == 1000
        for b in bins:
            assert 0.0 <= b["mean_predicted"] <= 1.0
            assert 0.0 <= b["realized_rate"] <= 1.0

    def test_slope_near_one_for_honest_probabilities(self):
        p_true = RNG.uniform(0.05, 0.95, size=20000)
        y = (RNG.uniform(size=len(p_true)) < p_true).astype(int)
        d = calibration_slope_intercept(p_true, y)
        assert d["slope"] == pytest.approx(1.0, abs=0.1)
        assert d["intercept"] == pytest.approx(0.0, abs=0.1)

    def test_one_class_slope_is_none(self):
        d = calibration_slope_intercept(np.array([0.5, 0.6]),
                                        np.array([1, 1]))
        assert d["slope"] is None
