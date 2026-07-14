"""
tests/test_conformal_returns.py
===============================
V3 Part 2 PR12 — split conformal return intervals (§44).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.conformal import SplitConformalCalibrator
from prediction.return_distribution import ReturnDistribution


def test_correction_uses_calibration_only():
    y = np.array([0.0, 0.01, -0.01, 0.02, -0.015, 0.005])
    lo = np.full_like(y, -0.005)
    hi = np.full_like(y, 0.005)
    sessions = [f"S{i}" for i in range(len(y))]
    cal = SplitConformalCalibrator(nominal_coverage=0.9).fit(
        y, lo, hi, sessions)
    assert cal.fitted
    assert cal.support_rows == 6
    assert set(cal.calibration_sessions) == set(sessions)
    y2 = y * 3
    cal2 = SplitConformalCalibrator(nominal_coverage=0.9).fit(
        y2, lo, hi, sessions)
    assert cal2.correction >= cal.correction - 1e-12


def test_test_labels_cannot_change_correction():
    y = np.array([0.0, 0.01, -0.01, 0.02])
    lo = np.full(4, -0.005)
    hi = np.full(4, 0.005)
    sessions = ["A", "B", "C", "D"]
    cal = SplitConformalCalibrator(nominal_coverage=0.8).fit(
        y, lo, hi, sessions)
    before = cal.correction
    y[:] = 999
    assert cal.correction == before
    interval = cal.apply(-0.005, 0.005)
    assert interval.correction == before


def test_corrected_bounds_ordered():
    cal = SplitConformalCalibrator(nominal_coverage=0.9)
    cal.fit([0.02, -0.02, 0.015], [-0.01, -0.01, -0.01],
            [0.01, 0.01, 0.01], ["s1", "s2", "s3"])
    iv = cal.apply(-0.01, 0.01)
    assert iv.lower <= iv.upper


def test_ood_flagged_coverage_limited():
    cal = SplitConformalCalibrator(
        nominal_coverage=0.9, ood_threshold=0.5, ood_multiplier=2.0)
    cal.fit([0.02, -0.02], [-0.005, -0.005], [0.005, 0.005], ["a", "b"])
    normal = cal.apply(-0.005, 0.005, ood_score=0.1)
    ood = cal.apply(-0.005, 0.005, ood_score=0.9)
    assert ood.diagnostics["coverage_limited"] is True
    assert normal.diagnostics["coverage_limited"] is False
    assert ood.correction > normal.correction


def test_attach_to_distribution():
    dist = ReturnDistribution(
        horizon="30m",
        quantiles={0.05: -0.02, 0.5: 0.0, 0.95: 0.02},
        expected_return=0.0,
        variance=0.0001,
        conformal_intervals={},
        conformal_support_rows=0,
        conformal_support_sessions=0,
        uncertainty=0.1,
        ood_score=None,
        model_version="v3",
    )
    cal = SplitConformalCalibrator(nominal_coverage=0.9)
    cal.fit([0.03, -0.025, 0.01], [-0.02, -0.02, -0.02],
            [0.02, 0.02, 0.02], ["x", "y", "z"])
    out = cal.attach_to_distribution(dist, lower_q=0.05, upper_q=0.95)
    assert "nominal_90" in out.conformal_intervals
    lo, hi = out.conformal_intervals["nominal_90"]
    assert lo <= hi
    assert out.conformal_support_rows == 3
    assert dist.conformal_intervals == {}


def test_support_recorded():
    cal = SplitConformalCalibrator(nominal_coverage=0.8)
    cal.fit([0.0, 0.01], [-0.01, -0.01], [0.01, 0.01], ["s1", "s1"])
    assert cal.support_rows == 2
    assert cal.support_sessions == 1
