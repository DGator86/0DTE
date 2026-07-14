"""
tests/test_return_distribution_v3.py
====================================
V3 Part 2 PR11 — expanded return quantile grid (§16 / §44 prep).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.base import rearrange_quantile_grid
from prediction.return_distribution import (
    QUANTILES,
    ExpandedReturnQuantileModel,
    ReturnDistribution,
    moments_from_quantiles,
)


def test_quantile_grid():
    assert QUANTILES[0] == 0.05
    assert QUANTILES[-1] == 0.95
    assert 0.50 in QUANTILES
    assert len(QUANTILES) == 11


def test_rearrange_quantile_grid_orders():
    n = 5
    raw = {
        0.1: np.array([0.05, 0.02, 0.0, -0.01, 0.03]),
        0.5: np.array([0.01, 0.04, 0.02, 0.0, 0.01]),  # crosses q10 on some rows
        0.9: np.array([0.02, 0.06, 0.05, 0.04, 0.07]),
    }
    ordered = rearrange_quantile_grid(raw)
    stacked = np.vstack([ordered[q] for q in sorted(ordered)])
    assert np.all(np.diff(stacked, axis=0) >= -1e-12)


def test_return_distribution_invariant():
    rd = ReturnDistribution(
        horizon="30m",
        quantiles={0.1: -0.01, 0.5: 0.0, 0.9: 0.01},
        expected_return=0.0,
        variance=0.0001,
        conformal_intervals={},
        conformal_support_rows=0,
        conformal_support_sessions=0,
        uncertainty=0.2,
        ood_score=None,
        model_version="v3",
    )
    assert rd.quantiles[0.1] <= rd.quantiles[0.5] <= rd.quantiles[0.9]
    with pytest.raises(ValueError):
        ReturnDistribution(
            horizon="30m",
            quantiles={0.1: 0.05, 0.5: 0.0, 0.9: 0.01},
            expected_return=None, variance=None,
            conformal_intervals={}, conformal_support_rows=0,
            conformal_support_sessions=0, uncertainty=0.0, ood_score=None,
            model_version="v3",
        )


def test_moments_diagnostic():
    qs = {q: float(np.quantile([-0.02, -0.01, 0.0, 0.01, 0.02], q))
          for q in QUANTILES}
    mean, var = moments_from_quantiles(qs)
    assert mean is not None and var is not None
    assert var >= 0


def test_expanded_model_fit_predict():
    rng = np.random.default_rng(7)
    rows, y, sessions = [], [], []
    for i in range(80):
        x = float(rng.normal())
        rows.append({"f1": x, "f2": float(rng.normal())})
        y.append(0.01 * x + float(rng.normal(0, 0.005)))
        sessions.append(f"S{i % 10:02d}")
    model = ExpandedReturnQuantileModel()
    # Use a thinner grid for speed in unit tests
    model.config.quantiles = (0.05, 0.10, 0.50, 0.90, 0.95)
    model.config.max_iter = 50
    model.fit(rows, y, sessions)
    dist = model.predict_distribution(rows[0])
    qs = sorted(dist.quantiles.items())
    for i in range(1, len(qs)):
        assert qs[i][1] >= qs[i - 1][1] - 1e-12
    assert dist.horizon == "30m"
    assert dist.conformal_intervals == {}
    metrics = model.evaluate(rows, y)
    assert metrics["n"] == 80
    assert "pinball_q50" in metrics
