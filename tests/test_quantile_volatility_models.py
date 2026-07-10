"""
tests/test_quantile_volatility_models.py
========================================
PR 4 acceptance — return-quantile and volatility models:
  * predicted quantiles are ALWAYS monotonically ordered (rearrangement);
  * pinball loss and 10-90 interval coverage are reported, sliceable by
    group (time of day / regime);
  * volatility forecasts are non-negative, log-target trained, with a
    bounded uncertainty and an implied-move ratio when available;
  * both models are deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.base import (interval_coverage, pinball_loss,
                                    rearrange_quantiles)
from prediction.models.return_quantiles import (ReturnQuantileConfig,
                                                ReturnQuantileModel)
from prediction.models.volatility import VolatilityModel, VolatilityModelConfig

RNG = np.random.default_rng(31)

Q_CFG = ReturnQuantileConfig(max_iter=50, min_samples_leaf=20)
V_CFG = VolatilityModelConfig(max_iter=50, min_samples_leaf=20)


def _quantile_data(n=600):
    rows, y = [], []
    for _ in range(n):
        sig = RNG.standard_normal()
        vol = RNG.uniform(0.5, 2.0)
        rows.append({"signal": sig, "vol_state": vol})
        y.append(0.002 * sig + RNG.standard_normal() * 0.001 * vol)
    return rows, np.array(y)


def _vol_data(n=600):
    rows, y = [], []
    for _ in range(n):
        vol = RNG.uniform(0.5, 2.0)
        rows.append({"vol_state": vol,
                     "implied_remaining_move": 0.004 * vol})
        y.append(abs(RNG.standard_normal()) * 0.003 * vol + 1e-4)
    return rows, np.array(y)


class TestRearrangement:
    def test_crossed_quantiles_are_sorted(self):
        q10, q50, q90 = rearrange_quantiles([0.5, -1.0], [0.0, 0.0],
                                            [-0.5, 1.0])
        assert np.all(q10 <= q50) and np.all(q50 <= q90)
        assert q10[0] == -0.5 and q90[0] == 0.5      # values preserved

    def test_pinball_asymmetry(self):
        # under-prediction hurts the q90 loss more than over-prediction
        y = np.array([1.0])
        assert (pinball_loss(y, np.array([0.0]), 0.9)
                > pinball_loss(y, np.array([2.0]), 0.9))

    def test_interval_coverage(self):
        y = np.array([0.0, 0.5, 1.0, 2.0])
        assert interval_coverage(y, np.zeros(4), np.ones(4)) == 0.75


class TestReturnQuantileModel:
    def test_ordering_holds_everywhere(self):
        rows, y = _quantile_data()
        m = ReturnQuantileModel(config=Q_CFG).fit(rows, y)
        p = m.predict(rows)
        assert np.all(p["q10"] <= p["q50"])
        assert np.all(p["q50"] <= p["q90"])

    def test_evaluation_metrics(self):
        rows, y = _quantile_data()
        m = ReturnQuantileModel(config=Q_CFG).fit(rows, y)
        ev = m.evaluate(rows, y)
        assert ev["n"] == len(y)
        # in-sample coverage should be in the right neighborhood of 80%
        assert 0.6 <= ev["coverage_10_90"] <= 1.0
        assert ev["pinball_q50"] > 0.0

    def test_coverage_by_group(self):
        rows, y = _quantile_data()
        groups = ["am" if i < len(rows) // 2 else "pm"
                  for i in range(len(rows))]
        ev = m = ReturnQuantileModel(config=Q_CFG).fit(rows, y).evaluate(
            rows, y, group_by=groups)
        assert set(ev["by_group"]) == {"am", "pm"}
        for g in ev["by_group"].values():
            assert 0.0 <= g["coverage_10_90"] <= 1.0

    def test_deterministic(self):
        rows, y = _quantile_data(300)
        p1 = ReturnQuantileModel(config=Q_CFG).fit(rows, y).predict(rows)
        p2 = ReturnQuantileModel(config=Q_CFG).fit(rows, y).predict(rows)
        for k in ("q10", "q50", "q90"):
            assert np.array_equal(p1[k], p2[k])

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            ReturnQuantileModel().predict([{}])


class TestVolatilityModel:
    def test_non_negative_and_bounded_uncertainty(self):
        rows, y = _vol_data()
        m = VolatilityModel(config=V_CFG).fit(rows, y)
        p = m.predict(rows)
        assert np.all(p["expected_move"] >= 0.0)
        assert np.all(p["move_q10"] <= p["expected_move"])
        assert np.all(p["expected_move"] <= p["move_q90"])
        assert np.all((p["uncertainty"] >= 0.0) & (p["uncertainty"] <= 1.0))

    def test_learns_volatility_state(self):
        rows, y = _vol_data(800)
        m = VolatilityModel(config=V_CFG).fit(rows, y)
        lo = m.predict([{"vol_state": 0.5, "implied_remaining_move": 0.002}])
        hi = m.predict([{"vol_state": 2.0, "implied_remaining_move": 0.008}])
        assert hi["expected_move"][0] > lo["expected_move"][0]

    def test_rv_iv_ratio(self):
        rows, y = _vol_data()
        m = VolatilityModel(config=V_CFG).fit(rows, y)
        p = m.predict([{"vol_state": 1.0, "implied_remaining_move": 0.004},
                       {"vol_state": 1.0, "implied_remaining_move": None},
                       {"vol_state": 1.0}])
        assert np.isfinite(p["rv_iv_ratio"][0])
        assert p["rv_iv_ratio"][0] == pytest.approx(
            p["expected_move"][0] / 0.004)
        assert np.isnan(p["rv_iv_ratio"][1])
        assert np.isnan(p["rv_iv_ratio"][2])

    def test_negative_targets_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VolatilityModel(config=V_CFG).fit([{"x": 1.0}] * 3,
                                              [0.001, -0.002, 0.003])

    def test_evaluate(self):
        rows, y = _vol_data()
        m = VolatilityModel(config=V_CFG).fit(rows, y)
        ev = m.evaluate(rows, y)
        assert ev["n"] == len(y)
        assert ev["mae"] >= 0.0
        assert 0.0 <= ev["coverage_10_90"] <= 1.0
