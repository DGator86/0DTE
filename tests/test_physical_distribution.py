"""
tests/test_physical_distribution.py
====================================
PR 5 acceptance — V2 independent physical density:
  * density integrates to one;
  * mean and variance match the forecast within tolerance;
  * high uncertainty blends toward the RND (mean shift shrinks, dispersion
    moves toward RN);
  * identical PhysicalForecast => identical density (determinism);
  * forecast_from_bundle lifts a usable forecast from a PredictionBundle;
  * moments / quality are recorded.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from prediction.contracts import PredictionBundle
from prediction.physical_distribution import (
    PHYSICAL_DIST_VERSION, PhysicalForecast, build_physical_density,
    density_moments, forecast_from_bundle,
)
from rnd_extractor import (
    ChainQuote, ChainSnapshot, _bs_call_fwd, extract_rnd,
)

F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
DF0 = math.exp(-R0 * T0)


def _chain(atm_s: float = 0.04) -> ChainSnapshot:
    qs = []
    for K in np.arange(F0 - 25, F0 + 26, 1.0):
        k = math.log(K / F0)
        s = max(atm_s - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)


def _forecast(**kw) -> PhysicalForecast:
    base = dict(
        expected_return=0.002,
        return_q10=-0.004, return_q50=0.002, return_q90=0.008,
        expected_realized_move=0.006,
        volatility_scale=1.0, skew_adjustment=0.0,
        uncertainty=0.0, model_version="test-v2",
    )
    base.update(kw)
    return PhysicalForecast(**base)


@pytest.fixture()
def rnd():
    return extract_rnd(_chain())


class TestBuildPhysicalDensity:
    def test_integrates_to_one(self, rnd):
        res = build_physical_density(rnd, _forecast())
        dx = res.grid[1] - res.grid[0]
        assert np.sum(res.density) * dx == pytest.approx(1.0, abs=1e-6)
        assert res.quality["integrate_error"] < 1e-6
        assert np.all(res.density >= 0.0)

    def test_mean_matches_forecast(self, rnd):
        f = _forecast(expected_return=0.003, uncertainty=0.0)
        res = build_physical_density(rnd, f)
        predicted = rnd.forward * math.exp(f.expected_return)
        assert res.moments["mean"] == pytest.approx(predicted, rel=0.02)
        assert res.quality["mean_error"] < 0.5          # dollars on a $600 underlier

    def test_std_matches_forecast_scale(self, rnd):
        f = _forecast(uncertainty=0.0)
        res = build_physical_density(rnd, f)
        # with uncertainty=0 the std should land near the (clipped) target
        assert res.quality["std_error"] / max(res.quality["target_std"], 1e-9) < 0.15
        assert res.moments["std"] > 0

    def test_bullish_forecast_shifts_mean_up(self, rnd):
        up = build_physical_density(rnd, _forecast(expected_return=0.005,
                                                   uncertainty=0.0))
        dn = build_physical_density(rnd, _forecast(expected_return=-0.005,
                                                   uncertainty=0.0))
        flat = build_physical_density(rnd, _forecast(expected_return=0.0,
                                                     uncertainty=0.0))
        assert up.moments["mean"] > flat.moments["mean"] > dn.moments["mean"]

    def test_high_uncertainty_blends_toward_rnd(self, rnd):
        f_certain = _forecast(expected_return=0.006, uncertainty=0.0)
        f_uncertain = _forecast(expected_return=0.006, uncertainty=0.9)
        certain = build_physical_density(rnd, f_certain)
        uncertain = build_physical_density(rnd, f_uncertain)
        rn_mean = certain.quality["rn_mean"]
        # uncertain mean is closer to the RN mean than the certain mean is
        assert abs(uncertain.moments["mean"] - rn_mean) < abs(
            certain.moments["mean"] - rn_mean)
        assert uncertain.quality["confidence_weight"] == pytest.approx(0.1)

    def test_identical_forecast_identical_density(self, rnd):
        f = _forecast()
        a = build_physical_density(rnd, f)
        b = build_physical_density(rnd, f)
        assert np.array_equal(a.density, b.density)
        assert a.moments == b.moments
        assert a.to_dict() == b.to_dict()

    def test_callable_matches_grid_density(self, rnd):
        res = build_physical_density(rnd, _forecast())
        pdf = res.as_callable()
        assert np.allclose(pdf(res.grid), res.density)
        # moments helper agrees
        m = density_moments(pdf, res.grid)
        assert m["mean"] == pytest.approx(res.moments["mean"], rel=1e-6)

    def test_moments_recorded(self, rnd):
        res = build_physical_density(rnd, _forecast())
        for k in ("mean", "std", "variance", "skew", "var_ratio",
                  "predicted_mean", "target_std", "rn_mean", "rn_std"):
            assert k in res.moments
        assert res.mode == "v2"
        assert res.model_version == "test-v2"

    def test_volatility_scale_widens(self, rnd):
        # Disable the live-loop scale clip so the test isolates volatility_scale
        # itself (the clip is covered by the mean/std match tests above).
        kw = dict(return_q10=-0.001, return_q50=0.0, return_q90=0.001,
                  expected_return=0.0, expected_realized_move=0.001,
                  uncertainty=0.0)
        narrow = build_physical_density(
            rnd, _forecast(volatility_scale=1.0, **kw),
            scale_min=0.01, scale_max=10.0)
        wide = build_physical_density(
            rnd, _forecast(volatility_scale=1.4, **kw),
            scale_min=0.01, scale_max=10.0)
        assert narrow.quality["scale"] < wide.quality["scale"]
        assert wide.moments["std"] > narrow.moments["std"]
        assert wide.quality["scale"] / narrow.quality["scale"] == pytest.approx(
            1.4, rel=0.05)

    def test_rejects_bad_forecast(self):
        with pytest.raises(ValueError):
            PhysicalForecast(
                expected_return=0.0, return_q10=-0.01, return_q50=0.0,
                return_q90=0.01, expected_realized_move=0.01, uncertainty=1.5)
        with pytest.raises(ValueError):
            PhysicalForecast(
                expected_return=0.0, return_q10=-0.01, return_q50=0.0,
                return_q90=0.01, expected_realized_move=-0.01)


class TestForecastFromBundle:
    def _bundle(self, **kw) -> PredictionBundle:
        base = dict(
            snapshot_id="abc", ts="2026-07-10T15:00:00-04:00",
            session_date="2026-07-10", symbol="SPY",
            expected_return_close=0.0015,
            return_q10_close=-0.004, return_q50_close=0.0015,
            return_q90_close=0.007,
            expected_realized_move_close=0.005,
            uncertainty=0.25,
            model_versions={"group": "v2.0.0-pr4", "volatility": "remaining_realized_move"},
        )
        base.update(kw)
        return PredictionBundle(**base)

    def test_lifts_close_horizon(self):
        f = forecast_from_bundle(self._bundle())
        assert f is not None
        assert f.expected_return == pytest.approx(0.0015)
        assert f.return_q10 <= f.return_q50 <= f.return_q90
        assert f.uncertainty == pytest.approx(0.25)
        assert f.model_version == "v2.0.0-pr4"

    def test_falls_back_to_30m(self):
        f = forecast_from_bundle(self._bundle(
            expected_return_close=None, return_q10_close=None,
            return_q50_close=None, return_q90_close=None,
            expected_realized_move_close=None,
            expected_return_30m=0.001,
            return_q10_30m=-0.003, return_q50_30m=0.001, return_q90_30m=0.005,
            expected_realized_move_30m=0.004))
        assert f is not None
        assert f.expected_return == pytest.approx(0.001)

    def test_none_without_returns(self):
        assert forecast_from_bundle(self._bundle(
            expected_return_close=None, return_q10_close=None,
            return_q50_close=None, return_q90_close=None,
            expected_realized_move_close=None)) is None

    def test_rearranges_crossed_quantiles(self):
        f = forecast_from_bundle(self._bundle(
            return_q10_close=0.01, return_q50_close=0.0, return_q90_close=-0.01))
        assert f.return_q10 <= f.return_q50 <= f.return_q90
