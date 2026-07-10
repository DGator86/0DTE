"""
tests/test_range_model.py
=========================
PR 7 acceptance — RangeSurvivalModel:
  * wall-channel / short-strike / breakeven kinds × 15m/30m/60m/close;
  * probabilities in [0, 1], deterministic, OOS-calibrated via inner split;
  * learnable width/vol signal beats the base rate out of sample;
  * feature row encodes barrier width normalized by volatility.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.base import brier_score
from prediction.models.range_survival import (
    RANGE_HORIZONS, RANGE_KINDS, RangeSurvivalConfig, RangeSurvivalModel,
    range_feature_row,
)


SMALL = RangeSurvivalConfig(
    kind="wall_channel", horizon="close",
    c_grid=(0.1, 1.0), l1_ratio_grid=(0.0, 0.5),
    class_weight_options=(None,), max_iter=500,
)


def _synth(n_sessions=12, per_session=40, strength=2.2, seed=29):
    rng = np.random.default_rng(seed)
    rows, y, sessions = [], [], []
    for s in range(n_sessions):
        date = f"2026-07-{s + 1:02d}"
        for _ in range(per_session):
            width = 0.004 + abs(rng.standard_normal()) * 0.008
            vol = 0.004 + abs(rng.standard_normal()) * 0.006
            noise = rng.standard_normal()
            # wider channel vs vol → higher survival
            ratio = width / vol
            logit = strength * (ratio - 1.0) + 0.15 * noise
            p = 1.0 / (1.0 + np.exp(-logit))
            half = width * 600.0 / 2.0
            rows.append(range_feature_row(
                spot=600.0, lower=600.0 - half, upper=600.0 + half,
                minutes_to_close=90.0, expected_realized_move=vol,
                move_consumed=0.3, net_gex=1e9, wall_stability=0.6,
                adx=16.0, cvd_slope=0.0))
            rows[-1]["noise"] = noise
            y.append(int(rng.uniform() < p))
            sessions.append(date)
    return rows, np.array(y), sessions


class TestRangeFeatureRow:
    def test_width_over_vol(self):
        row = range_feature_row(
            spot=600.0, lower=594.0, upper=606.0,
            expected_realized_move=0.01, minutes_to_close=60.0)
        assert row["barrier_width"] == pytest.approx(12.0 / 600.0)
        assert row["barrier_width_over_vol"] == pytest.approx(
            (12.0 / 600.0) / 0.01)
        assert row["dist_lower"] == pytest.approx(6.0 / 600.0)
        assert row["dist_upper"] == pytest.approx(6.0 / 600.0)


class TestRangeSurvivalModel:
    def test_kinds_and_horizons(self):
        assert set(RANGE_KINDS) == {"wall_channel", "short_strike", "breakeven"}
        assert set(RANGE_HORIZONS) == {"15m", "30m", "60m", "close"}

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown range kind"):
            RangeSurvivalModel(
                config=RangeSurvivalConfig(kind="nope")).fit(
                [{"barrier_width": 0.01}], [1], ["s0"])

    def test_bounds_determinism_calibration(self):
        rows, y, sessions = _synth()
        m1 = RangeSurvivalModel(config=SMALL).fit(rows, y, sessions)
        m2 = RangeSurvivalModel(config=SMALL).fit(rows, y, sessions)
        p1, p2 = m1.predict_proba(rows), m2.predict_proba(rows)
        assert np.all((p1 >= 0.0) & (p1 <= 1.0))
        assert np.array_equal(p1, p2)
        meta = m1.metadata
        assert meta["kind"] == "wall_channel"
        assert meta["horizon"] == "close"
        assert set(meta["fit_sessions"]).isdisjoint(meta["calibration_sessions"])
        assert "brier_calibrated" in meta["calibration_metrics"] or (
            "note" in meta["calibration_metrics"])

    def test_learns_signal_oos(self):
        rows, y, sessions = _synth(n_sessions=14, strength=2.8)
        train_s = {f"2026-07-{i:02d}" for i in range(1, 11)}
        test_s = {f"2026-07-{i:02d}" for i in range(12, 15)}
        tr = [i for i, s in enumerate(sessions) if s in train_s]
        te = [i for i, s in enumerate(sessions) if s in test_s]
        m = RangeSurvivalModel(config=SMALL).fit(
            [rows[i] for i in tr], y[tr], [sessions[i] for i in tr])
        p = m.predict_proba([rows[i] for i in te])
        base = np.full(len(te), float(np.mean(y[tr])))
        assert brier_score(y[te], p) < brier_score(y[te], base)

    def test_each_kind_horizon_fits(self):
        rows, y, sessions = _synth(n_sessions=8, per_session=25)
        for kind in RANGE_KINDS:
            for horizon in ("30m", "close"):
                cfg = RangeSurvivalConfig(
                    kind=kind, horizon=horizon,
                    c_grid=(1.0,), l1_ratio_grid=(0.5,),
                    class_weight_options=(None,), max_iter=400)
                m = RangeSurvivalModel(config=cfg).fit(rows, y, sessions)
                p = m.predict_proba(rows[:3])
                assert p.shape == (3,)
                assert np.all((p >= 0.0) & (p <= 1.0))
