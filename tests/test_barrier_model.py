"""
tests/test_barrier_model.py
===========================
PR 7 acceptance — BarrierTouchModel + selector V2 touch wiring:
  * probabilities stay in [0, 1] and are deterministic;
  * a learnable signal beats the base rate out of sample;
  * calibration stays inside training sessions (embargoed inner split);
  * candidate touch risk uses V2 when touch_probability_fn is set;
  * legacy RND reflection remains the fallback and is journaled for shadow.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.base import brier_score
from prediction.models.barrier_touch import (
    BARRIER_TARGETS, BarrierTouchConfig, BarrierTouchModel,
    barrier_feature_row, path_features,
)
from prediction.path_model import PathEventResult, PATH_MODEL_VERSION


SMALL = BarrierTouchConfig(
    target="touch_call_wall",
    c_grid=(0.1, 1.0), l1_ratio_grid=(0.0, 0.5),
    class_weight_options=(None,), max_iter=500,
)


def _synth(n_sessions=12, per_session=40, strength=2.0, seed=19):
    rng = np.random.default_rng(seed)
    rows, y, sessions = [], [], []
    for s in range(n_sessions):
        date = f"2026-07-{s + 1:02d}"
        for _ in range(per_session):
            dist = abs(rng.standard_normal()) * 0.01
            noise = rng.standard_normal()
            # nearer wall → higher touch prob
            logit = strength * (0.015 - dist) / 0.01 + 0.2 * noise
            p = 1.0 / (1.0 + np.exp(-logit))
            rows.append(barrier_feature_row(
                spot=600.0, call_wall=600.0 + dist * 600.0,
                put_wall=595.0, gamma_flip=599.0,
                minutes_to_close=120.0, expected_realized_move=0.008,
                net_gex=1e9, wall_stability=0.7, adx=18.0))
            rows[-1]["noise"] = noise
            y.append(int(rng.uniform() < p))
            sessions.append(date)
    return rows, np.array(y), sessions


class TestBarrierFeatureRow:
    def test_distances_and_path_features(self):
        events = PathEventResult(
            p_target_first=0.4, p_stop_first=0.3, p_neither=0.3,
            p_touch_call_wall=0.55, p_touch_put_wall=0.2,
            p_cross_gamma_flip=0.1, p_call_wall_first=0.4,
            p_put_wall_first=0.2, p_neither_wall=0.4,
            p_range_survive=0.5, terminal_mean=601.0, terminal_std=2.0,
            mfe_mean=0.01, mae_mean=-0.008, n_paths=100, n_steps=60,
            ambiguous_same_step_rate=0.02,
        )
        row = barrier_feature_row(
            spot=600.0, call_wall=608.0, put_wall=592.0, gamma_flip=599.0,
            minutes_to_close=90.0, expected_realized_move=0.01,
            path_events=events)
        assert row["dist_call_wall"] == pytest.approx(8.0 / 600.0)
        assert "path_p_touch_call_wall" in row
        assert path_features(events)["path_p_touch_call_wall"] == 0.55
        assert events.model_version == PATH_MODEL_VERSION


class TestBarrierTouchModel:
    def test_targets_enumerated(self):
        assert "stop_before_target" in BARRIER_TARGETS
        assert "cross_gamma_flip" in BARRIER_TARGETS

    def test_bounds_determinism_and_calibration_split(self):
        rows, y, sessions = _synth()
        m1 = BarrierTouchModel(config=SMALL).fit(rows, y, sessions)
        m2 = BarrierTouchModel(config=SMALL).fit(rows, y, sessions)
        p1, p2 = m1.predict_proba(rows), m2.predict_proba(rows)
        assert np.all((p1 >= 0.0) & (p1 <= 1.0))
        assert np.array_equal(p1, p2)
        meta = m1.metadata
        assert set(meta["calibration_sessions"]) <= set(meta["train_sessions"])
        assert meta["calibration_metrics"].get("crossfit") is True
        assert "calibration_metrics" in meta

    def test_learns_signal_oos(self):
        rows, y, sessions = _synth(n_sessions=14, strength=2.5)
        train_s = {f"2026-07-{i:02d}" for i in range(1, 11)}
        test_s = {f"2026-07-{i:02d}" for i in range(12, 15)}
        tr = [i for i, s in enumerate(sessions) if s in train_s]
        te = [i for i, s in enumerate(sessions) if s in test_s]
        m = BarrierTouchModel(config=SMALL).fit(
            [rows[i] for i in tr], y[tr], [sessions[i] for i in tr])
        p = m.predict_proba([rows[i] for i in te])
        base = np.full(len(te), float(np.mean(y[tr])))
        assert brier_score(y[te], p) < brier_score(y[te], base)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            BarrierTouchModel().predict_proba([{"dist_call_wall": 0.01}])


def _selector_fixture():
    """Reuse the PR 6 synthetic chain shape for selector touch wiring tests."""
    import math
    from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd, extract_rnd
    from spread_selector import GammaContext, Leg

    F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
    DF0 = math.exp(-R0 * T0)
    qs = []
    for K in np.arange(F0 - 10, F0 + 11, 1.0):
        k = math.log(K / F0)
        s = max(0.04 - 0.030 * k, 0.0008)
        cm = max(_bs_call_fwd(F0, K, s) * DF0, 0.01)
        pm = max(cm - DF0 * (F0 - K), 0.01)
        h = 0.02
        qs.append(ChainQuote(float(K), max(cm - h, 0.0), cm + h,
                             max(pm - h, 0.0), pm + h))
    chain = ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)
    rnd = extract_rnd(chain)
    dx = rnd.grid[1] - rnd.grid[0]
    phys = rnd.pdf / max(np.sum(rnd.pdf) * dx, 1e-12)
    ctx = GammaContext(spot=F0, call_wall=F0 + 5, put_wall=F0 - 5,
                       gamma_flip=F0 - 1, net_gex=1e9, gex_pct_rank=0.7)
    legs = (Leg(599.0, "P", -1), Leg(598.0, "P", 1))
    return chain, rnd, phys, ctx, legs


class TestSelectorV2Touch:
    def test_v2_overrides_reflection_and_journals_legacy(self):
        from spread_selector import SelectorConfig, _evaluate

        chain, rnd, phys, ctx, legs = _selector_fixture()
        v2_touch = 0.1234
        cfg = SelectorConfig(
            min_ev=-1e9, min_credit=0.0, min_liquidity=0.0,
            max_touch_short=1.0, veto_short_below_flip=False,
            touch_probability_fn=lambda k: v2_touch,
            journal_legacy_touch=True,
        )
        cand = _evaluate("put_credit", legs, chain, rnd, phys, ctx, cfg, {})
        assert cand is not None
        assert cand.touch_source == "v2"
        assert cand.prob_touch_short == pytest.approx(v2_touch, abs=1e-4)
        assert cand.touch_safety == pytest.approx(1.0 - v2_touch, abs=1e-4)
        assert cand.legacy_prob_touch_short is not None
        assert cand.legacy_prob_touch_short != pytest.approx(v2_touch, abs=1e-3)

    def test_reflection_fallback_when_fn_absent(self):
        from spread_selector import SelectorConfig, _evaluate

        chain, rnd, phys, ctx, legs = _selector_fixture()
        cfg = SelectorConfig(
            min_ev=-1e9, min_credit=0.0, min_liquidity=0.0,
            max_touch_short=1.0, veto_short_below_flip=False,
        )
        cand = _evaluate("put_credit", legs, chain, rnd, phys, ctx, cfg, {})
        assert cand is not None
        assert cand.touch_source == "reflection"
        assert cand.legacy_prob_touch_short is None
        assert cand.prob_touch_short == pytest.approx(rnd.prob_touch(599.0),
                                                      abs=1e-4)
