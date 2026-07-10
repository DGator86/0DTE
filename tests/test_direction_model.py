"""
tests/test_direction_model.py
=============================
PR 4 acceptance — calibrated elastic-net direction models:
  * probability outputs remain within [0, 1];
  * predictions are deterministic given the same data and config;
  * a learnable signal beats the base rate out of sample;
  * hyperparameter selection and calibration stay INSIDE training sessions
    (embargoed inner split, recorded in metadata);
  * missing feature values are handled explicitly (never crash, never
    silently neutral);
  * the required naive baselines are available and sane.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.base import FeatureVectorizer, brier_score
from prediction.models.direction import (DirectionModel, DirectionModelConfig,
                                         baseline_base_rate,
                                         baseline_legacy_composite,
                                         baseline_prev_sign, baseline_random,
                                         evaluate_probabilities,
                                         split_train_calibration)

RNG = np.random.default_rng(23)

SMALL_CFG = DirectionModelConfig(
    horizon="30m",
    c_grid=(0.1, 1.0), l1_ratio_grid=(0.0, 0.5),
    class_weight_options=(None,), max_iter=500,
)


def _synth(n_sessions=12, per_session=40, signal_strength=2.0, seed=23):
    """Sessions of rows where feature 'signal' drives P(up)."""
    rng = np.random.default_rng(seed)
    rows, y, sessions = [], [], []
    for s in range(n_sessions):
        date = f"2026-07-{s + 1:02d}"
        for _ in range(per_session):
            sig = rng.standard_normal()
            noise = rng.standard_normal()
            p_up = 1.0 / (1.0 + np.exp(-signal_strength * sig))
            rows.append({"signal": sig, "noise": noise,
                         "sometimes_missing": (rng.standard_normal()
                                               if rng.uniform() > 0.3 else None)})
            y.append(int(rng.uniform() < p_up))
            sessions.append(date)
    return rows, np.array(y), sessions


class TestVectorizer:
    def test_missingness_columns(self):
        v = FeatureVectorizer()
        X = v.fit_transform([{"a": 1.0, "b": None}, {"a": None, "b": 2.0}])
        assert X.shape == (2, 4)
        names = v.column_names()
        a_val, b_val = names.index("val:a"), names.index("val:b")
        a_miss, b_miss = names.index("miss:a"), names.index("miss:b")
        assert X[0, a_miss] == 0.0 and X[0, b_miss] == 1.0
        assert X[1, a_miss] == 1.0 and X[1, b_miss] == 0.0
        # imputed value is the training median of the observed values
        assert X[1, a_val] == 1.0
        assert X[0, b_val] == 2.0

    def test_frozen_column_order(self):
        v = FeatureVectorizer().fit([{"b": 1.0, "a": 2.0}])
        X1 = v.transform([{"a": 5.0, "b": 6.0}])
        X2 = v.transform([{"b": 6.0, "a": 5.0}])       # dict order irrelevant
        assert np.array_equal(X1, X2)

    def test_unseen_features_ignored(self):
        v = FeatureVectorizer().fit([{"a": 1.0}])
        X = v.transform([{"a": 2.0, "new_thing": 99.0}])
        assert X.shape == (1, 2)


class TestInnerSplit:
    def test_embargoed_session_split(self):
        sessions = [f"s{i:02d}" for i in range(10)]
        fit_s, cal_s = split_train_calibration(sessions, 0.3, 1)
        assert fit_s == sessions[:6]
        assert cal_s == sessions[7:]                   # s06 embargoed
        assert not set(fit_s) & set(cal_s)

    def test_too_few_sessions_no_cal_slice(self):
        fit_s, cal_s = split_train_calibration(["a", "b"], 0.25, 1)
        assert cal_s == []


class TestDirectionModel:
    def test_bounds_and_determinism(self):
        rows, y, sessions = _synth()
        m1 = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        m2 = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        p1, p2 = m1.predict_proba(rows), m2.predict_proba(rows)
        assert np.all((p1 >= 0.0) & (p1 <= 1.0))
        assert np.array_equal(p1, p2)                  # same data -> same model

    def test_learns_signal_beats_base_rate(self):
        rows, y, sessions = _synth(n_sessions=14)
        # train on the first 10 sessions, test on the last 3 (session 11 gap)
        train_s = {f"2026-07-{i:02d}" for i in range(1, 11)}
        test_s = {f"2026-07-{i:02d}" for i in range(12, 15)}
        tr = [i for i, s in enumerate(sessions) if s in train_s]
        te = [i for i, s in enumerate(sessions) if s in test_s]
        m = DirectionModel(config=SMALL_CFG).fit(
            [rows[i] for i in tr], y[tr], [sessions[i] for i in tr])
        p = m.predict_proba([rows[i] for i in te])
        base = baseline_base_rate(y[tr], len(te))
        assert brier_score(y[te], p) < brier_score(y[te], base)

    def test_calibration_stayed_inside_training_sessions(self):
        rows, y, sessions = _synth()
        m = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        meta = m.metadata
        fit_s = set(meta["fit_sessions"])
        cal_s = set(meta["calibration_sessions"])
        train_s = set(meta["train_sessions"])
        assert cal_s and fit_s
        assert not fit_s & cal_s                       # embargoed inner split
        assert fit_s | cal_s <= train_s
        assert meta["calibration_metrics"]["n"] > 0
        assert meta["decision_threshold"] == pytest.approx(0.58)
        assert 0.0 <= meta["uncertainty"] <= 1.0

    def test_handles_missing_features_at_predict_time(self):
        rows, y, sessions = _synth()
        m = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        p = m.predict_proba([{"signal": None, "noise": None,
                              "sometimes_missing": None}, {}])
        assert p.shape == (2,)
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_degenerate_one_class_training(self):
        rows = [{"x": float(i)} for i in range(30)]
        y = np.ones(30, dtype=int)
        sessions = [f"2026-07-{(i % 3) + 1:02d}" for i in range(30)]
        m = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        p = m.predict_proba(rows[:5])
        assert np.all((p >= 0.0) & (p <= 1.0))         # base rate, no crash

    def test_predict_label_threshold(self):
        rows, y, sessions = _synth()
        m = DirectionModel(config=SMALL_CFG).fit(rows, y, sessions)
        p = m.predict_proba(rows)
        lab = m.predict_label(rows)
        assert set(np.unique(lab)) <= {-1, 0, 1}
        assert np.all(lab[p >= 0.58] == 1)
        assert np.all(lab[p <= 0.42] == -1)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            DirectionModel().predict_proba([{}])

    def test_hgb_challenger_same_interface(self):
        rows, y, sessions = _synth(n_sessions=8)
        cfg = DirectionModelConfig(
            estimator="hgb",
            hgb_learning_rate_grid=(0.1,), hgb_max_leaf_nodes_grid=(7,),
            hgb_max_depth_grid=(2,), hgb_min_samples_leaf_grid=(50,),
            hgb_l2_grid=(0.0,))
        m = DirectionModel(config=cfg).fit(rows, y, sessions)
        p = m.predict_proba(rows[:10])
        assert np.all((p >= 0.0) & (p <= 1.0))


class TestBaselines:
    def test_base_rate(self):
        p = baseline_base_rate([1, 1, 0, 0, 1], 4)
        assert np.allclose(p, 0.6)

    def test_prev_sign(self):
        p = baseline_prev_sign([0.001, -0.002, 0.0, None], [1, 0, 1, 0])
        assert p[0] == 1.0 and p[1] == 0.0
        assert p[2] == pytest.approx(0.5)              # zero -> base rate
        assert p[3] == pytest.approx(0.5)              # missing -> base rate

    def test_legacy_composite(self):
        p = baseline_legacy_composite([58.0, 42.0, None], [1, 0])
        assert p[0] == pytest.approx(0.58)
        assert p[1] == pytest.approx(0.42)
        assert p[2] == pytest.approx(0.5)

    def test_random_seeded(self):
        assert np.array_equal(baseline_random(10), baseline_random(10))

    def test_evaluate_panel(self):
        y = np.array([1, 0, 1, 1])
        m = evaluate_probabilities(y, np.array([0.9, 0.1, 0.8, 0.7]))
        assert m["n"] == 4
        assert m["hit_rate_at_half"] == 1.0
        assert m["brier"] < 0.1
