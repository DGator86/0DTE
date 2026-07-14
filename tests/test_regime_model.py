"""
tests/test_regime_model.py
==========================
V3 Part 2 PR9 — regime probability model (§42).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.regime_moe import (
    REGIME_MODEL_VERSION,
    RegimeProbabilityModel,
    RegimeModelConfig,
    normalized_entropy,
    renormalize_probs,
)
from prediction.regime_labels import REGIME_CLASSES


def _synth_rows(n_per_class: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows, labels, sessions = [], [], []
    # Controlled features: trend_score high → trend; pin_score high → pin
    for ci, cname in enumerate(REGIME_CLASSES):
        for j in range(n_per_class):
            sess = f"2026-01-{(ci * n_per_class + j) % 20 + 1:02d}"
            base = {
                "trend_score": float(rng.normal(0, 0.3)),
                "pin_score": float(rng.normal(0, 0.3)),
                "flip_score": float(rng.normal(0, 0.3)),
                "vol_score": float(rng.normal(0, 0.3)),
                "gex_disagreement": float(rng.uniform(0, 0.5)),
                "minutes_to_close": float(rng.uniform(30, 300)),
            }
            if cname == "short_gamma_trend":
                base["trend_score"] = float(rng.normal(2.0, 0.3))
            elif cname == "long_gamma_pin":
                base["pin_score"] = float(rng.normal(2.0, 0.3))
            elif cname == "flip_transition":
                base["flip_score"] = float(rng.normal(2.0, 0.3))
            else:
                base["vol_score"] = float(rng.normal(2.0, 0.3))
            rows.append(base)
            labels.append(cname)
            sessions.append(sess)
    return rows, labels, sessions


def test_probabilities_sum_to_one():
    rows, labels, sessions = _synth_rows()
    model = RegimeProbabilityModel(
        RegimeModelConfig(estimator="multinomial", calibration="sigmoid")
    ).fit(rows, labels, sessions)
    rp = model.predict(rows[0])
    s = sum(rp.as_dict().values())
    assert abs(s - 1.0) <= 1e-6
    for v in rp.as_dict().values():
        assert 0.0 <= v <= 1.0


def test_probabilities_bounded_batch():
    rows, labels, sessions = _synth_rows()
    model = RegimeProbabilityModel().fit(rows, labels, sessions)
    proba = model.predict_proba(rows[:10])
    assert proba.shape == (10, 4)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_trend_evidence_raises_trend_probability():
    rows, labels, sessions = _synth_rows(40)
    model = RegimeProbabilityModel(
        RegimeModelConfig(estimator="multinomial", calibration="identity")
    ).fit(rows, labels, sessions)
    low = model.predict({"trend_score": 0.0, "pin_score": 0.0,
                         "flip_score": 0.0, "vol_score": 0.0,
                         "gex_disagreement": 0.2, "minutes_to_close": 120})
    high = model.predict({"trend_score": 3.0, "pin_score": 0.0,
                          "flip_score": 0.0, "vol_score": 0.0,
                          "gex_disagreement": 0.2, "minutes_to_close": 120})
    assert high.short_gamma_trend >= low.short_gamma_trend - 1e-9


def test_pin_evidence_raises_pin_probability():
    rows, labels, sessions = _synth_rows(40)
    model = RegimeProbabilityModel(
        RegimeModelConfig(estimator="multinomial", calibration="identity")
    ).fit(rows, labels, sessions)
    low = model.predict({"trend_score": 0.0, "pin_score": 0.0,
                         "flip_score": 0.0, "vol_score": 0.0,
                         "gex_disagreement": 0.2, "minutes_to_close": 120})
    high = model.predict({"trend_score": 0.0, "pin_score": 3.0,
                          "flip_score": 0.0, "vol_score": 0.0,
                          "gex_disagreement": 0.2, "minutes_to_close": 120})
    assert high.long_gamma_pin >= low.long_gamma_pin - 1e-9


def test_high_entropy_increases_uncertainty():
    flat = renormalize_probs({c: 0.25 for c in REGIME_CLASSES})
    peaked = renormalize_probs({
        "long_gamma_pin": 0.85,
        "short_gamma_trend": 0.05,
        "flip_transition": 0.05,
        "volatility_expansion": 0.05,
    })
    assert normalized_entropy(list(flat.values())) > normalized_entropy(
        list(peaked.values()))


def test_renormalize_invariant():
    p = renormalize_probs({"long_gamma_pin": 2.0, "short_gamma_trend": 2.0,
                           "flip_transition": 0.0, "volatility_expansion": 0.0})
    assert abs(sum(p.values()) - 1.0) <= 1e-6


def test_hgb_challenger_fits():
    rows, labels, sessions = _synth_rows(25)
    model = RegimeProbabilityModel(
        RegimeModelConfig(estimator="hgb", calibration="identity")
    ).fit(rows, labels, sessions)
    rp = model.predict(rows[0])
    assert abs(sum(rp.as_dict().values()) - 1.0) <= 1e-6
    assert rp.model_version == REGIME_MODEL_VERSION


def test_evaluate_metrics():
    rows, labels, sessions = _synth_rows(30)
    model = RegimeProbabilityModel().fit(rows, labels, sessions)
    metrics = model.evaluate(rows, labels)
    assert metrics["n"] > 0
    assert "log_loss" in metrics
    assert "macro_brier" in metrics
    assert set(metrics["brier_by_class"]) == set(REGIME_CLASSES)


def test_dominant_is_convenience_only():
    rows, labels, sessions = _synth_rows()
    model = RegimeProbabilityModel().fit(rows, labels, sessions)
    rp = model.predict(rows[0])
    assert rp.dominant_regime == max(rp.as_dict(), key=rp.as_dict().get)
    # Full vector still present
    assert len(rp.as_dict()) == 4
