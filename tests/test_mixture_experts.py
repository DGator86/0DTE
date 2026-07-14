"""
tests/test_mixture_experts.py
=============================
V3 Part 2 PR10 — mixture-of-experts blending (§43).
"""
from __future__ import annotations

import pytest

from prediction.models.mixture_experts import (
    MixtureExpertsConfig,
    MixtureOfExperts,
    between_expert_disagreement,
    blend_mixture,
    expert_shrinkage_weight,
    shrink_prediction,
)
from prediction.regime_labels import REGIME_CLASSES


def test_regime_probabilities_control_blend():
    shrunk = {
        "long_gamma_pin": 0.2,
        "short_gamma_trend": 0.8,
        "flip_transition": 0.5,
        "volatility_expansion": 0.5,
    }
    one_hot = {c: 0.0 for c in REGIME_CLASSES}
    one_hot["short_gamma_trend"] = 1.0
    assert blend_mixture(one_hot, shrunk) == pytest.approx(0.8)
    one_hot = {c: 0.0 for c in REGIME_CLASSES}
    one_hot["long_gamma_pin"] = 1.0
    assert blend_mixture(one_hot, shrunk) == pytest.approx(0.2)


def test_low_support_shrinks_toward_global():
    w = expert_shrinkage_weight(10, shrinkage_sessions=40)
    assert w == pytest.approx(10 / 50)
    shrunk = shrink_prediction(1.0, 0.0, w)
    assert shrunk == pytest.approx(w)


def test_missing_experts_fallback_explicit():
    moe = MixtureOfExperts(target="p_up", horizon="30m",
                           cfg=MixtureExpertsConfig(minimum_effective_sessions=20))
    moe.register_global(lambda row: 0.55)
    # Register with insufficient support
    moe.register_regime_expert(
        "long_gamma_pin", lambda row: 0.90, support_sessions=5)
    probs = {c: 0.25 for c in REGIME_CLASSES}
    out = moe.predict({"x": 1.0}, probs)
    assert out.expert_weights["long_gamma_pin"] == 0.0
    assert out.shrunk_expert_predictions["long_gamma_pin"] == pytest.approx(0.55)
    assert any(f["reason"] == "insufficient_support"
               for f in out.diagnostics["fallbacks"])


def test_expert_disagreement_raises_uncertainty():
    moe = MixtureOfExperts(target="p_up", horizon="30m",
                           cfg=MixtureExpertsConfig(minimum_effective_sessions=1,
                                                    shrinkage_sessions=0))
    moe.register_global(lambda row: 0.5)
    for c, v in zip(REGIME_CLASSES, (0.1, 0.9, 0.2, 0.8)):
        moe.register_regime_expert(c, lambda row, vv=v: vv, support_sessions=100)
    probs = {c: 0.25 for c in REGIME_CLASSES}
    out = moe.predict({"x": 1}, probs)
    assert out.disagreement > 0
    # Equal experts → zero disagreement
    moe2 = MixtureOfExperts(target="p_up", horizon="30m",
                            cfg=MixtureExpertsConfig(minimum_effective_sessions=1,
                                                     shrinkage_sessions=0))
    moe2.register_global(lambda row: 0.5)
    for c in REGIME_CLASSES:
        moe2.register_regime_expert(c, lambda row: 0.5, support_sessions=100)
    out2 = moe2.predict({"x": 1}, probs)
    assert out2.disagreement == pytest.approx(0.0)
    assert out.uncertainty > out2.uncertainty


def test_one_hot_reproduces_shrunk_expert():
    moe = MixtureOfExperts(target="p_up", horizon="30m",
                           cfg=MixtureExpertsConfig(minimum_effective_sessions=1,
                                                    shrinkage_sessions=40))
    moe.register_global(lambda row: 0.4)
    moe.register_regime_expert(
        "flip_transition", lambda row: 0.8, support_sessions=40)
    # Other regimes unavailable → global
    probs = {c: 0.0 for c in REGIME_CLASSES}
    probs["flip_transition"] = 1.0
    out = moe.predict({"x": 1}, probs)
    w = expert_shrinkage_weight(40, shrinkage_sessions=40)
    expected = shrink_prediction(0.8, 0.4, w)
    assert out.final_prediction == pytest.approx(expected)


def test_deterministic():
    moe = MixtureOfExperts(target="expected_return", horizon="30m",
                           cfg=MixtureExpertsConfig(minimum_effective_sessions=1))
    moe.register_global(lambda row: 0.01)
    for c in REGIME_CLASSES:
        moe.register_regime_expert(c, lambda row, cc=c: hash(cc) % 7 / 100.0,
                                   support_sessions=50)
    probs = {"long_gamma_pin": 0.4, "short_gamma_trend": 0.3,
             "flip_transition": 0.2, "volatility_expansion": 0.1}
    a = moe.predict({"f": 1}, probs)
    b = moe.predict({"f": 1}, probs)
    assert a.to_dict() == b.to_dict()


def test_candidate_features_rejected():
    moe = MixtureOfExperts(target="p_up", horizon="30m")
    moe.register_global(lambda row: 0.5)
    with pytest.raises(ValueError, match="candidate"):
        moe.predict({"candidate_id": "x", "f": 1},
                    {c: 0.25 for c in REGIME_CLASSES})


def test_equal_predictions_zero_disagreement():
    preds = {c: 0.33 for c in REGIME_CLASSES}
    probs = {c: 0.25 for c in REGIME_CLASSES}
    d = between_expert_disagreement(probs, preds, 0.33)
    assert d == pytest.approx(0.0)
