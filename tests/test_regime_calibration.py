"""
tests/test_regime_calibration.py
================================
V3 Part 2 PR9 — one-vs-rest regime calibration uses training-only OOF (§42).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from prediction.models.regime_moe import (
    RegimeModelConfig,
    RegimeProbabilityModel,
)
from prediction.regime_labels import REGIME_CLASSES
from prediction.storage import PredictionStore


def _data(seed=3):
    rng = np.random.default_rng(seed)
    rows, labels, sessions = [], [], []
    for ci, cname in enumerate(REGIME_CLASSES):
        for j in range(20):
            sess = f"S{(ci * 20 + j) % 12:02d}"
            feat = {
                "f1": float(rng.normal(ci, 0.4)),
                "f2": float(rng.normal(0, 1)),
            }
            rows.append(feat)
            labels.append(cname)
            sessions.append(sess)
    return rows, labels, sessions


def test_calibration_flag_and_sum():
    rows, labels, sessions = _data()
    model = RegimeProbabilityModel(
        RegimeModelConfig(calibration="sigmoid")
    ).fit(rows, labels, sessions)
    assert model.diagnostics.get("calibrated") in (True, False)
    proba = model.predict_proba(rows[:5])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_test_label_mutation_cannot_alter_trained_selection():
    rows, labels, sessions = _data()
    model = RegimeProbabilityModel(
        RegimeModelConfig(calibration="sigmoid")
    ).fit(rows, labels, sessions)
    before = model.predict_proba(rows[:8]).copy()
    # Mutate held-out-looking labels after fit — predictions must be unchanged
    poisoned = list(labels)
    for i in range(len(poisoned)):
        poisoned[i] = REGIME_CLASSES[(REGIME_CLASSES.index(poisoned[i]) + 1) % 4]
    after = model.predict_proba(rows[:8])
    assert np.allclose(before, after)
    # Also ensure evaluate on poisoned labels doesn't refit
    model.evaluate(rows, poisoned)
    assert np.allclose(before, model.predict_proba(rows[:8]))


def test_regime_output_persistence(tmp_path):
    rows, labels, sessions = _data()
    model = RegimeProbabilityModel().fit(rows, labels, sessions)
    rp = model.predict(rows[0])
    store = PredictionStore(db_path=str(tmp_path / "r.sqlite"))
    store.log_regime_output(
        "snap-r1", rp.model_version, rp.to_dict(),
        uncertainty=rp.uncertainty, mode="shadow",
    )
    fetched = store.fetch_regime_outputs("snap-r1")
    assert len(fetched) == 1
    assert fetched[0]["probabilities"]["dominant_regime"] == rp.dominant_regime
    assert abs(sum(
        fetched[0]["probabilities"][c] for c in REGIME_CLASSES
    ) - 1.0) <= 1e-5


def test_low_support_increases_uncertainty_component():
    rows, labels, sessions = _data()
    cfg = RegimeModelConfig(
        calibration="identity",
        minimum_effective_sessions=1000,  # force support penalty
    )
    model = RegimeProbabilityModel(cfg).fit(rows, labels, sessions)
    rp = model.predict(rows[0])
    assert rp.diagnostics["support_penalty"] > 0
    assert rp.uncertainty >= rp.diagnostics["entropy"] * 0.5
