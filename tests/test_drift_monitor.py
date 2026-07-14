"""
tests/test_drift_monitor.py
===========================
V3 Part 3 PR28 — drift monitoring (§24–§27 / §51).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.drift import (
    compute_drift_status, drift_weight_penalty, mean_shift_score,
    missingness_shift, population_stability_index, severity_from_composite,
)


def test_feature_shift_raises_psi():
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 500)
    shifted = rng.normal(2, 1, 500)
    psi = population_stability_index(ref, shifted)
    assert psi > population_stability_index(ref, ref)


def test_missingness_shift():
    assert missingness_shift(0.0, 0.5) > missingness_shift(0.1, 0.1)


def test_mean_shift():
    assert mean_shift_score([0, 0, 0, 0], [5, 5, 5, 5]) > 1.0


def test_missing_economic_not_zero_composite():
    # Only residual provided — composite equals residual (reweighted)
    st = compute_drift_status(
        model_id="m1", as_of_session="s",
        residual_drift=0.7,
        economic_drift=None,
    )
    assert st.composite == pytest.approx(0.7)
    assert "economic" in st.diagnostics["missing_components"]
    assert st.severity == "DEGRADED"


def test_severity_bands_and_actions():
    assert severity_from_composite(0.2) == "NORMAL"
    assert severity_from_composite(0.5) == "WATCH"
    assert severity_from_composite(0.7) == "DEGRADED"
    assert severity_from_composite(0.9) == "FREEZE"
    assert drift_weight_penalty("WATCH") < 1.0
    assert drift_weight_penalty("FREEZE") == 0.0


def test_force_freeze():
    st = compute_drift_status(
        model_id="m", as_of_session="s",
        feature_drift=0.0, force_freeze=True,
    )
    assert st.severity == "FREEZE"
