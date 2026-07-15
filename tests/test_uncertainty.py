"""
tests/test_uncertainty.py
=========================
V3 Part 1 §7 / §12 — composite uncertainty, missing-component reweighting,
and monotonicity.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.uncertainty import (
    ABSTAIN_SHADOW_THRESHOLD,
    compose_uncertainty,
    data_quality_uncertainty,
    ensemble_uncertainty_classification,
    ensemble_uncertainty_regression,
    model_age_uncertainty,
)


def test_missing_components_reweighted_not_zero():
    u = compose_uncertainty(
        ensemble=0.4,
        out_of_distribution=0.6,
        # conformal / calibration / dq / age missing
    )
    assert "missing_conformal_component" in u.reasons
    assert "missing_calibration_component" in u.reasons
    assert u.conformal is None
    # Composite is weighted mean of available only (0.4 and 0.6)
    # weights 0.25 and 0.25 → renorm 0.5 each → 0.5
    assert u.composite == pytest.approx(0.5)
    assert u.diagnostics["renormalized_weights"]["ensemble"] == pytest.approx(0.5)


def test_no_silent_zero_when_all_missing():
    u = compose_uncertainty()
    assert u.composite == 1.0
    assert "no_uncertainty_components_available" in u.reasons


def test_greater_ensemble_disagreement_does_not_reduce_uncertainty():
    low = ensemble_uncertainty_classification([[0.5, 0.51, 0.49]])
    # reshape: pass as (n_estimators,) for one obs — use list of lists as columns
    tight = ensemble_uncertainty_classification([
        [0.50], [0.51], [0.49], [0.50], [0.50],
    ])
    wide = ensemble_uncertainty_classification([
        [0.1], [0.9], [0.2], [0.8], [0.5],
    ])
    assert wide >= tight - 1e-12
    assert 0.0 <= low <= 1.0


def test_greater_ood_does_not_reduce_composite():
    a = compose_uncertainty(out_of_distribution=0.2, ensemble=0.3)
    b = compose_uncertainty(out_of_distribution=0.9, ensemble=0.3)
    assert b.composite >= a.composite - 1e-12


def test_worse_data_quality_does_not_reduce_uncertainty():
    good, _ = data_quality_uncertainty(
        feature_coverage=0.95, required_field_coverage=1.0,
        max_source_age_sec=1.0, arbitrage_violations=0)
    bad, reasons = data_quality_uncertainty(
        feature_coverage=0.4, required_field_coverage=0.5,
        max_source_age_sec=60.0, arbitrage_violations=3, feed_failover=True)
    assert bad >= good - 1e-12
    assert reasons


def test_calibration_degradation_monotonic_in_composite():
    a = compose_uncertainty(calibration=0.1, ensemble=0.2)
    b = compose_uncertainty(calibration=0.8, ensemble=0.2)
    assert b.composite >= a.composite - 1e-12


def test_model_age_increases_uncertainty():
    young, _ = model_age_uncertainty(artifact_age_days=1.0)
    old, reasons = model_age_uncertainty(
        artifact_age_days=40.0, days_since_last_eval=10.0,
        missing_recent_sessions=3)
    assert old >= young - 1e-12
    assert "artifact_age" in reasons or "stale_evaluation" in reasons


def test_abstain_shadow_flag():
    u = compose_uncertainty(ensemble=0.95, out_of_distribution=0.95)
    assert u.composite >= ABSTAIN_SHADOW_THRESHOLD
    assert "ABSTAIN_SHADOW" in u.reasons


def test_regression_iqr_uncertainty():
    tight = ensemble_uncertainty_regression([
        [1.0], [1.01], [0.99], [1.0], [1.02],
    ])
    wide = ensemble_uncertainty_regression([
        [0.0], [5.0], [-2.0], [3.0], [1.0],
    ])
    assert wide >= tight - 1e-12


def test_determinism_compose():
    a = compose_uncertainty(ensemble=0.3, conformal=0.4, out_of_distribution=0.5)
    b = compose_uncertainty(ensemble=0.3, conformal=0.4, out_of_distribution=0.5)
    assert a.to_dict() == b.to_dict()


def test_session_bootstrap_ensemble_clamps_to_5_9():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from prediction.uncertainty import SessionBootstrapEnsemble

    rng = np.random.default_rng(0)
    rows, y, sessions = [], [], []
    for s in range(6):
        date = f"2026-09-{s + 1:02d}"
        for j in range(5):
            x = float(rng.standard_normal())
            rows.append({"x": x})
            y.append(int(x > 0))
            sessions.append(date)

    def factory():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=200,
                                       random_state=0)),
        ])

    low = SessionBootstrapEnsemble(n_estimators=2, seed=1).fit(
        rows, y, sessions, factory)
    mid = SessionBootstrapEnsemble(n_estimators=7, seed=1).fit(
        rows, y, sessions, factory)
    high = SessionBootstrapEnsemble(n_estimators=20, seed=1).fit(
        rows, y, sessions, factory)
    assert 5 <= len(low.estimators) <= 9
    assert len(mid.estimators) == 7
    assert 5 <= len(high.estimators) <= 9
    assert len(high.estimators) == 9
