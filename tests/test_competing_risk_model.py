"""
tests/test_competing_risk_model.py
==================================
V3 Part 2 PR13 — competing-risk hazards / incidence (§45).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.competing_risk import (
    CompetingRiskForecast,
    CompetingRiskModel,
    expected_event_time,
    hazards_to_incidence,
)


def test_hazards_bounded_and_incidence_identity():
    ht = [0.1, 0.05, 0.02, 0.0]
    hs = [0.05, 0.05, 0.01, 0.0]
    out = hazards_to_incidence(ht, hs)
    assert np.all(out["h_target"] + out["h_stop"] <= 1.0 + 1e-12)
    assert np.all(np.diff(out["survival"]) <= 1e-12)  # non-increasing
    assert abs(
        out["p_target_first"] + out["p_stop_first"] + out["p_neither"] - 1.0
    ) <= 1e-6
    # Cumulative incidence monotone in cumsum
    assert np.all(np.diff(np.cumsum(out["ci_target"])) >= -1e-12)


def test_expected_time_none_when_negligible():
    assert expected_event_time([0.0, 0.0], 0.0) is None
    et = expected_event_time([0.1, 0.2, 0.1], 0.4)
    assert et == pytest.approx((1 * 0.1 + 2 * 0.2 + 3 * 0.1) / 0.4)


def test_forecast_contract_sum():
    with pytest.raises(ValueError):
        CompetingRiskForecast(
            p_target_first=0.5, p_stop_first=0.5, p_neither=0.5,
            expected_time_target=None, expected_time_stop=None,
            target_cumulative_incidence=(0.5,),
            stop_cumulative_incidence=(0.5,),
            survival_curve=(1.0, 0.0),
            uncertainty=0.0, support_rows=0, support_sessions=0,
            model_version="t",
        )


def test_model_fit_predict():
    rng = np.random.default_rng(0)
    rows, labels, sessions = [], [], []
    for i in range(120):
        dist_t = float(rng.uniform(0.5, 3.0))
        dist_s = float(rng.uniform(0.5, 3.0))
        # Closer target → more target events
        p_t = 0.4 * (1.0 / dist_t)
        p_s = 0.4 * (1.0 / dist_s)
        u = rng.random()
        if u < p_t:
            lab = 1
        elif u < p_t + p_s:
            lab = 2
        else:
            lab = 0
        rows.append({
            "target_distance_expected_move": dist_t,
            "stop_distance_expected_move": dist_s,
            "future_minute": float(i % 10),
            "time_fraction": (i % 10) / 10.0,
        })
        labels.append(lab)
        sessions.append(f"S{i % 8}")
    model = CompetingRiskModel().fit(rows, labels, sessions)
    # Build a 5-step path of features
    steps = [
        {"target_distance_expected_move": 1.0,
         "stop_distance_expected_move": 2.0,
         "future_minute": float(t), "time_fraction": t / 5.0}
        for t in range(5)
    ]
    fc = model.forecast_from_path_features(steps)
    assert abs(fc.p_target_first + fc.p_stop_first + fc.p_neither - 1.0) <= 1e-5
    assert len(fc.survival_curve) == 6
    assert fc.survival_curve[0] == pytest.approx(1.0)
    for i in range(1, len(fc.survival_curve)):
        assert fc.survival_curve[i] <= fc.survival_curve[i - 1] + 1e-12
