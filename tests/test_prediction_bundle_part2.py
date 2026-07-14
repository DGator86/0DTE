"""
tests/test_prediction_bundle_part2.py
=====================================
V3 Part 2 PR16 — PredictionBundle Part 2 field extensions (§34).
"""
from __future__ import annotations

from prediction.contracts import PredictionBundle


def _base(**kwargs):
    d = dict(
        snapshot_id="s1", ts="2026-07-14T15:00:00Z",
        session_date="2026-07-14", symbol="SPY",
    )
    d.update(kwargs)
    return PredictionBundle(**d)


def test_part2_defaults_safe():
    b = _base()
    assert b.regime_probabilities == {}
    assert b.regime_uncertainty is None
    assert b.dominant_regime is None
    assert b.return_distributions == {}
    assert b.competing_risk_forecasts == {}
    assert b.path_forecasts == {}
    assert b.ensemble_forecasts == {}
    assert b.structural_state_version is None


def test_old_bundle_dict_still_loads():
    # Simulate a Part 1 serialized bundle without Part 2 keys
    legacy = {
        "snapshot_id": "old", "ts": "t", "session_date": "d", "symbol": "SPY",
        "p_up_30m": 0.55, "uncertainty": 0.3,
    }
    b = PredictionBundle.from_dict(legacy)
    assert b.p_up_30m == 0.55
    assert b.regime_probabilities == {}


def test_part2_fields_roundtrip():
    b = _base(
        regime_probabilities={
            "long_gamma_pin": 0.4, "short_gamma_trend": 0.3,
            "flip_transition": 0.2, "volatility_expansion": 0.1,
        },
        regime_uncertainty=0.4,
        dominant_regime="long_gamma_pin",
        return_distributions={"30m": {"q50": 0.0}},
        structural_state_version="v3.0.0",
        forecast_model_group_version="v3.part2",
    )
    d = b.to_dict()
    b2 = PredictionBundle.from_dict(d)
    assert b2.dominant_regime == "long_gamma_pin"
    assert b2.structural_state_version == "v3.0.0"
    assert abs(sum(b2.regime_probabilities.values()) - 1.0) <= 1e-9
