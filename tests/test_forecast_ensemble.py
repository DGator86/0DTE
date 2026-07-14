"""
tests/test_forecast_ensemble.py
===============================
V3 Part 2 PR15 — forecast ensemble (§47).
"""
from __future__ import annotations

import pytest

from prediction.ensemble import (
    EnsembleConfig,
    ForecastEnsemble,
    enforce_max_weight,
    weighted_disagreement,
)


def test_weights_sum_to_one():
    ens = ForecastEnsemble(target="p_up", horizon="30m")
    out = ens.combine(
        {"global": 0.55, "moe": 0.60, "path": 0.52},
        oos_losses={"global": 0.3, "moe": 0.25, "path": 0.4},
    )
    assert abs(sum(out.component_weights.values()) - 1.0) <= 1e-6


def test_unavailable_excluded_and_renormalized():
    ens = ForecastEnsemble(target="p_up", horizon="30m")
    out = ens.combine(
        {"global": 0.55, "moe": None, "path": 0.52},
        oos_losses={"global": 0.3, "path": 0.4},
    )
    assert "moe" in out.missing_components
    assert "moe" not in out.component_weights
    assert abs(sum(out.component_weights.values()) - 1.0) <= 1e-6


def test_max_component_weight_enforced():
    w = enforce_max_weight({"a": 0.9, "b": 0.05, "c": 0.05}, maximum=0.6)
    assert w["a"] <= 0.6 + 1e-9
    assert abs(sum(w.values()) - 1.0) <= 1e-6


def test_negative_skill_fallback_only():
    ens = ForecastEnsemble(
        target="p_up", horizon="30m",
        cfg=EnsembleConfig(negative_skill_fallback_only=True,
                           fallback_weight=1e-6),
    )
    out = ens.combine(
        {"good": 0.6, "bad": 0.9},
        oos_losses={"good": 0.2, "bad": 0.1},
        negative_skill=["bad"],
    )
    assert out.component_weights["bad"] < out.component_weights["good"]
    assert out.component_weights["bad"] < 1e-3


def test_disagreement_raises_uncertainty():
    ens = ForecastEnsemble(target="p_up", horizon="30m")
    agree = ens.combine(
        {"a": 0.5, "b": 0.5},
        oos_losses={"a": 0.3, "b": 0.3},
        component_uncertainties={"a": 0.1, "b": 0.1},
    )
    differ = ens.combine(
        {"a": 0.1, "b": 0.9},
        oos_losses={"a": 0.3, "b": 0.3},
        component_uncertainties={"a": 0.1, "b": 0.1},
    )
    assert differ.disagreement > agree.disagreement
    assert differ.composite_uncertainty >= agree.composite_uncertainty


def test_identical_predictions_zero_disagreement():
    d = weighted_disagreement({"a": 0.4, "b": 0.4}, {"a": 0.5, "b": 0.5}, 0.4)
    assert d == pytest.approx(0.0)


def test_artifact_failure_visible():
    ens = ForecastEnsemble(target="p_up", horizon="30m")
    out = ens.combine(
        {"global": 0.55, "moe": 0.6, "legacy_monte_carlo": 0.5},
        oos_losses={"global": 0.3, "legacy_monte_carlo": 0.5},
        artifact_load_failures=["moe"],
    )
    assert "moe" in out.missing_components
    assert "moe" in out.diagnostics["artifact_load_failures"]


def test_legacy_mc_sole_survivor_flagged():
    ens = ForecastEnsemble(target="p_up", horizon="30m")
    out = ens.combine(
        {"moe": None, "path": None, "legacy_monte_carlo": 0.5},
        artifact_load_failures=["moe", "path"],
    )
    assert out.diagnostics.get("silent_dominance_prevented") is True
    assert out.composite_uncertainty >= 0.5
