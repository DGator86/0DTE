"""
tests/test_dynamic_weights.py
=============================
V3 Part 3 PR27 — dynamic ensemble weights (§22 / §50).
"""
from __future__ import annotations

import pytest

from prediction.dynamic_weights import (
    DynamicWeightConfig, update_dynamic_weights,
)


def test_weights_sum_to_one_and_bounds():
    state = update_dynamic_weights(
        target="direction",
        as_of_session="2026-07-10",
        prior_weights={"a": 0.5, "b": 0.5},
        losses_20={"a": 0.1, "b": 0.5},
        losses_60={"a": 0.2, "b": 0.4},
        losses_full={"a": 0.2, "b": 0.4},
        cfg=DynamicWeightConfig(minimum_weight=0.05, maximum_weight=0.60),
    )
    assert pytest.approx(sum(state.weights.values()), abs=1e-9) == 1.0
    assert state.weights["a"] > state.weights["b"]
    assert all(v <= 0.60 + 1e-9 for v in state.weights.values())
    assert all(v >= 0.05 - 1e-9 for v in state.weights.values())


def test_worse_loss_lowers_weight():
    state = update_dynamic_weights(
        target="x", as_of_session="s",
        prior_weights={"good": 0.5, "bad": 0.5},
        losses_20={"good": 0.05, "bad": 1.0},
        losses_60={"good": 0.05, "bad": 1.0},
        losses_full={"good": 0.05, "bad": 1.0},
    )
    assert state.weights["good"] > state.weights["bad"]


def test_freeze_excluded():
    state = update_dynamic_weights(
        target="x", as_of_session="s",
        prior_weights={"a": 0.5, "b": 0.5},
        losses_20={"a": 0.1, "b": 0.1},
        losses_60={"a": 0.1, "b": 0.1},
        losses_full={"a": 0.1, "b": 0.1},
        drift_severity={"b": "FREEZE"},
    )
    assert "b" in state.excluded_models
    assert "b" not in state.weights
    assert pytest.approx(sum(state.weights.values())) == 1.0


def test_drift_watch_penalizes():
    base = update_dynamic_weights(
        target="x", as_of_session="s",
        prior_weights={"a": 0.5, "b": 0.5},
        losses_20={"a": 0.2, "b": 0.2},
        losses_60={"a": 0.2, "b": 0.2},
        losses_full={"a": 0.2, "b": 0.2},
    )
    watch = update_dynamic_weights(
        target="x", as_of_session="s",
        prior_weights={"a": 0.5, "b": 0.5},
        losses_20={"a": 0.2, "b": 0.2},
        losses_60={"a": 0.2, "b": 0.2},
        losses_full={"a": 0.2, "b": 0.2},
        drift_severity={"b": "WATCH"},
    )
    assert watch.weights["b"] < base.weights["b"]


def test_identical_history_identical_weights():
    kwargs = dict(
        target="x", as_of_session="s",
        prior_weights={"a": 0.5, "b": 0.5},
        losses_20={"a": 0.3, "b": 0.4},
        losses_60={"a": 0.3, "b": 0.4},
        losses_full={"a": 0.3, "b": 0.4},
    )
    assert update_dynamic_weights(**kwargs).to_dict() == \
        update_dynamic_weights(**kwargs).to_dict()
