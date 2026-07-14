"""
tests/test_session_bootstrap.py
===============================
V3 Part 1 §11 — session-level bootstrap CIs (tick bootstrap prohibited).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.session_bootstrap import bootstrap_metric_by_session


def test_determinism():
    sessions = [f"s{i // 3}" for i in range(30)]
    values = list(np.linspace(0, 1, 30))
    a = bootstrap_metric_by_session(
        sessions, values, lambda xs: float(np.mean(xs)),
        n_bootstrap=200, seed=42)
    b = bootstrap_metric_by_session(
        sessions, values, lambda xs: float(np.mean(xs)),
        n_bootstrap=200, seed=42)
    assert a == b
    assert a["n_sessions"] == 10
    assert a["n_rows"] == 30
    assert a["lower"] <= a["point_estimate"] <= a["upper"]


def test_single_session_collapses():
    out = bootstrap_metric_by_session(
        ["s0", "s0", "s0"], [1.0, 2.0, 3.0],
        lambda xs: float(np.mean(xs)), n_bootstrap=50, seed=1)
    assert out["n_sessions"] == 1
    assert out["lower"] == out["upper"] == out["point_estimate"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_metric_by_session(["a"], [1.0, 2.0], lambda xs: 0.0)
