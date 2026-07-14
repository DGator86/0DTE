"""
tests/test_deployment_pointer.py
================================
V3 Part 3 PR29 — deployment pointer (§38 / §53).
"""
from __future__ import annotations

import json

import pytest

from prediction.deployment import (
    DeploymentError, configuration_hash, load_deployment_pointer,
    mode_may_alter_legacy, mode_may_place_orders, validate_deployment_pointer,
    write_deployment_pointer,
)


def _ptr(**kw):
    base = {
        "mode": "shadow",
        "prediction_model_group": "g1",
        "candidate_value_model": "cv1",
        "candidate_rank_model": "cr1",
        "fill_probability_model": "fp1",
        "fill_concession_model": "fc1",
        "meta_model": "mm1",
    }
    base.update(kw)
    return base


def test_atomic_write_and_hash(tmp_path):
    path = str(tmp_path / "deployment.json")
    ch = write_deployment_pointer(path, _ptr())
    loaded = load_deployment_pointer(path)
    assert loaded["configuration_hash"] == ch
    assert configuration_hash(loaded) == ch


def test_invalid_mode_fails_closed():
    with pytest.raises(DeploymentError):
        validate_deployment_pointer(_ptr(mode="live_trading"))


def test_hash_mismatch_fails_closed(tmp_path):
    path = str(tmp_path / "deployment.json")
    write_deployment_pointer(path, _ptr())
    with open(path) as f:
        data = json.load(f)
    data["configuration_hash"] = "deadbeef"
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(DeploymentError, match="hash"):
        load_deployment_pointer(path)


def test_advisory_cannot_place_orders():
    assert mode_may_place_orders("advisory") is False
    assert mode_may_place_orders("shadow") is False
    assert mode_may_alter_legacy("shadow") is False
