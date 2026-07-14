"""
tests/test_registry_mode_permissions.py
=======================================
V3 Part 3 PR29 — registry mode permissions (§30 / §52).
"""
from __future__ import annotations

import pytest

from prediction.deployment import DeploymentError, assert_mode_permission
from prediction.registry import RegistryError, assert_load_mode_allowed


def test_shadow_cannot_load_as_champion():
    with pytest.raises(DeploymentError):
        assert_mode_permission("shadow", "champion")


def test_candidate_ok_for_advisory():
    assert_mode_permission("candidate", "advisory")


def test_assert_load_mode_allowed():
    with pytest.raises(RegistryError):
        assert_load_mode_allowed({"model_id": "m", "status": "shadow"}, "champion")
