"""
tests/test_deployment_bundle_artifacts.py
=========================================
PR E — fail-closed artifact loading + registry group validation +
atomic rollback pointer.

Covers: missing model, wrong hash, wrong feature/label version,
unsupported status, atomic rollback.
"""
from __future__ import annotations

import json
import os

import pytest

from prediction.deployment import (
    DeploymentBundle, DeploymentError, load_deployment_bundle,
    load_deployment_pointer, rollback_deployment, validate_bundle_artifacts,
    write_deployment_bundle,
)
from prediction.registry import ModelRegistry, RegistryError


def _save_model(registry: ModelRegistry, *, name: str,
                feature_version: str = "v2.0.0",
                label_version: str = "v2.0.0",
                status: str = "shadow") -> str:
    return registry.save(
        {"kind": name},
        model_type=name,
        target="t",
        horizon=None,
        feature_version=feature_version,
        label_version=label_version,
        status=status,
        hyperparameters={"k": name},
        data_hash=f"hash-{name}",
        author="pytest",
    )


def _bundle_with_ids(**kw) -> DeploymentBundle:
    base = dict(
        deployment_id="dep-live",
        mode="shadow",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        structural_state_version="v3.0.0",
        policy_version="v1",
        execution_version="v3",
        risk_version="v1",
        fallback_policy="abstain",
        reference_account_id="legacy",
    )
    base.update(kw)
    return DeploymentBundle(**base)


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(directory=str(tmp_path / "models"))


@pytest.fixture()
def filled(registry):
    """Registry with a valid group + Part-3 models at shadow status."""
    comps = {
        "direction": _save_model(registry, name="direction"),
        "range": _save_model(registry, name="range"),
    }
    group = registry.save_group(
        component_model_ids=comps,
        feature_version="v2.0.0",
        label_version="v2.0.0",
        structural_state_version="v3.0.0",
        status="shadow",
        group_id="grp-shadow",
    )
    ids = {
        "prediction_model_group_id": group.group_id,
        "candidate_value_model_id": _save_model(registry, name="cv"),
        "candidate_rank_model_id": _save_model(registry, name="cr"),
        "fill_probability_model_id": _save_model(registry, name="fp"),
        "fill_concession_model_id": _save_model(registry, name="fc"),
        "meta_model_id": _save_model(registry, name="mm"),
    }
    return ids


def test_validate_bundle_artifacts_ok(registry, filled):
    b = _bundle_with_ids(**filled)
    validate_bundle_artifacts(b, registry)


def test_missing_model_fails_closed(registry, filled):
    bad = dict(filled)
    bad["candidate_value_model_id"] = "does-not-exist"
    b = _bundle_with_ids(**bad)
    with pytest.raises(DeploymentError, match="missing model"):
        validate_bundle_artifacts(b, registry)


def test_missing_group_fails_closed(registry, filled):
    bad = dict(filled)
    bad["prediction_model_group_id"] = "no-such-group"
    b = _bundle_with_ids(**bad)
    with pytest.raises(DeploymentError, match="group"):
        validate_bundle_artifacts(b, registry)


def test_wrong_hash_fails_closed(registry, filled, tmp_path):
    mid = filled["meta_model_id"]
    artifact = os.path.join(registry.directory, f"{mid}.joblib")
    with open(artifact, "ab") as f:
        f.write(b"tamper")
    b = _bundle_with_ids(**filled)
    with pytest.raises(DeploymentError, match="fail-closed load|hash"):
        validate_bundle_artifacts(b, registry)


def test_wrong_feature_version_fails_closed(registry, filled):
    b = _bundle_with_ids(feature_version="v9.9.9", **filled)
    with pytest.raises(DeploymentError, match="feature"):
        validate_bundle_artifacts(b, registry)


def test_wrong_label_version_fails_closed(registry, filled):
    b = _bundle_with_ids(label_version="v9.9.9", **filled)
    with pytest.raises(DeploymentError, match="label"):
        validate_bundle_artifacts(b, registry)


def test_unsupported_status_fails_closed(registry, filled):
    # Research-status group cannot load into candidate mode.
    comps = {
        "direction": _save_model(registry, name="dir2", status="research"),
    }
    group = registry.save_group(
        component_model_ids=comps,
        feature_version="v2.0.0",
        label_version="v2.0.0",
        status="research",
        group_id="grp-research",
    )
    ids = dict(filled)
    ids["prediction_model_group_id"] = group.group_id
    # Promote part3 models to candidate so only the group status blocks.
    for slot in (
        "candidate_value_model_id", "candidate_rank_model_id",
        "fill_probability_model_id", "fill_concession_model_id",
        "meta_model_id",
    ):
        registry.set_status(ids[slot], "candidate")
    b = _bundle_with_ids(
        mode="candidate", candidate_account_id="cand", **ids)
    with pytest.raises(DeploymentError):
        validate_bundle_artifacts(b, registry)


def test_candidate_missing_slot_fails_closed(registry, filled):
    ids = dict(filled)
    ids["meta_model_id"] = None
    b = _bundle_with_ids(
        mode="candidate", candidate_account_id="cand", **ids)
    with pytest.raises(DeploymentError, match="requires trained|missing"):
        validate_bundle_artifacts(b, registry)


def test_group_component_feature_conflict(registry):
    a = _save_model(registry, name="a", feature_version="v2.0.0")
    b = _save_model(registry, name="b", feature_version="v2.1.0")
    with pytest.raises(RegistryError, match="feature"):
        registry.save_group(
            component_model_ids={"a": a, "b": b},
            feature_version="v2.0.0",
            label_version="v2.0.0",
            status="shadow",
        )


def test_atomic_rollback_restores_complete_bundle(tmp_path, registry, filled):
    path = str(tmp_path / "deployment.json")
    prior = _bundle_with_ids(deployment_id="prior", **filled)
    current_ids = dict(filled)
    # New group for current so rollback is observable.
    comps = {
        "direction": _save_model(registry, name="dir-cur"),
    }
    group = registry.save_group(
        component_model_ids=comps,
        feature_version="v2.0.0",
        label_version="v2.0.0",
        status="shadow",
        group_id="grp-current",
    )
    current_ids["prediction_model_group_id"] = group.group_id
    current = _bundle_with_ids(
        deployment_id="current",
        rollback_deployment_id="prior",
        **current_ids,
    )
    write_deployment_bundle(path, prior)
    prior_loaded = load_deployment_pointer(path)
    write_deployment_bundle(path, current)
    audit = rollback_deployment(
        path, prior_pointer=prior_loaded, reason="bad_oos",
        trigger_source="human", registry=registry)
    restored = load_deployment_bundle(path)
    assert restored.deployment_id == "prior"
    assert restored.prediction_model_group_id == filled[
        "prediction_model_group_id"]
    assert restored.feature_version == "v2.0.0"
    assert restored.fallback_policy == "abstain"
    assert audit["configuration_hash"] == prior_loaded["configuration_hash"]
    assert audit["human_or_automatic"] == "human"


def test_rollback_strict_missing_registry_artifact_fails(
        tmp_path, registry, filled):
    path = str(tmp_path / "deployment.json")
    prior = _bundle_with_ids(
        deployment_id="prior",
        mode="candidate",
        candidate_account_id="cand",
        **filled,
    )
    # Promote artifacts so candidate mode is valid at write time.
    registry.set_group_status(filled["prediction_model_group_id"], "candidate")
    for slot in (
        "candidate_value_model_id", "candidate_rank_model_id",
        "fill_probability_model_id", "fill_concession_model_id",
        "meta_model_id",
    ):
        registry.set_status(filled[slot], "candidate")
    for mid in registry.load_group(
            filled["prediction_model_group_id"]).component_model_ids.values():
        registry.set_status(mid, "candidate")
    write_deployment_bundle(path, prior)
    prior_ptr = load_deployment_pointer(path)
    # Delete a referenced artifact so rollback with registry fails closed.
    mid = filled["meta_model_id"]
    os.unlink(os.path.join(registry.directory, f"{mid}.json"))
    os.unlink(os.path.join(registry.directory, f"{mid}.joblib"))
    with pytest.raises(DeploymentError, match="missing from registry"):
        rollback_deployment(
            path, prior_pointer=prior_ptr, reason="test",
            registry=registry)
