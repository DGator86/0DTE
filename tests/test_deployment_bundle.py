"""
tests/test_deployment_bundle.py
================================
UNIFIED PR1 — DeploymentBundle, complete hashing, strict modes.
"""
from __future__ import annotations

import pytest

from prediction.deployment import (
    DeploymentBundle, DeploymentError, configuration_hash,
    load_deployment_bundle, validate_deployment_bundle,
    write_deployment_bundle,
)


def _bundle(**kw) -> DeploymentBundle:
    base = dict(
        deployment_id="dep-1",
        mode="shadow",
        prediction_model_group_id="g1",
        candidate_value_model_id="cv1",
        candidate_rank_model_id="cr1",
        fill_probability_model_id="fp1",
        fill_concession_model_id="fc1",
        meta_model_id="mm1",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        structural_state_version="v3.0.0",
        policy_version="v1",
        execution_version="v3",
        risk_version="v1",
        authority_source="legacy",
        fallback_policy="abstain",
    )
    base.update(kw)
    return DeploymentBundle(**base)


def test_valid_bundle_loads(tmp_path):
    path = str(tmp_path / "deployment.json")
    b = _bundle()
    ch = write_deployment_bundle(path, b)
    loaded = load_deployment_bundle(path)
    assert loaded.deployment_id == "dep-1"
    assert loaded.configuration_hash == ch
    assert configuration_hash(loaded.to_dict()) == ch


def test_hash_changes_when_feature_version_changes():
    a = _bundle(feature_version="v2.0.0")
    b = _bundle(feature_version="v2.1.0")
    assert configuration_hash(a.to_dict()) != configuration_hash(b.to_dict())


def test_hash_changes_when_fallback_policy_changes():
    a = _bundle(fallback_policy="abstain")
    b = _bundle(fallback_policy="legacy")
    assert configuration_hash(a.to_dict()) != configuration_hash(b.to_dict())


def test_champion_requires_trained_artifacts():
    b = _bundle(mode="champion", prediction_model_group_id=None,
                candidate_value_model_id=None, candidate_rank_model_id=None,
                fill_probability_model_id=None, fill_concession_model_id=None,
                meta_model_id=None, approved_review_id="rev-1",
                rollback_deployment_id="prior")
    with pytest.raises(DeploymentError, match="requires trained"):
        validate_deployment_bundle(b)


def test_champion_requires_approved_review_id():
    b = _bundle(mode="champion", approved_review_id=None,
                rollback_deployment_id="prior")
    with pytest.raises(DeploymentError, match="approved_review_id"):
        validate_deployment_bundle(b)


def test_candidate_cannot_use_heuristic_flag():
    b = _bundle(mode="candidate")
    assert b.allows_heuristic_fallback() is False
    assert b.requires_trained_artifacts() is True


def test_shadow_allows_heuristic_flag():
    b = _bundle(mode="shadow", prediction_model_group_id=None,
                candidate_value_model_id=None, candidate_rank_model_id=None,
                fill_probability_model_id=None, fill_concession_model_id=None,
                meta_model_id=None)
    validate_deployment_bundle(b)
    assert b.allows_heuristic_fallback() is True


def test_invalid_fallback_policy():
    b = _bundle(fallback_policy="magic")
    with pytest.raises(DeploymentError, match="fallback_policy"):
        validate_deployment_bundle(b)
