"""
tests/test_prediction_runtime_loader.py
=======================================
UNIFIED PR1 — PredictionRuntime.from_deployment_bundle fail-closed loading.
"""
from __future__ import annotations

import pytest

from prediction.deployment import DeploymentBundle, write_deployment_bundle
from prediction.registry import ModelRegistry, RegistryError
from prediction.runtime import PredictionRuntime, PredictionRuntimeError


class _DummyModel:
    pass


def _save_v1_model(reg: ModelRegistry, model_id_hint: str = "m") -> str:
    # Minimal v1-compatible save via registry.save
    return reg.save(
        _DummyModel(),
        model_type="test",
        target="dummy",
        horizon="30m",
        feature_version="v2.0.0",
        status="shadow",
        label_version="v2.0.0",
        crossfit_config={"n_folds": 2},
        fold_hash="foldhash",
        oof_metrics={"brier": 0.2},
        calibration_artifact={"method": "none"},
        uncertainty_method="none",
        training_feature_distribution_hash="abc",
        required_input_fields=[],
        dependency_versions={},
        git_commit="test",
    )


def test_shadow_loads_without_group(tmp_path):
    reg = ModelRegistry(str(tmp_path / "models"))
    bundle = DeploymentBundle(
        deployment_id="d1",
        mode="shadow",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="legacy",
        fallback_policy="abstain",
    )
    rt = PredictionRuntime.from_deployment_bundle(bundle, reg)
    assert rt.is_heuristic is True
    assert rt.bundle.mode == "shadow"


def test_champion_rejects_heuristic(tmp_path):
    reg = ModelRegistry(str(tmp_path / "models"))
    bundle = DeploymentBundle(
        deployment_id="d1",
        mode="champion",
        prediction_model_group_id="missing",
        candidate_value_model_id="cv",
        candidate_rank_model_id="cr",
        fill_probability_model_id="fp",
        fill_concession_model_id="fc",
        meta_model_id="mm",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="v3",
        fallback_policy="abstain",
    )
    with pytest.raises(PredictionRuntimeError):
        PredictionRuntime.from_deployment_bundle(bundle, reg, strict=True)


def test_feature_version_mismatch_fails(tmp_path):
    reg = ModelRegistry(str(tmp_path / "models"))
    mid = _save_v1_model(reg)
    g = reg.save_group(
        component_model_ids={"direction": mid},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        status="shadow",
    )
    bundle = DeploymentBundle(
        deployment_id="d1",
        mode="shadow",
        prediction_model_group_id=g.group_id,
        feature_version="v9.9.9",
        label_version="v2.0.0",
        authority_source="legacy",
        fallback_policy="abstain",
    )
    with pytest.raises(PredictionRuntimeError, match="feature-version"):
        PredictionRuntime.from_deployment_bundle(bundle, reg, strict=True)


def test_status_permission_mismatch(tmp_path):
    reg = ModelRegistry(str(tmp_path / "models"))
    mid = _save_v1_model(reg)
    reg.set_status(mid, "research")
    g = reg.save_group(
        component_model_ids={"direction": mid},
        feature_version="v2.0.0",
        label_version="v2.0.0",
        status="research",
    )
    bundle = DeploymentBundle(
        deployment_id="d1",
        mode="candidate",
        prediction_model_group_id=g.group_id,
        candidate_value_model_id=mid,
        candidate_rank_model_id=mid,
        fill_probability_model_id=mid,
        fill_concession_model_id=mid,
        meta_model_id=mid,
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="v3",
        fallback_policy="abstain",
    )
    with pytest.raises(PredictionRuntimeError):
        PredictionRuntime.from_deployment_bundle(bundle, reg, strict=True)


def test_missing_required_artifact_fails(tmp_path):
    reg = ModelRegistry(str(tmp_path / "models"))
    bundle = DeploymentBundle(
        deployment_id="d1",
        mode="candidate",
        prediction_model_group_id="nope",
        candidate_value_model_id="nope",
        candidate_rank_model_id="nope",
        fill_probability_model_id="nope",
        fill_concession_model_id="nope",
        meta_model_id="nope",
        feature_version="v2.0.0",
        label_version="v2.0.0",
        authority_source="v3",
        fallback_policy="abstain",
    )
    with pytest.raises(PredictionRuntimeError):
        PredictionRuntime.from_deployment_bundle(bundle, reg)
