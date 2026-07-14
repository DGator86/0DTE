"""
tests/test_registry_v2.py
=========================
V3 Part 1 §10 — registry schema version 2 with v1 read-only compatibility.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from prediction.models.direction import DirectionModel, DirectionModelConfig
from prediction.registry import (
    SCHEMA_VERSION,
    ModelRegistry,
    RegistryError,
)


CFG = DirectionModelConfig(
    horizon="30m", c_grid=(1.0,), l1_ratio_grid=(0.0,),
    class_weight_options=(None,), max_iter=300,
)


def _model():
    rng = np.random.default_rng(9)
    rows, y, sessions = [], [], []
    for s in range(6):
        for _ in range(15):
            x = rng.standard_normal()
            rows.append({"x": float(x)})
            y.append(int(rng.uniform() < 1 / (1 + np.exp(-2 * x))))
            sessions.append(f"2026-10-{s + 1:02d}")
    return DirectionModel(config=CFG).fit(rows, y, sessions)


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(directory=str(tmp_path / "models"))


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_save_load_v2_roundtrip(registry):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v2.0.0", label_version="v2.0.0",
        hyperparameters=model.metadata["best_params"],
        training_sessions=model.metadata["train_sessions"],
        calibration_sessions=model.metadata["calibration_sessions"],
        crossfit_config=model.metadata.get("crossfit_config", {}),
        oof_metrics={"log_loss": 0.5},
        calibration_artifact={"method": "sigmoid"},
        uncertainty_method="composite_v3",
        git_commit="abc",
        dependency_versions={"sklearn": "1.0"},
        required_input_fields=["x"],
    )
    loaded, meta = registry.load(mid, expected_feature_version="v2.0.0",
                                 expected_target="up_30m")
    assert meta["schema_version"] == 2
    assert meta["label_version"] == "v2.0.0"
    assert meta["calibration_artifact"]["method"] == "sigmoid"
    assert np.allclose(
        model.predict_proba([{"x": 0.1}]),
        loaded.predict_proba([{"x": 0.1}]))


def test_v1_metadata_readable_readonly(registry, tmp_path):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v2.0.0")
    # Downgrade metadata to v1 (strip v2 fields)
    path = registry._meta_path(mid)
    with open(path) as f:
        meta = json.load(f)
    meta["schema_version"] = 1
    for k in ("label_version", "crossfit_config", "fold_hash", "oof_metrics",
              "calibration_artifact", "uncertainty_method",
              "training_feature_distribution_hash", "required_input_fields",
              "optional_input_fields", "dependency_versions", "git_commit",
              "slice_metrics", "feature_schema_hash"):
        meta.pop(k, None)
    with open(path, "w") as f:
        json.dump(meta, f)
    loaded_meta = registry.load_metadata(mid, validate_v2=False)
    assert loaded_meta["schema_version"] == 1
    # load() should also work for v1 without v2 validation
    obj, _ = registry.load(mid)
    assert obj is not None


def test_v2_missing_calibration_fails_closed(registry):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v2.0.0",
        calibration_artifact={"method": "sigmoid"})
    path = registry._meta_path(mid)
    with open(path) as f:
        meta = json.load(f)
    meta["calibration_artifact"] = {}
    with open(path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(RegistryError, match="calibration artifact"):
        registry.load(mid)


def test_v2_missing_oof_metrics_fails_closed(registry):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v2.0.0")
    path = registry._meta_path(mid)
    with open(path) as f:
        meta = json.load(f)
    del meta["oof_metrics"]
    with open(path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(RegistryError, match="oof_metrics|OOF|missing required"):
        registry.load(mid)


def test_required_feature_absent_fails_closed(registry):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v2.0.0",
        required_input_fields=["signal", "x"])
    with pytest.raises(RegistryError, match="required feature absent"):
        registry.load(mid, required_input_fields=["x"])


def test_newer_feature_version_than_live_fails(registry):
    model = _model()
    mid = registry.save(
        model, model_type="direction", target="up_30m", horizon="30m",
        feature_version="v9.0.0")
    with pytest.raises(RegistryError, match="newer than live"):
        registry.load(mid, live_feature_version="v2.0.0")
