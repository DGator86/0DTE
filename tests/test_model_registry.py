"""
tests/test_model_registry.py
============================
PR 4 acceptance — model registry (handoff §19):
  * save/load round trip reproduces identical predictions (determinism
    given artifact and features);
  * loads FAIL CLOSED on tampered artifacts, missing metadata, unsupported
    schema versions, and feature-version/target mismatches;
  * metadata carries hashes, sessions, hyperparameters, and status history;
  * status transitions are recorded, not destructive.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from prediction.models.direction import DirectionModel, DirectionModelConfig
from prediction.registry import (SCHEMA_VERSION, STATUSES, ModelRegistry,
                                 RegistryError)

CFG = DirectionModelConfig(horizon="30m", c_grid=(1.0,), l1_ratio_grid=(0.0,),
                           class_weight_options=(None,), max_iter=300)


def _tiny_model():
    rng = np.random.default_rng(5)
    rows, y, sessions = [], [], []
    for s in range(6):
        for _ in range(20):
            x = rng.standard_normal()
            rows.append({"x": x})
            y.append(int(rng.uniform() < 1 / (1 + np.exp(-2 * x))))
            sessions.append(f"2026-07-{s + 1:02d}")
    return DirectionModel(config=CFG).fit(rows, y, sessions), rows


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(directory=str(tmp_path / "models"))


def _save(registry, model):
    return registry.save(
        model, model_type="direction_elasticnet", target="up_30m",
        horizon="30m", feature_version="v2.0.0",
        hyperparameters=model.metadata["best_params"],
        metrics=model.metadata["calibration_metrics"],
        training_sessions=model.metadata["train_sessions"],
        calibration_sessions=model.metadata["calibration_sessions"],
        data_hash="deadbeef", author="pytest")


class TestRoundTrip:
    def test_identical_predictions_after_reload(self, registry):
        model, rows = _tiny_model()
        mid = _save(registry, model)
        loaded, meta = registry.load(mid)
        assert np.array_equal(model.predict_proba(rows),
                              loaded.predict_proba(rows))
        assert meta["model_id"] == mid

    def test_metadata_fields(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        meta = registry.load_metadata(mid)
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["target"] == "up_30m"
        assert meta["feature_version"] == "v2.0.0"
        assert meta["artifact_hash"] and len(meta["artifact_hash"]) == 64
        assert meta["configuration_hash"]
        assert meta["data_hash"] == "deadbeef"
        assert meta["training_sessions"] == model.metadata["train_sessions"]
        assert meta["training_start"] == "2026-07-01"
        assert meta["training_end"] == "2026-07-06"
        assert meta["status"] == "research"
        assert meta["status_history"][0]["status"] == "research"

    def test_model_id_embeds_artifact_hash(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        meta = registry.load_metadata(mid)
        assert mid.endswith(meta["artifact_hash"][:12])


class TestFailClosed:
    def test_missing_metadata(self, registry):
        with pytest.raises(RegistryError, match="no metadata"):
            registry.load("ghost-model")

    def test_tampered_artifact(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        path = registry._artifact_path(mid)
        with open(path, "ab") as f:
            f.write(b"tamper")
        with pytest.raises(RegistryError, match="hash mismatch"):
            registry.load(mid)

    def test_missing_artifact(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        os.unlink(registry._artifact_path(mid))
        with pytest.raises(RegistryError, match="missing artifact"):
            registry.load(mid)

    def test_unsupported_schema_version(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        meta_path = registry._meta_path(mid)
        with open(meta_path) as f:
            meta = json.load(f)
        meta["schema_version"] = 999
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with pytest.raises(RegistryError, match="unsupported registry schema"):
            registry.load(mid)

    def test_feature_version_mismatch(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        with pytest.raises(RegistryError, match="feature-version mismatch"):
            registry.load(mid, expected_feature_version="v9.9.9")

    def test_target_mismatch(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        with pytest.raises(RegistryError, match="target mismatch"):
            registry.load(mid, expected_target="up_close")

    def test_corrupt_metadata_json(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        with open(registry._meta_path(mid), "w") as f:
            f.write("{not json")
        with pytest.raises(RegistryError, match="unreadable metadata"):
            registry.load(mid)

    def test_unknown_status_rejected(self, registry):
        model, _ = _tiny_model()
        with pytest.raises(RegistryError, match="unknown status"):
            registry.save(model, model_type="d", target="t", horizon=None,
                          feature_version="v", status="production!!")


class TestStatus:
    def test_transition_history(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        registry.set_status(mid, "shadow", note="starting shadow run")
        meta = registry.set_status(mid, "candidate", note="passed folds")
        assert meta["status"] == "candidate"
        hist = [h["status"] for h in meta["status_history"]]
        assert hist == ["research", "shadow", "candidate"]
        assert meta["status_history"][1]["note"] == "starting shadow run"

    def test_all_statuses_accepted(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        for st in STATUSES:
            assert registry.set_status(mid, st)["status"] == st

    def test_invalid_status_rejected(self, registry):
        model, _ = _tiny_model()
        mid = _save(registry, model)
        with pytest.raises(RegistryError):
            registry.set_status(mid, "live")

    def test_list_models_filters(self, registry):
        m1, _ = _tiny_model()
        id1 = _save(registry, m1)
        registry.set_status(id1, "shadow")
        assert [m["model_id"] for m in registry.list_models("shadow")] == [id1]
        assert registry.list_models("champion") == []
        assert len(registry.list_models()) == 1
