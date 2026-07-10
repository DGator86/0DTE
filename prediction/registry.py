"""
prediction/registry.py
======================
Model registry: versioned, hashed, auditable model artifacts
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §19).

Layout — one pair of files per model under the registry directory:

    <model_id>.joblib   the pickled model object (atomic write)
    <model_id>.json     metadata: target, horizon, feature version,
                        training/calibration sessions, hyperparameters,
                        metrics, artifact SHA256, schema version, status
                        (+ full status history)

Loading FAILS CLOSED (§19.4): a missing metadata file, artifact-hash
mismatch, unsupported schema version, or feature-version/target mismatch
raises RegistryError instead of quietly serving a wrong or tampered model.
Statuses follow the §19.2 vocabulary; promotion remains a human action that
just moves a status pointer.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

SCHEMA_VERSION = 1

STATUSES = ("research", "shadow", "candidate", "pending_review",
            "champion", "rejected", "archived")


class RegistryError(RuntimeError):
    """Fail-closed load/save error — never serve a questionable artifact."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: str, write_fn) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".reg_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@dataclass
class ModelRegistry:
    directory: str = "models"

    def __post_init__(self):
        os.makedirs(self.directory, exist_ok=True)

    # -- paths -------------------------------------------------------------------
    def _artifact_path(self, model_id: str) -> str:
        return os.path.join(self.directory, f"{model_id}.joblib")

    def _meta_path(self, model_id: str) -> str:
        return os.path.join(self.directory, f"{model_id}.json")

    # -- save ---------------------------------------------------------------------
    def save(self, model, *, model_type: str, target: str,
             horizon: Optional[str], feature_version: str,
             hyperparameters: Optional[dict] = None,
             metrics: Optional[dict] = None,
             training_sessions: Optional[list] = None,
             calibration_sessions: Optional[list] = None,
             training_start: Optional[str] = None,
             training_end: Optional[str] = None,
             data_hash: Optional[str] = None,
             author: str = "",
             status: str = "research") -> str:
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        import joblib

        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        # dump to a temp file first so the hash names the artifact
        directory = self.directory
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".model_",
                                   suffix=".joblib.tmp")
        os.close(fd)
        try:
            joblib.dump(model, tmp)
            artifact_hash = _sha256_file(tmp)
            model_id = f"{model_type}-{target}-{artifact_hash[:12]}"
            artifact_path = self._artifact_path(model_id)
            os.replace(tmp, artifact_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

        sessions = sorted(training_sessions or [])
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "model_type": model_type,
            "target": target,
            "horizon": horizon,
            "feature_version": feature_version,
            "training_start": training_start or (sessions[0] if sessions else None),
            "training_end": training_end or (sessions[-1] if sessions else None),
            "training_sessions": sessions,
            "calibration_sessions": sorted(calibration_sessions or []),
            "data_hash": data_hash,
            "configuration_hash": _config_hash(hyperparameters or {}),
            "hyperparameters": hyperparameters or {},
            "metrics": metrics or {},
            "artifact_path": artifact_path,
            "artifact_hash": artifact_hash,
            "created_at": created_at,
            "author": author,
            "status": status,
            "status_history": [{"status": status, "at": created_at,
                                "note": "created"}],
        }
        payload = json.dumps(metadata, indent=2, sort_keys=True,
                             default=str).encode("utf-8")
        _atomic_write_bytes(self._meta_path(model_id),
                            lambda f: f.write(payload))
        return model_id

    # -- load (fail closed) ---------------------------------------------------------
    def load_metadata(self, model_id: str) -> dict:
        path = self._meta_path(model_id)
        if not os.path.exists(path):
            raise RegistryError(f"no metadata for model {model_id!r}")
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(f"unreadable metadata for {model_id!r}: {exc}")
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise RegistryError(
                f"unsupported registry schema {meta.get('schema_version')!r} "
                f"for {model_id!r} (supported: {SCHEMA_VERSION})")
        return meta

    def load(self, model_id: str, *,
             expected_feature_version: Optional[str] = None,
             expected_target: Optional[str] = None) -> tuple:
        """
        Return (model, metadata). Raises RegistryError when metadata is
        missing/corrupt, the artifact hash does not match, the schema version
        is unsupported, or the feature version / target is incompatible.
        """
        meta = self.load_metadata(model_id)
        artifact = self._artifact_path(model_id)
        if not os.path.exists(artifact):
            raise RegistryError(f"missing artifact for {model_id!r}")
        actual = _sha256_file(artifact)
        if actual != meta.get("artifact_hash"):
            raise RegistryError(
                f"artifact hash mismatch for {model_id!r}: metadata says "
                f"{meta.get('artifact_hash')!r}, file is {actual!r}")
        if (expected_feature_version is not None
                and meta.get("feature_version") != expected_feature_version):
            raise RegistryError(
                f"feature-version mismatch for {model_id!r}: model trained "
                f"on {meta.get('feature_version')!r}, live pipeline is "
                f"{expected_feature_version!r}")
        if expected_target is not None and meta.get("target") != expected_target:
            raise RegistryError(
                f"target mismatch for {model_id!r}: {meta.get('target')!r} "
                f"!= expected {expected_target!r}")
        import joblib
        return joblib.load(artifact), meta

    # -- status ------------------------------------------------------------------
    def set_status(self, model_id: str, status: str, note: str = "") -> dict:
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        meta = self.load_metadata(model_id)
        meta["status"] = status
        meta["status_history"].append({
            "status": status,
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": note,
        })
        payload = json.dumps(meta, indent=2, sort_keys=True,
                             default=str).encode("utf-8")
        _atomic_write_bytes(self._meta_path(model_id),
                            lambda f: f.write(payload))
        return meta

    def list_models(self, status: Optional[str] = None) -> list:
        out = []
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            try:
                meta = self.load_metadata(name[:-5])
            except RegistryError:
                continue                             # unreadable: not listed
            if status is None or meta.get("status") == status:
                out.append(meta)
        return out
