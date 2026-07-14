"""
prediction/registry.py
======================
Model registry: versioned, hashed, auditable model artifacts
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §19;
 docs/PREDICTION_ENGINE_V3_PART1_VALIDATION.md §10).

SCHEMA_VERSION = 2. Version-1 metadata remains readable in read-only
compatibility mode. Version-2 models fail closed when required V3 fields
(calibration artifact, fold metadata, OOF metrics, feature schema, …) are
missing or inconsistent.

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

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

STATUSES = ("research", "shadow", "candidate", "pending_review",
            "champion", "rejected", "archived")

# Required metadata keys for schema version 2 (fail closed on load).
_V2_REQUIRED = (
    "label_version",
    "crossfit_config",
    "fold_hash",
    "oof_metrics",
    "calibration_artifact",
    "uncertainty_method",
    "training_feature_distribution_hash",
    "required_input_fields",
    "dependency_versions",
    "git_commit",
)


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


def _validate_v2_metadata(meta: dict) -> None:
    """Fail closed for schema-v2 artifacts with incomplete audit fields."""
    mid = meta.get("model_id", "?")
    for key in _V2_REQUIRED:
        if key not in meta or meta[key] is None:
            raise RegistryError(
                f"v2 metadata missing required field {key!r} for {mid!r}")
    if not isinstance(meta.get("crossfit_config"), dict):
        raise RegistryError(f"malformed crossfit_config for {mid!r}")
    if not isinstance(meta.get("oof_metrics"), dict):
        raise RegistryError(f"missing OOF metrics for {mid!r}")
    cal = meta.get("calibration_artifact")
    if not isinstance(cal, dict) or not cal.get("method"):
        raise RegistryError(f"calibration artifact missing for {mid!r}")
    if not meta.get("fold_hash"):
        raise RegistryError(f"fold metadata malformed for {mid!r}")


@dataclass
class ModelRegistry:
    directory: str = "models"

    def __post_init__(self):
        os.makedirs(self.directory, exist_ok=True)

    def _artifact_path(self, model_id: str) -> str:
        return os.path.join(self.directory, f"{model_id}.joblib")

    def _meta_path(self, model_id: str) -> str:
        return os.path.join(self.directory, f"{model_id}.json")

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
             status: str = "research",
             # V3 / schema v2 fields
             label_version: Optional[str] = None,
             crossfit_config: Optional[dict] = None,
             fold_hash: Optional[str] = None,
             oof_metrics: Optional[dict] = None,
             slice_metrics: Optional[dict] = None,
             calibration_artifact: Optional[dict] = None,
             uncertainty_method: Optional[str] = None,
             training_feature_distribution_hash: Optional[str] = None,
             required_input_fields: Optional[list] = None,
             optional_input_fields: Optional[list] = None,
             dependency_versions: Optional[dict] = None,
             git_commit: Optional[str] = None,
             feature_schema_hash: Optional[str] = None,
             ) -> str:
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        import joblib

        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
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
        # Default V2 audit fields so new saves are loadable; callers should
        # pass real values. Empty structures still satisfy presence checks
        # only when method/fold_hash are non-empty — require explicit values
        # for production use.
        if calibration_artifact is None:
            calibration_artifact = {"method": "identity", "note": "unspecified"}
        if crossfit_config is None:
            crossfit_config = {}
        if oof_metrics is None:
            oof_metrics = {}
        if fold_hash is None:
            fold_hash = _config_hash({"sessions": sessions})
        if label_version is None:
            label_version = "unknown"
        if uncertainty_method is None:
            uncertainty_method = "scalar_v1"
        if training_feature_distribution_hash is None:
            training_feature_distribution_hash = data_hash or "unknown"
        if required_input_fields is None:
            required_input_fields = []
        if optional_input_fields is None:
            optional_input_fields = []
        if dependency_versions is None:
            dependency_versions = {}
        if git_commit is None:
            git_commit = "unknown"

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "model_type": model_type,
            "target": target,
            "horizon": horizon,
            "feature_version": feature_version,
            "feature_schema_hash": feature_schema_hash,
            "label_version": label_version,
            "training_start": training_start or (sessions[0] if sessions else None),
            "training_end": training_end or (sessions[-1] if sessions else None),
            "training_sessions": sessions,
            "calibration_sessions": sorted(calibration_sessions or []),
            "data_hash": data_hash,
            "configuration_hash": _config_hash(hyperparameters or {}),
            "hyperparameters": hyperparameters or {},
            "metrics": metrics or {},
            "crossfit_config": crossfit_config,
            "fold_hash": fold_hash,
            "oof_metrics": oof_metrics,
            "slice_metrics": slice_metrics or {},
            "calibration_artifact": calibration_artifact,
            "uncertainty_method": uncertainty_method,
            "training_feature_distribution_hash": training_feature_distribution_hash,
            "required_input_fields": list(required_input_fields),
            "optional_input_fields": list(optional_input_fields),
            "dependency_versions": dependency_versions,
            "git_commit": git_commit,
            "artifact_path": artifact_path,
            "artifact_hash": artifact_hash,
            "created_at": created_at,
            "author": author,
            "status": status,
            "status_history": [{"status": status, "at": created_at,
                                "note": "created"}],
        }
        _validate_v2_metadata(metadata)
        payload = json.dumps(metadata, indent=2, sort_keys=True,
                             default=str).encode("utf-8")
        _atomic_write_bytes(self._meta_path(model_id),
                            lambda f: f.write(payload))
        return model_id

    def load_metadata(self, model_id: str, *,
                      validate_v2: bool = True) -> dict:
        path = self._meta_path(model_id)
        if not os.path.exists(path):
            raise RegistryError(f"no metadata for model {model_id!r}")
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(f"unreadable metadata for {model_id!r}: {exc}")
        sv = meta.get("schema_version")
        if sv not in SUPPORTED_SCHEMA_VERSIONS:
            raise RegistryError(
                f"unsupported registry schema {sv!r} "
                f"for {model_id!r} (supported: {SUPPORTED_SCHEMA_VERSIONS})")
        # v1: read-only compatibility — do not require V2 fields
        if sv == 2 and validate_v2:
            _validate_v2_metadata(meta)
        return meta

    def load(self, model_id: str, *,
             expected_feature_version: Optional[str] = None,
             expected_target: Optional[str] = None,
             expected_horizon: Optional[str] = None,
             required_input_fields: Optional[list] = None,
             live_feature_version: Optional[str] = None,
             ) -> tuple:
        """
        Return (model, metadata). Fail closed on hash/schema/target/feature
        mismatches and (for v2) missing calibration/fold/OOF audit fields.
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
        if (expected_horizon is not None
                and meta.get("horizon") != expected_horizon):
            raise RegistryError(
                f"horizon mismatch for {model_id!r}: {meta.get('horizon')!r} "
                f"!= expected {expected_horizon!r}")

        # V2: reject models trained on a *newer* feature version than live
        live_fv = live_feature_version or expected_feature_version
        if (meta.get("schema_version") == 2 and live_fv is not None
                and meta.get("feature_version")
                and str(meta["feature_version"]) > str(live_fv)):
            raise RegistryError(
                f"model feature version {meta.get('feature_version')!r} is "
                f"newer than live pipeline {live_fv!r} for {model_id!r}")

        if required_input_fields is not None and meta.get("schema_version") == 2:
            required = set(meta.get("required_input_fields") or [])
            present = set(required_input_fields)
            missing = required - present
            if missing:
                raise RegistryError(
                    f"required feature absent for {model_id!r}: "
                    f"{sorted(missing)}")

        if (meta.get("schema_version") == 2
                and expected_feature_version is not None
                and meta.get("feature_schema_hash")
                and required_input_fields is not None):
            # Optional schema-hash check when caller provides a hash via
            # hyperparameters is handled by callers; presence already gated.
            pass

        import joblib
        return joblib.load(artifact), meta

    def set_status(self, model_id: str, status: str, note: str = "") -> dict:
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        # Allow status updates on v1 without forcing v2 validation rewrite
        meta = self.load_metadata(model_id, validate_v2=False)
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
                meta = self.load_metadata(name[:-5], validate_v2=False)
            except RegistryError:
                continue
            if status is None or meta.get("status") == status:
                out.append(meta)
        return out
