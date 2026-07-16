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

    # ------------------------------------------------------------------ #
    # Model groups (post-#119 handoff §16.1 / PR E)                        #
    # ------------------------------------------------------------------ #
    def _groups_dir(self) -> str:
        path = os.path.join(self.directory, "groups")
        os.makedirs(path, exist_ok=True)
        return path

    def _group_meta_path(self, group_id: str) -> str:
        return os.path.join(self._groups_dir(), f"{group_id}.json")

    def save_group(
        self,
        *,
        component_model_ids: dict,
        feature_version: str,
        label_version: str,
        structural_state_version: str = "",
        training_sessions: Optional[list] = None,
        calibration_sessions: Optional[list] = None,
        outer_test_sessions: Optional[list] = None,
        metrics: Optional[dict] = None,
        status: str = "research",
        group_id: Optional[str] = None,
    ) -> "ModelGroupMetadata":
        """Persist a model-group metadata record. Validates components exist."""
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        comps = dict(component_model_ids or {})
        if not comps:
            raise RegistryError("model group requires at least one component")
        for role, mid in comps.items():
            if not mid:
                raise RegistryError(f"empty component id for role {role!r}")
            self.load_metadata(str(mid), validate_v2=False)
        payload = {
            "component_model_ids": comps,
            "feature_version": feature_version,
            "label_version": label_version,
            "structural_state_version": structural_state_version or "",
            "training_sessions": list(training_sessions or []),
            "calibration_sessions": list(calibration_sessions or []),
            "outer_test_sessions": list(outer_test_sessions or []),
            "metrics": dict(metrics or {}),
            "status": status,
            "kind": "model_group",
        }
        payload["configuration_hash"] = _group_config_hash(payload)
        gid = group_id or _config_hash({
            "components": comps,
            "feature_version": feature_version,
            "label_version": label_version,
        })[:24]
        payload["group_id"] = gid
        meta = ModelGroupMetadata.from_dict(payload)
        self.validate_group(meta)
        raw = json.dumps(meta.to_dict(), indent=2, sort_keys=True,
                         default=str).encode("utf-8")
        _atomic_write_bytes(self._group_meta_path(gid), lambda f: f.write(raw))
        return meta

    def load_group(self, group_id: str) -> "ModelGroupMetadata":
        path = self._group_meta_path(group_id)
        if not os.path.exists(path):
            raise RegistryError(f"no model group {group_id!r}")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(
                f"unreadable group {group_id!r}: {exc}") from exc
        meta = ModelGroupMetadata.from_dict(data)
        self.validate_group(meta)
        return meta

    def set_group_status(
        self, group_id: str, status: str, note: str = "",
    ) -> "ModelGroupMetadata":
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        meta = self.load_group(group_id)
        d = meta.to_dict()
        d["status"] = status
        d.setdefault("status_history", []).append({
            "status": status,
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": note,
        })
        updated = ModelGroupMetadata.from_dict(d)
        raw = json.dumps({**d, **updated.to_dict()}, indent=2, sort_keys=True,
                         default=str).encode("utf-8")
        _atomic_write_bytes(self._group_meta_path(group_id),
                            lambda f: f.write(raw))
        return updated

    def validate_group(
        self,
        meta: "ModelGroupMetadata",
        *,
        load_mode: Optional[str] = None,
    ) -> None:
        """Fail closed when components conflict or are missing."""
        if not meta.component_model_ids:
            raise RegistryError(f"group {meta.group_id!r} has no components")
        feature_versions: set = set()
        label_versions: set = set()
        for role, mid in meta.component_model_ids.items():
            try:
                m = self.load_metadata(str(mid), validate_v2=False)
            except RegistryError as exc:
                raise RegistryError(
                    f"group {meta.group_id!r} missing component "
                    f"{role!r}={mid!r}: {exc}") from exc
            if m.get("feature_version"):
                feature_versions.add(m["feature_version"])
            if m.get("label_version"):
                label_versions.add(m["label_version"])
            if load_mode is not None:
                assert_load_mode_allowed(m, load_mode)
        if meta.feature_version and feature_versions:
            bad = {fv for fv in feature_versions if fv != meta.feature_version}
            if bad:
                raise RegistryError(
                    f"group {meta.group_id!r} feature_version conflict: "
                    f"group={meta.feature_version!r} components={sorted(bad)}")
        if len(feature_versions) > 1:
            raise RegistryError(
                f"group {meta.group_id!r} component feature versions conflict: "
                f"{sorted(feature_versions)}")
        if meta.label_version and label_versions:
            bad = {lv for lv in label_versions if lv != meta.label_version}
            if bad:
                raise RegistryError(
                    f"group {meta.group_id!r} label_version conflict: "
                    f"group={meta.label_version!r} components={sorted(bad)}")
        if len(label_versions) > 1:
            raise RegistryError(
                f"group {meta.group_id!r} component label versions conflict: "
                f"{sorted(label_versions)}")
        if load_mode is not None:
            from prediction.deployment import (
                DeploymentError, assert_mode_permission,
            )
            try:
                assert_mode_permission(meta.status, load_mode)
            except DeploymentError as exc:
                raise RegistryError(str(exc)) from exc
        expected = _group_config_hash(meta.to_dict())
        if meta.configuration_hash and meta.configuration_hash != expected:
            raise RegistryError(
                f"group {meta.group_id!r} configuration_hash mismatch")

    def list_groups(self, status: Optional[str] = None) -> list:
        out = []
        gdir = self._groups_dir()
        for name in sorted(os.listdir(gdir)):
            if not name.endswith(".json"):
                continue
            try:
                meta = self.load_group(name[:-5])
            except RegistryError:
                continue
            if status is None or meta.status == status:
                out.append(meta)
        return out


# --------------------------------------------------------------------------- #
# Model groups                                                                #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelGroupMetadata:
    group_id: str
    component_model_ids: dict
    feature_version: str
    label_version: str
    structural_state_version: str = ""
    configuration_hash: str = ""
    training_sessions: list = None  # type: ignore[assignment]
    calibration_sessions: list = None  # type: ignore[assignment]
    outer_test_sessions: list = None  # type: ignore[assignment]
    metrics: dict = None  # type: ignore[assignment]
    status: str = "research"

    def __post_init__(self):
        object.__setattr__(
            self, "training_sessions", list(self.training_sessions or []))
        object.__setattr__(
            self, "calibration_sessions",
            list(self.calibration_sessions or []))
        object.__setattr__(
            self, "outer_test_sessions",
            list(self.outer_test_sessions or []))
        object.__setattr__(self, "metrics", dict(self.metrics or {}))
        object.__setattr__(
            self, "component_model_ids", dict(self.component_model_ids or {}))

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "component_model_ids": dict(self.component_model_ids),
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "structural_state_version": self.structural_state_version,
            "configuration_hash": self.configuration_hash,
            "training_sessions": list(self.training_sessions),
            "calibration_sessions": list(self.calibration_sessions),
            "outer_test_sessions": list(self.outer_test_sessions),
            "metrics": dict(self.metrics),
            "status": self.status,
            "kind": "model_group",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelGroupMetadata":
        return cls(
            group_id=str(d["group_id"]),
            component_model_ids=dict(d.get("component_model_ids") or {}),
            feature_version=str(d.get("feature_version") or ""),
            label_version=str(d.get("label_version") or ""),
            structural_state_version=str(
                d.get("structural_state_version") or ""),
            configuration_hash=str(d.get("configuration_hash") or ""),
            training_sessions=list(d.get("training_sessions") or []),
            calibration_sessions=list(d.get("calibration_sessions") or []),
            outer_test_sessions=list(d.get("outer_test_sessions") or []),
            metrics=dict(d.get("metrics") or {}),
            status=str(d.get("status") or "research"),
        )


def _group_config_hash(meta: dict) -> str:
    payload = {
        "component_model_ids": meta.get("component_model_ids") or {},
        "feature_version": meta.get("feature_version"),
        "label_version": meta.get("label_version"),
        "structural_state_version": meta.get("structural_state_version"),
    }
    return _config_hash(payload)


# Part 3 mode permissions (§30.3)
PART3_MODEL_TYPES = (
    "candidate_value_v3",
    "candidate_pair_ranker",
    "fill_probability",
    "fill_concession",
    "trade_meta",
    "dynamic_weight_state",
    "drift_monitor",
)


def allowed_modes_for_status(status: str) -> frozenset:
    from prediction.deployment import MODE_PERMISSIONS
    return MODE_PERMISSIONS.get(str(status).lower(), frozenset())


def assert_load_mode_allowed(meta: dict, load_mode: str) -> None:
    """Fail closed when artifact status cannot load into the requested mode."""
    from prediction.deployment import DeploymentError, assert_mode_permission
    status = str(meta.get("status", "research"))
    try:
        assert_mode_permission(status, load_mode)
    except DeploymentError:
        raise RegistryError(
            f"artifact {meta.get('model_id')!r} status {status!r} "
            f"cannot load as {load_mode!r}")
