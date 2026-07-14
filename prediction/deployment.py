"""
prediction/deployment.py
========================
Deployment modes, pointer, and atomic rollback
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §31, §38–§39).

Fail closed on invalid configuration or artifact mismatch.
Shadow cannot alter legacy authority. Advisory cannot place orders.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Optional

DEPLOYMENT_MODES = (
    "research", "shadow", "advisory", "candidate", "champion",
)

REQUIRED_POINTER_KEYS = (
    "mode",
    "prediction_model_group",
    "candidate_value_model",
    "candidate_rank_model",
    "fill_probability_model",
    "fill_concession_model",
    "meta_model",
)

# Registry status → modes the artifact may be loaded into.
MODE_PERMISSIONS = {
    "research": frozenset({"research"}),
    "shadow": frozenset({"research", "shadow"}),
    "advisory": frozenset({"research", "shadow", "advisory"}),
    "candidate": frozenset({"research", "shadow", "advisory", "candidate"}),
    "pending_review": frozenset({"research", "shadow", "advisory", "candidate"}),
    "champion": frozenset({
        "research", "shadow", "advisory", "candidate", "champion"}),
    "rejected": frozenset({"research"}),
    "archived": frozenset({"research"}),
}


class DeploymentError(RuntimeError):
    """Fail-closed deployment / rollback error."""


@dataclass
class DeploymentPointer:
    mode: str = "shadow"
    prediction_model_group: Optional[str] = None
    candidate_value_model: Optional[str] = None
    candidate_rank_model: Optional[str] = None
    fill_probability_model: Optional[str] = None
    fill_concession_model: Optional[str] = None
    meta_model: Optional[str] = None
    previous_deployment_id: Optional[str] = None
    rollback_deployment_id: Optional[str] = None
    approved_review_id: Optional[str] = None
    configuration_hash: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "mode": self.mode,
            "prediction_model_group": self.prediction_model_group,
            "candidate_value_model": self.candidate_value_model,
            "candidate_rank_model": self.candidate_rank_model,
            "fill_probability_model": self.fill_probability_model,
            "fill_concession_model": self.fill_concession_model,
            "meta_model": self.meta_model,
            "previous_deployment_id": self.previous_deployment_id,
            "rollback_deployment_id": self.rollback_deployment_id,
            "approved_review_id": self.approved_review_id,
            "configuration_hash": self.configuration_hash,
        }
        d.update(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DeploymentPointer":
        known = {
            "mode", "prediction_model_group", "candidate_value_model",
            "candidate_rank_model", "fill_probability_model",
            "fill_concession_model", "meta_model",
            "previous_deployment_id", "rollback_deployment_id",
            "approved_review_id", "configuration_hash",
        }
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            mode=str(d.get("mode", "shadow")),
            prediction_model_group=d.get("prediction_model_group"),
            candidate_value_model=d.get("candidate_value_model"),
            candidate_rank_model=d.get("candidate_rank_model"),
            fill_probability_model=d.get("fill_probability_model"),
            fill_concession_model=d.get("fill_concession_model"),
            meta_model=d.get("meta_model"),
            previous_deployment_id=d.get("previous_deployment_id"),
            rollback_deployment_id=d.get("rollback_deployment_id"),
            approved_review_id=d.get("approved_review_id"),
            configuration_hash=d.get("configuration_hash"),
            extras=extras,
        )


def configuration_hash(pointer: dict) -> str:
    payload = {k: pointer.get(k) for k in REQUIRED_POINTER_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()


def validate_deployment_pointer(pointer: dict, *, fail_closed: bool = True) -> None:
    if not isinstance(pointer, dict):
        raise DeploymentError("deployment pointer must be a dict")
    mode = pointer.get("mode")
    if mode not in DEPLOYMENT_MODES:
        raise DeploymentError(f"invalid deployment mode: {mode!r}")
    missing = [k for k in REQUIRED_POINTER_KEYS if k not in pointer]
    if missing and fail_closed:
        raise DeploymentError(
            f"deployment pointer missing keys: {missing}")
    # Unknown critical keys — treat keys starting with required-looking
    # names that aren't known as errors if marked critical
    critical_unknown = [
        k for k in pointer
        if k.startswith("required_") or k.endswith("_must")
    ]
    if critical_unknown:
        raise DeploymentError(
            f"unknown critical configuration keys: {critical_unknown}")


def assert_mode_permission(artifact_status: str, load_mode: str) -> None:
    allowed = MODE_PERMISSIONS.get(str(artifact_status).lower(), frozenset())
    if load_mode not in allowed:
        raise DeploymentError(
            f"artifact status {artifact_status!r} cannot load as mode "
            f"{load_mode!r}")


def mode_may_place_orders(mode: str) -> bool:
    """Part 3 never authorizes live orders from these modes."""
    return False


def mode_may_alter_legacy(mode: str) -> bool:
    """Shadow must not alter legacy tickets/size/notifications."""
    return mode == "champion"  # still no live orders; authority flag only


def _atomic_write_json(path: str, obj: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".deploy_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_deployment_pointer(path: str, pointer: dict) -> str:
    """Validate, hash, atomically write. Returns configuration hash."""
    validate_deployment_pointer(pointer)
    ptr = dict(pointer)
    ch = configuration_hash(ptr)
    ptr["configuration_hash"] = ch
    _atomic_write_json(path, ptr)
    return ch


def load_deployment_pointer(path: str) -> dict:
    if not os.path.exists(path):
        raise DeploymentError(f"deployment pointer not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            ptr = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"unreadable deployment pointer: {exc}") from exc
    validate_deployment_pointer(ptr)
    expected = configuration_hash(ptr)
    stored = ptr.get("configuration_hash")
    if stored and stored != expected:
        raise DeploymentError(
            "deployment configuration_hash mismatch (fail closed)")
    return ptr


def rollback_deployment(
    path: str,
    *,
    prior_pointer: dict,
    reason: str,
    trigger_source: str = "human",
) -> dict:
    """
    Atomic pointer replacement to a prior deployment. Does not retrain.
    Returns audit record.
    """
    validate_deployment_pointer(prior_pointer)
    previous = None
    if os.path.exists(path):
        try:
            previous = load_deployment_pointer(path)
        except DeploymentError:
            previous = None
    ch = write_deployment_pointer(path, prior_pointer)
    return {
        "reason": reason,
        "trigger_source": trigger_source,
        "previous_deployment": previous,
        "restored_deployment": load_deployment_pointer(path),
        "configuration_hash": ch,
        "human_or_automatic": (
            "human" if trigger_source == "human" else "automatic"),
    }
