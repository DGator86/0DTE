"""
prediction/deployment.py
========================
Deployment modes, pointer/bundle, and atomic rollback
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §31, §38–§39;
 docs/UNIFIED_V1_V2_V3_HANDOFF.md §7.6, §8, PR1).

Fail closed on invalid configuration or artifact mismatch.
Shadow cannot alter legacy authority. Advisory cannot place orders.
Candidate and champion modes must never silently use heuristic fallback.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

DEPLOYMENT_MODES = (
    "research", "shadow", "advisory", "candidate", "champion",
)

AUTHORITY_SOURCES = ("legacy", "v3", "human")
FALLBACK_POLICIES = ("abstain", "legacy", "no_trade")

# Legacy pointer keys (still required for on-disk compatibility).
REQUIRED_POINTER_KEYS = (
    "mode",
    "prediction_model_group",
    "candidate_value_model",
    "candidate_rank_model",
    "fill_probability_model",
    "fill_concession_model",
    "meta_model",
)

# Every decision-relevant field enters the configuration hash.
HASH_FIELDS = (
    "deployment_id",
    "mode",
    "legacy_rule_config_id",
    "prediction_model_group",
    "candidate_value_model",
    "candidate_rank_model",
    "fill_probability_model",
    "fill_concession_model",
    "meta_model",
    "policy_version",
    "execution_version",
    "risk_version",
    "feature_version",
    "label_version",
    "structural_state_version",
    "authority_source",
    "fallback_policy",
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

# Modes that require trained artifacts (no silent heuristic substitution).
STRICT_ARTIFACT_MODES = frozenset({"candidate", "champion"})


class DeploymentError(RuntimeError):
    """Fail-closed deployment / rollback error."""


@dataclass
class DeploymentPointer:
    """Legacy on-disk pointer shape (backward compatible)."""

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


@dataclass(frozen=True)
class DeploymentBundle:
    """
    Complete decision-stack deployment (UNIFIED handoff §7.6).

    One pointer controls V1 rule config + V2/V3 model artifacts together.
    """

    deployment_id: str
    mode: str = "shadow"
    legacy_rule_config_id: Optional[str] = None
    prediction_model_group_id: Optional[str] = None
    candidate_value_model_id: Optional[str] = None
    candidate_rank_model_id: Optional[str] = None
    fill_probability_model_id: Optional[str] = None
    fill_concession_model_id: Optional[str] = None
    meta_model_id: Optional[str] = None
    policy_version: str = ""
    execution_version: str = ""
    risk_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    structural_state_version: str = ""
    authority_source: str = "legacy"
    fallback_policy: str = "abstain"
    previous_deployment_id: Optional[str] = None
    rollback_deployment_id: Optional[str] = None
    approved_review_id: Optional[str] = None
    configuration_hash: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "deployment_id": self.deployment_id,
            "mode": self.mode,
            "legacy_rule_config_id": self.legacy_rule_config_id,
            # Dual keys: new *_id names + legacy pointer names for on-disk compat.
            "prediction_model_group_id": self.prediction_model_group_id,
            "prediction_model_group": self.prediction_model_group_id,
            "candidate_value_model_id": self.candidate_value_model_id,
            "candidate_value_model": self.candidate_value_model_id,
            "candidate_rank_model_id": self.candidate_rank_model_id,
            "candidate_rank_model": self.candidate_rank_model_id,
            "fill_probability_model_id": self.fill_probability_model_id,
            "fill_probability_model": self.fill_probability_model_id,
            "fill_concession_model_id": self.fill_concession_model_id,
            "fill_concession_model": self.fill_concession_model_id,
            "meta_model_id": self.meta_model_id,
            "meta_model": self.meta_model_id,
            "policy_version": self.policy_version,
            "execution_version": self.execution_version,
            "risk_version": self.risk_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "structural_state_version": self.structural_state_version,
            "authority_source": self.authority_source,
            "fallback_policy": self.fallback_policy,
            "previous_deployment_id": self.previous_deployment_id,
            "rollback_deployment_id": self.rollback_deployment_id,
            "approved_review_id": self.approved_review_id,
            "configuration_hash": self.configuration_hash,
        }
        d.update(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DeploymentBundle":
        known = {
            "deployment_id", "mode", "legacy_rule_config_id",
            "prediction_model_group_id", "prediction_model_group",
            "candidate_value_model_id", "candidate_value_model",
            "candidate_rank_model_id", "candidate_rank_model",
            "fill_probability_model_id", "fill_probability_model",
            "fill_concession_model_id", "fill_concession_model",
            "meta_model_id", "meta_model",
            "policy_version", "execution_version", "risk_version",
            "feature_version", "label_version", "structural_state_version",
            "authority_source", "fallback_policy",
            "previous_deployment_id", "rollback_deployment_id",
            "approved_review_id", "configuration_hash",
        }
        extras = {k: v for k, v in d.items() if k not in known}
        dep_id = d.get("deployment_id") or str(uuid.uuid4())
        return cls(
            deployment_id=str(dep_id),
            mode=str(d.get("mode", "shadow")),
            legacy_rule_config_id=d.get("legacy_rule_config_id"),
            prediction_model_group_id=(
                d.get("prediction_model_group_id")
                or d.get("prediction_model_group")),
            candidate_value_model_id=(
                d.get("candidate_value_model_id")
                or d.get("candidate_value_model")),
            candidate_rank_model_id=(
                d.get("candidate_rank_model_id")
                or d.get("candidate_rank_model")),
            fill_probability_model_id=(
                d.get("fill_probability_model_id")
                or d.get("fill_probability_model")),
            fill_concession_model_id=(
                d.get("fill_concession_model_id")
                or d.get("fill_concession_model")),
            meta_model_id=d.get("meta_model_id") or d.get("meta_model"),
            policy_version=str(d.get("policy_version") or ""),
            execution_version=str(d.get("execution_version") or ""),
            risk_version=str(d.get("risk_version") or ""),
            feature_version=str(d.get("feature_version") or ""),
            label_version=str(d.get("label_version") or ""),
            structural_state_version=str(
                d.get("structural_state_version") or ""),
            authority_source=str(d.get("authority_source") or "legacy"),
            fallback_policy=str(d.get("fallback_policy") or "abstain"),
            previous_deployment_id=d.get("previous_deployment_id"),
            rollback_deployment_id=d.get("rollback_deployment_id"),
            approved_review_id=d.get("approved_review_id"),
            configuration_hash=str(d.get("configuration_hash") or ""),
            extras=extras,
        )

    def requires_trained_artifacts(self) -> bool:
        return self.mode in STRICT_ARTIFACT_MODES

    def allows_heuristic_fallback(self) -> bool:
        """Heuristic forecasts only in research/shadow (labeled baseline)."""
        return self.mode not in STRICT_ARTIFACT_MODES


def configuration_hash(pointer: dict) -> str:
    """
    Hash every decision-relevant field.

    Accepts either a legacy pointer dict or a DeploymentBundle.to_dict()
    payload. Legacy-only pointers still hash; missing extended fields become
    null so adding any decision-relevant field changes the hash.
    Empty strings are normalized to null for stable hashing across
    pointer ↔ bundle conversions.
    """
    def _norm(v):
        if v == "":
            return None
        return v

    payload = {}
    for k in HASH_FIELDS:
        # Prefer explicit *_id keys when present.
        if k == "prediction_model_group":
            payload[k] = _norm(
                pointer.get("prediction_model_group_id")
                if "prediction_model_group_id" in pointer
                else pointer.get("prediction_model_group"))
        elif k == "candidate_value_model":
            payload[k] = _norm(
                pointer.get("candidate_value_model_id")
                if "candidate_value_model_id" in pointer
                else pointer.get("candidate_value_model"))
        elif k == "candidate_rank_model":
            payload[k] = _norm(
                pointer.get("candidate_rank_model_id")
                if "candidate_rank_model_id" in pointer
                else pointer.get("candidate_rank_model"))
        elif k == "fill_probability_model":
            payload[k] = _norm(
                pointer.get("fill_probability_model_id")
                if "fill_probability_model_id" in pointer
                else pointer.get("fill_probability_model"))
        elif k == "fill_concession_model":
            payload[k] = _norm(
                pointer.get("fill_concession_model_id")
                if "fill_concession_model_id" in pointer
                else pointer.get("fill_concession_model"))
        elif k == "meta_model":
            payload[k] = _norm(
                pointer.get("meta_model_id")
                if "meta_model_id" in pointer
                else pointer.get("meta_model"))
        else:
            payload[k] = _norm(pointer.get(k))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


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
    critical_unknown = [
        k for k in pointer
        if k.startswith("required_") or k.endswith("_must")
    ]
    if critical_unknown:
        raise DeploymentError(
            f"unknown critical configuration keys: {critical_unknown}")
    auth = pointer.get("authority_source")
    if auth is not None and auth not in AUTHORITY_SOURCES:
        raise DeploymentError(f"invalid authority_source: {auth!r}")
    fb = pointer.get("fallback_policy")
    if fb is not None and fb not in FALLBACK_POLICIES:
        raise DeploymentError(f"invalid fallback_policy: {fb!r}")


def validate_deployment_bundle(bundle: DeploymentBundle) -> None:
    validate_deployment_pointer(bundle.to_dict())
    if bundle.authority_source not in AUTHORITY_SOURCES:
        raise DeploymentError(
            f"invalid authority_source: {bundle.authority_source!r}")
    if bundle.fallback_policy not in FALLBACK_POLICIES:
        raise DeploymentError(
            f"invalid fallback_policy: {bundle.fallback_policy!r}")
    if bundle.requires_trained_artifacts():
        required = {
            "prediction_model_group_id": bundle.prediction_model_group_id,
            "candidate_value_model_id": bundle.candidate_value_model_id,
            "candidate_rank_model_id": bundle.candidate_rank_model_id,
            "fill_probability_model_id": bundle.fill_probability_model_id,
            "fill_concession_model_id": bundle.fill_concession_model_id,
            "meta_model_id": bundle.meta_model_id,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise DeploymentError(
                f"mode {bundle.mode!r} requires trained artifacts; "
                f"missing: {missing}")
    if str(bundle.mode).lower() == "champion":
        if not bundle.approved_review_id:
            raise DeploymentError(
                "champion mode requires approved_review_id")
        if not bundle.rollback_deployment_id:
            raise DeploymentError(
                "champion mode requires rollback_deployment_id")


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
    if not ptr.get("deployment_id"):
        ptr["deployment_id"] = str(uuid.uuid4())
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


def write_deployment_bundle(path: str, bundle: DeploymentBundle) -> str:
    """Validate complete bundle, hash, atomically write."""
    validate_deployment_bundle(bundle)
    payload = bundle.to_dict()
    ch = configuration_hash(payload)
    payload["configuration_hash"] = ch
    _atomic_write_json(path, payload)
    return ch


def load_deployment_bundle(path: str) -> DeploymentBundle:
    """Load and validate a DeploymentBundle from disk."""
    ptr = load_deployment_pointer(path)
    bundle = DeploymentBundle.from_dict(ptr)
    validate_deployment_bundle(bundle)
    expected = configuration_hash(bundle.to_dict())
    if bundle.configuration_hash and bundle.configuration_hash != expected:
        raise DeploymentError(
            "deployment configuration_hash mismatch (fail closed)")
    # Re-bind computed hash when older files lacked extended fields but
    # still passed the pointer hash check above.
    if not bundle.configuration_hash:
        object.__setattr__(bundle, "configuration_hash", expected)
    return bundle


def rollback_deployment(
    path: str,
    *,
    prior_pointer: dict,
    reason: str,
    trigger_source: str = "human",
    registry: Any = None,
) -> dict:
    """
    Atomic pointer replacement to a prior deployment. Does not retrain.
    Validates the prior as a complete DeploymentBundle and verifies
    referenced model IDs are present in the pointer payload.
    When ``registry`` is provided, also verifies each referenced artifact
    still exists on disk.
    Returns audit record.
    """
    validate_deployment_pointer(prior_pointer)
    prior_bundle = DeploymentBundle.from_dict(prior_pointer)
    validate_deployment_bundle(prior_bundle)
    # Require that referenced artifact IDs are non-null when mode is strict.
    if prior_bundle.requires_trained_artifacts():
        for attr in (
            "prediction_model_group_id", "candidate_value_model_id",
            "candidate_rank_model_id", "fill_probability_model_id",
            "fill_concession_model_id", "meta_model_id",
        ):
            mid = getattr(prior_bundle, attr)
            if not mid:
                raise DeploymentError(
                    f"rollback target missing required {attr}")
            if registry is not None:
                exists = False
                if hasattr(registry, "exists"):
                    exists = bool(registry.exists(mid))
                elif hasattr(registry, "get"):
                    try:
                        exists = registry.get(mid) is not None
                    except Exception:
                        exists = False
                elif hasattr(registry, "load"):
                    try:
                        exists = registry.load(mid) is not None
                    except Exception:
                        exists = False
                if not exists:
                    raise DeploymentError(
                        f"rollback target artifact missing from registry: "
                        f"{attr}={mid!r}")
    previous = None
    if os.path.exists(path):
        try:
            previous = load_deployment_pointer(path)
        except DeploymentError:
            previous = None
    # Write the caller-supplied prior pointer dict (not a re-serialized
    # bundle) so the configuration_hash remains stable across rollback.
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
