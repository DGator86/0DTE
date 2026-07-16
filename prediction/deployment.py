"""
prediction/deployment.py
========================
Deployment modes, DeploymentBundle, pointer, and atomic rollback
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §31, §38–§39;
 post-#119 handoff §6.8 / PR E).

Fail closed on invalid configuration or artifact mismatch.
Shadow cannot alter legacy authority. Advisory cannot place orders.
Candidate and champion modes must never silently use heuristic fallback.

PR E: bundle + registry validation + fail-closed artifact checks + atomic
rollback. PredictionRuntime serving is PR F — not implemented here.

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

BUNDLE_SCHEMA_VERSION = "1"
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

# Every decision-relevant field enters the configuration hash (PR E / §6.8).
HASH_FIELDS = (
    "deployment_id",
    "bundle_schema_version",
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
    "api_schema_version",
    "fallback_policy",
    "reference_account_id",
    "candidate_account_id",
    "champion_account_id",
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

_MODEL_ID_ALIASES = (
    ("prediction_model_group", "prediction_model_group_id"),
    ("candidate_value_model", "candidate_value_model_id"),
    ("candidate_rank_model", "candidate_rank_model_id"),
    ("fill_probability_model", "fill_probability_model_id"),
    ("fill_concession_model", "fill_concession_model_id"),
    ("meta_model", "meta_model_id"),
)


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
    Complete decision-stack deployment (post-#119 handoff §6.8 / PR E).

    One pointer controls V1 rule config + V2/V3 model artifacts together.
    Candidate mode requires a non-null candidate_account_id distinct from
    reference_account_id.
    """

    deployment_id: str
    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION
    mode: str = "shadow"
    legacy_rule_config_id: str = ""
    prediction_model_group_id: Optional[str] = None
    candidate_value_model_id: Optional[str] = None
    candidate_rank_model_id: Optional[str] = None
    fill_probability_model_id: Optional[str] = None
    fill_concession_model_id: Optional[str] = None
    meta_model_id: Optional[str] = None
    feature_version: str = ""
    label_version: str = ""
    structural_state_version: str = ""
    policy_version: str = ""
    execution_version: str = ""
    risk_version: str = ""
    api_schema_version: str = "live.v1"
    fallback_policy: str = "abstain"
    reference_account_id: str = "legacy"
    candidate_account_id: Optional[str] = None
    champion_account_id: Optional[str] = None
    previous_deployment_id: Optional[str] = None
    rollback_deployment_id: Optional[str] = None
    approved_review_id: Optional[str] = None
    configuration_hash: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "deployment_id": self.deployment_id,
            "bundle_schema_version": self.bundle_schema_version,
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
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "structural_state_version": self.structural_state_version,
            "policy_version": self.policy_version,
            "execution_version": self.execution_version,
            "risk_version": self.risk_version,
            "api_schema_version": self.api_schema_version,
            "fallback_policy": self.fallback_policy,
            "reference_account_id": self.reference_account_id,
            "candidate_account_id": self.candidate_account_id,
            "champion_account_id": self.champion_account_id,
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
            "deployment_id", "bundle_schema_version", "mode",
            "legacy_rule_config_id",
            "prediction_model_group_id", "prediction_model_group",
            "candidate_value_model_id", "candidate_value_model",
            "candidate_rank_model_id", "candidate_rank_model",
            "fill_probability_model_id", "fill_probability_model",
            "fill_concession_model_id", "fill_concession_model",
            "meta_model_id", "meta_model",
            "feature_version", "label_version", "structural_state_version",
            "policy_version", "execution_version", "risk_version",
            "api_schema_version", "fallback_policy",
            "reference_account_id", "candidate_account_id",
            "champion_account_id",
            "previous_deployment_id", "rollback_deployment_id",
            "approved_review_id", "configuration_hash",
        }
        extras = {k: v for k, v in d.items() if k not in known}
        dep_id = d.get("deployment_id") or str(uuid.uuid4())
        return cls(
            deployment_id=str(dep_id),
            bundle_schema_version=str(
                d.get("bundle_schema_version") or BUNDLE_SCHEMA_VERSION),
            mode=str(d.get("mode", "shadow")),
            legacy_rule_config_id=str(d.get("legacy_rule_config_id") or ""),
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
            feature_version=str(d.get("feature_version") or ""),
            label_version=str(d.get("label_version") or ""),
            structural_state_version=str(
                d.get("structural_state_version") or ""),
            policy_version=str(d.get("policy_version") or ""),
            execution_version=str(d.get("execution_version") or ""),
            risk_version=str(d.get("risk_version") or ""),
            api_schema_version=str(d.get("api_schema_version") or "live.v1"),
            fallback_policy=str(d.get("fallback_policy") or "abstain"),
            reference_account_id=str(d.get("reference_account_id") or "legacy"),
            candidate_account_id=d.get("candidate_account_id"),
            champion_account_id=d.get("champion_account_id"),
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
        id_alias = None
        for legacy, new in _MODEL_ID_ALIASES:
            if k == legacy:
                id_alias = new
                break
        if id_alias is not None:
            if id_alias in pointer:
                payload[k] = _norm(pointer.get(id_alias))
            else:
                payload[k] = _norm(pointer.get(k))
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
    # Accept either legacy or *_id keys for required model slots.
    missing = []
    for legacy, new in _MODEL_ID_ALIASES:
        if legacy not in pointer and new not in pointer:
            missing.append(legacy)
    if "mode" not in pointer:
        missing.append("mode")
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
    fb = pointer.get("fallback_policy")
    if fb is not None and fb not in FALLBACK_POLICIES:
        raise DeploymentError(f"invalid fallback_policy: {fb!r}")


def validate_deployment_bundle(bundle: DeploymentBundle) -> None:
    validate_deployment_pointer(bundle.to_dict())
    if bundle.bundle_schema_version != BUNDLE_SCHEMA_VERSION:
        raise DeploymentError(
            f"unsupported bundle_schema_version: "
            f"{bundle.bundle_schema_version!r}")
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
    if str(bundle.mode).lower() == "candidate":
        if not bundle.candidate_account_id:
            raise DeploymentError(
                "candidate mode requires candidate_account_id")
        if bundle.candidate_account_id == bundle.reference_account_id:
            raise DeploymentError(
                "candidate_account_id must differ from reference_account_id")
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
    if not bundle.configuration_hash:
        object.__setattr__(bundle, "configuration_hash", expected)
    return bundle


def _registry_has_model(registry: Any, model_id: str) -> bool:
    """Best-effort existence check without requiring a public exists() API."""
    if hasattr(registry, "exists"):
        try:
            return bool(registry.exists(model_id))
        except Exception:
            return False
    if hasattr(registry, "load_metadata"):
        try:
            registry.load_metadata(model_id, validate_v2=False)
            return True
        except Exception:
            return False
    if hasattr(registry, "load"):
        try:
            registry.load(model_id)
            return True
        except Exception:
            return False
    return False


def validate_bundle_artifacts(
    bundle: DeploymentBundle,
    registry: Any,
) -> None:
    """
    Fail-closed artifact loading checks (PR E).

    Verifies referenced groups/models exist, artifact hashes load cleanly,
    feature/label versions match the bundle, and registry status permits the
    deployment mode. Does **not** serve forecasts (that is PR F).

    Research/shadow may omit optional components (heuristic fallback allowed).
    Candidate/champion require every slot and fail closed on any mismatch.
    """
    from prediction.registry import RegistryError, assert_load_mode_allowed

    validate_deployment_bundle(bundle)
    mode = bundle.mode
    strict = bundle.requires_trained_artifacts()

    def _check_model(mid: Optional[str], slot: str) -> None:
        if not mid:
            if strict:
                raise DeploymentError(
                    f"mode {mode!r} missing required artifact {slot}")
            return
        try:
            meta = registry.load_metadata(str(mid), validate_v2=False)
        except RegistryError as exc:
            raise DeploymentError(
                f"missing model for {slot}={mid!r}: {exc}") from exc
        try:
            assert_load_mode_allowed(meta, mode)
        except RegistryError as exc:
            raise DeploymentError(str(exc)) from exc
        fv = meta.get("feature_version")
        if bundle.feature_version and fv and fv != bundle.feature_version:
            raise DeploymentError(
                f"feature version mismatch for {slot}={mid!r}: "
                f"bundle={bundle.feature_version!r} artifact={fv!r}")
        lv = meta.get("label_version")
        if bundle.label_version and lv and lv != bundle.label_version:
            raise DeploymentError(
                f"label version mismatch for {slot}={mid!r}: "
                f"bundle={bundle.label_version!r} artifact={lv!r}")
        # Full load exercises artifact hash / schema fail-closed paths.
        try:
            registry.load(str(mid))
        except RegistryError as exc:
            raise DeploymentError(
                f"fail-closed load for {slot}={mid!r}: {exc}") from exc

    if bundle.prediction_model_group_id:
        try:
            group = registry.load_group(bundle.prediction_model_group_id)
            registry.validate_group(group, load_mode=mode)
        except RegistryError as exc:
            raise DeploymentError(
                f"model group validation failed: {exc}") from exc
        if (bundle.feature_version and group.feature_version
                and group.feature_version != bundle.feature_version):
            raise DeploymentError(
                f"group feature_version mismatch: "
                f"bundle={bundle.feature_version!r} "
                f"group={group.feature_version!r}")
        if (bundle.label_version and group.label_version
                and group.label_version != bundle.label_version):
            raise DeploymentError(
                f"group label_version mismatch: "
                f"bundle={bundle.label_version!r} "
                f"group={group.label_version!r}")
        for role, mid in group.component_model_ids.items():
            _check_model(mid, f"group.{role}")
    elif strict:
        raise DeploymentError(
            f"mode {mode!r} requires prediction_model_group_id")

    for slot, mid in (
        ("candidate_value_model_id", bundle.candidate_value_model_id),
        ("candidate_rank_model_id", bundle.candidate_rank_model_id),
        ("fill_probability_model_id", bundle.fill_probability_model_id),
        ("fill_concession_model_id", bundle.fill_concession_model_id),
        ("meta_model_id", bundle.meta_model_id),
    ):
        _check_model(mid, slot)


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

    Validates the prior as a complete DeploymentBundle. When ``registry`` is
    provided and the prior mode is strict, verifies each referenced artifact
    still exists. Returns an audit record.
    """
    validate_deployment_pointer(prior_pointer)
    prior_bundle = DeploymentBundle.from_dict(prior_pointer)
    validate_deployment_bundle(prior_bundle)
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
            if registry is not None and not _registry_has_model(registry, mid):
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
