"""
prediction/runtime.py
=====================
PredictionRuntime — load a DeploymentBundle from the registry and produce
forecasts / candidate evaluations (docs/UNIFIED_V1_V2_V3_HANDOFF.md §9.1).

Research/shadow may degrade with explicit unavailable labels.
Candidate/champion fail closed — never silently substitute heuristics.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from prediction.deployment import (
    DeploymentBundle,
    DeploymentError,
    STRICT_ARTIFACT_MODES,
    assert_mode_permission,
)
from prediction.registry import ModelRegistry, RegistryError


class PredictionRuntimeError(RuntimeError):
    """Structured runtime load / forecast failure."""


@dataclass
class LoadedArtifacts:
    """Resolved artifacts for one deployment."""

    bundle: DeploymentBundle
    model_group_meta: Optional[Any] = None
    model_group: Optional[Any] = None  # PredictionModelGroup when available
    candidate_value: Optional[Any] = None
    candidate_rank: Optional[Any] = None
    fill_probability: Optional[Any] = None
    fill_concession: Optional[Any] = None
    meta_model: Optional[Any] = None
    component_meta: dict = field(default_factory=dict)
    heuristic: bool = False
    load_errors: list = field(default_factory=list)


@dataclass
class PredictionRuntime:
    """
    Fail-closed loader + forecast/evaluation facade.

    Tick integration (forecast / evaluate_candidates) is completed in later
    PRs; from_deployment_bundle is the PR1 contract.
    """

    bundle: DeploymentBundle
    registry: ModelRegistry
    artifacts: LoadedArtifacts
    strict: bool = False

    @classmethod
    def from_deployment_bundle(
        cls,
        bundle: DeploymentBundle,
        registry: ModelRegistry,
        *,
        strict: Optional[bool] = None,
    ) -> "PredictionRuntime":
        """
        Load every required artifact. Verify hashes, status permissions,
        feature/label versions. Never silently mutate deployment state.
        """
        if not isinstance(bundle, DeploymentBundle):
            raise PredictionRuntimeError(
                "from_deployment_bundle requires a DeploymentBundle")
        mode = bundle.mode
        strict_mode = (
            bool(strict) if strict is not None
            else mode in STRICT_ARTIFACT_MODES)
        errors: list[dict] = []
        component_meta: dict = {}
        loaded = LoadedArtifacts(bundle=bundle)

        def _fail(msg: str) -> None:
            if strict_mode:
                raise PredictionRuntimeError(msg)
            errors.append({
                "component": "runtime",
                "stage": "load",
                "message": msg,
                "required_or_optional": (
                    "required" if strict_mode else "optional"),
                "fallback_action": (
                    "abstain" if not bundle.allows_heuristic_fallback()
                    else "heuristic_baseline"),
                "deployment_mode": mode,
            })

        # --- prediction model group ---
        gid = bundle.prediction_model_group_id
        if gid:
            try:
                gmeta = registry.load_group(gid)
                registry.validate_group(gmeta, load_mode=mode)
                if bundle.feature_version and gmeta.feature_version:
                    if gmeta.feature_version != bundle.feature_version:
                        raise PredictionRuntimeError(
                            f"feature-version mismatch: group "
                            f"{gmeta.feature_version!r} != deployment "
                            f"{bundle.feature_version!r}")
                if bundle.label_version and gmeta.label_version:
                    if gmeta.label_version != bundle.label_version:
                        raise PredictionRuntimeError(
                            f"label-version mismatch: group "
                            f"{gmeta.label_version!r} != deployment "
                            f"{bundle.label_version!r}")
                loaded.model_group_meta = gmeta
                component_meta["prediction_model_group"] = gmeta.to_dict()
                # Optional: load a pickled PredictionModelGroup if stored
                # under a well-known component role.
                group_artifact_id = gmeta.component_model_ids.get(
                    "model_group") or gmeta.component_model_ids.get("bundle")
                if group_artifact_id:
                    model, meta = registry.load(
                        group_artifact_id,
                        expected_feature_version=(
                            bundle.feature_version or None),
                    )
                    assert_mode_permission(meta.get("status", "research"), mode)
                    loaded.model_group = model
                    component_meta["model_group_artifact"] = meta
            except (RegistryError, DeploymentError, PredictionRuntimeError) as exc:
                _fail(f"prediction model group load failed: {exc}")
                if strict_mode:
                    raise
        elif strict_mode:
            raise PredictionRuntimeError(
                f"mode {mode!r} requires prediction_model_group_id")

        # --- Part 3 decision models ---
        slot_map = (
            ("candidate_value", bundle.candidate_value_model_id),
            ("candidate_rank", bundle.candidate_rank_model_id),
            ("fill_probability", bundle.fill_probability_model_id),
            ("fill_concession", bundle.fill_concession_model_id),
            ("meta_model", bundle.meta_model_id),
        )
        for attr, mid in slot_map:
            if not mid:
                if strict_mode:
                    raise PredictionRuntimeError(
                        f"mode {mode!r} requires {attr}_id")
                continue
            try:
                model, meta = registry.load(
                    mid,
                    expected_feature_version=(
                        bundle.feature_version or None),
                )
                assert_mode_permission(meta.get("status", "research"), mode)
                setattr(loaded, attr, model)
                component_meta[attr] = meta
            except (RegistryError, DeploymentError) as exc:
                _fail(f"{attr} load failed: {exc}")
                if strict_mode:
                    raise PredictionRuntimeError(str(exc)) from exc

        # Heuristic only when explicitly allowed and no trained group.
        if loaded.model_group is None and not gid:
            if not bundle.allows_heuristic_fallback():
                raise PredictionRuntimeError(
                    f"mode {mode!r} forbids heuristic fallback; "
                    "no trained prediction_model_group_id")
            loaded.heuristic = True
            errors.append({
                "component": "forecast",
                "stage": "load",
                "message": "no trained group; heuristic baseline labeled",
                "required_or_optional": "optional",
                "fallback_action": "heuristic_baseline",
                "deployment_mode": mode,
            })

        loaded.component_meta = component_meta
        loaded.load_errors = errors
        return cls(
            bundle=bundle,
            registry=registry,
            artifacts=loaded,
            strict=strict_mode,
        )

    # ------------------------------------------------------------------ #
    # Forecast / evaluation (stubs filled by later PRs; safe defaults)   #
    # ------------------------------------------------------------------ #

    def forecast(self, snapshot: Any) -> Any:
        """Produce a ForecastBundle / PredictionBundle from a CanonicalSnapshot."""
        from prediction.forecast_assembly import build_v3_forecast
        return build_v3_forecast(
            snapshot=snapshot,
            runtime=self,
            mode=self.bundle.mode,
        )

    def evaluate_candidates(
        self,
        snapshot: Any,
        forecast: Any,
        universe: Any,
    ) -> tuple:
        """Evaluate a shared candidate universe with V3 economics."""
        from prediction.part3_decision import build_v3_candidate_evaluations
        return build_v3_candidate_evaluations(
            snapshot=snapshot,
            forecast=forecast,
            universe=universe,
            runtime=self,
            mode=self.bundle.mode,
        )

    @property
    def is_heuristic(self) -> bool:
        return bool(self.artifacts.heuristic)

    @property
    def deployment_id(self) -> str:
        return self.bundle.deployment_id

    def component_ids(self) -> dict:
        b = self.bundle
        return {
            "deployment_id": b.deployment_id,
            "legacy_rule_config_id": b.legacy_rule_config_id,
            "prediction_model_group_id": b.prediction_model_group_id,
            "candidate_value_model_id": b.candidate_value_model_id,
            "candidate_rank_model_id": b.candidate_rank_model_id,
            "fill_probability_model_id": b.fill_probability_model_id,
            "fill_concession_model_id": b.fill_concession_model_id,
            "meta_model_id": b.meta_model_id,
            "feature_version": b.feature_version,
            "label_version": b.label_version,
            "fallback_policy": b.fallback_policy,
            "mode": b.mode,
            "heuristic": self.is_heuristic,
        }
