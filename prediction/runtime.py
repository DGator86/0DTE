"""Versioned, fail-closed serving runtime for the prediction stack."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from prediction.contracts import PredictionBundle
from prediction.deployment import (
    DeploymentBundle,
    DeploymentError,
    configuration_hash,
    load_deployment_bundle,
    validate_bundle_artifacts,
    validate_deployment_bundle,
)
from prediction.registry import ModelRegistry, RegistryError
from prediction.training import PredictionModelGroup, build_prediction_bundle
from prediction.uncertainty import ABSTAIN_SHADOW_THRESHOLD, compose_uncertainty

RUNTIME_VERSION = "v4.0.0-pr-f"
HEURISTIC_FALLBACK_MODES = frozenset({"research", "shadow"})
_DECISION_SLOTS = (
    "candidate_value_model",
    "candidate_rank_model",
    "fill_probability_model",
    "fill_concession_model",
    "meta_model",
)


class PredictionRuntimeError(RuntimeError):
    """Startup or inference failure that must fail closed."""


@dataclass(frozen=True)
class RuntimeHealth:
    status: str
    deployment_id: str
    deployment_mode: str
    configuration_hash: str
    source: str = "trained_artifacts"
    loaded_components: tuple[str, ...] = ()
    component_errors: Mapping[str, str] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    inference_latency_ms: Optional[float] = None
    reasons: tuple[str, ...] = ()
    runtime_version: str = RUNTIME_VERSION

    @property
    def actionable(self) -> bool:
        return self.status in {"OK", "DEGRADED"}

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PredictionRuntimeResult:
    bundle: PredictionBundle
    health: RuntimeHealth

    @property
    def actionable(self) -> bool:
        return self.health.actionable


class PredictionRuntime:
    """Load one deployment bundle and serve one canonical forecast per snapshot."""

    def __init__(
        self,
        bundle: DeploymentBundle,
        registry: ModelRegistry,
        *,
        heuristic_provider: Optional[Callable[..., Any]] = None,
        store: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.bundle = bundle
        self.registry = registry
        self.heuristic_provider = heuristic_provider
        self.store = store
        self._clock = clock
        self._hashes: dict[str, str] = {}
        self._model_ids: dict[str, str] = {}
        self._required_fields: dict[str, tuple[str, ...]] = {}
        self._decision_models: dict[str, Any] = {}
        self._regime_model: Any = None
        self._forecast_group: Optional[PredictionModelGroup] = None
        self._validate_bundle_hash()
        try:
            validate_deployment_bundle(bundle)
            validate_bundle_artifacts(bundle, registry)
            self._load_artifacts()
        except (DeploymentError, RegistryError, OSError, ValueError) as exc:
            raise PredictionRuntimeError(
                f"prediction runtime startup failed closed: {exc}") from exc

    @classmethod
    def from_path(
        cls, path: str, registry: ModelRegistry, **kwargs: Any,
    ) -> "PredictionRuntime":
        return cls(load_deployment_bundle(path), registry, **kwargs)

    @property
    def decision_models(self) -> Mapping[str, Any]:
        return dict(self._decision_models)

    @property
    def artifact_manifest(self) -> Mapping[str, str]:
        return dict(self._hashes)

    def infer(
        self,
        *,
        snapshot_id: str,
        ts: str,
        session_date: str,
        symbol: str,
        feature_row: Mapping[str, Any],
        structural: Optional[Mapping[str, Any]] = None,
        quality: Optional[Mapping[str, Any]] = None,
        uncertainty_components: Optional[Mapping[str, Any]] = None,
        ood_score: Optional[float] = None,
        ood_percentile: Optional[float] = None,
        calibration_support: Optional[float] = None,
    ) -> PredictionRuntimeResult:
        started = self._clock()
        quality = dict(quality or {})
        errors: dict[str, str] = {}
        bundle: Optional[PredictionBundle] = None

        if self._forecast_group is None:
            errors["prediction_model_group"] = "not loaded"
        else:
            try:
                self._check_required_inputs(feature_row, structural or {})
                bundle = build_prediction_bundle(
                    self._forecast_group,
                    dict(feature_row),
                    snapshot_id=snapshot_id,
                    ts=ts,
                    session_date=session_date,
                    symbol=symbol,
                    quality=quality,
                    structural=dict(structural or {}),
                    ood_score=ood_score,
                    ood_percentile=ood_percentile,
                    calibration_support=calibration_support,
                )
                bundle = self._attach_regime(bundle, feature_row)
            except Exception as exc:  # fail closed, but keep runtime alive
                errors["prediction_model_group"] = (
                    f"{type(exc).__name__}: {exc}")
                bundle = None

        if errors or bundle is None:
            fallback = self._heuristic_fallback(
                snapshot_id=snapshot_id,
                ts=ts,
                session_date=session_date,
                symbol=symbol,
                feature_row=dict(feature_row),
                structural=dict(structural or {}),
                quality=quality,
                errors=errors,
                started=started,
            )
            if fallback is not None:
                return fallback
            return self._abstain(
                snapshot_id, ts, session_date, symbol, quality, errors,
                started, ("required_prediction_component_failed",))

        coverage = _bounded_or_none(
            quality.get("feature_coverage", quality.get("data_quality")))
        data_quality = _bounded_or_none(quality.get("data_quality", coverage))
        supplied = dict(uncertainty_components or {})
        ensemble = supplied.get("ensemble")
        if ensemble is None:
            ensemble = bundle.uncertainty
        data_uncertainty = supplied.get("data_quality")
        if data_uncertainty is None and data_quality is not None:
            data_uncertainty = 1.0 - data_quality
        composed = compose_uncertainty(
            ensemble=ensemble,
            conformal=supplied.get("conformal"),
            out_of_distribution=supplied.get(
                "out_of_distribution", ood_score),
            calibration=supplied.get("calibration"),
            data_quality=data_uncertainty,
            model_age=supplied.get("model_age"),
            extra_reasons=tuple(supplied.get("reasons") or ()),
            diagnostics={"runtime_version": RUNTIME_VERSION},
        )
        status = (
            "ABSTAIN"
            if composed.composite >= ABSTAIN_SHADOW_THRESHOLD else "OK")
        latency = (self._clock() - started) * 1000.0
        data = bundle.to_dict()
        diagnostics = dict(data.get("diagnostics") or {})
        diagnostics.update({
            "source": "trained_artifacts",
            "runtime_status": status,
            "runtime_version": RUNTIME_VERSION,
            "deployment_id": self.bundle.deployment_id,
            "deployment_mode": self.bundle.mode,
            "configuration_hash": self.bundle.configuration_hash,
            "artifact_hashes": dict(self._hashes),
            "inference_latency_ms": latency,
        })
        data.update({
            "uncertainty": composed.composite,
            "data_quality": data_quality,
            "feature_coverage": coverage,
            "feature_version": self.bundle.feature_version,
            "model_versions": self._model_versions(),
            "diagnostics": diagnostics,
            "uncertainty_components": {
                "ensemble": composed.ensemble,
                "conformal": composed.conformal,
                "out_of_distribution": composed.out_of_distribution,
                "calibration": composed.calibration,
                "data_quality": composed.data_quality,
                "model_age": composed.model_age,
                "composite": composed.composite,
            },
            "uncertainty_reasons": composed.reasons,
            "ood_score": _bounded_or_none(ood_score),
            "ood_percentile": _bounded_or_none(ood_percentile),
            "calibration_support": _bounded_or_none(calibration_support),
            "structural_state_version": (
                self.bundle.structural_state_version or None),
            "forecast_model_group_version": (
                self.bundle.prediction_model_group_id or None),
        })
        final_bundle = PredictionBundle.from_dict(data)
        health = RuntimeHealth(
            status=status,
            deployment_id=self.bundle.deployment_id,
            deployment_mode=self.bundle.mode,
            configuration_hash=self.bundle.configuration_hash,
            loaded_components=tuple(sorted(self._hashes)),
            artifact_hashes=dict(self._hashes),
            inference_latency_ms=latency,
            reasons=(("uncertainty_abstention",) if status == "ABSTAIN" else ()),
        )
        self._journal(final_bundle, health)
        return PredictionRuntimeResult(final_bundle, health)

    def predict_bundle(self, **kwargs: Any) -> PredictionBundle:
        return self.infer(**kwargs).bundle

    def _validate_bundle_hash(self) -> None:
        if (self.bundle.configuration_hash
                and self.bundle.configuration_hash
                != configuration_hash(self.bundle.to_dict())):
            raise PredictionRuntimeError(
                "deployment configuration_hash mismatch (fail closed)")

    def _load_artifacts(self) -> None:
        direction: dict[str, Any] = {}
        quantiles: dict[str, Any] = {}
        range_survival: dict[str, Any] = {}
        volatility = None
        group_id = self.bundle.prediction_model_group_id
        if group_id:
            group = self.registry.load_group(group_id)
            self.registry.validate_group(group, load_mode=self.bundle.mode)
            for role, model_id in group.component_model_ids.items():
                model, meta = self._load_one(str(role), str(model_id))
                target = str(meta.get("target") or "")
                horizon = str(meta.get("horizon") or _horizon(target, role) or "")
                if target.startswith("up_"):
                    direction[horizon] = model
                elif target.startswith("fwd_return_"):
                    quantiles[horizon] = model
                elif target == "remaining_realized_move":
                    volatility = model
                elif target.startswith("range_survive_"):
                    range_survival[horizon] = model
                elif "regime" in target.lower() or "regime" in str(role).lower():
                    self._regime_model = model
                else:
                    raise PredictionRuntimeError(
                        f"unsupported prediction-group component "
                        f"{role!r} target={target!r}")
            self._forecast_group = PredictionModelGroup(
                direction=direction,
                quantiles=quantiles,
                volatility=volatility,
                range_survival=range_survival,
                feature_version=self.bundle.feature_version,
                group_version=group.group_id,
            )
        for slot in _DECISION_SLOTS:
            model_id = getattr(self.bundle, f"{slot}_id")
            if model_id:
                model, _ = self._load_one(slot, str(model_id))
                self._decision_models[slot] = model

    def _load_one(self, slot: str, model_id: str) -> tuple[Any, dict]:
        model, meta = self.registry.load(
            model_id,
            expected_feature_version=(self.bundle.feature_version or None),
            live_feature_version=(self.bundle.feature_version or None),
        )
        artifact_hash = str(meta.get("artifact_hash") or "")
        if not artifact_hash:
            raise RegistryError(f"artifact {model_id!r} has no artifact_hash")
        self._hashes[slot] = artifact_hash
        self._model_ids[slot] = model_id
        self._required_fields[slot] = tuple(
            str(name) for name in (meta.get("required_input_fields") or ()))
        return model, dict(meta)

    def _check_required_inputs(
        self, row: Mapping[str, Any], structural: Mapping[str, Any],
    ) -> None:
        for role, required in self._required_fields.items():
            if role in _DECISION_SLOTS:
                continue
            missing = [
                name for name in required
                if row.get(name) is None and structural.get(name) is None
            ]
            if missing:
                raise PredictionRuntimeError(
                    f"{role}: missing required input fields {sorted(missing)}")

    def _attach_regime(
        self, bundle: PredictionBundle, row: Mapping[str, Any],
    ) -> PredictionBundle:
        if self._regime_model is None:
            return bundle
        pred = self._regime_model.predict(dict(row))
        data = bundle.to_dict()
        data["regime_probabilities"] = pred.as_dict()
        data["regime_uncertainty"] = getattr(pred, "uncertainty", None)
        data["dominant_regime"] = getattr(pred, "dominant_regime", None)
        return PredictionBundle.from_dict(data)

    def _heuristic_fallback(
        self, *, snapshot_id: str, ts: str, session_date: str, symbol: str,
        feature_row: dict, structural: dict, quality: dict,
        errors: Mapping[str, str], started: float,
    ) -> Optional[PredictionRuntimeResult]:
        if (self.bundle.mode not in HEURISTIC_FALLBACK_MODES
                or self.bundle.fallback_policy != "legacy"
                or self.heuristic_provider is None):
            return None
        try:
            raw = self.heuristic_provider(
                snapshot_id=snapshot_id, ts=ts, session_date=session_date,
                symbol=symbol, feature_row=feature_row,
                structural=structural, quality=quality)
            data = raw.to_dict() if isinstance(raw, PredictionBundle) else dict(raw)
            data.update({"snapshot_id": snapshot_id, "ts": ts,
                         "session_date": session_date, "symbol": symbol,
                         "feature_version": self.bundle.feature_version})
            diagnostics = dict(data.get("diagnostics") or {})
            diagnostics.update({
                "source": "heuristic_fallback",
                "runtime_status": "DEGRADED",
                "runtime_version": RUNTIME_VERSION,
                "deployment_id": self.bundle.deployment_id,
                "deployment_mode": self.bundle.mode,
                "component_errors": dict(errors),
            })
            data["diagnostics"] = diagnostics
            bundle = PredictionBundle.from_dict(data)
        except Exception as exc:
            all_errors = dict(errors)
            all_errors["heuristic_fallback"] = f"{type(exc).__name__}: {exc}"
            return self._abstain(
                snapshot_id, ts, session_date, symbol, quality, all_errors,
                started, ("heuristic_fallback_failed",))
        health = RuntimeHealth(
            status="DEGRADED",
            deployment_id=self.bundle.deployment_id,
            deployment_mode=self.bundle.mode,
            configuration_hash=self.bundle.configuration_hash,
            source="heuristic_fallback",
            loaded_components=tuple(sorted(self._hashes)),
            component_errors=dict(errors),
            artifact_hashes=dict(self._hashes),
            inference_latency_ms=(self._clock() - started) * 1000.0,
            reasons=("explicit_heuristic_fallback",),
        )
        self._journal(bundle, health)
        return PredictionRuntimeResult(bundle, health)

    def _abstain(
        self, snapshot_id: str, ts: str, session_date: str, symbol: str,
        quality: Mapping[str, Any], errors: Mapping[str, str], started: float,
        reasons: tuple[str, ...],
    ) -> PredictionRuntimeResult:
        latency = (self._clock() - started) * 1000.0
        coverage = _bounded_or_none(
            quality.get("feature_coverage", quality.get("data_quality")))
        data_quality = _bounded_or_none(quality.get("data_quality", coverage))
        bundle = PredictionBundle(
            snapshot_id=snapshot_id, ts=ts, session_date=session_date,
            symbol=symbol, uncertainty=1.0, data_quality=data_quality,
            feature_coverage=coverage, feature_version=self.bundle.feature_version,
            model_versions=self._model_versions(),
            diagnostics={
                "source": "abstain",
                "runtime_status": "ABSTAIN",
                "runtime_version": RUNTIME_VERSION,
                "deployment_id": self.bundle.deployment_id,
                "deployment_mode": self.bundle.mode,
                "configuration_hash": self.bundle.configuration_hash,
                "component_errors": dict(errors),
                "artifact_hashes": dict(self._hashes),
                "inference_latency_ms": latency,
            },
            uncertainty_components={"composite": 1.0},
            uncertainty_reasons=reasons,
            structural_state_version=self.bundle.structural_state_version or None,
            forecast_model_group_version=(
                self.bundle.prediction_model_group_id or None),
        )
        health = RuntimeHealth(
            status="ABSTAIN",
            deployment_id=self.bundle.deployment_id,
            deployment_mode=self.bundle.mode,
            configuration_hash=self.bundle.configuration_hash,
            source="abstain",
            loaded_components=tuple(sorted(self._hashes)),
            component_errors=dict(errors),
            artifact_hashes=dict(self._hashes),
            inference_latency_ms=latency,
            reasons=reasons,
        )
        self._journal(bundle, health)
        return PredictionRuntimeResult(bundle, health)

    def _model_versions(self) -> dict:
        return {
            "runtime": RUNTIME_VERSION,
            "deployment": self.bundle.deployment_id,
            **dict(self._model_ids),
        }

    def _journal(self, bundle: PredictionBundle, health: RuntimeHealth) -> None:
        if self.store is None:
            return
        if hasattr(self.store, "log_prediction"):
            try:
                self.store.log_prediction(
                    bundle.snapshot_id,
                    self.bundle.prediction_model_group_id or RUNTIME_VERSION,
                    bundle.to_dict(),
                    uncertainty=bundle.uncertainty,
                    generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    mode=self.bundle.mode,
                )
            except Exception:
                pass
        if hasattr(self.store, "log_prediction_runtime_health"):
            try:
                self.store.log_prediction_runtime_health(
                    bundle.snapshot_id, health.to_dict())
            except Exception:
                pass


def make_runtime_bundle_provider(
    runtime: PredictionRuntime, *, symbol: str = "SPY",
) -> Callable:
    """Adapt PredictionRuntime to UnifiedOrchestrator's provider hook."""
    def provider(snap, signals, intent=None, regime_state=None):
        from prediction.dataset import FEATURE_VERSION, make_snapshot_id, session_metadata
        from prediction.inference import live_feature_row

        signals = dict(signals or {})
        market = snap.market
        now = market.now
        meta = session_metadata(now)
        snapshot_id = signals.get("_snapshot_id") or make_snapshot_id(
            symbol, now, runtime.bundle.feature_version or FEATURE_VERSION, 0)
        structural = {
            key: getattr(market, key, None)
            for key in ("spot", "put_wall", "call_wall", "gamma_flip",
                        "net_gex", "adx", "cvd_slope")
        }
        structural.update({
            "move_consumed": signals.get("move_consumed"),
            "minutes_to_close": meta.get("minutes_to_close"),
        })
        has_chain = getattr(snap, "chain", None) is not None
        quality = {
            "feature_coverage": 0.85 if has_chain else 0.55,
            "data_quality": 0.85 if has_chain else 0.55,
        }
        return runtime.predict_bundle(
            snapshot_id=str(snapshot_id),
            ts=now.isoformat(),
            session_date=str(meta.get("session_date") or now.date().isoformat()),
            symbol=symbol,
            feature_row=live_feature_row(snap, signals),
            structural=structural,
            quality=quality,
        )
    return provider


def _horizon(target: str, role: str) -> Optional[str]:
    text = f"{target} {role}".lower()
    for horizon in ("5m", "15m", "30m", "60m", "close"):
        if horizon in text:
            return horizon
    return None


def _bounded_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"bounded value outside [0,1]: {value!r}")
    return value
