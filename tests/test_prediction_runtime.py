"""Tests for the fail-closed PredictionRuntime serving layer."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from prediction.contracts import PredictionBundle
from prediction.deployment import DeploymentBundle, configuration_hash
from prediction.registry import ModelRegistry
from prediction.runtime import PredictionRuntime, PredictionRuntimeError

FEATURE_VERSION = "v2.0.0"
LABEL_VERSION = "v3.0.0"


class StubDirectionModel:
    def __init__(self, value: float = 0.63, *, fail: bool = False):
        self.value = value
        self.fail = fail
        self.config = SimpleNamespace(horizon="30m")
        self.metadata = {"uncertainty": 0.10}

    def predict_proba(self, rows):
        if self.fail:
            raise RuntimeError("direction failure")
        return np.full(len(rows), self.value)


class StubQuantileModel:
    def __init__(self):
        self.config = SimpleNamespace(horizon="30m")
        self.metadata = {"uncertainty": 0.15}

    def predict(self, rows):
        n = len(rows)
        return {
            "q10": np.full(n, -0.004),
            "q50": np.full(n, 0.001),
            "q90": np.full(n, 0.006),
        }


class StubVolatilityModel:
    metadata = {"uncertainty": 0.20}

    def predict(self, rows):
        n = len(rows)
        return {
            "expected_move": np.full(n, 0.005),
            "uncertainty": np.full(n, 0.20),
        }


class StubRangeModel:
    def __init__(self):
        self.config = SimpleNamespace(horizon="30m")
        self.metadata = {"uncertainty": 0.12}

    def predict_proba(self, rows):
        return np.full(len(rows), 0.72)


class PlaceholderModel:
    """Serializable artifact used for decision-model slots in strict modes."""


def _save(
    registry: ModelRegistry,
    model,
    *,
    model_type: str,
    target: str,
    horizon: str | None = None,
    status: str = "shadow",
    required_input_fields: list[str] | None = None,
) -> str:
    return registry.save(
        model,
        model_type=model_type,
        target=target,
        horizon=horizon,
        feature_version=FEATURE_VERSION,
        label_version=LABEL_VERSION,
        status=status,
        hyperparameters={"test": model_type},
        data_hash=f"data-{model_type}",
        author="pytest",
        required_input_fields=required_input_fields or [],
    )


def _save_group(
    registry: ModelRegistry,
    *,
    status: str = "shadow",
    direction_fail: bool = False,
    direction_required: list[str] | None = None,
) -> str:
    components = {
        "direction_30m": _save(
            registry,
            StubDirectionModel(fail=direction_fail),
            model_type="direction",
            target="up_30m",
            horizon="30m",
            status=status,
            required_input_fields=direction_required,
        ),
        "quantiles_30m": _save(
            registry,
            StubQuantileModel(),
            model_type="return_quantiles",
            target="fwd_return_30m",
            horizon="30m",
            status=status,
        ),
        "volatility": _save(
            registry,
            StubVolatilityModel(),
            model_type="volatility",
            target="remaining_realized_move",
            status=status,
        ),
        "range_30m": _save(
            registry,
            StubRangeModel(),
            model_type="range_survival",
            target="range_survive_wall_channel_30m",
            horizon="30m",
            status=status,
        ),
    }
    return registry.save_group(
        component_model_ids=components,
        feature_version=FEATURE_VERSION,
        label_version=LABEL_VERSION,
        structural_state_version="v3.0.0",
        status=status,
        group_id=f"group-{status}-{int(direction_fail)}-{len(direction_required or [])}",
    ).group_id


def _decision_ids(registry: ModelRegistry, *, status: str) -> dict:
    out = {}
    for slot in (
        "candidate_value_model_id",
        "candidate_rank_model_id",
        "fill_probability_model_id",
        "fill_concession_model_id",
        "meta_model_id",
    ):
        out[slot] = _save(
            registry,
            PlaceholderModel(),
            model_type=slot.removesuffix("_id"),
            target="decision_support",
            status=status,
        )
    return out


def _bundle(*, group_id: str | None, mode: str = "shadow", **kwargs):
    values = dict(
        deployment_id=f"dep-{mode}",
        mode=mode,
        prediction_model_group_id=group_id,
        feature_version=FEATURE_VERSION,
        label_version=LABEL_VERSION,
        structural_state_version="v3.0.0",
        policy_version="v3",
        execution_version="v3",
        risk_version="v1",
        fallback_policy="abstain",
        reference_account_id="legacy",
    )
    values.update(kwargs)
    bundle = DeploymentBundle(**values)
    return dataclasses.replace(
        bundle, configuration_hash=configuration_hash(bundle.to_dict()))


def _infer(runtime: PredictionRuntime):
    return runtime.infer(
        snapshot_id="snap-1",
        ts="2026-07-17T14:30:00-04:00",
        session_date="2026-07-17",
        symbol="SPY",
        feature_row={
            "spot": 630.0,
            "put_wall": 625.0,
            "call_wall": 635.0,
            "adx": 18.0,
            "net_gex": 1.0,
        },
        structural={
            "spot": 630.0,
            "put_wall": 625.0,
            "call_wall": 635.0,
            "adx": 18.0,
            "net_gex": 1.0,
            "minutes_to_close": 120.0,
        },
        quality={"feature_coverage": 0.95, "data_quality": 0.92},
    )


def test_runtime_loads_group_and_serves_canonical_bundle(tmp_path):
    registry = ModelRegistry(directory=str(tmp_path / "models"))
    group_id = _save_group(registry)
    runtime = PredictionRuntime(_bundle(group_id=group_id), registry)

    result = _infer(runtime)

    assert result.actionable
    assert result.health.status == "OK"
    assert result.bundle.p_up_30m == pytest.approx(0.63)
    assert result.bundle.return_q10_30m == pytest.approx(-0.004)
    assert result.bundle.return_q50_30m == pytest.approx(0.001)
    assert result.bundle.return_q90_30m == pytest.approx(0.006)
    assert result.bundle.expected_realized_move_30m == pytest.approx(0.005)
    assert result.bundle.p_range_survive_30m == pytest.approx(0.72)
    assert result.bundle.diagnostics["source"] == "trained_artifacts"
    assert result.bundle.diagnostics["artifact_hashes"]


def test_shadow_heuristic_fallback_is_explicit_and_labeled(tmp_path):
    registry = ModelRegistry(directory=str(tmp_path / "models"))
    group_id = _save_group(registry, direction_fail=True)
    calls = []

    def heuristic(**kwargs):
        calls.append(kwargs["snapshot_id"])
        return PredictionBundle(
            snapshot_id=kwargs["snapshot_id"],
            ts=kwargs["ts"],
            session_date=kwargs["session_date"],
            symbol=kwargs["symbol"],
            p_up_30m=0.51,
            uncertainty=0.60,
            data_quality=0.70,
        )

    runtime = PredictionRuntime(
        _bundle(group_id=group_id, fallback_policy="legacy"),
        registry,
        heuristic_provider=heuristic,
    )
    result = _infer(runtime)

    assert calls == ["snap-1"]
    assert result.health.status == "DEGRADED"
    assert result.health.source == "heuristic_fallback"
    assert result.bundle.p_up_30m == pytest.approx(0.51)
    assert result.bundle.diagnostics["source"] == "heuristic_fallback"
    assert "prediction_model_group" in result.bundle.diagnostics[
        "component_errors"]


def test_advisory_mode_never_uses_heuristic_fallback(tmp_path):
    registry = ModelRegistry(directory=str(tmp_path / "models"))
    group_id = _save_group(
        registry, status="candidate", direction_fail=True)
    called = False

    def heuristic(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("advisory must not invoke heuristic fallback")

    runtime = PredictionRuntime(
        _bundle(
            group_id=group_id,
            mode="advisory",
            fallback_policy="legacy",
        ),
        registry,
        heuristic_provider=heuristic,
    )
    result = _infer(runtime)

    assert called is False
    assert result.actionable is False
    assert result.health.status == "ABSTAIN"
    assert result.bundle.uncertainty == 1.0
    assert result.bundle.diagnostics["source"] == "abstain"


def test_candidate_mode_fails_closed_on_missing_required_feature(tmp_path):
    registry = ModelRegistry(directory=str(tmp_path / "models"))
    group_id = _save_group(
        registry, status="candidate", direction_required=["dealer_positioning"])
    decision_ids = _decision_ids(registry, status="candidate")
    bundle = _bundle(
        group_id=group_id,
        mode="candidate",
        fallback_policy="legacy",
        candidate_account_id="candidate-paper",
        **decision_ids,
    )
    runtime = PredictionRuntime(bundle, registry)

    result = _infer(runtime)

    assert result.actionable is False
    assert result.health.status == "ABSTAIN"
    assert "prediction_model_group" in result.health.component_errors
    assert "dealer_positioning" in result.health.component_errors[
        "prediction_model_group"]


def test_runtime_rejects_configuration_hash_mismatch(tmp_path):
    registry = ModelRegistry(directory=str(tmp_path / "models"))
    group_id = _save_group(registry)
    bad = dataclasses.replace(
        _bundle(group_id=group_id), configuration_hash="not-the-real-hash")

    with pytest.raises(PredictionRuntimeError, match="configuration_hash"):
        PredictionRuntime(bad, registry)
