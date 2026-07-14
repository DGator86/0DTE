"""
prediction/dynamic_weights.py
=============================
Dynamic out-of-sample ensemble weighting
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §22–§23).

Updates only after completed/settled sessions. Never promotes models.
Never uses current-session unresolved labels.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class DynamicWeightConfig:
    eta: float = 1.0
    session_half_life: float = 20.0
    minimum_weight: float = 0.05
    maximum_weight: float = 0.60
    uncertainty_penalty: float = 0.50
    drift_penalty_watch: float = 0.15
    drift_penalty_degraded: float = 0.50
    allow_shadow_models: bool = True


@dataclass(frozen=True)
class DynamicWeightState:
    target: str
    horizon: Optional[str]
    as_of_session: str
    weights: dict
    recent_losses: dict
    full_losses: dict
    penalties: dict
    excluded_models: tuple
    configuration_hash: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["excluded_models"] = list(self.excluded_models)
        return d


def _config_hash(cfg: DynamicWeightConfig) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()


def composite_loss(
    loss_20: float,
    loss_60: float,
    loss_full: float,
) -> float:
    return 0.50 * float(loss_20) + 0.30 * float(loss_60) + 0.20 * float(loss_full)


def update_dynamic_weights(
    *,
    target: str,
    as_of_session: str,
    prior_weights: dict,
    losses_20: dict,
    losses_60: dict,
    losses_full: dict,
    drift_severity: Optional[dict] = None,
    uncertainty: Optional[dict] = None,
    model_status: Optional[dict] = None,
    horizon: Optional[str] = None,
    cfg: Optional[DynamicWeightConfig] = None,
) -> DynamicWeightState:
    """
    Compute new weights from completed-session losses only.

    model_status values: research|shadow|candidate|champion|FREEZE|...
    FREEZE / degraded-freeze models are excluded from decision-facing weights.
    """
    cfg = cfg or DynamicWeightConfig()
    drift_severity = drift_severity or {}
    uncertainty = uncertainty or {}
    model_status = model_status or {}

    models = sorted(set(prior_weights) | set(losses_20) | set(losses_full))
    raw = {}
    penalties = {}
    excluded = []
    for m in models:
        status = str(model_status.get(m, "shadow")).upper()
        sev = str(drift_severity.get(m, "NORMAL")).upper()
        if sev == "FREEZE" or status == "FREEZE":
            excluded.append(m)
            penalties[m] = {"reason": "freeze", "factor": 0.0}
            continue
        if not cfg.allow_shadow_models and status.lower() == "shadow":
            excluded.append(m)
            penalties[m] = {"reason": "shadow_disallowed", "factor": 0.0}
            continue
        prior = float(prior_weights.get(m, 1.0 / max(len(models), 1)))
        loss = composite_loss(
            float(losses_20.get(m, losses_full.get(m, 1.0))),
            float(losses_60.get(m, losses_full.get(m, 1.0))),
            float(losses_full.get(m, 1.0)),
        )
        w = prior * math.exp(-cfg.eta * loss)
        pen = 1.0
        reasons = []
        u = float(uncertainty.get(m, 0.0))
        if u > 0:
            pen *= max(0.0, 1.0 - cfg.uncertainty_penalty * u)
            reasons.append("uncertainty")
        if sev == "WATCH":
            pen *= max(0.0, 1.0 - cfg.drift_penalty_watch)
            reasons.append("drift_watch")
        elif sev == "DEGRADED":
            pen *= max(0.0, 1.0 - cfg.drift_penalty_degraded)
            reasons.append("drift_degraded")
        w *= pen
        raw[m] = max(w, 0.0)
        penalties[m] = {"factor": pen, "reasons": reasons, "loss": loss}

    # Clamp and normalize
    if not raw:
        # Fall back to equal prior among non-excluded if everything frozen
        weights = {}
    else:
        # Apply min/max after preliminary normalize
        total = sum(raw.values()) or 1.0
        weights = {m: v / total for m, v in raw.items()}
        # Enforce max
        for m in list(weights):
            if weights[m] > cfg.maximum_weight:
                weights[m] = cfg.maximum_weight
        # Enforce min among included
        for m in list(weights):
            if 0.0 < weights[m] < cfg.minimum_weight:
                weights[m] = cfg.minimum_weight
        total = sum(weights.values()) or 1.0
        weights = {m: float(v / total) for m, v in weights.items()}
        # Re-clamp max after renormalize once
        overflow = 0.0
        for m in list(weights):
            if weights[m] > cfg.maximum_weight:
                overflow += weights[m] - cfg.maximum_weight
                weights[m] = cfg.maximum_weight
        if overflow > 0:
            others = [m for m in weights if weights[m] < cfg.maximum_weight]
            if others:
                add = overflow / len(others)
                for m in others:
                    weights[m] = min(cfg.maximum_weight, weights[m] + add)
            total = sum(weights.values()) or 1.0
            weights = {m: float(v / total) for m, v in weights.items()}

    return DynamicWeightState(
        target=target,
        horizon=horizon,
        as_of_session=as_of_session,
        weights=weights,
        recent_losses={m: float(losses_20.get(m, 0.0)) for m in models},
        full_losses={m: float(losses_full.get(m, 0.0)) for m in models},
        penalties=penalties,
        excluded_models=tuple(sorted(excluded)),
        configuration_hash=_config_hash(cfg),
        diagnostics={
            "n_models": len(models),
            "n_excluded": len(excluded),
        },
    )
