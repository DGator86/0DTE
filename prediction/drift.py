"""
prediction/drift.py
===================
Feature / prediction / residual / execution / economic drift monitoring
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §24–§27).

Drift may penalize or FREEZE models. Drift never promotes models and never
deletes artifacts.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

SEVERITIES = ("NORMAL", "WATCH", "DEGRADED", "FREEZE")


@dataclass
class DriftConfig:
    watch_threshold: float = 0.40
    degraded_threshold: float = 0.60
    freeze_threshold: float = 0.80
    w_feature: float = 0.20
    w_prediction: float = 0.15
    w_residual: float = 0.25
    w_execution: float = 0.15
    w_economic: float = 0.25


@dataclass(frozen=True)
class DriftStatus:
    model_id: str
    as_of_session: str
    feature_drift: float
    prediction_drift: float
    residual_drift: float
    execution_drift: float
    economic_drift: float
    composite: float
    severity: str
    affected_slices: tuple
    recommended_action: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["affected_slices"] = list(self.affected_slices)
        return d


def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    n_bins: int = 10,
) -> float:
    """PSI between two univariate samples (clipped)."""
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) < 2 or len(act) < 2:
        return 0.0
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(exp, qs))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(exp, bins=edges)
    a_hist, _ = np.histogram(act, bins=edges)
    e_pct = e_hist / max(e_hist.sum(), 1)
    a_pct = a_hist / max(a_hist.sum(), 1)
    e_pct = np.clip(e_pct, 1e-4, None)
    a_pct = np.clip(a_pct, 1e-4, None)
    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return float(np.clip(psi, 0.0, 5.0))


def mean_shift_score(ref: Sequence[float], cur: Sequence[float]) -> float:
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if len(r) < 2 or len(c) < 1:
        return 0.0
    sd = float(np.std(r)) + 1e-9
    return float(np.clip(abs(np.mean(c) - np.mean(r)) / sd, 0.0, 5.0))


def missingness_shift(ref_miss: float, cur_miss: float) -> float:
    return float(np.clip(abs(float(cur_miss) - float(ref_miss)) * 5.0, 0.0, 5.0))


def severity_from_composite(composite: float, cfg: Optional[DriftConfig] = None,
                            *, force_freeze: bool = False) -> str:
    if force_freeze:
        return "FREEZE"
    cfg = cfg or DriftConfig()
    c = float(composite)
    if c >= cfg.freeze_threshold:
        return "FREEZE"
    if c >= cfg.degraded_threshold:
        return "DEGRADED"
    if c >= cfg.watch_threshold:
        return "WATCH"
    return "NORMAL"


def recommended_action_for(severity: str) -> str:
    return {
        "NORMAL": "continue",
        "WATCH": "journal_warning_and_penalize_weights",
        "DEGRADED": "prevent_promotion_reduce_weight_require_review",
        "FREEZE": "remove_from_decision_ensemble_require_human_reactivation",
    }.get(severity, "continue")


def compute_drift_status(
    *,
    model_id: str,
    as_of_session: str,
    feature_drift: Optional[float] = None,
    prediction_drift: Optional[float] = None,
    residual_drift: Optional[float] = None,
    execution_drift: Optional[float] = None,
    economic_drift: Optional[float] = None,
    affected_slices: Sequence[str] = (),
    cfg: Optional[DriftConfig] = None,
    force_freeze: bool = False,
) -> DriftStatus:
    """
    Composite drift with reweighting when components are missing.
    Missing economic labels are NOT treated as zero drift.
    """
    cfg = cfg or DriftConfig()
    components = {
        "feature": (feature_drift, cfg.w_feature),
        "prediction": (prediction_drift, cfg.w_prediction),
        "residual": (residual_drift, cfg.w_residual),
        "execution": (execution_drift, cfg.w_execution),
        "economic": (economic_drift, cfg.w_economic),
    }
    present = {k: (float(v), w) for k, (v, w) in components.items()
               if v is not None}
    missing = tuple(sorted(k for k, (v, _) in components.items() if v is None))
    if not present:
        composite = 0.0
        vals = {k: 0.0 for k in components}
    else:
        wsum = sum(w for _, w in present.values()) or 1.0
        composite = sum(v * (w / wsum) for v, w in present.values())
        vals = {k: (float(v) if v is not None else float("nan"))
                for k, (v, _) in components.items()}
    severity = severity_from_composite(
        composite, cfg, force_freeze=force_freeze)
    return DriftStatus(
        model_id=model_id,
        as_of_session=as_of_session,
        feature_drift=float(vals["feature"]) if vals["feature"] == vals["feature"] else 0.0,
        prediction_drift=float(vals["prediction"]) if vals["prediction"] == vals["prediction"] else 0.0,
        residual_drift=float(vals["residual"]) if vals["residual"] == vals["residual"] else 0.0,
        execution_drift=float(vals["execution"]) if vals["execution"] == vals["execution"] else 0.0,
        economic_drift=float(vals["economic"]) if vals["economic"] == vals["economic"] else 0.0,
        composite=float(composite),
        severity=severity,
        affected_slices=tuple(affected_slices),
        recommended_action=recommended_action_for(severity),
        diagnostics={
            "missing_components": list(missing),
            "reweighted": bool(missing),
            "force_freeze": force_freeze,
        },
    )


def drift_weight_penalty(severity: str, cfg: Optional[DriftConfig] = None) -> float:
    """Multiplicative weight penalty factor in [0, 1]. FREEZE → 0."""
    from prediction.dynamic_weights import DynamicWeightConfig
    dw = DynamicWeightConfig()
    sev = severity.upper()
    if sev == "FREEZE":
        return 0.0
    if sev == "DEGRADED":
        return max(0.0, 1.0 - dw.drift_penalty_degraded)
    if sev == "WATCH":
        return max(0.0, 1.0 - dw.drift_penalty_watch)
    return 1.0
