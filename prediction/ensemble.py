"""
prediction/ensemble.py
======================
Forecast-level ensemble weighting (V3 Part 2 §32–§33, PR 15).

Weights derive from historical out-of-sample loss only. Unavailable /
failed components are excluded and recorded. Maximum component weight is
enforced. Legacy Monte Carlo cannot silently dominate after artifact
load failures.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence, Union

from prediction.models.base import clip_probability

ENSEMBLE_VERSION = "v3.0.0-ensemble"
PredictionValue = Union[float, dict]

# Soft prior so a brand-new component is not zeroed
_DEFAULT_PRIOR = 1.0
_FALLBACK_WEIGHT = 1e-6


@dataclass(frozen=True)
class EnsembleForecast:
    target: str
    horizon: str
    prediction: PredictionValue
    component_predictions: dict[str, PredictionValue]
    component_weights: dict[str, float]
    component_uncertainties: dict[str, float]
    disagreement: float
    composite_uncertainty: float
    missing_components: tuple[str, ...]
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnsembleConfig:
    maximum_component_weight: float = 0.60
    minimum_eligible_weight: float = 0.0
    eta: float = 1.0
    fallback_weight: float = _FALLBACK_WEIGHT
    negative_skill_fallback_only: bool = True
    legacy_mc_name: str = "legacy_monte_carlo"
    epsilon: float = 1e-12


def oos_component_weight(
    historical_oos_loss: float,
    *,
    prior_weight: float = _DEFAULT_PRIOR,
    eta: float = 1.0,
) -> float:
    """raw_weight = prior * exp(-eta * loss) (§32.4)."""
    return float(prior_weight * math.exp(-float(eta) * float(historical_oos_loss)))


def enforce_max_weight(
    weights: Mapping[str, float],
    *,
    maximum: float = 0.60,
    epsilon: float = 1e-12,
    protect_below: float = 1e-4,
) -> dict[str, float]:
    """Cap any single weight and renormalize the remainder."""
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    if not w:
        return {}
    # Tiny fallback weights stay tiny — never boosted by redistribution
    protected = {k for k, v in w.items() if v <= protect_below}
    for _ in range(len(w) + 2):
        total = sum(w.values())
        if total <= epsilon:
            n = len(w)
            return {k: 1.0 / n for k in w}
        w = {k: v / total for k, v in w.items()}
        over = [k for k, v in w.items()
                if v > maximum + epsilon and k not in protected]
        if not over:
            break
        capped_mass = sum(maximum for _ in over)
        # Keep protected at their current (tiny) normalized values
        prot_mass = sum(w[k] for k in protected)
        free = {k: v for k, v in w.items()
                if k not in over and k not in protected}
        for k in over:
            w[k] = maximum
        remain = max(0.0, 1.0 - capped_mass - prot_mass)
        free_sum = sum(free.values())
        if free and free_sum > epsilon and remain > epsilon:
            for k in free:
                w[k] = free[k] / free_sum * remain
        elif not free and not protected:
            n = len(w)
            w = {k: 1.0 / n for k in w}
            break
        # else: leave protected as-is; capped hold maximum; done
        break
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > epsilon else w


def weighted_blend(
    predictions: Mapping[str, PredictionValue],
    weights: Mapping[str, float],
) -> PredictionValue:
    keys = [k for k in predictions if k in weights and weights[k] > 0]
    if not keys:
        raise ValueError("no eligible components to blend")
    sample = predictions[keys[0]]
    if isinstance(sample, dict):
        out_keys = sorted({kk for k in keys for kk in predictions[k]})  # type: ignore
        return {
            kk: float(sum(
                float(weights[k]) * float(predictions[k].get(kk, 0.0))  # type: ignore
                for k in keys))
            for kk in out_keys
        }
    return float(sum(float(weights[k]) * float(predictions[k]) for k in keys))


def weighted_disagreement(
    predictions: Mapping[str, PredictionValue],
    weights: Mapping[str, float],
    final: PredictionValue,
) -> float:
    keys = [k for k in predictions if k in weights and weights[k] > 0]
    if len(keys) <= 1:
        return 0.0
    if isinstance(final, dict):
        vals = []
        for kk in final:
            mu = float(final[kk])
            var = sum(
                float(weights[k]) * (
                    float(predictions[k].get(kk, mu)) - mu) ** 2  # type: ignore
                for k in keys)
            vals.append(var)
        return float(sum(vals) / max(len(vals), 1))
    mu = float(final)
    return float(sum(
        float(weights[k]) * (float(predictions[k]) - mu) ** 2
        for k in keys))


@dataclass
class ForecastEnsemble:
    """Assemble an EnsembleForecast from component predictions + OOS losses."""

    target: str
    horizon: str
    cfg: EnsembleConfig = field(default_factory=EnsembleConfig)
    model_version: str = ENSEMBLE_VERSION

    def combine(
        self,
        component_predictions: Mapping[str, Optional[PredictionValue]],
        *,
        oos_losses: Optional[Mapping[str, float]] = None,
        prior_weights: Optional[Mapping[str, float]] = None,
        component_uncertainties: Optional[Mapping[str, float]] = None,
        negative_skill: Optional[Sequence[str]] = None,
        ood_scores: Optional[Mapping[str, float]] = None,
        artifact_load_failures: Optional[Sequence[str]] = None,
    ) -> EnsembleForecast:
        oos_losses = dict(oos_losses or {})
        prior_weights = dict(prior_weights or {})
        component_uncertainties = dict(component_uncertainties or {})
        negative_skill = set(negative_skill or ())
        ood_scores = dict(ood_scores or {})
        failures = set(artifact_load_failures or ())

        missing = []
        eligible_preds: dict[str, PredictionValue] = {}
        for name, pred in component_predictions.items():
            if pred is None or name in failures:
                missing.append(name)
                continue
            eligible_preds[name] = pred

        diagnostics: dict = {
            "artifact_load_failures": sorted(failures),
            "negative_skill": sorted(negative_skill),
        }

        if not eligible_preds:
            raise RuntimeError(
                "no eligible ensemble components "
                f"(missing={missing}, failures={sorted(failures)})")

        raw_weights: dict[str, float] = {}
        for name in eligible_preds:
            if name in negative_skill and self.cfg.negative_skill_fallback_only:
                raw_weights[name] = float(self.cfg.fallback_weight)
                continue
            loss = float(oos_losses.get(name, 1.0))
            prior = float(prior_weights.get(name, _DEFAULT_PRIOR))
            w = oos_component_weight(loss, prior_weight=prior, eta=self.cfg.eta)
            # Extreme OOD → reduce weight
            ood = float(ood_scores.get(name, 0.0))
            if ood >= 0.8:
                w *= 0.25
            raw_weights[name] = max(w, float(self.cfg.minimum_eligible_weight))

        # Prevent legacy MC from dominating when it is the only survivor of
        # artifact failures — keep it but flag; if others exist, OK.
        if (self.cfg.legacy_mc_name in raw_weights
                and failures
                and set(eligible_preds) == {self.cfg.legacy_mc_name}):
            diagnostics["legacy_mc_sole_survivor"] = True
            diagnostics["legacy_mc_dominance_blocked"] = False
            # Still allow but mark — caller may abstain. Spec: must not
            # silently become dominant merely because others failed to load.
            # We keep the prediction but set a high uncertainty penalty.
            diagnostics["silent_dominance_prevented"] = True

        weights = enforce_max_weight(
            raw_weights,
            maximum=self.cfg.maximum_component_weight,
            epsilon=self.cfg.epsilon,
        )
        # If legacy MC would be sole component after failures, force explicit
        # uncertainty spike rather than pretending it's a healthy ensemble.
        unc_penalty = 0.0
        if diagnostics.get("silent_dominance_prevented"):
            unc_penalty = 0.5

        final = weighted_blend(eligible_preds, weights)
        disagree = weighted_disagreement(eligible_preds, weights, final)
        # Composite uncertainty
        mean_comp_unc = 0.0
        if weights:
            mean_comp_unc = sum(
                weights[k] * float(component_uncertainties.get(k, 0.0))
                for k in weights)
        missing_pen = 0.1 * len(missing)
        composite = float(clip_probability(
            0.4 * mean_comp_unc
            + 0.3 * math.sqrt(max(0.0, disagree))
            + 0.2 * missing_pen
            + unc_penalty
        ))

        return EnsembleForecast(
            target=self.target,
            horizon=self.horizon,
            prediction=final,
            component_predictions=dict(eligible_preds),
            component_weights=weights,
            component_uncertainties={
                k: float(component_uncertainties.get(k, 0.0)) for k in weights
            },
            disagreement=float(disagree),
            composite_uncertainty=composite,
            missing_components=tuple(sorted(missing)),
            model_version=self.model_version,
            diagnostics=diagnostics,
        )
