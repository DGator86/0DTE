"""
prediction/models/mixture_experts.py
====================================
Regime mixture-of-experts blending (V3 Part 2 §14–§15, PR 10).

Combines a global expert with regime-specialized experts using the full
regime probability vector. Low-support experts shrink toward the global
model. Missing experts fall back explicitly to the global prediction.

Forecasts remain independent of candidate selection.

Research / shadow only.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from prediction.models.base import clip_probability
from prediction.regime_labels import REGIME_CLASSES

MIXTURE_MODEL_VERSION = "v3.0.0-moe"

PredictionValue = Union[float, dict]


@dataclass
class MixtureExpertsConfig:
    shrinkage_sessions: float = 40.0
    minimum_sessions: float = 40.0
    minimum_effective_sessions: float = 20.0
    minimum_rows: float = 500.0
    epsilon: float = 1e-12


@dataclass(frozen=True)
class MixtureForecast:
    target: str
    horizon: str
    final_prediction: PredictionValue
    regime_probabilities: dict[str, float]
    raw_expert_predictions: dict[str, PredictionValue]
    shrunk_expert_predictions: dict[str, PredictionValue]
    expert_support: dict[str, float]
    expert_weights: dict[str, float]
    global_prediction: PredictionValue
    uncertainty: float
    disagreement: float
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def expert_shrinkage_weight(
    effective_support_sessions: float,
    *,
    shrinkage_sessions: float = 40.0,
) -> float:
    """
    expert_weight = support / (support + shrinkage_sessions) (§14.6).
    """
    s = max(0.0, float(effective_support_sessions))
    k = max(0.0, float(shrinkage_sessions))
    if s + k <= 0:
        return 0.0
    return float(s / (s + k))


def shrink_prediction(
    expert_prediction: PredictionValue,
    global_prediction: PredictionValue,
    expert_weight: float,
) -> PredictionValue:
    """Blend expert toward global by expert_weight."""
    w = float(clip_probability(expert_weight))
    if isinstance(expert_prediction, dict) or isinstance(global_prediction, dict):
        ep = dict(expert_prediction) if isinstance(expert_prediction, dict) else {}
        gp = dict(global_prediction) if isinstance(global_prediction, dict) else {}
        keys = sorted(set(ep) | set(gp))
        return {
            k: w * float(ep.get(k, gp.get(k, 0.0)))
            + (1.0 - w) * float(gp.get(k, ep.get(k, 0.0)))
            for k in keys
        }
    return float(
        w * float(expert_prediction) + (1.0 - w) * float(global_prediction)
    )


def blend_mixture(
    regime_probabilities: Mapping[str, float],
    shrunk_expert_predictions: Mapping[str, PredictionValue],
    *,
    classes: Sequence[str] = REGIME_CLASSES,
    epsilon: float = 1e-12,
) -> PredictionValue:
    """
    Soft mixture: sum_r p(r) * shrunk_expert[r] (§14.5).
    """
    probs = {c: max(0.0, float(regime_probabilities.get(c, 0.0))) for c in classes}
    total = sum(probs.values())
    if total <= epsilon:
        probs = {c: 1.0 / len(classes) for c in classes}
    else:
        probs = {c: probs[c] / total for c in classes}

    sample = next(iter(shrunk_expert_predictions.values()))
    if isinstance(sample, dict):
        keys = sorted({k for v in shrunk_expert_predictions.values()
                       if isinstance(v, dict) for k in v})
        out: dict[str, float] = {}
        for k in keys:
            out[k] = float(sum(
                probs[c] * float(shrunk_expert_predictions.get(c, {}).get(k, 0.0))
                for c in classes
            ))
        return out
    return float(sum(
        probs[c] * float(shrunk_expert_predictions.get(c, 0.0))
        for c in classes
    ))


def between_expert_disagreement(
    regime_probabilities: Mapping[str, float],
    expert_predictions: Mapping[str, PredictionValue],
    final_prediction: PredictionValue,
    *,
    classes: Sequence[str] = REGIME_CLASSES,
) -> float:
    """
    Between-expert variance for scalars; mean key-wise variance for dicts.
    """
    probs = {c: max(0.0, float(regime_probabilities.get(c, 0.0))) for c in classes}
    total = sum(probs.values()) or 1.0
    probs = {c: probs[c] / total for c in classes}

    if isinstance(final_prediction, dict):
        keys = list(final_prediction.keys())
        if not keys:
            return 0.0
        vars_ = []
        for k in keys:
            fp = float(final_prediction[k])
            var = sum(
                probs[c] * (
                    float(expert_predictions.get(c, {}).get(k, fp)) - fp
                ) ** 2
                for c in classes
            )
            vars_.append(var)
        return float(sum(vars_) / len(vars_))

    fp = float(final_prediction)
    return float(sum(
        probs[c] * (float(expert_predictions.get(c, fp)) - fp) ** 2
        for c in classes
    ))


def mixture_uncertainty(
    *,
    within_expert: float,
    disagreement: float,
    gating_entropy: float,
    support_uncertainty: float,
    ood_score: float = 0.0,
) -> float:
    """Composite mixture uncertainty (§14.9) — clipped to [0, 1]."""
    raw = (
        0.35 * float(within_expert)
        + 0.25 * math.sqrt(max(0.0, float(disagreement)))
        + 0.20 * float(gating_entropy)
        + 0.15 * float(support_uncertainty)
        + 0.05 * float(ood_score)
    )
    return float(clip_probability(raw))


@dataclass
class MixtureOfExperts:
    """
    Blend global + regime experts for one target/horizon.

    Experts are callables: row -> float|dict, or pre-bound prediction maps.
    Training membership uses hard regime labels (initial Part 2 form).
    """

    target: str
    horizon: str
    cfg: MixtureExpertsConfig = field(default_factory=MixtureExpertsConfig)
    global_expert: Optional[Callable[[dict], PredictionValue]] = None
    regime_experts: dict[str, Callable[[dict], PredictionValue]] = field(
        default_factory=dict)
    expert_support: dict[str, float] = field(default_factory=dict)
    expert_available: dict[str, bool] = field(default_factory=dict)
    model_version: str = MIXTURE_MODEL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def register_global(self, expert: Callable[[dict], PredictionValue]) -> None:
        self.global_expert = expert

    def register_regime_expert(
        self,
        regime: str,
        expert: Callable[[dict], PredictionValue],
        *,
        support_sessions: float,
        n_rows: float = 0.0,
    ) -> None:
        if regime not in REGIME_CLASSES:
            raise ValueError(f"unknown regime {regime!r}")
        ok = support_sessions >= self.cfg.minimum_effective_sessions
        self.regime_experts[regime] = expert
        self.expert_support[regime] = float(support_sessions)
        self.expert_available[regime] = bool(ok)
        if not ok:
            self.diagnostics.setdefault("fallbacks", []).append({
                "regime": regime,
                "reason": "insufficient_support",
                "support_sessions": float(support_sessions),
                "n_rows": float(n_rows),
            })

    def predict(
        self,
        row: dict,
        regime_probabilities: Mapping[str, float],
        *,
        within_expert_uncertainty: float = 0.0,
        ood_score: float = 0.0,
    ) -> MixtureForecast:
        if self.global_expert is None:
            raise RuntimeError("global expert not registered")
        # Candidate leakage guard — refuse known candidate keys
        banned = {"candidate_id", "family", "legs", "candidate_rank",
                  "gate_result", "policy_choice", "pnl"}
        leak = banned.intersection(row.keys())
        if leak:
            raise ValueError(f"candidate information in expert features: {leak}")

        global_pred = self.global_expert(row)
        raw: dict[str, PredictionValue] = {}
        shrunk: dict[str, PredictionValue] = {}
        weights: dict[str, float] = {}
        support = dict(self.expert_support)

        for regime in REGIME_CLASSES:
            available = self.expert_available.get(regime, False)
            expert = self.regime_experts.get(regime)
            if not available or expert is None:
                raw[regime] = global_pred
                shrunk[regime] = global_pred
                weights[regime] = 0.0
                support.setdefault(regime, 0.0)
                continue
            pred = expert(row)
            raw[regime] = pred
            w = expert_shrinkage_weight(
                support.get(regime, 0.0),
                shrinkage_sessions=self.cfg.shrinkage_sessions,
            )
            weights[regime] = w
            shrunk[regime] = shrink_prediction(pred, global_pred, w)

        probs = {
            c: max(0.0, float(regime_probabilities.get(c, 0.0)))
            for c in REGIME_CLASSES
        }
        z = sum(probs.values())
        if z <= self.cfg.epsilon:
            probs = {c: 1.0 / len(REGIME_CLASSES) for c in REGIME_CLASSES}
        else:
            probs = {c: probs[c] / z for c in REGIME_CLASSES}

        final = blend_mixture(probs, shrunk, classes=REGIME_CLASSES,
                              epsilon=self.cfg.epsilon)
        disagree = between_expert_disagreement(probs, shrunk, final)
        # Gating entropy
        ent = 0.0
        for p in probs.values():
            if p > 0:
                ent -= p * math.log(p)
        gating_entropy = ent / math.log(len(REGIME_CLASSES))
        # Support uncertainty: weight by regime prob of low-support experts
        support_u = 0.0
        for c in REGIME_CLASSES:
            s = support.get(c, 0.0)
            pen = 1.0 - min(1.0, s / max(self.cfg.shrinkage_sessions, 1.0))
            support_u += probs[c] * pen
        unc = mixture_uncertainty(
            within_expert=within_expert_uncertainty,
            disagreement=disagree,
            gating_entropy=gating_entropy,
            support_uncertainty=support_u,
            ood_score=ood_score,
        )
        return MixtureForecast(
            target=self.target,
            horizon=self.horizon,
            final_prediction=final,
            regime_probabilities=probs,
            raw_expert_predictions=raw,
            shrunk_expert_predictions=shrunk,
            expert_support=support,
            expert_weights=weights,
            global_prediction=global_pred,
            uncertainty=unc,
            disagreement=float(disagree),
            model_version=self.model_version,
            diagnostics={
                "gating_entropy": gating_entropy,
                "support_uncertainty": support_u,
                "fallbacks": list(self.diagnostics.get("fallbacks") or []),
            },
        )


def hard_label_expert_membership(
    labels: Sequence[Optional[str]],
    sessions: Sequence[str],
) -> dict[str, dict[str, float]]:
    """
    Initial Part 2 membership: hard regime labels → per-regime support
    (sessions + rows). Soft OOF weights are a later preferred form.
    """
    out = {c: {"sessions": set(), "rows": 0} for c in REGIME_CLASSES}
    for lab, sess in zip(labels, sessions):
        if lab not in REGIME_CLASSES:
            continue
        out[lab]["sessions"].add(sess)
        out[lab]["rows"] += 1
    return {
        c: {
            "support_sessions": float(len(v["sessions"])),
            "n_rows": float(v["rows"]),
        }
        for c, v in out.items()
    }
