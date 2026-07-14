"""
prediction/models/trade_meta.py
================================
Trade / no-edge / abstain meta-model
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §18–§21).

Statistical actions: TRADE | NO_EDGE | ABSTAIN.
Hard vetoes are applied AFTER this logic by the risk gate.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import RANDOM_STATE, FeatureVectorizer, clip_probability

META_MODEL_VERSION = "v3.0.0"

STATISTICAL_ACTIONS = ("TRADE", "NO_EDGE", "ABSTAIN")


@dataclass
class MetaThresholdConfig:
    uncertainty_abstain_threshold: float = 0.75
    ood_abstain_threshold: float = 0.95
    minimum_data_quality: float = 0.60
    minimum_candidate_utility: float = 0.0
    minimum_expected_order_value: float = 0.0
    minimum_trade_probability: float = 0.58


@dataclass(frozen=True)
class MetaDecision:
    action: str
    p_positive_utility: float
    expected_order_value: float
    selected_candidate_id: Optional[str]
    composite_uncertainty: float
    threshold_used: float
    uncertainty_threshold: float
    ood_threshold: float
    reasons: tuple
    model_version: str = META_MODEL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "p_positive_utility": self.p_positive_utility,
            "expected_order_value": self.expected_order_value,
            "selected_candidate_id": self.selected_candidate_id,
            "composite_uncertainty": self.composite_uncertainty,
            "threshold_used": self.threshold_used,
            "uncertainty_threshold": self.uncertainty_threshold,
            "ood_threshold": self.ood_threshold,
            "reasons": list(self.reasons),
            "model_version": self.model_version,
            "diagnostics": dict(self.diagnostics),
        }


def meta_features_from_inputs(
    *,
    forecast: Optional[dict] = None,
    candidate: Optional[dict] = None,
    execution: Optional[dict] = None,
    context: Optional[dict] = None,
) -> dict:
    """Allowed features only — never include realized outcomes."""
    forecast = forecast or {}
    candidate = candidate or {}
    execution = execution or {}
    context = context or {}
    prohibited = {
        "realized_pnl", "realized_utility", "future_fill", "settlement",
        "human_action", "user_agreed", "future_drift", "post_trade_notes",
    }
    for src in (forecast, candidate, execution, context):
        for k in prohibited:
            if k in src:
                raise ValueError(f"prohibited meta feature present: {k}")
    row = {}
    for k, v in forecast.items():
        row[f"f_{k}"] = v
    for k, v in candidate.items():
        row[f"c_{k}"] = v
    for k, v in execution.items():
        row[f"e_{k}"] = v
    for k, v in context.items():
        row[f"x_{k}"] = v
    return row


def decide_meta_action(
    *,
    p_positive_utility: float,
    expected_order_value: float,
    selected_candidate_id: Optional[str],
    selected_candidate_utility: Optional[float],
    composite_uncertainty: float,
    ood_score: float,
    data_quality: float,
    cfg: Optional[MetaThresholdConfig] = None,
) -> tuple:
    """
    Returns (action, reasons). Pure threshold logic — no hard vetoes here.
    """
    cfg = cfg or MetaThresholdConfig()
    reasons: list = []
    if composite_uncertainty >= cfg.uncertainty_abstain_threshold:
        reasons.append("high_model_uncertainty")
        return "ABSTAIN", tuple(reasons)
    if ood_score >= cfg.ood_abstain_threshold:
        reasons.append("extreme_ood")
        return "ABSTAIN", tuple(reasons)
    if data_quality <= cfg.minimum_data_quality:
        reasons.append("low_data_quality")
        return "ABSTAIN", tuple(reasons)
    if selected_candidate_id is None:
        reasons.append("no_feasible_candidate")
        return "NO_EDGE", tuple(reasons)
    util = float(selected_candidate_utility
                 if selected_candidate_utility is not None else -1e9)
    if util <= cfg.minimum_candidate_utility:
        reasons.append("negative_candidate_utility")
        return "NO_EDGE", tuple(reasons)
    if expected_order_value <= cfg.minimum_expected_order_value:
        reasons.append("negative_expected_order_value")
        return "NO_EDGE", tuple(reasons)
    if p_positive_utility < cfg.minimum_trade_probability:
        reasons.append("meta_probability_below_threshold")
        return "NO_EDGE", tuple(reasons)
    reasons.extend([
        "positive_expected_order_value",
        "candidate_utility_above_threshold",
        "meta_probability_above_threshold",
        "uncertainty_below_threshold",
    ])
    return "TRADE", tuple(reasons)


def apply_hard_vetoes(
    statistical_action: str,
    hard_vetoes: Sequence[str],
) -> tuple:
    """Any hard veto converts the final action to HARD_VETO."""
    vetoes = tuple(hard_vetoes or ())
    if vetoes:
        return "HARD_VETO", vetoes
    return statistical_action, vetoes


def select_thresholds_nested(
    fold_scores: Sequence[dict],
    *,
    prob_grid: Sequence[float] = (0.50, 0.55, 0.58, 0.60, 0.65),
    unc_grid: Sequence[float] = (0.70, 0.75, 0.80),
) -> MetaThresholdConfig:
    """
    Choose thresholds from inner-fold scores only (no outer-test leakage).
    fold_scores: [{prob, unc, utility, shortfall, drew_down, traded}, ...]
    Research objective: mean utility - penalties.
    """
    best = None
    best_score = -1e18
    for pmin in prob_grid:
        for unc in unc_grid:
            utils, shorts, trades, abstains = [], [], 0, 0
            for row in fold_scores:
                if row["unc"] >= unc:
                    abstains += 1
                    continue
                if row["prob"] < pmin:
                    continue
                trades += 1
                utils.append(float(row["utility"]))
                shorts.append(float(row.get("shortfall", 0.0)))
            if trades == 0:
                score = -1.0  # avoid zero-trade "success"
            else:
                mean_u = float(np.mean(utils))
                mean_s = float(np.mean(shorts)) if shorts else 0.0
                trade_rate = trades / max(len(fold_scores), 1)
                abstain_rate = abstains / max(len(fold_scores), 1)
                score = (mean_u - 0.5 * mean_s
                         - 0.1 * max(trade_rate - 0.5, 0.0)
                         - 0.1 * max(abstain_rate - 0.5, 0.0))
            if score > best_score:
                best_score = score
                best = MetaThresholdConfig(
                    minimum_trade_probability=float(pmin),
                    uncertainty_abstain_threshold=float(unc),
                )
    return best or MetaThresholdConfig()


@dataclass
class TradeMetaModel:
    estimator: str = "logistic"
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    model: object = None
    fitted: bool = False
    thresholds: MetaThresholdConfig = field(default_factory=MetaThresholdConfig)
    model_version: str = META_MODEL_VERSION

    def _make(self):
        if self.estimator == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=10, random_state=RANDOM_STATE)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ])

    def fit(self, feature_rows: Sequence[dict], labels: Sequence[int],
            ) -> "TradeMetaModel":
        if len(feature_rows) < 4:
            raise ValueError("need at least 4 meta rows")
        y = np.asarray(labels, dtype=int)
        self.vectorizer.fit(list(feature_rows))
        X = self.vectorizer.transform(list(feature_rows))
        est = self._make()
        if y.min() == y.max():
            self.model = ("constant", float(y.mean()))
        else:
            est.fit(X, y)
            self.model = ("est", est)
        self.fitted = True
        return self

    def predict_proba(self, features: dict) -> float:
        if not self.fitted:
            raise RuntimeError("TradeMetaModel.predict before fit")
        kind, obj = self.model
        if kind == "constant":
            return clip_probability(obj)
        X = self.vectorizer.transform([features])
        return clip_probability(float(obj.predict_proba(X)[0, 1]))

    def decide(
        self,
        features: dict,
        *,
        expected_order_value: float,
        selected_candidate_id: Optional[str],
        selected_candidate_utility: Optional[float],
        composite_uncertainty: float,
        ood_score: float,
        data_quality: float,
        hard_vetoes: Sequence[str] = (),
    ) -> MetaDecision:
        p = self.predict_proba(features) if self.fitted else 0.5
        statistical, reasons = decide_meta_action(
            p_positive_utility=p,
            expected_order_value=expected_order_value,
            selected_candidate_id=selected_candidate_id,
            selected_candidate_utility=selected_candidate_utility,
            composite_uncertainty=composite_uncertainty,
            ood_score=ood_score,
            data_quality=data_quality,
            cfg=self.thresholds,
        )
        final, vetoes = apply_hard_vetoes(statistical, hard_vetoes)
        out_reasons = reasons
        if final == "HARD_VETO":
            out_reasons = reasons + tuple(f"hard_veto:{v}" for v in vetoes)
        return MetaDecision(
            action=final,
            p_positive_utility=p,
            expected_order_value=float(expected_order_value),
            selected_candidate_id=selected_candidate_id,
            composite_uncertainty=float(composite_uncertainty),
            threshold_used=self.thresholds.minimum_trade_probability,
            uncertainty_threshold=self.thresholds.uncertainty_abstain_threshold,
            ood_threshold=self.thresholds.ood_abstain_threshold,
            reasons=out_reasons,
            model_version=self.model_version,
            diagnostics={
                "statistical_action": statistical,
                "hard_vetoes": list(vetoes),
            },
        )
