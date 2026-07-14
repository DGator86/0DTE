"""
prediction/models/fill_concession.py
====================================
Conditional fill-concession model (Stage 2)
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §14).

Trains only on valid fills. Baseline: Huber regression.
Challenger: HistGradientBoostingRegressor (quantile loss for q heads).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from execution.fill_records import fill_fraction as compute_fill_fraction
from prediction.fill_training import blend_with_prior, fallback_level, stage2_fills
from prediction.models.base import RANDOM_STATE, FeatureVectorizer
from prediction.models.fill import fill_fraction_for
from prediction.models.fill_probability import _features_from_record

FILL_CONCESSION_VERSION = "v3.0.0"


@dataclass(frozen=True)
class FillConcessionForecast:
    expected_fill_fraction: float
    fill_q10: float
    fill_q50: float
    fill_q90: float
    conservative_fill_fraction: float
    support_rows: int
    support_sessions: int
    family_support: int
    uncertainty: float
    model_version: str = FILL_CONCESSION_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "expected_fill_fraction": self.expected_fill_fraction,
            "fill_q10": self.fill_q10,
            "fill_q50": self.fill_q50,
            "fill_q90": self.fill_q90,
            "conservative_fill_fraction": self.conservative_fill_fraction,
            "support_rows": self.support_rows,
            "support_sessions": self.support_sessions,
            "family_support": self.family_support,
            "uncertainty": self.uncertainty,
            "model_version": self.model_version,
            "diagnostics": dict(self.diagnostics),
        }


def _ordered_quantiles(q10, q50, q90, expected) -> tuple:
    vals = sorted([float(q10), float(q50), float(q90)])
    return vals[0], vals[1], vals[2], float(expected)


@dataclass
class FillConcessionModel:
    estimator: str = "huber"  # huber | hgb
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    expected_model: object = None
    q_models: dict = field(default_factory=dict)
    fitted: bool = False
    support_rows: int = 0
    support_sessions: int = 0
    family_counts: dict = field(default_factory=dict)
    y_train: Optional[np.ndarray] = None
    model_version: str = FILL_CONCESSION_VERSION

    def _make_expected(self):
        if self.estimator == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                loss="squared_error", max_depth=3, learning_rate=0.05,
                max_leaf_nodes=15, min_samples_leaf=5,
                random_state=RANDOM_STATE)
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(max_iter=500)

    def _make_quantile(self, alpha: float):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha, max_depth=3,
            learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5,
            random_state=RANDOM_STATE)

    def fit(self, records: Sequence) -> "FillConcessionModel":
        fills = stage2_fills(list(records))
        if len(fills) < 4:
            raise ValueError("need at least 4 valid fills for Stage 2")
        rows, ys = [], []
        sessions = set()
        for r in fills:
            rows.append(_features_from_record(r))
            raw, clipped = compute_fill_fraction(
                r.mid_credit_at_submit, r.natural_credit_at_submit,
                r.fill_credit, side=r.side)
            # Train on clipped for stability; retain raw in diagnostics upstream
            ys.append(clipped)
            sessions.add(r.session_date)
            self.family_counts[r.family] = self.family_counts.get(r.family, 0) + 1
        self.vectorizer.fit(rows)
        X = self.vectorizer.transform(rows)
        y = np.asarray(ys, dtype=float)
        self.y_train = y
        self.expected_model = self._make_expected()
        self.expected_model.fit(X, y)
        self.q_models = {}
        for alpha, name in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
            qm = self._make_quantile(alpha)
            qm.fit(X, y)
            self.q_models[name] = qm
        self.support_rows = len(fills)
        self.support_sessions = len(sessions)
        self.fitted = True
        return self

    def predict(
        self,
        features: dict,
        *,
        family: str = "",
        n_legs: int = 2,
        prior_kwargs: Optional[dict] = None,
    ) -> FillConcessionForecast:
        if not self.fitted:
            raise RuntimeError("FillConcessionModel.predict before fit")
        X = self.vectorizer.transform([features])
        expected_emp = float(self.expected_model.predict(X)[0])
        q10 = float(self.q_models["q10"].predict(X)[0])
        q50 = float(self.q_models["q50"].predict(X)[0])
        q90 = float(self.q_models["q90"].predict(X)[0])
        # Retain raw extremes diagnostically before operational clip
        raw_diag = {
            "expected_raw": expected_emp, "q10_raw": q10,
            "q50_raw": q50, "q90_raw": q90,
        }
        q10, q50, q90, expected_emp = _ordered_quantiles(q10, q50, q90, expected_emp)
        prior_kw = dict(prior_kwargs or {})
        prior, prior_diag = fill_fraction_for(family, n_legs=n_legs, **prior_kw)
        prior = float(prior)
        fam_sup = int(self.family_counts.get(family, 0))
        blended, emp_w = blend_with_prior(
            expected_emp, prior, self.support_rows)
        q10 = float(np.clip(q10, 0.0, 1.5))
        q50 = float(np.clip(q50, 0.0, 1.5))
        q90 = float(np.clip(q90, 0.0, 1.5))
        blended = float(np.clip(blended, 0.0, 1.0))
        conservative = max(q90, blended)
        conservative_op = float(np.clip(conservative, 0.0, 1.0))
        level = fallback_level(
            family_support=fam_sup, global_support=self.support_rows)
        unc = float(np.clip(1.0 - self.support_rows / 100.0, 0.0, 1.0))
        return FillConcessionForecast(
            expected_fill_fraction=blended,
            fill_q10=float(np.clip(q10, 0.0, 1.0)),
            fill_q50=float(np.clip(q50, 0.0, 1.0)),
            fill_q90=float(np.clip(q90, 0.0, 1.0)),
            conservative_fill_fraction=conservative_op,
            support_rows=self.support_rows,
            support_sessions=self.support_sessions,
            family_support=fam_sup,
            uncertainty=unc,
            model_version=self.model_version,
            diagnostics={
                "empirical_weight": emp_w,
                "fallback_level": level,
                "prior": prior,
                "prior_diagnostics": prior_diag,
                "raw": raw_diag,
                "conservative_unclipped": conservative,
                "estimator": self.estimator,
            },
        )
