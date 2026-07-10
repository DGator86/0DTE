"""
prediction/models/volatility.py
===============================
Realized-move forecast model (docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.3).

Primary target: the remaining-session realized move
(prediction/labels.py `remaining_realized_move` — max simple-return
excursion from the observation to the close). Positive targets are trained
in log space, target = log(realized_measure + epsilon), which keeps the
regressor honest about the heavy right tail.

Outputs per observation:
  * expected realized move (point forecast, >= 0);
  * q10/q90 move range (monotone with the point forecast);
  * a [0, 1] uncertainty from the relative interval width;
  * forecast / implied-remaining-move ratio when the implied feature is
    available (the long-vol edge signal).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import (RANDOM_STATE, FeatureVectorizer,
                                    rearrange_quantiles)


@dataclass
class VolatilityModelConfig:
    target: str = "remaining_realized_move"
    epsilon: float = 1e-6
    quantiles: tuple = (0.1, 0.9)
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    max_depth: Optional[int] = 3
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_iter: int = 200
    # feature carrying the option-implied remaining move (decimal), used for
    # the forecast/implied ratio; None disables the ratio output
    implied_feature: Optional[str] = "implied_remaining_move"


@dataclass
class VolatilityModel:
    config: VolatilityModelConfig = field(default_factory=VolatilityModelConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    point_estimator: object = None
    quantile_estimators: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False

    def _make_estimator(self, quantile: Optional[float]):
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(learning_rate=self.config.learning_rate,
                  max_leaf_nodes=self.config.max_leaf_nodes,
                  max_depth=self.config.max_depth,
                  min_samples_leaf=self.config.min_samples_leaf,
                  l2_regularization=self.config.l2_regularization,
                  max_iter=self.config.max_iter,
                  random_state=RANDOM_STATE)
        if quantile is None:
            return HistGradientBoostingRegressor(loss="squared_error", **kw)
        return HistGradientBoostingRegressor(loss="quantile",
                                             quantile=quantile, **kw)

    def fit(self, rows: Sequence[dict], y: Sequence[float],
            sessions: Optional[Sequence[str]] = None) -> "VolatilityModel":
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("volatility targets must be non-negative")
        y_log = np.log(y + self.config.epsilon)
        X = self.vectorizer.fit_transform(list(rows))
        self.point_estimator = self._make_estimator(None)
        self.point_estimator.fit(X, y_log)
        self.quantile_estimators = {}
        for q in self.config.quantiles:
            est = self._make_estimator(q)
            est.fit(X, y_log)
            self.quantile_estimators[q] = est
        self.metadata = {
            "target": self.config.target,
            "epsilon": self.config.epsilon,
            "n_train_rows": int(len(y)),
            "train_sessions": sorted(set(sessions)) if sessions else None,
            "train_move_median": float(np.median(y)),
        }
        self.fitted = True
        return self

    def _from_log(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(z) - self.config.epsilon, 0.0)

    def predict(self, rows: Sequence[dict]) -> dict:
        """
        {"expected_move", "move_q10", "move_q90", "uncertainty",
         "rv_iv_ratio"} — arrays aligned with rows. rv_iv_ratio entries are
        NaN when the implied feature is missing for that row.
        """
        if not self.fitted:
            raise RuntimeError("VolatilityModel used before fit")
        rows = list(rows)
        X = self.vectorizer.transform(rows)
        point = self._from_log(self.point_estimator.predict(X))
        lo_q, hi_q = min(self.config.quantiles), max(self.config.quantiles)
        q_lo = self._from_log(self.quantile_estimators[lo_q].predict(X))
        q_hi = self._from_log(self.quantile_estimators[hi_q].predict(X))
        # monotone: q10 <= point <= q90 after rearrangement
        q_lo, point, q_hi = rearrange_quantiles(q_lo, point, q_hi)
        uncertainty = np.clip((q_hi - q_lo) / (point + self.config.epsilon)
                              / 4.0, 0.0, 1.0)

        ratio = np.full(len(rows), np.nan)
        feat = self.config.implied_feature
        if feat:
            for i, r in enumerate(rows):
                iv = r.get(feat)
                if isinstance(iv, (int, float)) and iv and np.isfinite(iv):
                    ratio[i] = point[i] / float(iv)
        return {"expected_move": point, "move_q10": q_lo, "move_q90": q_hi,
                "uncertainty": uncertainty, "rv_iv_ratio": ratio}

    def evaluate(self, rows: Sequence[dict], y: Sequence[float]) -> dict:
        y = np.asarray(y, dtype=float)
        p = self.predict(rows)
        err = p["expected_move"] - y
        cover = np.mean((y >= p["move_q10"]) & (y <= p["move_q90"]))
        return {"n": int(len(y)),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "coverage_10_90": float(cover),
                "mean_predicted_move": float(np.mean(p["expected_move"])),
                "mean_realized_move": float(np.mean(y))}
