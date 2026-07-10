"""
prediction/models/return_quantiles.py
=====================================
Forward-return quantile models (docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.2):
q10 / q50 / q90 gradient-boosted quantile regression per horizon
(30m, 60m, close), trained on the canonical dataset's log-return labels.

Contract requirements enforced here:
  * quantiles are monotonically ordered after prediction — independently
    fitted heads can cross, so every prediction passes through rearrangement;
  * evaluation reports pinball loss per quantile and 10-90 interval
    coverage, sliceable by an arbitrary grouping key (time of day, regime).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import (RANDOM_STATE, FeatureVectorizer,
                                    interval_coverage, pinball_loss,
                                    rearrange_quantiles)

QUANTILE_HORIZONS = ("30m", "60m", "close")
QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class ReturnQuantileConfig:
    horizon: str = "30m"
    quantiles: tuple = QUANTILES
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    max_depth: Optional[int] = 3
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_iter: int = 200


@dataclass
class ReturnQuantileModel:
    """q10/q50/q90 forward-log-return forecasts for one horizon."""
    config: ReturnQuantileConfig = field(default_factory=ReturnQuantileConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimators: dict = field(default_factory=dict)    # quantile -> regressor
    metadata: dict = field(default_factory=dict)
    fitted: bool = False

    def _make_estimator(self, quantile: float):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            max_iter=self.config.max_iter,
            random_state=RANDOM_STATE)

    def fit(self, rows: Sequence[dict], y: Sequence[float],
            sessions: Optional[Sequence[str]] = None) -> "ReturnQuantileModel":
        y = np.asarray(y, dtype=float)
        X = self.vectorizer.fit_transform(list(rows))
        self.estimators = {}
        for q in self.config.quantiles:
            est = self._make_estimator(q)
            est.fit(X, y)
            self.estimators[q] = est
        self.metadata = {
            "target": f"fwd_return_{self.config.horizon}",
            "horizon": self.config.horizon,
            "quantiles": list(self.config.quantiles),
            "n_train_rows": int(len(y)),
            "train_sessions": sorted(set(sessions)) if sessions else None,
        }
        self.fitted = True
        return self

    def predict(self, rows: Sequence[dict]) -> dict:
        """{"q10": ..., "q50": ..., "q90": ...} — ALWAYS ordered (§11.2)."""
        if not self.fitted:
            raise RuntimeError("ReturnQuantileModel used before fit")
        X = self.vectorizer.transform(list(rows))
        preds = [self.estimators[q].predict(X) for q in self.config.quantiles]
        q10, q50, q90 = rearrange_quantiles(*preds)
        return {"q10": q10, "q50": q50, "q90": q90}

    def evaluate(self, rows: Sequence[dict], y: Sequence[float],
                 group_by: Optional[Sequence] = None) -> dict:
        """Pinball loss per quantile + 10-90 coverage, optionally per group
        (pass time-of-day buckets or regime labels as group_by)."""
        y = np.asarray(y, dtype=float)
        p = self.predict(rows)
        out = {
            "n": int(len(y)),
            "pinball_q10": pinball_loss(y, p["q10"], 0.1),
            "pinball_q50": pinball_loss(y, p["q50"], 0.5),
            "pinball_q90": pinball_loss(y, p["q90"], 0.9),
            "coverage_10_90": interval_coverage(y, p["q10"], p["q90"]),
            "median_abs_error": float(np.median(np.abs(y - p["q50"]))),
            "bias": float(np.mean(p["q50"] - y)),
        }
        if group_by is not None:
            groups: dict = {}
            gb = list(group_by)
            for g in sorted(set(gb)):
                mask = np.array([x == g for x in gb])
                if mask.sum() >= 3:
                    groups[str(g)] = {
                        "n": int(mask.sum()),
                        "coverage_10_90": interval_coverage(
                            y[mask], p["q10"][mask], p["q90"][mask]),
                        "pinball_q50": pinball_loss(
                            y[mask], p["q50"][mask], 0.5),
                    }
            out["by_group"] = groups
        return out
