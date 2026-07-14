"""
prediction/return_distribution.py
=================================
Expanded return-distribution forecasts (V3 Part 2 §16 / §18, PR 11).

Quantile grid 0.05…0.95 with monotone rearrangement. Moments derived from
the quantile curve are diagnostic only.

Conformal intervals are attached by PR 12 (`prediction.conformal`); this
module leaves them empty by default.

Research / shadow only.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    interval_coverage,
    pinball_loss,
    rearrange_quantile_grid,
)

RETURN_DIST_VERSION = "v3.0.0-return"
QUANTILES = (
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)
RETURN_DIST_HORIZONS = ("15m", "30m", "60m", "close")


@dataclass(frozen=True)
class ReturnDistribution:
    horizon: str
    quantiles: dict[float, float]
    expected_return: Optional[float]
    variance: Optional[float]
    conformal_intervals: dict[str, tuple[float, float]]
    conformal_support_rows: int
    conformal_support_sessions: int
    uncertainty: float
    ood_score: Optional[float]
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        qs = sorted((float(k), float(v)) for k, v in self.quantiles.items())
        for i in range(1, len(qs)):
            if qs[i][1] + 1e-12 < qs[i - 1][1]:
                raise ValueError(
                    f"quantiles not ordered at {qs[i-1]} -> {qs[i]}")
        for name, interval in self.conformal_intervals.items():
            lo, hi = interval
            if lo > hi:
                raise ValueError(
                    f"conformal interval {name} has lower > upper: {interval}")

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON-friendly quantile keys
        d["quantiles"] = {str(k): v for k, v in self.quantiles.items()}
        d["conformal_intervals"] = {
            k: list(v) for k, v in self.conformal_intervals.items()
        }
        return d


def moments_from_quantiles(
    quantiles: dict[float, float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Diagnostic mean/variance via trapezoidal interpolation of the quantile
    curve. Not exact unless the reconstruction method is documented.
    """
    if not quantiles:
        return None, None
    qs = sorted((float(q), float(v)) for q, v in quantiles.items())
    # Approximate E[X] ≈ ∫ Q(u) du over available grid
    mean = 0.0
    for i in range(1, len(qs)):
        q0, v0 = qs[i - 1]
        q1, v1 = qs[i]
        mean += 0.5 * (v0 + v1) * (q1 - q0)
    # Pad tails crudely if grid doesn't span [0,1]
    if qs[0][0] > 0:
        mean += qs[0][1] * qs[0][0]
    if qs[-1][0] < 1:
        mean += qs[-1][1] * (1.0 - qs[-1][0])
    # Second moment similarly
    m2 = 0.0
    for i in range(1, len(qs)):
        q0, v0 = qs[i - 1]
        q1, v1 = qs[i]
        m2 += 0.5 * (v0 * v0 + v1 * v1) * (q1 - q0)
    if qs[0][0] > 0:
        m2 += (qs[0][1] ** 2) * qs[0][0]
    if qs[-1][0] < 1:
        m2 += (qs[-1][1] ** 2) * (1.0 - qs[-1][0])
    var = max(0.0, m2 - mean * mean)
    return float(mean), float(var)


@dataclass
class ExpandedReturnQuantileConfig:
    horizon: str = "30m"
    quantiles: tuple = QUANTILES
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    max_depth: Optional[int] = 3
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    max_iter: int = 150


@dataclass
class ExpandedReturnQuantileModel:
    """Expanded quantile-grid return model (shadow only)."""

    config: ExpandedReturnQuantileConfig = field(
        default_factory=ExpandedReturnQuantileConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimators: dict = field(default_factory=dict)
    model_version: str = RETURN_DIST_VERSION
    fitted: bool = False
    diagnostics: dict = field(default_factory=dict)

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
            random_state=RANDOM_STATE,
        )

    def fit(
        self,
        rows: Sequence[dict],
        y: Sequence[float],
        sessions: Optional[Sequence[str]] = None,
    ) -> "ExpandedReturnQuantileModel":
        y_arr = np.asarray(y, dtype=float)
        X = self.vectorizer.fit_transform(list(rows))
        self.estimators = {}
        for q in self.config.quantiles:
            est = self._make_estimator(float(q))
            est.fit(X, y_arr)
            self.estimators[float(q)] = est
        self.diagnostics = {
            "horizon": self.config.horizon,
            "quantiles": list(self.config.quantiles),
            "n_train_rows": int(len(y_arr)),
            "train_sessions": sorted(set(sessions)) if sessions else None,
        }
        self.fitted = True
        return self

    def predict_quantiles(self, rows: Sequence[dict]) -> dict[float, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("ExpandedReturnQuantileModel used before fit")
        X = self.vectorizer.transform(list(rows))
        raw = {q: est.predict(X) for q, est in self.estimators.items()}
        return rearrange_quantile_grid(raw)

    def predict_distribution(
        self,
        row: dict,
        *,
        uncertainty: float = 0.0,
        ood_score: Optional[float] = None,
    ) -> ReturnDistribution:
        qmap = self.predict_quantiles([row])
        quantiles = {q: float(arr[0]) for q, arr in qmap.items()}
        # Ensure ordered (rearrangement already applied)
        mean, var = moments_from_quantiles(quantiles)
        return ReturnDistribution(
            horizon=self.config.horizon,
            quantiles=quantiles,
            expected_return=mean,
            variance=var,
            conformal_intervals={},
            conformal_support_rows=0,
            conformal_support_sessions=0,
            uncertainty=float(uncertainty),
            ood_score=ood_score,
            model_version=self.model_version,
            diagnostics={"moments_are_diagnostic": True},
        )

    def evaluate(
        self,
        rows: Sequence[dict],
        y: Sequence[float],
    ) -> dict:
        y_arr = np.asarray(y, dtype=float)
        qmap = self.predict_quantiles(rows)
        pinballs = {
            f"pinball_q{int(q * 100):02d}": pinball_loss(y_arr, qmap[q], q)
            for q in sorted(qmap)
        }
        # Coverage for central 80% and 90% if present
        coverages = {}
        if 0.10 in qmap and 0.90 in qmap:
            coverages["coverage_80"] = interval_coverage(
                y_arr, qmap[0.10], qmap[0.90])
        if 0.05 in qmap and 0.95 in qmap:
            coverages["coverage_90"] = interval_coverage(
                y_arr, qmap[0.05], qmap[0.95])
        return {"n": int(len(y_arr)), **pinballs, **coverages}
