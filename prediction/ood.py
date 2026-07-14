"""
prediction/ood.py
=================
Out-of-distribution detection for Prediction Engine V3 Part 1 §7.4.

Two-layer, dependency-light detector:
  1. Robust standardized feature-range checks.
  2. Nearest-neighbor distance in a curated feature space, calibrated to a
     training percentile.

Scores: 0 = in support, 1 = extreme OOD.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Recommended OOD features (V3 §7.4) — used when present in the row.
DEFAULT_OOD_FEATURES = (
    "dist_vwap",
    "distance_to_vwap",
    "dist_gamma_flip",
    "distance_to_gamma_flip",
    "dist_call_wall",
    "dist_put_wall",
    "gex_rank",
    "gex_pct_rank",
    "gex_disagreement",
    "realized_vol",
    "expected_realized_move",
    "implied_remaining_move",
    "adx",
    "cvd_slope",
    "breadth_alignment",
    "minutes_to_close",
    "vvix_state",
    "data_age",
)

# Explicit exclusions
_EXCLUDED_PREFIXES = ("snapshot_id", "candidate_id", "ts", "session",
                      "label_", "y_", "human_", "policy_", "gate_")


@dataclass(frozen=True)
class OODResult:
    score: float
    percentile: float
    nearest_distance: float | None
    support_count: int
    state_bucket: str
    reasons: tuple[str, ...] = ()


@dataclass
class OODDetectorConfig:
    features: tuple[str, ...] = DEFAULT_OOD_FEATURES
    robust_z_clip: float = 6.0
    range_weight: float = 0.35
    nn_weight: float = 0.65
    k_neighbors: int = 5
    # Percentile interpretation defaults (configurable)
    normal_max: float = 0.80
    reduced_max: float = 0.95
    high_max: float = 0.99
    random_state: int = 42


def _as_float(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _select_feature_names(rows: Sequence[dict], cfg: OODDetectorConfig) -> list[str]:
    present = set()
    for r in rows:
        present.update(r.keys())
    names = [f for f in cfg.features if f in present]
    if names:
        return names
    # Fallback: numeric keys excluding IDs/labels
    numeric = []
    for k in sorted(present):
        if any(k.startswith(p) or k == p for p in _EXCLUDED_PREFIXES):
            continue
        vals = [_as_float(r.get(k)) for r in rows[:50]]
        if any(np.isfinite(v) for v in vals):
            numeric.append(k)
    return numeric[:24]


def _matrix(rows: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    X = np.empty((len(rows), len(names)), dtype=float)
    for i, r in enumerate(rows):
        for j, n in enumerate(names):
            X[i, j] = _as_float(r.get(n))
    return X


@dataclass
class OODDetector:
    """Fit on training feature rows; score new observations."""
    config: OODDetectorConfig = field(default_factory=OODDetectorConfig)
    feature_names: list[str] = field(default_factory=list)
    medians: np.ndarray = field(default_factory=lambda: np.array([]))
    mads: np.ndarray = field(default_factory=lambda: np.array([]))
    train_X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    train_nn_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted: bool = False

    def fit(self, rows: Sequence[dict]) -> "OODDetector":
        rows = list(rows)
        self.feature_names = _select_feature_names(rows, self.config)
        if not self.feature_names or not rows:
            self.fitted = True
            return self
        X = _matrix(rows, self.feature_names)
        self.medians = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.medians), axis=0)
        self.mads = np.where(mad > 1e-12, mad, 1.0)
        # Impute for NN
        X_imp = np.where(np.isfinite(X), X, self.medians)
        self.train_X = self._robust_scale(X_imp)
        # Leave-one-out-ish NN distance distribution on training set
        dists = []
        k = min(self.config.k_neighbors, max(1, len(self.train_X) - 1))
        for i in range(len(self.train_X)):
            d = np.linalg.norm(self.train_X - self.train_X[i], axis=1)
            d[i] = np.inf
            nn = np.partition(d, k - 1)[:k]
            dists.append(float(np.mean(nn)))
        self.train_nn_distances = np.asarray(dists, dtype=float)
        self.fitted = True
        return self

    def _robust_scale(self, X: np.ndarray) -> np.ndarray:
        z = 0.6745 * (X - self.medians) / self.mads
        return np.clip(z, -self.config.robust_z_clip, self.config.robust_z_clip)

    def score_one(self, row: dict) -> OODResult:
        if not self.fitted:
            raise RuntimeError("OODDetector used before fit")
        if not self.feature_names or len(self.train_X) == 0:
            return OODResult(
                score=0.5, percentile=0.5, nearest_distance=None,
                support_count=0, state_bucket="unknown",
                reasons=("no_training_support",))

        x = np.array([_as_float(row.get(n)) for n in self.feature_names],
                     dtype=float)
        reasons: list[str] = []
        # Layer 1: robust range / missingness
        missing = ~np.isfinite(x)
        if missing.any():
            reasons.append(f"missing_features:{int(missing.sum())}")
        x_imp = np.where(np.isfinite(x), x, self.medians)
        z = self._robust_scale(x_imp.reshape(1, -1))[0]
        range_frac = float(np.mean(np.abs(z) >= self.config.robust_z_clip * 0.85))
        if range_frac > 0:
            reasons.append(f"range_violation_frac={range_frac:.2f}")

        # Layer 2: NN distance
        d = np.linalg.norm(self.train_X - z, axis=1)
        k = min(self.config.k_neighbors, len(d))
        nn = np.partition(d, k - 1)[:k]
        nearest = float(np.min(d))
        mean_nn = float(np.mean(nn))
        # Percentile vs training NN distances
        pct = float(np.mean(self.train_nn_distances <= mean_nn))
        pct = float(np.clip(pct, 0.0, 1.0))

        # Composite OOD score
        range_u = float(np.clip(range_frac, 0.0, 1.0))
        nn_u = pct
        score = float(np.clip(
            self.config.range_weight * range_u
            + self.config.nn_weight * nn_u, 0.0, 1.0))

        if pct <= self.config.normal_max:
            bucket = "normal_support"
        elif pct <= self.config.reduced_max:
            bucket = "reduced_support"
        elif pct <= self.config.high_max:
            bucket = "high_ood"
        else:
            bucket = "extreme_ood"
            reasons.append("extreme_ood")

        return OODResult(
            score=score,
            percentile=pct,
            nearest_distance=nearest,
            support_count=int(len(self.train_X)),
            state_bucket=bucket,
            reasons=tuple(reasons),
        )

    def score_many(self, rows: Sequence[dict]) -> list[OODResult]:
        return [self.score_one(r) for r in rows]
