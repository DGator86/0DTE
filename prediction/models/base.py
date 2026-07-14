"""
prediction/models/base.py
=========================
Shared plumbing for the V2 model suite (docs/PREDICTION_ENGINE_V2_HANDOFF.md
§11): deterministic feature vectorization with EXPLICIT missingness.

The canonical dataset stores raw features as dicts with None for missing
values (spec §8.7: missing must never be silently imputed to a neutral
value without the model knowing). FeatureVectorizer therefore emits, for
every feature, BOTH a value column (median-imputed for estimators that
cannot handle NaN) and a 0/1 missingness column — the model always sees
whether a neutral-looking value was observed or imputed.

Everything here is deterministic: feature order is fixed at fit time,
medians come from training data only, and no randomness is involved.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

RANDOM_STATE = 7        # one seed repo-wide for the sklearn estimators


def _as_float(v) -> float:
    """None / non-numeric / non-finite -> NaN; else float."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    return float("nan")


@dataclass
class FeatureVectorizer:
    """
    dict-rows -> fixed-width numeric matrix [values | missing flags].

    fit() freezes the feature-name order and learns per-feature training
    medians; transform() never adds or reorders columns, so a persisted
    model keeps producing identical vectors (determinism acceptance).
    """
    feature_names: list = field(default_factory=list)
    medians: dict = field(default_factory=dict)
    fitted: bool = False

    def fit(self, rows: Sequence[dict]) -> "FeatureVectorizer":
        names: set = set()
        for r in rows:
            names.update(r.keys())
        self.feature_names = sorted(names)
        cols = {n: [] for n in self.feature_names}
        for r in rows:
            for n in self.feature_names:
                cols[n].append(_as_float(r.get(n)))
        self.medians = {}
        for n in self.feature_names:
            arr = np.asarray(cols[n], dtype=float)
            finite = arr[np.isfinite(arr)]
            self.medians[n] = float(np.median(finite)) if len(finite) else 0.0
        self.fitted = True
        return self

    def transform(self, rows: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureVectorizer.transform before fit")
        n_feat = len(self.feature_names)
        out = np.empty((len(rows), 2 * n_feat), dtype=float)
        for i, r in enumerate(rows):
            for j, name in enumerate(self.feature_names):
                v = _as_float(r.get(name))
                missing = not math.isfinite(v)
                out[i, j] = self.medians[name] if missing else v
                out[i, n_feat + j] = 1.0 if missing else 0.0
        return out

    def fit_transform(self, rows: Sequence[dict]) -> np.ndarray:
        return self.fit(rows).transform(rows)

    @property
    def n_columns(self) -> int:
        return 2 * len(self.feature_names)

    def column_names(self) -> list:
        return ([f"val:{n}" for n in self.feature_names]
                + [f"miss:{n}" for n in self.feature_names])


def clip_probability(p) -> np.ndarray:
    """Hard [0,1] bound on any probability output (contract §6.2)."""
    return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


def rearrange_quantiles(q10, q50, q90) -> tuple:
    """
    Monotone rearrangement (spec §11.2): independently fitted quantile heads
    can cross; sorting per row restores q10 <= q50 <= q90.
    """
    stacked = np.sort(np.vstack([np.asarray(q10, dtype=float),
                                 np.asarray(q50, dtype=float),
                                 np.asarray(q90, dtype=float)]), axis=0)
    return stacked[0], stacked[1], stacked[2]


def rearrange_quantile_grid(
    quantile_preds: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    """
    Sort predicted quantile values per row so the quantile axis is monotone
    (V3 Part 2 §16.3 — quantile rearrangement).
    """
    qs = sorted(quantile_preds.keys())
    if not qs:
        return {}
    stacked = np.vstack([np.asarray(quantile_preds[q], dtype=float) for q in qs])
    stacked = np.sort(stacked, axis=0)
    return {q: stacked[i] for i, q in enumerate(qs)}


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Mean pinball (quantile) loss — the quantile model's fit metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of outcomes inside [lower, upper] — target 0.80 for q10/q90."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= np.asarray(lower, dtype=float))
                         & (y_true <= np.asarray(upper, dtype=float))))


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def brier_skill(y_true, p) -> Optional[float]:
    """1 - Brier/Brier(base rate); None when the base rate is degenerate."""
    y_true = np.asarray(y_true, dtype=float)
    base = float(np.mean(y_true))
    ref = brier_score(y_true, np.full_like(y_true, base))
    if ref <= 0.0:
        return None
    return 1.0 - brier_score(y_true, p) / ref


def log_loss_score(y_true, p, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
