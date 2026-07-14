"""
prediction/models/fill_probability.py
=====================================
Empirical P(fill within horizon) models
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §13).

Baseline: calibrated logistic regression.
Challenger: HistGradientBoostingClassifier.
Horizon probabilities are monotonically rearranged when needed.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import RANDOM_STATE, FeatureVectorizer, clip_probability

FILL_PROBABILITY_VERSION = "v3.0.0"
DEFAULT_HORIZONS = (15, 30, 60)


@dataclass(frozen=True)
class FillProbabilityForecast:
    p_fill_15s: float
    p_fill_30s: float
    p_fill_60s: float
    p_fill_before_cancel: float
    calibration_support: int
    family_support: int
    uncertainty: float
    model_version: str = FILL_PROBABILITY_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "p_fill_15s": self.p_fill_15s,
            "p_fill_30s": self.p_fill_30s,
            "p_fill_60s": self.p_fill_60s,
            "p_fill_before_cancel": self.p_fill_before_cancel,
            "calibration_support": self.calibration_support,
            "family_support": self.family_support,
            "uncertainty": self.uncertainty,
            "model_version": self.model_version,
            "diagnostics": dict(self.diagnostics),
        }


def enforce_horizon_order(probs: Sequence[float]) -> list:
    """Monotone non-decreasing rearrangement across horizons."""
    out = [clip_probability(float(p)) for p in probs]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def fill_horizon_labels(
    records: Sequence,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict:
    """
    Build y vectors per horizon. Unfilled/cancelled attempts are valid
    negatives (or censored if no resolved timing — still labeled 0).
    """
    ys = {f"y_{h}": [] for h in horizons}
    ys["y_before_cancel"] = []
    for rec in records:
        if isinstance(rec, dict):
            filled = bool(rec.get("filled"))
            sec = rec.get("seconds_to_first_fill")
            cancelled = bool(rec.get("cancelled") or rec.get("expired_unfilled"))
        else:
            filled = bool(getattr(rec, "filled", False))
            sec = getattr(rec, "seconds_to_first_fill", None)
            cancelled = bool(getattr(rec, "cancelled", False)
                             or getattr(rec, "expired_unfilled", False))
        for h in horizons:
            ys[f"y_{h}"].append(int(
                filled and sec is not None and float(sec) <= float(h)))
        ys["y_before_cancel"].append(int(filled and not cancelled))
    return {k: np.asarray(v, dtype=int) for k, v in ys.items()}


def fill_features_from_attempt(rec) -> dict:
    """
    Canonical fill-attempt feature builder (train AND live serve).

    Feature names and units must remain identical across training
    (`FillProbabilityModel.fit` / `FillConcessionModel.fit`) and live
    inference. Do not invent alternate live schemas.
    """
    if isinstance(rec, dict):
        d = rec
    else:
        d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec.__dict__)
    mid = float(d.get("mid_credit_at_submit") or 0.0)
    nat = float(d.get("natural_credit_at_submit") or 0.0)
    lim = float(d.get("limit_credit") if d.get("limit_credit") is not None else mid)
    return {
        "n_legs": float(d.get("n_legs") or 0),
        "is_credit": 1.0 if (d.get("side") or "credit").lower()
        in ("credit", "sell") else 0.0,
        "dist_limit_mid": abs(lim - mid),
        "dist_limit_natural": abs(lim - nat),
        "relative_spread": float(d.get("relative_spread") or 0.0),
        "absolute_spread": float(d.get("absolute_spread") or 0.0),
        "option_price_scale": float(d.get("option_price_scale") or 0.0),
        "quote_age_seconds": float(d.get("quote_age_seconds") or 0.0),
        "minutes_to_close": float(d.get("minutes_to_close") or 0.0),
        "realized_volatility": d.get("realized_volatility"),
        "implied_remaining_move": d.get("implied_remaining_move"),
        "data_quality": d.get("data_quality"),
        "replacement_count": float(d.get("replacement_count") or 0),
        "requested_quantity": float(d.get("requested_quantity") or 0),
        "family_code": float(
            sum(ord(c) for c in str(d.get("family") or "")) % 97),
    }


# Backward-compatible alias used by existing training code.
_features_from_record = fill_features_from_attempt


@dataclass
class FillProbabilityModel:
    horizons: tuple = DEFAULT_HORIZONS
    estimator: str = "logistic"  # logistic | hgb
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    models: dict = field(default_factory=dict)
    fitted: bool = False
    calibration_support: int = 0
    family_counts: dict = field(default_factory=dict)
    model_version: str = FILL_PROBABILITY_VERSION

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

    def fit(self, records: Sequence) -> "FillProbabilityModel":
        if len(records) < 4:
            raise ValueError("need at least 4 fill attempts for Stage 1")
        rows = [_features_from_record(r) for r in records]
        self.vectorizer.fit(rows)
        X = self.vectorizer.transform(rows)
        labels = fill_horizon_labels(records, self.horizons)
        self.models = {}
        for key, y in labels.items():
            est = self._make()
            if y.min() == y.max():
                # Constant label — store prior
                self.models[key] = ("constant", float(y.mean()))
            else:
                est.fit(X, y)
                self.models[key] = ("est", est)
        self.calibration_support = len(records)
        self.family_counts = {}
        for r in records:
            fam = (r.get("family") if isinstance(r, dict)
                   else getattr(r, "family", "unknown"))
            self.family_counts[fam] = self.family_counts.get(fam, 0) + 1
        self.fitted = True
        return self

    def _predict_one(self, key: str, X: np.ndarray) -> float:
        kind, obj = self.models[key]
        if kind == "constant":
            return clip_probability(obj)
        if hasattr(obj, "predict_proba"):
            return clip_probability(float(obj.predict_proba(X)[0, 1]))
        return clip_probability(0.5)

    def predict(
        self,
        features: dict,
        *,
        family: Optional[str] = None,
        cancel_horizon_s: float = 120.0,
    ) -> FillProbabilityForecast:
        if not self.fitted:
            raise RuntimeError("FillProbabilityModel.predict before fit")
        X = self.vectorizer.transform([features])
        raw = []
        for h in self.horizons:
            raw.append(self._predict_one(f"y_{h}", X))
        p_cancel = self._predict_one("y_before_cancel", X)
        ordered = enforce_horizon_order(raw + [max(p_cancel, raw[-1])])
        p15, p30, p60, p_bc = ordered[0], ordered[1], ordered[2], ordered[3]
        fam_sup = int(self.family_counts.get(family or "", 0))
        # Uncertainty high when support is thin
        unc = float(np.clip(1.0 - self.calibration_support / 200.0, 0.0, 1.0))
        return FillProbabilityForecast(
            p_fill_15s=p15,
            p_fill_30s=p30,
            p_fill_60s=p60,
            p_fill_before_cancel=p_bc,
            calibration_support=self.calibration_support,
            family_support=fam_sup,
            uncertainty=unc,
            model_version=self.model_version,
            diagnostics={"estimator": self.estimator},
        )
