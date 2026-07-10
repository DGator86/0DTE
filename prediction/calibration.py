"""
prediction/calibration.py
=========================
Probability calibration for the V2 model suite
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.7).

Default is sigmoid/Platt scaling — a one-dimensional logistic fit on the
model's raw scores. Isotonic regression is available but GATED: it is only
selected when the calibration sample is large enough (samples AND independent
sessions) and inner comparison shows it actually improves Brier score;
otherwise it degenerates into an unstable step function on small samples.

The calibrator is a separate object fitted on training data only — the
training pipeline (prediction/training.py) fits it on a held-in calibration
slice of the TRAIN sessions, never on test or holdout periods. Nothing in
this module ever sees a test label by construction; enforcing the split is
the trainer's job and is covered by tests/test_grouped_training.py.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from prediction.models.base import brier_score, clip_probability

# Gates for isotonic selection (§11.7). Research defaults, configurable.
ISOTONIC_MIN_SAMPLES = 2000
ISOTONIC_MIN_SESSIONS = 40


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class SigmoidCalibrator:
    """Platt scaling: logistic regression on the raw score's log-odds."""
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, p_raw, y) -> "SigmoidCalibrator":
        from sklearn.linear_model import LogisticRegression
        x = _logit(p_raw).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            # degenerate calibration set: identity mapping, never a crash
            self.a, self.b = 1.0, 0.0
            self.fitted = True
            return self
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, p_raw) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator used before fit")
        z = self.a * _logit(p_raw) + self.b
        # numerically stable sigmoid: exp only ever sees non-positive input
        out = np.where(z >= 0,
                       1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
        return clip_probability(out)

    def to_dict(self) -> dict:
        return {"method": "sigmoid", "a": self.a, "b": self.b}


@dataclass
class IsotonicCalibrator:
    """Isotonic regression p_raw -> p_cal; monotone, clipped to [0, 1]."""
    _iso: object = field(default=None, repr=False)
    fitted: bool = False

    def fit(self, p_raw, y) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")
        self._iso.fit(np.asarray(p_raw, dtype=float),
                      np.asarray(y, dtype=float))
        self.fitted = True
        return self

    def transform(self, p_raw) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator used before fit")
        return clip_probability(self._iso.predict(
            np.asarray(p_raw, dtype=float)))

    def to_dict(self) -> dict:
        return {"method": "isotonic"}


@dataclass
class IdentityCalibrator:
    """No-op fallback (still clips to [0, 1])."""
    fitted: bool = True

    def fit(self, p_raw, y) -> "IdentityCalibrator":
        return self

    def transform(self, p_raw) -> np.ndarray:
        return clip_probability(p_raw)

    def to_dict(self) -> dict:
        return {"method": "identity"}


def fit_calibrator(p_raw, y, method: str = "sigmoid"):
    """Fit one named calibrator ('sigmoid' | 'isotonic' | 'identity')."""
    cal = {"sigmoid": SigmoidCalibrator,
           "isotonic": IsotonicCalibrator,
           "identity": IdentityCalibrator}.get(method)
    if cal is None:
        raise ValueError(f"unknown calibration method {method!r}")
    return cal().fit(p_raw, y)


def select_calibrator(p_raw, y, n_sessions: int, *,
                      min_samples: int = ISOTONIC_MIN_SAMPLES,
                      min_sessions: int = ISOTONIC_MIN_SESSIONS):
    """
    Choose sigmoid vs isotonic per §11.7. Sigmoid is the default; isotonic
    is used only when the calibration sample is large (rows AND independent
    sessions) and it beats sigmoid on the same slice's Brier score.
    Returns (calibrator, diagnostics dict).
    """
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    sig = SigmoidCalibrator().fit(p_raw, y)
    diag = {"chosen": "sigmoid", "n": int(len(y)),
            "n_sessions": int(n_sessions),
            "brier_sigmoid": brier_score(y, sig.transform(p_raw)),
            "brier_isotonic": None}
    if len(y) >= min_samples and n_sessions >= min_sessions:
        iso = IsotonicCalibrator().fit(p_raw, y)
        diag["brier_isotonic"] = brier_score(y, iso.transform(p_raw))
        if diag["brier_isotonic"] < diag["brier_sigmoid"]:
            diag["chosen"] = "isotonic"
            return iso, diag
    return sig, diag


def reliability_bins(p, y, n_bins: int = 10) -> list:
    """Reliability table: mean predicted vs realized rate per probability bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                        "mean_predicted": float(p[mask].mean()),
                        "realized_rate": float(y[mask].mean())})
    return out


def calibration_slope_intercept(p, y) -> dict:
    """
    Logistic recalibration slope/intercept (promotion criteria §22.2:
    slope ~1, intercept ~0 for an honest probability).
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None}
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0])}
