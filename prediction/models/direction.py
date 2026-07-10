"""
prediction/models/direction.py
==============================
Calibrated direction-probability models (docs/PREDICTION_ENGINE_V2_HANDOFF.md
§11.1): one binary model per horizon (up_5m/15m/30m/60m/close).

Baseline estimator: elastic-net logistic regression (saga), hyperparameters
selected on an inner, session-grouped, embargoed calibration slice of the
TRAINING data. Challenger: HistGradientBoostingClassifier behind the same
interface. Probability calibration (sigmoid by default, isotonic gated —
prediction/calibration.py) is fitted on the inner slice's out-of-sample
scores, never on test data.

Also provides the REQUIRED naive baselines (§11.1) every direction model
must beat before anyone takes it seriously: base rate, previous-return sign,
the legacy 0-100 direction composite, and a random/climatology reference.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (IdentityCalibrator, SigmoidCalibrator,
                                    calibration_slope_intercept,
                                    fit_calibrator, reliability_bins,
                                    select_calibrator)
from prediction.models.base import (RANDOM_STATE, FeatureVectorizer,
                                    brier_score, brier_skill,
                                    clip_probability, log_loss_score)

DIRECTION_HORIZONS = ("5m", "15m", "30m", "60m", "close")


@dataclass
class DirectionModelConfig:
    horizon: str = "30m"
    estimator: str = "elasticnet"            # "elasticnet" | "hgb" (challenger)
    # elastic-net grids (spec §11.1)
    c_grid: tuple = (0.01, 0.05, 0.1, 0.5, 1.0)
    l1_ratio_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    class_weight_options: tuple = (None, "balanced")
    max_iter: int = 2000
    # challenger grids (spec §11.1) — subsettable, iterated deterministically
    hgb_learning_rate_grid: tuple = (0.02, 0.05, 0.1)
    hgb_max_leaf_nodes_grid: tuple = (7, 15, 31)
    hgb_max_depth_grid: tuple = (2, 3, None)
    hgb_min_samples_leaf_grid: tuple = (50, 100, 250)
    hgb_l2_grid: tuple = (0.0, 0.1, 1.0, 10.0)
    # inner train/calibration split (fraction of TRAIN sessions, embargoed)
    calibration_frac: float = 0.25
    embargo_sessions: int = 1
    calibration: str = "auto"                # "auto" | "sigmoid" | "isotonic" | "identity"
    decision_threshold: float = 0.58         # legacy-compatible actionable cut


def _make_estimator(cfg: DirectionModelConfig, params: dict):
    if cfg.estimator == "elasticnet":
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        kw = dict(solver="saga", C=params["C"], l1_ratio=params["l1_ratio"],
                  class_weight=params["class_weight"],
                  max_iter=cfg.max_iter, random_state=RANDOM_STATE)
        # sklearn >= 1.8 selects elastic-net from l1_ratio alone and
        # deprecates the penalty kwarg; older versions require it.
        ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if ver < (1, 8):
            kw["penalty"] = "elasticnet"
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(**kw))])
    if cfg.estimator == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=RANDOM_STATE)
    raise ValueError(f"unknown estimator {cfg.estimator!r}")


def _param_grid(cfg: DirectionModelConfig) -> list:
    if cfg.estimator == "elasticnet":
        return [{"C": c, "l1_ratio": l1, "class_weight": cw}
                for c in cfg.c_grid
                for l1 in cfg.l1_ratio_grid
                for cw in cfg.class_weight_options]
    return [{"learning_rate": lr, "max_leaf_nodes": ln, "max_depth": md,
             "min_samples_leaf": ms, "l2_regularization": l2}
            for lr in cfg.hgb_learning_rate_grid
            for ln in cfg.hgb_max_leaf_nodes_grid
            for md in cfg.hgb_max_depth_grid
            for ms in cfg.hgb_min_samples_leaf_grid
            for l2 in cfg.hgb_l2_grid]


def split_train_calibration(sessions: Sequence[str], calibration_frac: float,
                            embargo_sessions: int) -> tuple:
    """
    Session-grouped inner split of TRAIN data: the LAST calibration_frac of
    unique sessions become the calibration slice, separated from the fit
    slice by `embargo_sessions` whole dropped sessions. Never splits a
    session; falls back to (all, none) when there are too few sessions.
    """
    uniq = sorted(set(sessions))
    n_cal = max(1, int(round(len(uniq) * calibration_frac)))
    n_fit = len(uniq) - n_cal - embargo_sessions
    if n_fit < 1:
        return uniq, []
    return uniq[:n_fit], uniq[n_fit + embargo_sessions:]


@dataclass
class DirectionModel:
    """One calibrated binary direction model for one horizon."""
    config: DirectionModelConfig = field(default_factory=DirectionModelConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = None
    calibrator: object = field(default_factory=IdentityCalibrator)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False

    # -- training ---------------------------------------------------------------
    def fit(self, rows: Sequence[dict], y: Sequence[int],
            sessions: Sequence[str]) -> "DirectionModel":
        y = np.asarray(y, dtype=int)
        sessions = list(sessions)
        fit_sessions, cal_sessions = split_train_calibration(
            sessions, self.config.calibration_frac,
            self.config.embargo_sessions)
        fit_mask = np.array([s in set(fit_sessions) for s in sessions])
        cal_mask = np.array([s in set(cal_sessions) for s in sessions])

        fit_rows = [r for r, m in zip(rows, fit_mask) if m]
        X_fit = self.vectorizer.fit_transform(fit_rows)
        y_fit = y[fit_mask]

        grid = _param_grid(self.config)
        best_params, best_est, best_loss = grid[0], None, math.inf
        if cal_mask.any() and len(np.unique(y_fit)) >= 2:
            cal_rows = [r for r, m in zip(rows, cal_mask) if m]
            X_cal = self.vectorizer.transform(cal_rows)
            y_cal = y[cal_mask]
            for params in grid:
                est = _make_estimator(self.config, params)
                est.fit(X_fit, y_fit)
                loss = log_loss_score(y_cal, self._raw_from(est, X_cal))
                if loss < best_loss:
                    best_params, best_est, best_loss = params, est, loss
            p_cal_raw = self._raw_from(best_est, X_cal)
            if self.config.calibration == "auto":
                self.calibrator, cal_diag = select_calibrator(
                    p_cal_raw, y_cal, n_sessions=len(cal_sessions))
            else:
                self.calibrator = fit_calibrator(
                    p_cal_raw, y_cal, self.config.calibration)
                cal_diag = self.calibrator.to_dict()
            p_cal = self.calibrator.transform(p_cal_raw)
            cal_metrics = {
                "n": int(len(y_cal)),
                "brier_raw": brier_score(y_cal, p_cal_raw),
                "brier_calibrated": brier_score(y_cal, p_cal),
                "brier_skill": brier_skill(y_cal, p_cal),
                "log_loss": log_loss_score(y_cal, p_cal),
                **calibration_slope_intercept(p_cal, y_cal),
                "reliability": reliability_bins(p_cal, y_cal, n_bins=5),
                "calibration_diag": cal_diag,
            }
        else:
            # too few sessions (or one-class fit data): fit everything, no
            # calibration claim — identity mapping, flagged in metadata
            X_fit = self.vectorizer.fit_transform(list(rows))
            y_fit = y
            best_est = _make_estimator(self.config, best_params)
            if len(np.unique(y_fit)) < 2:
                best_est = None                        # degenerate: base rate
            else:
                best_est.fit(X_fit, y_fit)
            self.calibrator = IdentityCalibrator()
            cal_metrics = {"note": "no calibration slice; identity calibrator"}
            fit_sessions, cal_sessions = sorted(set(sessions)), []

        self.estimator = best_est
        self._base_rate = float(np.mean(y)) if len(y) else 0.5
        self.metadata = {
            "target": f"up_{self.config.horizon}",
            "horizon": self.config.horizon,
            "estimator": self.config.estimator,
            "best_params": {k: (v if v is None or isinstance(v, (int, float))
                                else str(v))
                            for k, v in best_params.items()},
            "train_sessions": sorted(set(sessions)),
            "fit_sessions": sorted(set(fit_sessions)),
            "calibration_sessions": sorted(set(cal_sessions)),
            "n_train_rows": int(len(y)),
            "base_rate": self._base_rate,
            "calibration_metrics": cal_metrics,
            "decision_threshold": self.config.decision_threshold,
        }
        # crude v1 model-uncertainty scalar: 1 - clipped calibration Brier
        # skill (uncertain until proven skilled; see handoff §6.2/§18.13)
        skill = (cal_metrics.get("brier_skill")
                 if isinstance(cal_metrics, dict) else None)
        self.metadata["uncertainty"] = float(
            np.clip(1.0 - max(skill or 0.0, 0.0), 0.0, 1.0))
        self.fitted = True
        return self

    # -- inference ----------------------------------------------------------------
    @staticmethod
    def _raw_from(estimator, X: np.ndarray) -> np.ndarray:
        return clip_probability(estimator.predict_proba(X)[:, 1])

    def predict_raw(self, rows: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("DirectionModel used before fit")
        if self.estimator is None:                     # degenerate training set
            return np.full(len(rows), self._base_rate)
        return self._raw_from(self.estimator, self.vectorizer.transform(rows))

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        """Calibrated P(up) in [0, 1] — the PredictionBundle input."""
        return clip_probability(self.calibrator.transform(
            self.predict_raw(rows)))

    def predict_label(self, rows: Sequence[dict]) -> np.ndarray:
        """+1 / 0 / -1 at the symmetric decision threshold (default 58/42)."""
        p = self.predict_proba(rows)
        thr = self.config.decision_threshold
        return np.where(p >= thr, 1, np.where(p <= 1.0 - thr, -1, 0))


# --------------------------------------------------------------------------- #
# Required naive baselines (§11.1)                                             #
# --------------------------------------------------------------------------- #
def baseline_base_rate(y_train, n: int) -> np.ndarray:
    """Climatology: the training base rate as a constant probability."""
    base = float(np.mean(np.asarray(y_train, dtype=float))) if len(y_train) else 0.5
    return np.full(n, base)

def baseline_prev_sign(prev_returns, y_train) -> np.ndarray:
    """Previous-return sign; missing/zero previous returns fall back to the
    training base rate."""
    base = float(np.mean(np.asarray(y_train, dtype=float))) if len(y_train) else 0.5
    out = np.full(len(prev_returns), base)
    for i, r in enumerate(prev_returns):
        if isinstance(r, (int, float)) and math.isfinite(r) and r != 0.0:
            out[i] = 1.0 if r > 0 else 0.0
    return clip_probability(out)

def baseline_legacy_composite(bias_values, y_train) -> np.ndarray:
    """The existing 0-100 direction composite read as P(up) = value/100;
    missing composites fall back to the training base rate."""
    base = float(np.mean(np.asarray(y_train, dtype=float))) if len(y_train) else 0.5
    out = np.full(len(bias_values), base)
    for i, v in enumerate(bias_values):
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[i] = float(v) / 100.0
    return clip_probability(out)

def baseline_random(n: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Seeded uniform-random probabilities — the floor any model must clear."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)


def evaluate_probabilities(y_true, p) -> dict:
    """Standard direction-metric panel for one probability series."""
    return {"n": int(len(y_true)),
            "brier": brier_score(y_true, p),
            "brier_skill": brier_skill(y_true, p),
            "log_loss": log_loss_score(y_true, p),
            "hit_rate_at_half": float(np.mean(
                (np.asarray(p) >= 0.5) == (np.asarray(y_true) == 1)))}
