"""
prediction/models/direction.py
==============================
Calibrated direction-probability models (docs/PREDICTION_ENGINE_V2_HANDOFF.md
§11.1; V3 Part 1 §5): one binary model per horizon (up_5m/15m/30m/60m/close).

Hyperparameters are selected by inner session-grouped out-of-fold log loss.
The probability calibrator is fitted on cross-fitted raw scores from the
training window only — never on the same slice used for hyperparameter
selection as a joint objective, and never on outer test labels.

Baseline estimator: elastic-net logistic regression (saga). Challenger:
HistGradientBoostingClassifier. Also provides the required naive baselines.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (
    IdentityCalibrator,
    build_calibration_artifact,
    calibration_slope_intercept,
    fit_calibrator,
    reliability_bins,
    select_calibrator,
    slice_calibration_report,
)
from prediction.crossfit import NestedCrossFitConfig, inner_folds_for_train
from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    brier_score,
    brier_skill,
    clip_probability,
    log_loss_score,
)

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
    # retained for backward-compatible metadata / small-data fallbacks
    calibration_frac: float = 0.25
    embargo_sessions: int = 1
    calibration: str = "auto"                # "auto" | "sigmoid" | "isotonic" | "identity"
    decision_threshold: float = 0.58
    # V3 nested cross-fit knobs (adapted downward for small samples)
    inner_folds: int = 3
    min_train_sessions: int = 8
    min_validation_sessions: int = 3
    random_state: int = RANDOM_STATE


def _make_estimator(cfg: DirectionModelConfig, params: dict):
    if cfg.estimator == "elasticnet":
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        kw = dict(solver="saga", C=params["C"], l1_ratio=params["l1_ratio"],
                  class_weight=params["class_weight"],
                  max_iter=cfg.max_iter, random_state=RANDOM_STATE)
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
    Session-grouped inner split of TRAIN data (legacy helper retained for
    tests and callers). Prefer nested cross-fitting via DirectionModel.fit.
    """
    uniq = sorted(set(sessions))
    n_cal = max(1, int(round(len(uniq) * calibration_frac)))
    n_fit = len(uniq) - n_cal - embargo_sessions
    if n_fit < 1:
        return uniq, []
    return uniq[:n_fit], uniq[n_fit + embargo_sessions:]


def _adaptive_crossfit_cfg(sessions: Sequence[str],
                           cfg: DirectionModelConfig) -> NestedCrossFitConfig:
    n = len(set(sessions))
    min_train = min(cfg.min_train_sessions, max(3, n // 3))
    min_val = min(cfg.min_validation_sessions, max(2, n // 6))
    return NestedCrossFitConfig(
        outer_folds=max(2, min(4, max(1, n // (min_train + min_val + 1)))),
        inner_folds=min(cfg.inner_folds, max(2, n // (min_train + min_val))),
        embargo_sessions=cfg.embargo_sessions,
        min_train_sessions=min_train,
        min_validation_sessions=min_val,
        retain_fold_models=False,
        random_state=cfg.random_state,
    )


def _select_params_oof(
    rows: Sequence[dict],
    y: np.ndarray,
    sessions: Sequence[str],
    cfg: DirectionModelConfig,
    xfit_cfg: NestedCrossFitConfig,
) -> tuple[dict, dict]:
    """Select hyperparameters by mean inner-OOF log loss (FeatureVectorizer)."""
    grid = _param_grid(cfg)
    inner = inner_folds_for_train(sorted(set(sessions)), xfit_cfg)
    if not inner:
        return dict(grid[0]), {"note": "insufficient_sessions_for_inner_cv",
                               "selected": dict(grid[0])}

    scores: list[tuple[float, dict, list]] = []
    for params in grid:
        fold_losses = []
        for inf in inner:
            tr_set = set(inf["train_sessions"])
            va_set = set(inf["test_sessions"])
            tr_idx = [i for i, s in enumerate(sessions) if s in tr_set]
            va_idx = [i for i, s in enumerate(sessions) if s in va_set]
            if not tr_idx or not va_idx:
                continue
            y_tr = y[tr_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            vec = FeatureVectorizer().fit([rows[i] for i in tr_idx])
            est = _make_estimator(cfg, params)
            est.fit(vec.transform([rows[i] for i in tr_idx]), y_tr)
            p = clip_probability(
                est.predict_proba(vec.transform([rows[i] for i in va_idx]))[:, 1])
            fold_losses.append(log_loss_score(y[va_idx], p))
        if fold_losses:
            scores.append((float(np.mean(fold_losses)), dict(params), fold_losses))

    if not scores:
        return dict(grid[0]), {"note": "inner_cv_produced_no_scores",
                               "selected": dict(grid[0])}
    scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
    best_loss, best_params, best_folds = scores[0]
    return best_params, {
        "selection_metric": "log_loss",
        "best_log_loss": best_loss,
        "fold_log_losses": best_folds,
        "n_param_candidates_scored": len(scores),
        "selected": best_params,
    }


def _oof_raw_probabilities(
    rows: Sequence[dict],
    y: np.ndarray,
    sessions: Sequence[str],
    params: dict,
    cfg: DirectionModelConfig,
    xfit_cfg: NestedCrossFitConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Cross-fitted raw probabilities on training sessions for calibrator fitting.
    Returns (p_oof, y_oof, sessions_oof) aligned arrays (only covered rows).
    """
    inner = inner_folds_for_train(sorted(set(sessions)), xfit_cfg)
    n = len(rows)
    pred = np.full(n, np.nan, dtype=float)
    if not inner:
        return np.array([]), np.array([]), []

    for inf in inner:
        tr_set = set(inf["train_sessions"])
        va_set = set(inf["test_sessions"])
        tr_idx = [i for i, s in enumerate(sessions) if s in tr_set]
        va_idx = [i for i, s in enumerate(sessions) if s in va_set]
        if not tr_idx or not va_idx:
            continue
        y_tr = y[tr_idx]
        if len(np.unique(y_tr)) < 2:
            pred[va_idx] = float(np.mean(y_tr)) if len(y_tr) else 0.5
            continue
        vec = FeatureVectorizer().fit([rows[i] for i in tr_idx])
        est = _make_estimator(cfg, params)
        est.fit(vec.transform([rows[i] for i in tr_idx]), y_tr)
        pred[va_idx] = clip_probability(
            est.predict_proba(vec.transform([rows[i] for i in va_idx]))[:, 1])

    covered = np.isfinite(pred)
    idx = np.flatnonzero(covered)
    return pred[covered], y[covered], [sessions[i] for i in idx]


@dataclass
class DirectionModel:
    """One calibrated binary direction model for one horizon."""
    config: DirectionModelConfig = field(default_factory=DirectionModelConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = None
    calibrator: object = field(default_factory=IdentityCalibrator)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False
    calibration_artifact: object = None

    def fit(self, rows: Sequence[dict], y: Sequence[int],
            sessions: Sequence[str]) -> "DirectionModel":
        """
        V3 sequence (§5.2, training-window only):
          1. Select hyperparameters via inner OOF log loss.
          2. Generate cross-fitted raw probabilities on eligible sessions.
          3. Fit probability calibrator on those OOF probabilities.
          4. Train final base estimator on all eligible sessions.
          5. Attach the training-only calibrator.
        Outer test evaluation is the caller's job (prediction/training.py).
        """
        y = np.asarray(y, dtype=int)
        sessions = list(sessions)
        rows = list(rows)
        uniq = sorted(set(sessions))
        xfit_cfg = _adaptive_crossfit_cfg(sessions, self.config)
        grid = _param_grid(self.config)
        best_params = dict(grid[0])
        inner_diag: dict = {"note": "skipped"}
        cal_metrics: dict = {"note": "no calibration; identity calibrator"}
        fit_sessions, cal_sessions = uniq, []

        if len(uniq) >= 2 and len(np.unique(y)) >= 2:
            best_params, inner_diag = _select_params_oof(
                rows, y, sessions, self.config, xfit_cfg)
            p_oof, y_oof, sess_oof = _oof_raw_probabilities(
                rows, y, sessions, best_params, self.config, xfit_cfg)

            if len(p_oof) >= 5 and len(np.unique(y_oof)) >= 2:
                cal_sessions = sorted(set(sess_oof))
                # Sessions used purely for HP inner-train prefixes (approx):
                # everything not solely in the last cal-like holdouts.
                fit_sessions = [s for s in uniq if s in set(sess_oof) or True]
                # Record HP-fit sessions as those appearing in any inner train
                inner = inner_folds_for_train(uniq, xfit_cfg)
                hp_fit = sorted({s for inf in inner
                                 for s in inf["train_sessions"]})
                if hp_fit:
                    fit_sessions = hp_fit

                if self.config.calibration == "auto":
                    self.calibrator, cal_diag = select_calibrator(
                        p_oof, y_oof, n_sessions=len(set(sess_oof)),
                        sessions=sess_oof,
                        embargo_sessions=self.config.embargo_sessions)
                else:
                    self.calibrator = fit_calibrator(
                        p_oof, y_oof, self.config.calibration)
                    cal_diag = self.calibrator.to_dict()

                p_cal = self.calibrator.transform(p_oof)
                slice_report = slice_calibration_report(
                    p_cal, y_oof, sessions=sess_oof, rows=None)
                art = build_calibration_artifact(
                    self.calibrator, p_oof, y_oof,
                    training_sessions=cal_sessions,
                    diagnostics=cal_diag)
                self.calibration_artifact = art
                cal_metrics = {
                    "n": int(len(y_oof)),
                    "brier_raw": art.brier_before,
                    "brier_calibrated": art.brier_after,
                    "brier_skill": brier_skill(y_oof, p_cal),
                    "log_loss": art.log_loss_after,
                    "slope": art.slope,
                    "intercept": art.intercept,
                    "reliability": art.reliability_bins,
                    "calibration_diag": cal_diag,
                    "slice_report": slice_report,
                    "calibration_artifact": art.to_dict(),
                    "crossfit": True,
                }
            else:
                self.calibrator = IdentityCalibrator()
                self.calibration_artifact = None
                cal_metrics = {"note": "insufficient OOF rows for calibration"}
        else:
            self.calibrator = IdentityCalibrator()
            self.calibration_artifact = None

        # Final base estimator on ALL eligible sessions
        self.vectorizer = FeatureVectorizer()
        X_all = self.vectorizer.fit_transform(rows)
        if len(np.unique(y)) < 2:
            self.estimator = None
        else:
            self.estimator = _make_estimator(self.config, best_params)
            self.estimator.fit(X_all, y)

        self._base_rate = float(np.mean(y)) if len(y) else 0.5
        self.metadata = {
            "target": f"up_{self.config.horizon}",
            "horizon": self.config.horizon,
            "estimator": self.config.estimator,
            "best_params": {k: (v if v is None or isinstance(v, (int, float))
                                else str(v))
                            for k, v in best_params.items()},
            "train_sessions": uniq,
            "fit_sessions": sorted(set(fit_sessions)),
            "calibration_sessions": sorted(set(cal_sessions)),
            "n_train_rows": int(len(y)),
            "base_rate": self._base_rate,
            "calibration_metrics": cal_metrics,
            "decision_threshold": self.config.decision_threshold,
            "inner_selection": inner_diag,
            "crossfit_config": {
                "inner_folds": xfit_cfg.inner_folds,
                "embargo_sessions": xfit_cfg.embargo_sessions,
                "min_train_sessions": xfit_cfg.min_train_sessions,
                "min_validation_sessions": xfit_cfg.min_validation_sessions,
                "random_state": xfit_cfg.random_state,
            },
        }
        skill = (cal_metrics.get("brier_skill")
                 if isinstance(cal_metrics, dict) else None)
        self.metadata["uncertainty"] = float(
            np.clip(1.0 - max(skill or 0.0, 0.0), 0.0, 1.0))
        self.fitted = True
        return self

    @staticmethod
    def _raw_from(estimator, X: np.ndarray) -> np.ndarray:
        return clip_probability(estimator.predict_proba(X)[:, 1])

    def predict_raw(self, rows: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("DirectionModel used before fit")
        if self.estimator is None:
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
