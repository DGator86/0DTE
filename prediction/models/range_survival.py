"""
prediction/models/range_survival.py
===================================
Calibrated range-survival models
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.4).

Predict P(price stays strictly inside [lower, upper] through horizon) for:
  * wall-channel survival (put_wall, call_wall)
  * candidate short-strike survival
  * candidate breakeven survival

Horizons: 15m, 30m, 60m, close. One binary elastic-net logistic model per
(target_kind, horizon) pair, with an embargoed inner calibration split.

Inputs (§11.4): distance to each boundary, forecast volatility, time
remaining, GEX state, wall stability, trend/flow, expected-move consumed,
barrier width / volatility, plus optional path-model survival frequency.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (IdentityCalibrator, fit_calibrator,
                                    select_calibrator)
from prediction.models.base import (RANDOM_STATE, FeatureVectorizer,
                                    brier_score, brier_skill,
                                    clip_probability, log_loss_score)
from prediction.models.barrier_touch import path_features
from prediction.models.direction import split_train_calibration

RANGE_HORIZONS = ("15m", "30m", "60m", "close")
RANGE_KINDS = ("wall_channel", "short_strike", "breakeven")


@dataclass
class RangeSurvivalConfig:
    kind: str = "wall_channel"           # wall_channel | short_strike | breakeven
    horizon: str = "close"
    c_grid: tuple = (0.05, 0.1, 0.5, 1.0)
    l1_ratio_grid: tuple = (0.0, 0.5, 1.0)
    class_weight_options: tuple = (None, "balanced")
    max_iter: int = 1500
    calibration_frac: float = 0.25
    embargo_sessions: int = 1
    calibration: str = "auto"


def _make_estimator(cfg: RangeSurvivalConfig, params: dict):
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


def _param_grid(cfg: RangeSurvivalConfig) -> list:
    return [{"C": c, "l1_ratio": l1, "class_weight": cw}
            for c in cfg.c_grid
            for l1 in cfg.l1_ratio_grid
            for cw in cfg.class_weight_options]


@dataclass
class RangeSurvivalModel:
    config: RangeSurvivalConfig = field(default_factory=RangeSurvivalConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = None
    calibrator: object = field(default_factory=IdentityCalibrator)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False
    _base_rate: float = 0.5

    def fit(self, rows: Sequence[dict], y: Sequence[int],
            sessions: Sequence[str]) -> "RangeSurvivalModel":
        if self.config.kind not in RANGE_KINDS:
            raise ValueError(f"unknown range kind {self.config.kind!r}")
        if self.config.horizon not in RANGE_HORIZONS:
            raise ValueError(f"unknown horizon {self.config.horizon!r}")
        y = np.asarray(y, dtype=int)
        sessions = list(sessions)
        fit_s, cal_s = split_train_calibration(
            sessions, self.config.calibration_frac, self.config.embargo_sessions)
        fit_mask = np.array([s in set(fit_s) for s in sessions])
        cal_mask = np.array([s in set(cal_s) for s in sessions])

        fit_rows = [r for r, m in zip(rows, fit_mask) if m]
        X_fit = self.vectorizer.fit_transform(fit_rows)
        y_fit = y[fit_mask]

        grid = _param_grid(self.config)
        best_params, best_est, best_loss = grid[0], None, math.inf
        cal_metrics: dict = {"note": "no calibration slice; identity calibrator"}

        if cal_mask.any() and len(np.unique(y_fit)) >= 2:
            cal_rows = [r for r, m in zip(rows, cal_mask) if m]
            X_cal = self.vectorizer.transform(cal_rows)
            y_cal = y[cal_mask]
            for params in grid:
                est = _make_estimator(self.config, params)
                est.fit(X_fit, y_fit)
                loss = log_loss_score(y_cal, est.predict_proba(X_cal)[:, 1])
                if loss < best_loss:
                    best_params, best_est, best_loss = params, est, loss
            p_raw = clip_probability(best_est.predict_proba(X_cal)[:, 1])
            if self.config.calibration == "auto":
                self.calibrator, cal_diag = select_calibrator(
                    p_raw, y_cal, n_sessions=len(cal_s))
            else:
                self.calibrator = fit_calibrator(
                    p_raw, y_cal, self.config.calibration)
                cal_diag = self.calibrator.to_dict()
            p_cal = self.calibrator.transform(p_raw)
            cal_metrics = {
                "n": int(len(y_cal)),
                "brier_raw": brier_score(y_cal, p_raw),
                "brier_calibrated": brier_score(y_cal, p_cal),
                "brier_skill": brier_skill(y_cal, p_cal),
                "log_loss": log_loss_score(y_cal, p_cal),
                "calibration_diag": cal_diag,
            }
        else:
            X_fit = self.vectorizer.fit_transform(list(rows))
            y_fit = y
            best_est = _make_estimator(self.config, best_params)
            if len(np.unique(y_fit)) < 2:
                best_est = None
            else:
                best_est.fit(X_fit, y_fit)
            self.calibrator = IdentityCalibrator()
            fit_s, cal_s = sorted(set(sessions)), []

        self.estimator = best_est
        self._base_rate = float(np.mean(y)) if len(y) else 0.5
        skill = (cal_metrics.get("brier_skill")
                 if isinstance(cal_metrics, dict) else None)
        self.metadata = {
            "target": f"range_survive_{self.config.kind}_{self.config.horizon}",
            "kind": self.config.kind,
            "horizon": self.config.horizon,
            "best_params": {k: (v if v is None or isinstance(v, (int, float))
                                else str(v))
                            for k, v in best_params.items()},
            "train_sessions": sorted(set(sessions)),
            "fit_sessions": sorted(set(fit_s)),
            "calibration_sessions": sorted(set(cal_s)),
            "n_train_rows": int(len(y)),
            "base_rate": self._base_rate,
            "calibration_metrics": cal_metrics,
            "uncertainty": float(np.clip(1.0 - max(skill or 0.0, 0.0), 0.0, 1.0)),
        }
        self.fitted = True
        return self

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RangeSurvivalModel used before fit")
        if self.estimator is None:
            return np.full(len(rows), self._base_rate)
        raw = clip_probability(
            self.estimator.predict_proba(self.vectorizer.transform(rows))[:, 1])
        return clip_probability(self.calibrator.transform(raw))


def range_feature_row(
    *,
    spot: float,
    lower: float,
    upper: float,
    minutes_to_close: Optional[float] = None,
    expected_realized_move: Optional[float] = None,
    move_consumed: Optional[float] = None,
    net_gex: Optional[float] = None,
    wall_stability: Optional[float] = None,
    adx: Optional[float] = None,
    cvd_slope: Optional[float] = None,
    path_events: Optional[object] = None,
) -> dict:
    """Standard feature row for range-survival models (§11.4)."""
    width = (upper - lower) / spot if spot else None
    row = {
        "dist_lower": (spot - lower) / spot if spot else None,
        "dist_upper": (upper - spot) / spot if spot else None,
        "barrier_width": width,
    }
    if width is not None and expected_realized_move and expected_realized_move > 0:
        row["barrier_width_over_vol"] = width / expected_realized_move
    if minutes_to_close is not None:
        row["minutes_to_close"] = float(minutes_to_close)
    if expected_realized_move is not None:
        row["expected_realized_move"] = float(expected_realized_move)
    if move_consumed is not None:
        row["move_consumed"] = float(move_consumed)
    if net_gex is not None:
        row["net_gex_sign"] = float(np.sign(net_gex))
        row["net_gex"] = float(net_gex)
    if wall_stability is not None:
        row["wall_stability"] = float(wall_stability)
    if adx is not None:
        row["adx"] = float(adx)
    if cvd_slope is not None:
        row["cvd_slope"] = float(cvd_slope)
    if path_events is not None:
        row.update(path_features(path_events))
    return row
