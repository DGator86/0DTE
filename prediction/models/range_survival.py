"""
prediction/models/range_survival.py
===================================
Calibrated range-survival models
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.4; V3 Part 1 §5).

Hyperparameters selected by inner OOF log loss; calibrator fitted on
cross-fitted raw scores.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (
    IdentityCalibrator,
    build_calibration_artifact,
    fit_calibrator,
    select_calibrator,
    slice_calibration_report,
)
from prediction.crossfit import NestedCrossFitConfig, inner_folds_for_train
from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    brier_skill,
    clip_probability,
    log_loss_score,
)
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
    inner_folds: int = 3
    min_train_sessions: int = 8
    min_validation_sessions: int = 3
    random_state: int = RANDOM_STATE


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


def _adaptive_cfg(sessions, cfg: RangeSurvivalConfig) -> NestedCrossFitConfig:
    n = len(set(sessions))
    min_train = min(cfg.min_train_sessions, max(3, n // 3))
    min_val = min(cfg.min_validation_sessions, max(2, n // 6))
    return NestedCrossFitConfig(
        outer_folds=2, inner_folds=min(cfg.inner_folds, 3),
        embargo_sessions=cfg.embargo_sessions,
        min_train_sessions=min_train,
        min_validation_sessions=min_val,
        retain_fold_models=False,
        random_state=cfg.random_state,
    )


@dataclass
class RangeSurvivalModel:
    config: RangeSurvivalConfig = field(default_factory=RangeSurvivalConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = None
    calibrator: object = field(default_factory=IdentityCalibrator)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False
    _base_rate: float = 0.5
    calibration_artifact: object = None

    def fit(self, rows: Sequence[dict], y: Sequence[int],
            sessions: Sequence[str]) -> "RangeSurvivalModel":
        if self.config.kind not in RANGE_KINDS:
            raise ValueError(f"unknown range kind {self.config.kind!r}")
        if self.config.horizon not in RANGE_HORIZONS:
            raise ValueError(f"unknown horizon {self.config.horizon!r}")
        y = np.asarray(y, dtype=int)
        sessions = list(sessions)
        rows = list(rows)
        uniq = sorted(set(sessions))
        xfit_cfg = _adaptive_cfg(sessions, self.config)
        grid = _param_grid(self.config)
        best_params = dict(grid[0])
        cal_metrics: dict = {"note": "no calibration slice; identity calibrator"}
        fit_s, cal_s = uniq, []
        inner_diag: dict = {}

        if len(uniq) >= 2 and len(np.unique(y)) >= 2:
            inner = inner_folds_for_train(uniq, xfit_cfg)
            scores = []
            for params in grid:
                losses = []
                for inf in inner:
                    tr = [i for i, s in enumerate(sessions)
                          if s in set(inf["train_sessions"])]
                    va = [i for i, s in enumerate(sessions)
                          if s in set(inf["test_sessions"])]
                    if not tr or not va or len(np.unique(y[tr])) < 2:
                        continue
                    vec = FeatureVectorizer().fit([rows[i] for i in tr])
                    est = _make_estimator(self.config, params)
                    est.fit(vec.transform([rows[i] for i in tr]), y[tr])
                    p = clip_probability(
                        est.predict_proba(
                            vec.transform([rows[i] for i in va]))[:, 1])
                    losses.append(log_loss_score(y[va], p))
                if losses:
                    scores.append((float(np.mean(losses)), dict(params)))
            if scores:
                scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
                best_params = scores[0][1]
                inner_diag = {"selection_metric": "log_loss",
                              "best_log_loss": scores[0][0]}

            pred = np.full(len(rows), np.nan)
            for inf in inner:
                tr = [i for i, s in enumerate(sessions)
                      if s in set(inf["train_sessions"])]
                va = [i for i, s in enumerate(sessions)
                      if s in set(inf["test_sessions"])]
                if not tr or not va:
                    continue
                if len(np.unique(y[tr])) < 2:
                    pred[va] = float(np.mean(y[tr]))
                    continue
                vec = FeatureVectorizer().fit([rows[i] for i in tr])
                est = _make_estimator(self.config, best_params)
                est.fit(vec.transform([rows[i] for i in tr]), y[tr])
                pred[va] = clip_probability(
                    est.predict_proba(
                        vec.transform([rows[i] for i in va]))[:, 1])
            covered = np.isfinite(pred)
            if covered.sum() >= 5 and len(np.unique(y[covered])) >= 2:
                p_oof, y_oof = pred[covered], y[covered]
                sess_oof = [sessions[i] for i in np.flatnonzero(covered)]
                cal_s = sorted(set(sess_oof))
                fit_s = sorted({s for inf in inner
                                for s in inf["train_sessions"]}) or uniq
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
                art = build_calibration_artifact(
                    self.calibrator, p_oof, y_oof,
                    training_sessions=cal_s, diagnostics=cal_diag)
                self.calibration_artifact = art
                cal_metrics = {
                    "n": int(len(y_oof)),
                    "brier_raw": art.brier_before,
                    "brier_calibrated": art.brier_after,
                    "brier_skill": brier_skill(y_oof, p_cal),
                    "log_loss": art.log_loss_after,
                    "calibration_diag": cal_diag,
                    "slice_report": slice_calibration_report(
                        p_cal, y_oof, sessions=sess_oof),
                    "calibration_artifact": art.to_dict(),
                    "crossfit": True,
                }
            else:
                self.calibrator = IdentityCalibrator()
        else:
            self.calibrator = IdentityCalibrator()

        self.vectorizer = FeatureVectorizer()
        X = self.vectorizer.fit_transform(rows)
        if len(np.unique(y)) < 2:
            self.estimator = None
        else:
            self.estimator = _make_estimator(self.config, best_params)
            self.estimator.fit(X, y)

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
            "train_sessions": uniq,
            "fit_sessions": sorted(set(fit_s)),
            "calibration_sessions": sorted(set(cal_s)),
            "n_train_rows": int(len(y)),
            "base_rate": self._base_rate,
            "calibration_metrics": cal_metrics,
            "inner_selection": inner_diag,
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
