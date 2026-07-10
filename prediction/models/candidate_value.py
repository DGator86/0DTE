"""
prediction/models/candidate_value.py
====================================
Candidate-level value model (docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.6).

One row = one candidate at one snapshot. Predicts:
  * expected net P&L (elastic-net regression)
  * P(positive net P&L) (elastic-net logistic + calibration)
  * downside / median / upside P&L quantiles (HGB quantile regression)

Grouping: candidates that share a snapshot_id must never be split across
train/test (enforced by callers via prediction.candidate_dataset folds).

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
                                    clip_probability, log_loss_score,
                                    rearrange_quantiles)
from prediction.models.direction import split_train_calibration

CANDIDATE_VALUE_VERSION = "v2.0.0-pr8"
QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class CandidateForecast:
    """Ranking output for one candidate (§11.6)."""
    candidate_id: str
    expected_net_pnl: float
    p_profit: float
    pnl_q10: float
    pnl_q50: float
    pnl_q90: float
    expected_shortfall: float
    fill_uncertainty: float
    model_uncertainty: float
    utility_score: float
    model_version: str = CANDIDATE_VALUE_VERSION

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "expected_net_pnl": self.expected_net_pnl,
            "p_profit": self.p_profit,
            "pnl_q10": self.pnl_q10,
            "pnl_q50": self.pnl_q50,
            "pnl_q90": self.pnl_q90,
            "expected_shortfall": self.expected_shortfall,
            "fill_uncertainty": self.fill_uncertainty,
            "model_uncertainty": self.model_uncertainty,
            "utility_score": self.utility_score,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateForecast":
        return cls(
            candidate_id=str(d["candidate_id"]),
            expected_net_pnl=float(d["expected_net_pnl"]),
            p_profit=float(d["p_profit"]),
            pnl_q10=float(d["pnl_q10"]),
            pnl_q50=float(d["pnl_q50"]),
            pnl_q90=float(d["pnl_q90"]),
            expected_shortfall=float(d["expected_shortfall"]),
            fill_uncertainty=float(d.get("fill_uncertainty", 0.0)),
            model_uncertainty=float(d.get("model_uncertainty", 0.0)),
            utility_score=float(d.get("utility_score", 0.0)),
            model_version=str(d.get("model_version", CANDIDATE_VALUE_VERSION)),
        )


@dataclass
class CandidateValueConfig:
    c_grid: tuple = (0.05, 0.1, 0.5, 1.0)
    l1_ratio_grid: tuple = (0.0, 0.5, 1.0)
    max_iter: int = 1500
    calibration_frac: float = 0.25
    embargo_sessions: int = 1
    calibration: str = "auto"
    # Quantile heads (HistGradientBoostingRegressor)
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    max_depth: Optional[int] = 3
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    quantile_max_iter: int = 150
    quantiles: tuple = QUANTILES


def _elasticnet_regressor(cfg: CandidateValueConfig, params: dict):
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("en", ElasticNet(
            alpha=params["alpha"], l1_ratio=params["l1_ratio"],
            max_iter=cfg.max_iter, random_state=RANDOM_STATE)),
    ])


def _elasticnet_classifier(cfg: CandidateValueConfig, params: dict):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    kw = dict(solver="saga", C=params["C"], l1_ratio=params["l1_ratio"],
              max_iter=cfg.max_iter, random_state=RANDOM_STATE)
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 8):
        kw["penalty"] = "elasticnet"
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(**kw))])


def _quantile_regressor(cfg: CandidateValueConfig, quantile: float):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        max_iter=cfg.quantile_max_iter,
        random_state=RANDOM_STATE)


@dataclass
class CandidateValueModel:
    """Multi-head candidate value model (§11.6)."""
    config: CandidateValueConfig = field(default_factory=CandidateValueConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    pnl_estimator: object = None
    profit_estimator: object = None
    profit_calibrator: object = field(default_factory=IdentityCalibrator)
    quantile_estimators: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    fitted: bool = False
    _mean_pnl: float = 0.0
    _base_profit: float = 0.5
    _model_uncertainty: float = 0.5

    def fit(
        self,
        rows: Sequence[dict],
        *,
        y_pnl: Sequence[float],
        y_profit: Sequence[int],
        sessions: Sequence[str],
        group_ids: Optional[Sequence[str]] = None,
    ) -> "CandidateValueModel":
        """
        Fit on candidate rows. `sessions` drive the inner calibration split;
        `group_ids` (snapshot_id) are recorded for audit — callers must ensure
        snapshot groups are never split across outer train/test folds.
        """
        y_pnl = np.asarray(y_pnl, dtype=float)
        y_profit = np.asarray(y_profit, dtype=int)
        sessions = list(sessions)
        X = self.vectorizer.fit_transform(list(rows))

        # --- expected net P&L (elastic-net) ---
        best_pnl, best_mse = None, math.inf
        for alpha in (0.001, 0.01, 0.1):
            for l1 in self.config.l1_ratio_grid:
                est = _elasticnet_regressor(
                    self.config, {"alpha": alpha, "l1_ratio": l1})
                est.fit(X, y_pnl)
                pred = est.predict(X)
                mse = float(np.mean((pred - y_pnl) ** 2))
                if mse < best_mse:
                    best_mse, best_pnl = mse, est
        self.pnl_estimator = best_pnl
        self._mean_pnl = float(np.mean(y_pnl)) if len(y_pnl) else 0.0

        # --- P(profit) with embargoed inner calibration ---
        fit_s, cal_s = split_train_calibration(
            sessions, self.config.calibration_frac, self.config.embargo_sessions)
        fit_mask = np.array([s in set(fit_s) for s in sessions])
        cal_mask = np.array([s in set(cal_s) for s in sessions])
        cal_metrics: dict = {"note": "no calibration slice; identity calibrator"}
        best_params = {"C": 0.5, "l1_ratio": 0.5}

        if (cal_mask.any() and fit_mask.any()
                and len(np.unique(y_profit[fit_mask])) >= 2):
            X_fit, y_fit = X[fit_mask], y_profit[fit_mask]
            X_cal, y_cal = X[cal_mask], y_profit[cal_mask]
            best_loss, best_clf = math.inf, None
            for C in self.config.c_grid:
                for l1 in self.config.l1_ratio_grid:
                    params = {"C": C, "l1_ratio": l1}
                    clf = _elasticnet_classifier(self.config, params)
                    clf.fit(X_fit, y_fit)
                    loss = log_loss_score(y_cal, clf.predict_proba(X_cal)[:, 1])
                    if loss < best_loss:
                        best_loss, best_clf, best_params = loss, clf, params
            self.profit_estimator = best_clf
            p_raw = clip_probability(best_clf.predict_proba(X_cal)[:, 1])
            if self.config.calibration == "auto":
                self.profit_calibrator, cal_diag = select_calibrator(
                    p_raw, y_cal, n_sessions=len(cal_s))
            else:
                self.profit_calibrator = fit_calibrator(
                    p_raw, y_cal, self.config.calibration)
                cal_diag = self.profit_calibrator.to_dict()
            p_cal = self.profit_calibrator.transform(p_raw)
            skill = brier_skill(y_cal, p_cal)
            cal_metrics = {
                "n": int(len(y_cal)),
                "brier_raw": brier_score(y_cal, p_raw),
                "brier_calibrated": brier_score(y_cal, p_cal),
                "brier_skill": skill,
                "log_loss": log_loss_score(y_cal, p_cal),
                "calibration_diag": cal_diag,
            }
            self._model_uncertainty = float(
                np.clip(1.0 - max(skill or 0.0, 0.0), 0.0, 1.0))
        else:
            if len(np.unique(y_profit)) >= 2:
                clf = _elasticnet_classifier(self.config, best_params)
                clf.fit(X, y_profit)
                self.profit_estimator = clf
            else:
                self.profit_estimator = None
            self.profit_calibrator = IdentityCalibrator()
            fit_s, cal_s = sorted(set(sessions)), []
            self._model_uncertainty = 0.5

        self._base_profit = float(np.mean(y_profit)) if len(y_profit) else 0.5

        # --- quantile heads ---
        self.quantile_estimators = {}
        for q in self.config.quantiles:
            est = _quantile_regressor(self.config, q)
            est.fit(X, y_pnl)
            self.quantile_estimators[q] = est

        n_groups = len(set(group_ids)) if group_ids is not None else None
        self.metadata = {
            "target": "candidate_net_pnl",
            "model_version": CANDIDATE_VALUE_VERSION,
            "n_train_rows": int(len(y_pnl)),
            "n_train_snapshots": n_groups,
            "train_sessions": sorted(set(sessions)),
            "fit_sessions": sorted(set(fit_s)),
            "calibration_sessions": sorted(set(cal_s)),
            "profit_best_params": best_params,
            "calibration_metrics": cal_metrics,
            "mean_pnl": self._mean_pnl,
            "base_profit_rate": self._base_profit,
            "uncertainty": self._model_uncertainty,
        }
        self.fitted = True
        return self

    def predict_components(self, rows: Sequence[dict]) -> dict:
        """Raw component arrays (no utility)."""
        if not self.fitted:
            raise RuntimeError("CandidateValueModel used before fit")
        X = self.vectorizer.transform(list(rows))
        n = len(rows)
        if self.pnl_estimator is None:
            exp = np.full(n, self._mean_pnl)
        else:
            exp = np.asarray(self.pnl_estimator.predict(X), dtype=float)

        if self.profit_estimator is None:
            p_profit = np.full(n, self._base_profit)
        else:
            raw = clip_probability(
                self.profit_estimator.predict_proba(X)[:, 1])
            p_profit = clip_probability(self.profit_calibrator.transform(raw))

        preds = [self.quantile_estimators[q].predict(X)
                 for q in self.config.quantiles]
        q10, q50, q90 = rearrange_quantiles(*preds)
        # Expected shortfall ≈ magnitude of downside quantile loss
        shortfall = np.maximum(-q10, 0.0)
        return {
            "expected_net_pnl": exp,
            "p_profit": p_profit,
            "pnl_q10": q10,
            "pnl_q50": q50,
            "pnl_q90": q90,
            "expected_shortfall": shortfall,
            "model_uncertainty": np.full(n, self._model_uncertainty),
        }

    def predict(
        self,
        rows: Sequence[dict],
        *,
        candidate_ids: Optional[Sequence[str]] = None,
        fill_uncertainty: Optional[Sequence[float]] = None,
        capital: Optional[Sequence[float]] = None,
        utility_fn=None,
    ) -> list[CandidateForecast]:
        """
        Build CandidateForecast list. Utility is computed by `utility_fn`
        (default: prediction.candidate_ranker.candidate_utility) when provided;
        otherwise utility_score is set to expected_net_pnl as a placeholder.
        """
        comps = self.predict_components(rows)
        n = len(rows)
        ids = list(candidate_ids) if candidate_ids is not None else [
            f"cand_{i}" for i in range(n)]
        fills = (np.asarray(fill_uncertainty, dtype=float)
                 if fill_uncertainty is not None else np.zeros(n))
        caps = (np.asarray(capital, dtype=float)
                if capital is not None else np.zeros(n))

        out: list[CandidateForecast] = []
        for i in range(n):
            fc = CandidateForecast(
                candidate_id=ids[i],
                expected_net_pnl=float(comps["expected_net_pnl"][i]),
                p_profit=float(comps["p_profit"][i]),
                pnl_q10=float(comps["pnl_q10"][i]),
                pnl_q50=float(comps["pnl_q50"][i]),
                pnl_q90=float(comps["pnl_q90"][i]),
                expected_shortfall=float(comps["expected_shortfall"][i]),
                fill_uncertainty=float(fills[i]),
                model_uncertainty=float(comps["model_uncertainty"][i]),
                utility_score=0.0,
            )
            if utility_fn is not None:
                util = float(utility_fn(fc, capital=float(caps[i])))
            else:
                util = fc.expected_net_pnl
            out.append(CandidateForecast(
                **{**fc.to_dict(), "utility_score": util}))
        return out
