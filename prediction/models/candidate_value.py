"""
prediction/models/candidate_value.py
====================================
Candidate-level value model (docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.6;
docs/PREDICTION_ENGINE_V3_PART1_VALIDATION.md §6).

One row = one candidate at one snapshot. Predicts:
  * expected net P&L (OOF-selected ElasticNet / Huber / HGB regressor)
  * P(positive net P&L) (OOF elastic-net logistic + independent calibration)
  * downside / median / upside P&L quantiles (HGB quantile regression, OOF eval)

Grouping: candidates that share a snapshot_id must never be split across
train/validation (enforced via group_ids + session folds).

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (
    IdentityCalibrator,
    build_calibration_artifact,
    fit_calibrator,
    select_calibrator,
)
from prediction.crossfit import (
    NestedCrossFitConfig,
    downside_underprediction_penalty,
    huber_loss,
    inner_folds_for_train,
    regression_metrics,
    regression_selection_score,
)
from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    brier_score,
    brier_skill,
    clip_probability,
    interval_coverage,
    log_loss_score,
    pinball_loss,
    rearrange_quantiles,
)

CANDIDATE_VALUE_VERSION = "v3.0.0-part1"
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
    alpha_grid: tuple = (0.001, 0.01, 0.1)
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
    # V3 selection / cross-fit
    bias_weight: float = 0.25
    downside_weight: float = 0.25
    huber_delta: float = 1.0
    inner_folds: int = 3
    min_train_sessions: int = 8
    min_validation_sessions: int = 3
    random_state: int = RANDOM_STATE
    # Challenger grids (kept small for determinism / runtime)
    huber_epsilon_grid: tuple = (1.1, 1.35)
    hgb_learning_rate_grid: tuple = (0.05, 0.1)
    hgb_max_depth_grid: tuple = (2, 3)
    feature_version: str = "v2.0.0"
    label_version: str = "v2.0.0"


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


def _huber_regressor(cfg: CandidateValueConfig, params: dict):
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("hb", HuberRegressor(
            epsilon=params.get("epsilon", 1.35),
            alpha=params.get("alpha", 0.0001),
            max_iter=cfg.max_iter)),
    ])


def _hgb_regressor(cfg: CandidateValueConfig, params: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=params.get("learning_rate", cfg.learning_rate),
        max_leaf_nodes=params.get("max_leaf_nodes", cfg.max_leaf_nodes),
        max_depth=params.get("max_depth", cfg.max_depth),
        min_samples_leaf=params.get("min_samples_leaf", cfg.min_samples_leaf),
        l2_regularization=params.get("l2_regularization", cfg.l2_regularization),
        max_iter=cfg.quantile_max_iter,
        random_state=RANDOM_STATE)


def _pnl_estimator_factory(cfg: CandidateValueConfig, params: dict):
    kind = params.get("estimator", "elasticnet")
    if kind == "elasticnet":
        return _elasticnet_regressor(cfg, params)
    if kind == "huber":
        return _huber_regressor(cfg, params)
    if kind == "hgb":
        return _hgb_regressor(cfg, params)
    raise ValueError(f"unknown pnl estimator {kind!r}")


def _pnl_param_grid(cfg: CandidateValueConfig) -> list[dict]:
    grid: list[dict] = []
    for alpha in cfg.alpha_grid:
        for l1 in cfg.l1_ratio_grid:
            grid.append({"estimator": "elasticnet", "alpha": alpha,
                         "l1_ratio": l1})
    for eps in cfg.huber_epsilon_grid:
        for alpha in (0.0001, 0.001):
            grid.append({"estimator": "huber", "epsilon": eps, "alpha": alpha})
    for lr in cfg.hgb_learning_rate_grid:
        for md in cfg.hgb_max_depth_grid:
            grid.append({"estimator": "hgb", "learning_rate": lr,
                         "max_depth": md,
                         "max_leaf_nodes": cfg.max_leaf_nodes,
                         "min_samples_leaf": cfg.min_samples_leaf,
                         "l2_regularization": cfg.l2_regularization})
    return grid


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


def _adaptive_cfg(sessions, cfg: CandidateValueConfig) -> NestedCrossFitConfig:
    n = len(set(sessions))
    min_train = min(cfg.min_train_sessions, max(3, n // 3))
    min_val = min(cfg.min_validation_sessions, max(2, n // 6))
    return NestedCrossFitConfig(
        outer_folds=2,
        inner_folds=min(cfg.inner_folds, 3),
        embargo_sessions=cfg.embargo_sessions,
        min_train_sessions=min_train,
        min_validation_sessions=min_val,
        retain_fold_models=False,
        random_state=cfg.random_state,
        bias_weight=cfg.bias_weight,
        downside_weight=cfg.downside_weight,
        huber_delta=cfg.huber_delta,
    )


def _assert_groups_intact(group_ids, tr_idx, va_idx):
    if group_ids is None:
        return
    tr_g = {group_ids[i] for i in tr_idx}
    va_g = {group_ids[i] for i in va_idx}
    leaked = tr_g & va_g
    if leaked:
        raise AssertionError(
            f"snapshot groups split across folds: {sorted(leaked)[:5]}")


def _schema_hash(rows: Sequence[dict]) -> str:
    keys = sorted({k for r in rows for k in r.keys()})
    return hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _family_of(row: dict) -> str:
    return str(row.get("family") or row.get("option_family") or "unknown")


@dataclass
class CandidateValueModel:
    """Multi-head candidate value model (§11.6 / V3 §6)."""
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
    calibration_artifact: object = None

    def fit(
        self,
        rows: Sequence[dict],
        *,
        y_pnl: Sequence[float],
        y_profit: Sequence[int],
        sessions: Sequence[str],
        group_ids: Optional[Sequence[str]] = None,
        data_hash: Optional[str] = None,
        outcome_coverage: Optional[float] = None,
    ) -> "CandidateValueModel":
        """
        Fit with session-grouped OOF selection (no in-sample P&L HP search).
        `group_ids` (snapshot_id) are never split across train/validation.
        """
        y_pnl = np.asarray(y_pnl, dtype=float)
        y_profit = np.asarray(y_profit, dtype=int)
        sessions = list(sessions)
        rows = list(rows)
        groups = list(group_ids) if group_ids is not None else None
        uniq = sorted(set(sessions))
        xfit_cfg = _adaptive_cfg(sessions, self.config)
        schema_hash = _schema_hash(rows)

        # --- expected P&L: OOF selection across challengers ---
        pnl_params, pnl_oof_metrics, pnl_diag = self._select_pnl_oof(
            rows, y_pnl, sessions, groups, xfit_cfg)

        # --- P(profit): OOF HP + independent calibration ---
        profit_params, cal_metrics, fit_s, cal_s = self._fit_profit_oof(
            rows, y_profit, sessions, groups, xfit_cfg)

        # --- quantile heads: fit final + OOF evaluation ---
        quantile_oof = self._evaluate_quantiles_oof(
            rows, y_pnl, sessions, groups, xfit_cfg)

        # Final estimators on all rows
        self.vectorizer = FeatureVectorizer()
        X = self.vectorizer.fit_transform(rows)
        self.pnl_estimator = _pnl_estimator_factory(self.config, pnl_params)
        self.pnl_estimator.fit(X, y_pnl)
        self._mean_pnl = float(np.mean(y_pnl)) if len(y_pnl) else 0.0

        if len(np.unique(y_profit)) >= 2:
            self.profit_estimator = _elasticnet_classifier(
                self.config, profit_params)
            self.profit_estimator.fit(X, y_profit)
        else:
            self.profit_estimator = None
        self._base_profit = float(np.mean(y_profit)) if len(y_profit) else 0.5

        self.quantile_estimators = {}
        for q in self.config.quantiles:
            est = _quantile_regressor(self.config, q)
            est.fit(X, y_pnl)
            self.quantile_estimators[q] = est

        families = sorted({_family_of(r) for r in rows})
        n_groups = len(set(groups)) if groups is not None else None
        self.metadata = {
            "target": "candidate_net_pnl",
            "model_version": CANDIDATE_VALUE_VERSION,
            "feature_version": self.config.feature_version,
            "label_version": self.config.label_version,
            "candidate_feature_schema_hash": schema_hash,
            "n_train_rows": int(len(y_pnl)),
            "n_train_snapshots": n_groups,
            "snapshot_count": n_groups,
            "candidate_count": int(len(y_pnl)),
            "train_sessions": uniq,
            "fit_sessions": sorted(set(fit_s)),
            "calibration_sessions": sorted(set(cal_s)),
            "crossfit_config": {
                "inner_folds": xfit_cfg.inner_folds,
                "embargo_sessions": xfit_cfg.embargo_sessions,
                "min_train_sessions": xfit_cfg.min_train_sessions,
                "bias_weight": xfit_cfg.bias_weight,
                "downside_weight": xfit_cfg.downside_weight,
                "random_state": xfit_cfg.random_state,
            },
            "selected_estimator_per_head": {
                "expected_pnl": pnl_params.get("estimator"),
                "p_profit": "elasticnet_logistic",
                "quantiles": "hgb_quantile",
            },
            "selected_hyperparameters": {
                "expected_pnl": pnl_params,
                "p_profit": profit_params,
            },
            "profit_best_params": profit_params,
            "oof_metrics": {
                "expected_pnl": pnl_oof_metrics,
                "pnl_selection": pnl_diag,
                "quantiles": quantile_oof,
            },
            "calibration_metrics": cal_metrics,
            "calibration_artifact": (
                self.calibration_artifact.to_dict()
                if self.calibration_artifact is not None else None),
            "family_coverage": families,
            "outcome_coverage": outcome_coverage,
            "data_hash": data_hash,
            "mean_pnl": self._mean_pnl,
            "base_profit_rate": self._base_profit,
            "uncertainty": self._model_uncertainty,
            "insample_pnl_selection": False,
        }
        self.fitted = True
        return self

    def _select_pnl_oof(self, rows, y_pnl, sessions, groups, xfit_cfg):
        grid = _pnl_param_grid(self.config)
        inner = inner_folds_for_train(sorted(set(sessions)), xfit_cfg)
        if not inner:
            params = dict(grid[0])
            return params, {"note": "insufficient_sessions"}, {
                "note": "insufficient_sessions_for_inner_cv", "selected": params}

        scores: list[tuple[float, dict, dict]] = []
        for params in grid:
            fold_scores = []
            fold_metrics = []
            for inf in inner:
                tr = [i for i, s in enumerate(sessions)
                      if s in set(inf["train_sessions"])]
                va = [i for i, s in enumerate(sessions)
                      if s in set(inf["test_sessions"])]
                _assert_groups_intact(groups, tr, va)
                if not tr or not va:
                    continue
                vec = FeatureVectorizer().fit([rows[i] for i in tr])
                est = _pnl_estimator_factory(self.config, params)
                est.fit(vec.transform([rows[i] for i in tr]), y_pnl[tr])
                pred = np.asarray(
                    est.predict(vec.transform([rows[i] for i in va])),
                    dtype=float)
                m = regression_metrics(
                    y_pnl[va], pred, huber_delta=xfit_cfg.huber_delta)
                m["selection_score"] = regression_selection_score(m, xfit_cfg)
                fold_scores.append(m["selection_score"])
                fold_metrics.append(m)
            if fold_scores:
                scores.append((
                    float(np.mean(fold_scores)), dict(params),
                    {"fold_metrics": fold_metrics,
                     "mean_selection_score": float(np.mean(fold_scores))}))

        if not scores:
            params = dict(grid[0])
            return params, {"note": "no_scores"}, {
                "note": "inner_cv_produced_no_scores", "selected": params}

        scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
        best_score, best_params, best_detail = scores[0]
        # Aggregate OOF metrics for the winner
        agg = {
            "selection_score": best_score,
            "mae": float(np.mean([m["mae"] for m in best_detail["fold_metrics"]])),
            "huber": float(np.mean([m["huber"] for m in best_detail["fold_metrics"]])),
            "bias": float(np.mean([m["bias"] for m in best_detail["fold_metrics"]])),
            "downside_underprediction": float(np.mean([
                m["downside_underprediction"]
                for m in best_detail["fold_metrics"]])),
            "mse": float(np.mean([m["mse"] for m in best_detail["fold_metrics"]])),
        }
        diag = {
            "selection_metric": "huber_bias_downside",
            "best_selection_score": best_score,
            "n_param_candidates_scored": len(scores),
            "selected": best_params,
            "challengers_considered": sorted({
                p["estimator"] for _, p, _ in scores}),
        }
        return best_params, agg, diag

    def _fit_profit_oof(self, rows, y_profit, sessions, groups, xfit_cfg):
        uniq = sorted(set(sessions))
        grid = [{"C": c, "l1_ratio": l1}
                for c in self.config.c_grid
                for l1 in self.config.l1_ratio_grid]
        best_params = dict(grid[0])
        cal_metrics: dict = {"note": "no calibration; identity calibrator"}
        fit_s, cal_s = uniq, []
        inner = inner_folds_for_train(uniq, xfit_cfg)

        if len(uniq) >= 2 and len(np.unique(y_profit)) >= 2 and inner:
            scores = []
            for params in grid:
                losses = []
                for inf in inner:
                    tr = [i for i, s in enumerate(sessions)
                          if s in set(inf["train_sessions"])]
                    va = [i for i, s in enumerate(sessions)
                          if s in set(inf["test_sessions"])]
                    _assert_groups_intact(groups, tr, va)
                    if not tr or not va or len(np.unique(y_profit[tr])) < 2:
                        continue
                    vec = FeatureVectorizer().fit([rows[i] for i in tr])
                    clf = _elasticnet_classifier(self.config, params)
                    clf.fit(vec.transform([rows[i] for i in tr]), y_profit[tr])
                    p = clip_probability(
                        clf.predict_proba(
                            vec.transform([rows[i] for i in va]))[:, 1])
                    losses.append(log_loss_score(y_profit[va], p))
                if losses:
                    scores.append((float(np.mean(losses)), dict(params)))
            if scores:
                scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
                best_params = scores[0][1]

            pred = np.full(len(rows), np.nan)
            for inf in inner:
                tr = [i for i, s in enumerate(sessions)
                      if s in set(inf["train_sessions"])]
                va = [i for i, s in enumerate(sessions)
                      if s in set(inf["test_sessions"])]
                _assert_groups_intact(groups, tr, va)
                if not tr or not va:
                    continue
                if len(np.unique(y_profit[tr])) < 2:
                    pred[va] = float(np.mean(y_profit[tr]))
                    continue
                vec = FeatureVectorizer().fit([rows[i] for i in tr])
                clf = _elasticnet_classifier(self.config, best_params)
                clf.fit(vec.transform([rows[i] for i in tr]), y_profit[tr])
                pred[va] = clip_probability(
                    clf.predict_proba(
                        vec.transform([rows[i] for i in va]))[:, 1])
            covered = np.isfinite(pred)
            if covered.sum() >= 5 and len(np.unique(y_profit[covered])) >= 2:
                p_oof = pred[covered]
                y_oof = y_profit[covered]
                sess_oof = [sessions[i] for i in np.flatnonzero(covered)]
                cal_s = sorted(set(sess_oof))
                fit_s = sorted({s for inf in inner
                                for s in inf["train_sessions"]}) or uniq
                if self.config.calibration == "auto":
                    self.profit_calibrator, cal_diag = select_calibrator(
                        p_oof, y_oof, n_sessions=len(set(sess_oof)),
                        sessions=sess_oof,
                        embargo_sessions=self.config.embargo_sessions)
                else:
                    self.profit_calibrator = fit_calibrator(
                        p_oof, y_oof, self.config.calibration)
                    cal_diag = self.profit_calibrator.to_dict()
                p_cal = self.profit_calibrator.transform(p_oof)
                art = build_calibration_artifact(
                    self.profit_calibrator, p_oof, y_oof,
                    training_sessions=cal_s, diagnostics=cal_diag)
                self.calibration_artifact = art
                skill = brier_skill(y_oof, p_cal)
                cal_metrics = {
                    "n": int(len(y_oof)),
                    "brier_raw": art.brier_before,
                    "brier_calibrated": art.brier_after,
                    "brier_skill": skill,
                    "log_loss": art.log_loss_after,
                    "calibration_diag": cal_diag,
                    "calibration_artifact": art.to_dict(),
                    "crossfit": True,
                }
                self._model_uncertainty = float(
                    np.clip(1.0 - max(skill or 0.0, 0.0), 0.0, 1.0))
            else:
                self.profit_calibrator = IdentityCalibrator()
                self._model_uncertainty = 0.5
        else:
            self.profit_calibrator = IdentityCalibrator()
            self._model_uncertainty = 0.5

        return best_params, cal_metrics, fit_s, cal_s

    def _evaluate_quantiles_oof(self, rows, y_pnl, sessions, groups, xfit_cfg):
        inner = inner_folds_for_train(sorted(set(sessions)), xfit_cfg)
        if not inner:
            return {"note": "insufficient_sessions_for_quantile_oof"}

        pinballs = {q: [] for q in self.config.quantiles}
        coverages = []
        widths = []
        downside_miss = []
        crossings = []
        by_family: dict[str, list] = {}

        for inf in inner:
            tr = [i for i, s in enumerate(sessions)
                  if s in set(inf["train_sessions"])]
            va = [i for i, s in enumerate(sessions)
                  if s in set(inf["test_sessions"])]
            _assert_groups_intact(groups, tr, va)
            if not tr or not va:
                continue
            vec = FeatureVectorizer().fit([rows[i] for i in tr])
            X_tr = vec.transform([rows[i] for i in tr])
            X_va = vec.transform([rows[i] for i in va])
            raw = {}
            for q in self.config.quantiles:
                est = _quantile_regressor(self.config, q)
                est.fit(X_tr, y_pnl[tr])
                raw[q] = np.asarray(est.predict(X_va), dtype=float)
                pinballs[q].append(pinball_loss(y_pnl[va], raw[q], q))
            # Crossing before rearrangement
            crossed = np.mean(
                (raw[0.1] > raw[0.5]) | (raw[0.5] > raw[0.9])
                | (raw[0.1] > raw[0.9]))
            crossings.append(float(crossed))
            q10, q50, q90 = rearrange_quantiles(raw[0.1], raw[0.5], raw[0.9])
            coverages.append(interval_coverage(y_pnl[va], q10, q90))
            widths.append(float(np.mean(q90 - q10)))
            # Downside miss: realized below q10
            downside_miss.append(float(np.mean(y_pnl[va] < q10)))
            for j, idx in enumerate(va):
                fam = _family_of(rows[idx])
                by_family.setdefault(fam, []).append({
                    "y": float(y_pnl[idx]),
                    "q10": float(q10[j]),
                    "q90": float(q90[j]),
                })

        family_metrics = {}
        for fam, items in by_family.items():
            ys = np.array([it["y"] for it in items])
            lo = np.array([it["q10"] for it in items])
            hi = np.array([it["q90"] for it in items])
            family_metrics[fam] = {
                "n": len(items),
                "coverage": interval_coverage(ys, lo, hi),
                "width": float(np.mean(hi - lo)),
            }

        return {
            "pinball": {str(q): float(np.mean(v)) if v else None
                        for q, v in pinballs.items()},
            "interval_coverage": float(np.mean(coverages)) if coverages else None,
            "interval_width": float(np.mean(widths)) if widths else None,
            "downside_miss_rate": (
                float(np.mean(downside_miss)) if downside_miss else None),
            "quantile_crossing_rate_before_rearrangement": (
                float(np.mean(crossings)) if crossings else None),
            "by_option_family": family_metrics,
        }

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
