"""
prediction/models/regime_moe.py
===============================
Multiclass regime probability model (V3 Part 2 §12–§13, PR 9).

Baseline: multinomial logistic regression.
Challenger: HistGradientBoostingClassifier.

Calibration is one-vs-rest on cross-fitted (or held-out) raw class
probabilities, then renormalized so probabilities sum to 1 within 1e-6.

Downstream blending must use the full probability vector — dominant_regime
is a convenience label only.

Research / shadow only. No policy effect.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.calibration import (
    IdentityCalibrator,
    fit_calibrator,
    select_calibrator,
)
from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    brier_score,
    clip_probability,
    log_loss_score,
)
from prediction.regime_labels import REGIME_CLASSES

REGIME_MODEL_VERSION = "v3.0.0-regime"
_PROB_SUM_TOL = 1e-6


@dataclass(frozen=True)
class RegimeProbabilities:
    long_gamma_pin: float
    short_gamma_trend: float
    flip_transition: float
    volatility_expansion: float
    uncertainty: float
    dominant_regime: str
    class_support: dict[str, float]
    calibrated: bool
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "long_gamma_pin": self.long_gamma_pin,
            "short_gamma_trend": self.short_gamma_trend,
            "flip_transition": self.flip_transition,
            "volatility_expansion": self.volatility_expansion,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RegimeModelConfig:
    estimator: str = "multinomial"  # "multinomial" | "hgb"
    classes: tuple[str, ...] = REGIME_CLASSES
    c_grid: tuple = (0.1, 1.0, 10.0)
    max_iter: int = 2000
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 3
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 20
    calibration: str = "auto"  # auto | sigmoid | isotonic | identity
    inner_folds: int = 3
    embargo_sessions: int = 1
    min_train_sessions: int = 4
    min_validation_sessions: int = 2
    random_state: int = RANDOM_STATE
    minimum_sessions: int = 40
    minimum_effective_sessions: int = 20
    minimum_rows: int = 500


def normalized_entropy(probs: Sequence[float], *, epsilon: float = 1e-12) -> float:
    """Entropy / log(K) in [0, 1]."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        return 1.0
    p = p / s
    k = len(p)
    if k <= 1:
        return 0.0
    ent = -float(np.sum(p * np.log(p + epsilon)))
    return float(ent / math.log(k))


def renormalize_probs(
    probs: dict[str, float],
    classes: Sequence[str] = REGIME_CLASSES,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Clip invalid values and renormalize to sum ≈ 1."""
    raw = np.array([max(0.0, float(probs.get(c, 0.0))) for c in classes],
                   dtype=float)
    if not np.isfinite(raw).all() or raw.sum() <= epsilon:
        raw = np.ones(len(classes), dtype=float) / len(classes)
    else:
        raw = raw / raw.sum()
    out = {c: float(raw[i]) for i, c in enumerate(classes)}
    assert abs(sum(out.values()) - 1.0) <= _PROB_SUM_TOL + 1e-9
    return out


def _make_estimator(cfg: RegimeModelConfig):
    if cfg.estimator == "multinomial":
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        kw = dict(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
        # sklearn >= 1.5 deprecates multi_class; keep for older CI
        ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if ver >= (1, 7):
            kw.pop("multi_class", None)
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(**kw)),
        ])
    if cfg.estimator == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=cfg.hgb_learning_rate,
            max_depth=cfg.hgb_max_depth,
            max_leaf_nodes=cfg.hgb_max_leaf_nodes,
            min_samples_leaf=cfg.hgb_min_samples_leaf,
            random_state=cfg.random_state,
        )
    raise ValueError(f"unknown estimator {cfg.estimator!r}")


def _session_folds(
    sessions: Sequence[str],
    n_folds: int,
    embargo: int,
) -> list[tuple[list[str], list[str]]]:
    uniq = sorted(set(sessions))
    n = len(uniq)
    if n < 4 or n_folds < 2:
        return []
    fold_size = max(1, n // n_folds)
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = n if i == n_folds - 1 else (i + 1) * fold_size
        val = uniq[val_start:val_end]
        train = [s for j, s in enumerate(uniq)
                 if not (val_start - embargo <= j < val_end + embargo)]
        if len(train) >= 2 and len(val) >= 1:
            folds.append((train, val))
    return folds


@dataclass
class RegimeProbabilityModel:
    """Fit / predict multiclass regime probabilities."""

    cfg: RegimeModelConfig = field(default_factory=RegimeModelConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = field(default=None, repr=False)
    calibrators: dict = field(default_factory=dict)
    class_support: dict = field(default_factory=dict)
    class_to_index: dict = field(default_factory=dict)
    fitted: bool = False
    model_version: str = REGIME_MODEL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def fit(
        self,
        rows: Sequence[dict],
        labels: Sequence[str],
        sessions: Sequence[str],
    ) -> "RegimeProbabilityModel":
        classes = tuple(self.cfg.classes)
        y_raw = list(labels)
        keep = [i for i, lab in enumerate(y_raw) if lab in classes]
        if len(keep) < 4:
            raise ValueError("insufficient labeled rows for regime model")
        rows_k = [rows[i] for i in keep]
        y_str = [y_raw[i] for i in keep]
        sess_k = [sessions[i] for i in keep]

        self.vectorizer = FeatureVectorizer().fit(rows_k)
        X = self.vectorizer.transform(rows_k)
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        y = np.array([self.class_to_index[c] for c in y_str], dtype=int)

        support = {c: 0.0 for c in classes}
        sess_support: dict[str, set] = {c: set() for c in classes}
        for lab, sess in zip(y_str, sess_k):
            support[lab] += 1.0
            sess_support[lab].add(sess)
        self.class_support = {
            c: float(len(sess_support[c])) for c in classes
        }
        self.diagnostics["row_support"] = support
        self.diagnostics["n_sessions"] = len(set(sess_k))
        self.diagnostics["n_rows"] = len(rows_k)

        # OOF raw probabilities for calibration
        oof = np.full((len(rows_k), len(classes)), np.nan)
        folds = _session_folds(
            sess_k, self.cfg.inner_folds, self.cfg.embargo_sessions)
        if folds:
            for train_s, val_s in folds:
                tr = [i for i, s in enumerate(sess_k) if s in set(train_s)]
                va = [i for i, s in enumerate(sess_k) if s in set(val_s)]
                if len(tr) < 2 or not va:
                    continue
                if len(set(y[tr])) < 2:
                    continue
                est = _make_estimator(self.cfg)
                est.fit(X[tr], y[tr])
                proba = _predict_proba_aligned(est, X[va], classes, self.class_to_index)
                oof[va] = proba
        else:
            # Tiny-sample fallback: use in-sample only for calibrator
            # training after a 50/50 session split when possible.
            uniq = sorted(set(sess_k))
            mid = max(1, len(uniq) // 2)
            tr_s, va_s = set(uniq[:mid]), set(uniq[mid:])
            tr = [i for i, s in enumerate(sess_k) if s in tr_s]
            va = [i for i, s in enumerate(sess_k) if s in va_s]
            if len(tr) >= 2 and va and len(set(y[tr])) >= 2:
                est = _make_estimator(self.cfg)
                est.fit(X[tr], y[tr])
                oof[va] = _predict_proba_aligned(
                    est, X[va], classes, self.class_to_index)

        # Final estimator on all labeled rows
        self.estimator = _make_estimator(self.cfg)
        self.estimator.fit(X, y)

        # One-vs-rest calibrators on OOF rows only
        self.calibrators = {}
        mask = np.isfinite(oof[:, 0])
        calibrated = False
        if mask.sum() >= 4:
            for ci, cname in enumerate(classes):
                p_raw = oof[mask, ci]
                y_bin = (y[mask] == ci).astype(int)
                if len(np.unique(y_bin)) < 2:
                    cal = IdentityCalibrator().fit(p_raw, y_bin)
                elif self.cfg.calibration == "auto":
                    sess_oof = [sess_k[i] for i, m in enumerate(mask) if m]
                    cal, _ = select_calibrator(
                        p_raw, y_bin,
                        n_sessions=len(set(sess_oof)),
                        sessions=sess_oof,
                    )
                else:
                    cal = fit_calibrator(p_raw, y_bin, method=self.cfg.calibration)
                self.calibrators[cname] = cal
            calibrated = True
        else:
            for cname in classes:
                self.calibrators[cname] = IdentityCalibrator()

        self.fitted = True
        self.diagnostics["calibrated"] = calibrated
        self.diagnostics["oof_rows"] = int(mask.sum())
        self.diagnostics["estimator"] = self.cfg.estimator
        return self

    def predict_proba_raw(self, rows: Sequence[dict]) -> np.ndarray:
        self._require_fit()
        X = self.vectorizer.transform(rows)
        return _predict_proba_aligned(
            self.estimator, X, self.cfg.classes, self.class_to_index)

    def predict_proba(self, rows: Sequence[dict]) -> np.ndarray:
        """Calibrated + renormalized class probabilities (N, K)."""
        raw = self.predict_proba_raw(rows)
        out = np.zeros_like(raw)
        for i, cname in enumerate(self.cfg.classes):
            cal = self.calibrators.get(cname) or IdentityCalibrator()
            out[:, i] = cal.transform(raw[:, i])
        # Renormalize each row
        out = np.clip(out, 0.0, 1.0)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        out = out / row_sums
        return out

    def predict(self, row: dict) -> RegimeProbabilities:
        """Single-observation RegimeProbabilities contract (§12.7)."""
        proba = self.predict_proba([row])[0]
        classes = self.cfg.classes
        probs = {c: float(proba[i]) for i, c in enumerate(classes)}
        probs = renormalize_probs(probs, classes)
        dominant = max(probs, key=probs.get)
        ent = normalized_entropy([probs[c] for c in classes])
        # Support uncertainty: low class support → higher uncertainty
        supports = [float(self.class_support.get(c, 0.0)) for c in classes]
        max_sup = max(supports) if supports else 0.0
        support_pen = 0.0
        if max_sup < self.cfg.minimum_effective_sessions:
            support_pen = 1.0 - (max_sup / max(self.cfg.minimum_effective_sessions, 1))
        uncertainty = float(clip_probability(0.7 * ent + 0.3 * support_pen))
        return RegimeProbabilities(
            long_gamma_pin=probs["long_gamma_pin"],
            short_gamma_trend=probs["short_gamma_trend"],
            flip_transition=probs["flip_transition"],
            volatility_expansion=probs["volatility_expansion"],
            uncertainty=uncertainty,
            dominant_regime=dominant,
            class_support={c: float(self.class_support.get(c, 0.0))
                           for c in classes},
            calibrated=bool(self.diagnostics.get("calibrated")),
            model_version=self.model_version,
            diagnostics={
                "entropy": ent,
                "support_penalty": support_pen,
                "estimator": self.cfg.estimator,
            },
        )

    def evaluate(
        self,
        rows: Sequence[dict],
        labels: Sequence[str],
    ) -> dict:
        """Multiclass log loss, macro Brier, confusion support."""
        classes = self.cfg.classes
        keep = [i for i, lab in enumerate(labels) if lab in classes]
        if not keep:
            return {"n": 0}
        rows_k = [rows[i] for i in keep]
        y = np.array([self.class_to_index[labels[i]] for i in keep])
        proba = self.predict_proba(rows_k)
        # Multiclass log loss
        ll = 0.0
        for i, yi in enumerate(y):
            ll += -math.log(max(proba[i, yi], 1e-12))
        ll /= len(y)
        # Per-class Brier + macro
        briers = {}
        for ci, cname in enumerate(classes):
            y_bin = (y == ci).astype(float)
            briers[cname] = float(brier_score(proba[:, ci], y_bin))
        macro_brier = float(np.mean(list(briers.values())))
        # Confusion
        pred = proba.argmax(axis=1)
        confusion = {
            f"true_{classes[ti]}_pred_{classes[pj]}": int(
                ((y == ti) & (pred == pj)).sum())
            for ti in range(len(classes))
            for pj in range(len(classes))
        }
        return {
            "n": int(len(y)),
            "log_loss": float(ll),
            "brier_by_class": briers,
            "macro_brier": macro_brier,
            "confusion": confusion,
        }

    def _require_fit(self) -> None:
        if not self.fitted or self.estimator is None:
            raise RuntimeError("RegimeProbabilityModel used before fit")


def _predict_proba_aligned(est, X, classes, class_to_index) -> np.ndarray:
    """Map estimator predict_proba columns onto the canonical class order."""
    raw = est.predict_proba(X)
    # sklearn classes_ are integer labels we trained with
    est_classes = list(getattr(est, "classes_", range(raw.shape[1])))
    # If Pipeline, dig into final estimator
    if hasattr(est, "named_steps") and "lr" in est.named_steps:
        est_classes = list(est.named_steps["lr"].classes_)
    elif hasattr(est, "classes_"):
        est_classes = list(est.classes_)
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for col, cls_idx in enumerate(est_classes):
        # cls_idx is int index into classes
        if isinstance(cls_idx, (int, np.integer)):
            out[:, int(cls_idx)] = raw[:, col]
        elif cls_idx in class_to_index:
            out[:, class_to_index[cls_idx]] = raw[:, col]
    # Rows that got no probability mass (unseen class) → uniform
    sums = out.sum(axis=1)
    missing = sums <= 0
    if missing.any():
        out[missing] = 1.0 / len(classes)
        sums = out.sum(axis=1)
    out = out / sums.reshape(-1, 1)
    return out
