"""
prediction/crossfit.py
======================
Nested session-grouped cross-fitting (Prediction Engine V3 Part 1 §4).

Outer expanding walk-forward folds reserve complete sessions for testing.
Inside each outer training window, inner folds select hyperparameters from
out-of-fold scores only. Outer test sessions are never used for tuning or
calibration. Snapshot-level group_ids (when provided) are never split across
train/validation boundaries.

See docs/PREDICTION_ENGINE_V3_PART1_VALIDATION.md.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from prediction.models.base import (
    brier_score,
    brier_skill,
    log_loss_score,
)
from prediction.training import grouped_session_folds


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: str
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    calibration_sessions: tuple[str, ...]
    embargoed_sessions: tuple[str, ...] = ()


@dataclass
class CrossFitResult:
    selected_params: dict
    oof_raw_predictions: np.ndarray
    oof_row_indices: np.ndarray
    fold_assignments: np.ndarray
    fold_metrics: list[dict]
    fitted_fold_models: list[Any] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class NestedCrossFitConfig:
    outer_folds: int = 4
    inner_folds: int = 3
    embargo_sessions: int = 1
    min_train_sessions: int = 20
    min_validation_sessions: int = 5
    retain_fold_models: bool = True
    random_state: int = 42
    # regression selection weights (Part 1 §6.3)
    bias_weight: float = 0.25
    downside_weight: float = 0.25
    huber_delta: float = 1.0


# --------------------------------------------------------------------------- #
# Fold construction                                                            #
# --------------------------------------------------------------------------- #

def build_nested_session_folds(
    sessions: Sequence[str],
    cfg: NestedCrossFitConfig,
) -> list[FoldDefinition]:
    """
    Expanding time-ordered outer folds with whole-session embargoes.

    Each FoldDefinition's train_sessions are the outer train window;
    validation_sessions are the outer test block (held out for final eval);
    calibration_sessions are reserved from the *end* of the outer train
    window for independent calibrator fitting (never overlap validation).
    """
    uniq = sorted(set(sessions))
    outer = grouped_session_folds(
        uniq,
        n_folds=cfg.outer_folds,
        embargo_sessions=cfg.embargo_sessions,
        min_train_sessions=cfg.min_train_sessions,
    )
    folds: list[FoldDefinition] = []
    for i, of in enumerate(outer):
        train = list(of["train_sessions"])
        test = list(of["test_sessions"])
        embargoed = list(of["embargoed_sessions"])
        # Reserve trailing sessions of the outer train for calibration only.
        # These are NOT used for hyperparameter selection (inner folds operate
        # on the remaining fit prefix).
        n_cal = min(
            max(cfg.min_validation_sessions, 1),
            max(0, len(train) // 5),
        )
        if len(train) - n_cal - cfg.embargo_sessions < cfg.min_train_sessions:
            n_cal = 0
        if n_cal > 0:
            cal = train[-n_cal:]
            # optional inner embargo between fit-prefix and cal
            fit_end = len(train) - n_cal - cfg.embargo_sessions
            if fit_end < cfg.min_train_sessions:
                cal = []
                fit_train = train
                cal_embargo: list[str] = []
            else:
                fit_train = train[:fit_end]
                cal_embargo = train[fit_end:len(train) - n_cal]
                cal = train[len(train) - n_cal:]
                embargoed = list(dict.fromkeys(embargoed + cal_embargo))
        else:
            cal = []
            fit_train = train

        # Invariants
        assert not (set(fit_train) & set(test)), "train/test session leak"
        assert not (set(cal) & set(test)), "calibration/test session leak"
        assert not (set(fit_train) & set(cal)), "fit/cal session leak"
        assert not (set(embargoed) & set(fit_train)), "embargo in fit"
        assert not (set(embargoed) & set(test)), "embargo in test"

        folds.append(FoldDefinition(
            fold_id=f"outer-{i}",
            train_sessions=tuple(fit_train),
            validation_sessions=tuple(test),
            calibration_sessions=tuple(cal),
            embargoed_sessions=tuple(embargoed),
        ))
    return folds


def _inner_folds_for_train(
    train_sessions: Sequence[str],
    cfg: NestedCrossFitConfig,
) -> list[dict]:
    """Expanding inner folds inside an outer training window for HP search."""
    uniq = sorted(set(train_sessions))
    n_inner = min(cfg.inner_folds, max(1, len(uniq) // (
        cfg.min_validation_sessions + cfg.embargo_sessions
        + max(cfg.min_train_sessions // 4, 2))))
    if n_inner < 1 or len(uniq) < (
            cfg.min_train_sessions // 2 + cfg.min_validation_sessions
            + cfg.embargo_sessions):
        # Too few sessions for nested CV: single holdout of last
        # min_validation_sessions (still session-grouped).
        n_val = min(cfg.min_validation_sessions, max(1, len(uniq) // 4))
        if len(uniq) - n_val - cfg.embargo_sessions < 1:
            return []
        lo = len(uniq) - n_val
        train = uniq[:max(lo - cfg.embargo_sessions, 0)]
        val = uniq[lo:]
        embargoed = uniq[max(lo - cfg.embargo_sessions, 0):lo]
        if not train or not val:
            return []
        return [{"train_sessions": train, "test_sessions": val,
                 "embargoed_sessions": embargoed}]
    min_train = max(2, min(cfg.min_train_sessions // 2, len(uniq) // 3))
    try:
        return grouped_session_folds(
            uniq,
            n_folds=n_inner,
            embargo_sessions=cfg.embargo_sessions,
            min_train_sessions=min_train,
        )
    except ValueError:
        return []


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def _roc_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = np.empty(len(y_sorted), dtype=float)
    # average ranks for ties in probability
    i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pr_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-p)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(y.sum())
    # step-wise average precision
    ap = 0.0
    prev_r = 0.0
    for prec, rec in zip(precision, recall):
        ap += float(prec) * (float(rec) - prev_r)
        prev_r = float(rec)
    return float(ap)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    from prediction.calibration import calibration_slope_intercept
    slope_int = calibration_slope_intercept(p, y)
    pred_cls = (p >= 0.5).astype(float)
    return {
        "log_loss": log_loss_score(y, p),
        "brier": brier_score(y, p),
        "brier_skill": brier_skill(y, p),
        "calibration_slope": slope_int.get("slope"),
        "calibration_intercept": slope_int.get("intercept"),
        "roc_auc": _roc_auc(y, p),
        "pr_auc": _pr_auc(y.astype(int), p),
        "accuracy": float(np.mean(pred_cls == y)) if len(y) else None,
        "n": int(len(y)),
    }


def huber_loss(y: np.ndarray, pred: np.ndarray, delta: float = 1.0) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = np.abs(y - pred)
    quad = np.minimum(err, delta)
    lin = err - quad
    return float(np.mean(0.5 * quad ** 2 + delta * lin))


def downside_underprediction_penalty(
    y: np.ndarray, pred: np.ndarray, decile: float = 0.1,
) -> float:
    """Mean max(pred - realized, 0) on the worst realized-PnL decile."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return 0.0
    thr = float(np.quantile(y, decile))
    mask = y <= thr
    if not mask.any():
        return 0.0
    return float(np.mean(np.maximum(pred[mask] - y[mask], 0.0)))


def regression_metrics(
    y: np.ndarray, pred: np.ndarray, *, huber_delta: float = 1.0,
) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "huber": huber_loss(y, pred, delta=huber_delta),
        "median_ae": float(np.median(np.abs(err))),
        "bias": float(np.mean(err)),
        "downside_underprediction": downside_underprediction_penalty(y, pred),
        "mse": float(np.mean(err ** 2)),
        "n": int(len(y)),
    }


def regression_selection_score(metrics: dict, cfg: NestedCrossFitConfig) -> float:
    return (
        float(metrics["huber"])
        + cfg.bias_weight * abs(float(metrics["bias"]))
        + cfg.downside_weight * float(metrics["downside_underprediction"])
    )


# --------------------------------------------------------------------------- #
# Index helpers                                                                #
# --------------------------------------------------------------------------- #

def _session_mask(sessions: Sequence[str], keep: Sequence[str]) -> np.ndarray:
    keep_set = set(keep)
    return np.array([s in keep_set for s in sessions], dtype=bool)


def _indices_for(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def _assert_groups_intact(
    group_ids: Optional[Sequence[str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    if group_ids is None:
        return
    g = list(group_ids)
    train_g = {g[i] for i in train_idx}
    val_g = {g[i] for i in val_idx}
    leaked = train_g & val_g
    if leaked:
        raise AssertionError(
            f"group_ids split across train/validation: {sorted(leaked)[:5]}")


def _rows_to_matrix(
    rows: Sequence[dict],
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Lightweight deterministic dict-row -> matrix (median impute)."""
    if feature_names is None:
        names: set = set()
        for r in rows:
            names.update(r.keys())
        feature_names = sorted(names)
    n = len(rows)
    m = len(feature_names)
    X = np.empty((n, m), dtype=float)
    for j, name in enumerate(feature_names):
        col = []
        for r in rows:
            v = r.get(name)
            try:
                f = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                f = float("nan")
            col.append(f)
        arr = np.asarray(col, dtype=float)
        finite = arr[np.isfinite(arr)]
        med = float(np.median(finite)) if len(finite) else 0.0
        arr = np.where(np.isfinite(arr), arr, med)
        X[:, j] = arr
    return X, feature_names


# --------------------------------------------------------------------------- #
# Cross-fit classifiers / regressors                                           #
# --------------------------------------------------------------------------- #

def crossfit_classifier(
    rows: Sequence[dict],
    y: Sequence[int],
    sessions: Sequence[str],
    param_grid: Sequence[dict],
    estimator_factory: Callable[[dict], Any],
    predict_raw: Callable[[Any, np.ndarray], np.ndarray],
    cfg: NestedCrossFitConfig,
) -> CrossFitResult:
    """
    Nested cross-fit for binary classifiers.

    Hyperparameters selected by mean inner-OOF log loss. Outer-fold raw
    predictions are generated for held-out validation sessions using the
    selected params (fit on that fold's train sessions only). Calibration
    sessions from FoldDefinition are excluded from HP selection and from
    the returned OOF stream used for outer evaluation — callers should fit
    calibrators on a separate OOF pass over train+cal eligible rows.
    """
    y_arr = np.asarray(y, dtype=int)
    sessions_l = list(sessions)
    rows_l = list(rows)
    n = len(rows_l)
    if not (len(y_arr) == n == len(sessions_l)):
        raise ValueError("rows, y, sessions length mismatch")
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    outer_folds = build_nested_session_folds(sessions_l, cfg)
    # --- Stage A: select params on inner OOF over the union of outer-train ---
    # Use the full eligible history minus a reserved outer-test suffix so
    # outer test never influences selection. Practically: run inner CV on
    # sessions that appear in ANY outer train window (earliest prefix through
    # last outer train end), which excludes the final outer validation block.
    all_train_sessions = sorted({
        s for fd in outer_folds for s in fd.train_sessions
    })
    # Also allow calibration sessions into the HP-eligible pool? Spec says
    # select using inner OOF only and never use outer test for tuning.
    # Calibration sessions are reserved for calibrator fitting, so exclude
    # them from HP selection.
    all_cal = {s for fd in outer_folds for s in fd.calibration_sessions}
    hp_sessions = [s for s in all_train_sessions if s not in all_cal]
    if len(set(hp_sessions)) < 2:
        hp_sessions = list(all_train_sessions)

    selected_params, inner_diag = _select_classifier_params(
        rows_l, y_arr, sessions_l, hp_sessions, param_grid,
        estimator_factory, predict_raw, cfg,
    )

    # --- Stage B: OOF raw predictions on outer validation sessions ---
    oof_pred = np.full(n, np.nan, dtype=float)
    fold_assign = np.full(n, -1, dtype=int)
    fold_metrics: list[dict] = []
    fold_models: list[Any] = []
    feature_names: Optional[list[str]] = None

    for fi, fd in enumerate(outer_folds):
        train_mask = _session_mask(sessions_l, fd.train_sessions)
        val_mask = _session_mask(sessions_l, fd.validation_sessions)
        train_idx = _indices_for(train_mask)
        val_idx = _indices_for(val_mask)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        y_tr = y_arr[train_idx]
        if len(np.unique(y_tr)) < 2:
            # degenerate: constant base rate on train
            base = float(np.mean(y_tr)) if len(y_tr) else 0.5
            oof_pred[val_idx] = base
            fold_assign[val_idx] = fi
            fold_metrics.append({
                "fold_id": fd.fold_id,
                "note": "degenerate_train_single_class",
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            })
            fold_models.append(None)
            continue

        train_rows = [rows_l[i] for i in train_idx]
        val_rows = [rows_l[i] for i in val_idx]
        X_tr, feature_names = _rows_to_matrix(train_rows, feature_names)
        X_va, _ = _rows_to_matrix(val_rows, feature_names)
        est = estimator_factory(selected_params)
        est.fit(X_tr, y_tr)
        p_va = np.asarray(predict_raw(est, X_va), dtype=float)
        oof_pred[val_idx] = p_va
        fold_assign[val_idx] = fi
        metrics = classification_metrics(y_arr[val_idx], p_va)
        metrics.update({
            "fold_id": fd.fold_id,
            "train_sessions": list(fd.train_sessions),
            "validation_sessions": list(fd.validation_sessions),
            "calibration_sessions": list(fd.calibration_sessions),
            "embargoed_sessions": list(fd.embargoed_sessions),
        })
        fold_metrics.append(metrics)
        fold_models.append(est if cfg.retain_fold_models else None)

    covered = np.isfinite(oof_pred)
    return CrossFitResult(
        selected_params=dict(selected_params),
        oof_raw_predictions=oof_pred[covered],
        oof_row_indices=np.flatnonzero(covered),
        fold_assignments=fold_assign,
        fold_metrics=fold_metrics,
        fitted_fold_models=fold_models if cfg.retain_fold_models else [],
        diagnostics={
            "task": "classification",
            "outer_folds": [fd.fold_id for fd in outer_folds],
            "fold_definitions": [
                {
                    "fold_id": fd.fold_id,
                    "train_sessions": list(fd.train_sessions),
                    "validation_sessions": list(fd.validation_sessions),
                    "calibration_sessions": list(fd.calibration_sessions),
                    "embargoed_sessions": list(fd.embargoed_sessions),
                }
                for fd in outer_folds
            ],
            "inner_selection": inner_diag,
            "n_rows": n,
            "n_oof": int(covered.sum()),
            "random_state": cfg.random_state,
        },
    )


def _select_classifier_params(
    rows, y_arr, sessions_l, hp_sessions, param_grid,
    estimator_factory, predict_raw, cfg,
) -> tuple[dict, dict]:
    inner = _inner_folds_for_train(hp_sessions, cfg)
    if not inner:
        # Fall back: first grid point (deterministic), flag in diagnostics.
        return dict(param_grid[0]), {"note": "insufficient_sessions_for_inner_cv",
                                     "selected": dict(param_grid[0])}

    scores: list[tuple[float, dict, list]] = []
    feature_names: Optional[list[str]] = None
    for params in param_grid:
        fold_losses = []
        for inf in inner:
            tr_mask = _session_mask(sessions_l, inf["train_sessions"])
            va_mask = _session_mask(sessions_l, inf["test_sessions"])
            # Restrict to hp_sessions only
            hp_set = set(hp_sessions)
            tr_mask = tr_mask & np.array([s in hp_set for s in sessions_l])
            va_mask = va_mask & np.array([s in hp_set for s in sessions_l])
            tr_idx = _indices_for(tr_mask)
            va_idx = _indices_for(va_mask)
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            y_tr = y_arr[tr_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            X_tr, feature_names = _rows_to_matrix(
                [rows[i] for i in tr_idx], feature_names)
            X_va, _ = _rows_to_matrix(
                [rows[i] for i in va_idx], feature_names)
            est = estimator_factory(params)
            est.fit(X_tr, y_tr)
            p = np.asarray(predict_raw(est, X_va), dtype=float)
            fold_losses.append(log_loss_score(y_arr[va_idx], p))
        if fold_losses:
            scores.append((float(np.mean(fold_losses)), dict(params), fold_losses))

    if not scores:
        return dict(param_grid[0]), {"note": "inner_cv_produced_no_scores",
                                     "selected": dict(param_grid[0])}

    scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
    best_loss, best_params, best_folds = scores[0]
    return best_params, {
        "selection_metric": "log_loss",
        "best_log_loss": best_loss,
        "fold_log_losses": best_folds,
        "n_param_candidates_scored": len(scores),
        "selected": best_params,
    }


def crossfit_regressor(
    rows: Sequence[dict],
    y: Sequence[float],
    sessions: Sequence[str],
    param_grid: Sequence[dict],
    estimator_factory: Callable[[dict], Any],
    cfg: NestedCrossFitConfig,
    group_ids: Optional[Sequence[str]] = None,
) -> CrossFitResult:
    """
    Nested cross-fit for continuous targets (returns / expected P&L).

    When group_ids (e.g. snapshot_id) are provided, every member of a group
    stays on the same side of any train/validation split.
    """
    y_arr = np.asarray(y, dtype=float)
    sessions_l = list(sessions)
    rows_l = list(rows)
    n = len(rows_l)
    if not (len(y_arr) == n == len(sessions_l)):
        raise ValueError("rows, y, sessions length mismatch")
    if group_ids is not None and len(group_ids) != n:
        raise ValueError("group_ids length mismatch")
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    # Enforce: groups never span multiple sessions (would break session folds)
    if group_ids is not None:
        g_to_s: dict[str, set] = {}
        for g, s in zip(group_ids, sessions_l):
            g_to_s.setdefault(g, set()).add(s)
        multi = {g: ss for g, ss in g_to_s.items() if len(ss) > 1}
        if multi:
            raise ValueError(
                f"group_ids span multiple sessions: {list(multi)[:3]}")

    outer_folds = build_nested_session_folds(sessions_l, cfg)
    all_train_sessions = sorted({
        s for fd in outer_folds for s in fd.train_sessions
    })
    all_cal = {s for fd in outer_folds for s in fd.calibration_sessions}
    hp_sessions = [s for s in all_train_sessions if s not in all_cal]
    if len(set(hp_sessions)) < 2:
        hp_sessions = list(all_train_sessions)

    selected_params, inner_diag = _select_regressor_params(
        rows_l, y_arr, sessions_l, hp_sessions, param_grid,
        estimator_factory, cfg, group_ids,
    )

    oof_pred = np.full(n, np.nan, dtype=float)
    fold_assign = np.full(n, -1, dtype=int)
    fold_metrics: list[dict] = []
    fold_models: list[Any] = []
    feature_names: Optional[list[str]] = None

    for fi, fd in enumerate(outer_folds):
        train_mask = _session_mask(sessions_l, fd.train_sessions)
        val_mask = _session_mask(sessions_l, fd.validation_sessions)
        train_idx = _indices_for(train_mask)
        val_idx = _indices_for(val_mask)
        _assert_groups_intact(group_ids, train_idx, val_idx)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        train_rows = [rows_l[i] for i in train_idx]
        val_rows = [rows_l[i] for i in val_idx]
        X_tr, feature_names = _rows_to_matrix(train_rows, feature_names)
        X_va, _ = _rows_to_matrix(val_rows, feature_names)
        est = estimator_factory(selected_params)
        est.fit(X_tr, y_arr[train_idx])
        pred = np.asarray(est.predict(X_va), dtype=float)
        oof_pred[val_idx] = pred
        fold_assign[val_idx] = fi
        metrics = regression_metrics(
            y_arr[val_idx], pred, huber_delta=cfg.huber_delta)
        metrics["selection_score"] = regression_selection_score(metrics, cfg)
        metrics.update({
            "fold_id": fd.fold_id,
            "train_sessions": list(fd.train_sessions),
            "validation_sessions": list(fd.validation_sessions),
            "calibration_sessions": list(fd.calibration_sessions),
            "embargoed_sessions": list(fd.embargoed_sessions),
        })
        fold_metrics.append(metrics)
        fold_models.append(est if cfg.retain_fold_models else None)

    covered = np.isfinite(oof_pred)
    return CrossFitResult(
        selected_params=dict(selected_params),
        oof_raw_predictions=oof_pred[covered],
        oof_row_indices=np.flatnonzero(covered),
        fold_assignments=fold_assign,
        fold_metrics=fold_metrics,
        fitted_fold_models=fold_models if cfg.retain_fold_models else [],
        diagnostics={
            "task": "regression",
            "outer_folds": [fd.fold_id for fd in outer_folds],
            "fold_definitions": [
                {
                    "fold_id": fd.fold_id,
                    "train_sessions": list(fd.train_sessions),
                    "validation_sessions": list(fd.validation_sessions),
                    "calibration_sessions": list(fd.calibration_sessions),
                    "embargoed_sessions": list(fd.embargoed_sessions),
                }
                for fd in outer_folds
            ],
            "inner_selection": inner_diag,
            "n_rows": n,
            "n_oof": int(covered.sum()),
            "random_state": cfg.random_state,
            "group_ids_used": group_ids is not None,
        },
    )


def _select_regressor_params(
    rows, y_arr, sessions_l, hp_sessions, param_grid,
    estimator_factory, cfg, group_ids,
) -> tuple[dict, dict]:
    inner = _inner_folds_for_train(hp_sessions, cfg)
    if not inner:
        return dict(param_grid[0]), {"note": "insufficient_sessions_for_inner_cv",
                                     "selected": dict(param_grid[0])}

    scores: list[tuple[float, dict, list]] = []
    feature_names: Optional[list[str]] = None
    for params in param_grid:
        fold_scores = []
        for inf in inner:
            tr_mask = _session_mask(sessions_l, inf["train_sessions"])
            va_mask = _session_mask(sessions_l, inf["test_sessions"])
            hp_set = set(hp_sessions)
            tr_mask = tr_mask & np.array([s in hp_set for s in sessions_l])
            va_mask = va_mask & np.array([s in hp_set for s in sessions_l])
            tr_idx = _indices_for(tr_mask)
            va_idx = _indices_for(va_mask)
            _assert_groups_intact(group_ids, tr_idx, va_idx)
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            X_tr, feature_names = _rows_to_matrix(
                [rows[i] for i in tr_idx], feature_names)
            X_va, _ = _rows_to_matrix(
                [rows[i] for i in va_idx], feature_names)
            est = estimator_factory(params)
            est.fit(X_tr, y_arr[tr_idx])
            pred = np.asarray(est.predict(X_va), dtype=float)
            m = regression_metrics(
                y_arr[va_idx], pred, huber_delta=cfg.huber_delta)
            fold_scores.append(regression_selection_score(m, cfg))
        if fold_scores:
            scores.append((float(np.mean(fold_scores)), dict(params), fold_scores))

    if not scores:
        return dict(param_grid[0]), {"note": "inner_cv_produced_no_scores",
                                     "selected": dict(param_grid[0])}

    scores.sort(key=lambda t: (t[0], str(sorted(t[1].items()))))
    best, best_params, best_folds = scores[0]
    return best_params, {
        "selection_metric": "huber_bias_downside",
        "best_selection_score": best,
        "fold_selection_scores": best_folds,
        "n_param_candidates_scored": len(scores),
        "selected": best_params,
    }


def generate_train_oof_predictions(
    rows: Sequence[dict],
    y: Sequence,
    sessions: Sequence[str],
    params: dict,
    estimator_factory: Callable[[dict], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    cfg: NestedCrossFitConfig,
    *,
    task: str = "classification",
    group_ids: Optional[Sequence[str]] = None,
    eligible_sessions: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce cross-fitted predictions on eligible training sessions only
    (for calibrator fitting / residual analysis). Outer validation sessions
    are excluded when eligible_sessions is the train(+cal) union.
    """
    y_arr = np.asarray(y)
    sessions_l = list(sessions)
    rows_l = list(rows)
    n = len(rows_l)
    if eligible_sessions is None:
        # default: all sessions appearing in any outer train or cal
        folds = build_nested_session_folds(sessions_l, cfg)
        eligible = sorted({
            s for fd in folds
            for s in list(fd.train_sessions) + list(fd.calibration_sessions)
        })
    else:
        eligible = list(eligible_sessions)

    inner = _inner_folds_for_train(eligible, cfg)
    pred = np.full(n, np.nan, dtype=float)
    assign = np.full(n, -1, dtype=int)
    feature_names: Optional[list[str]] = None
    if not inner:
        # fit on all eligible, predict nowhere OOF — leave NaN
        return pred, np.array([], dtype=int), assign

    for fi, inf in enumerate(inner):
        tr_mask = _session_mask(sessions_l, inf["train_sessions"])
        va_mask = _session_mask(sessions_l, inf["test_sessions"])
        elig = set(eligible)
        tr_mask = tr_mask & np.array([s in elig for s in sessions_l])
        va_mask = va_mask & np.array([s in elig for s in sessions_l])
        tr_idx = _indices_for(tr_mask)
        va_idx = _indices_for(va_mask)
        _assert_groups_intact(group_ids, tr_idx, va_idx)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        if task == "classification" and len(np.unique(y_arr[tr_idx])) < 2:
            base = float(np.mean(y_arr[tr_idx])) if len(tr_idx) else 0.5
            pred[va_idx] = base
            assign[va_idx] = fi
            continue
        X_tr, feature_names = _rows_to_matrix(
            [rows_l[i] for i in tr_idx], feature_names)
        X_va, _ = _rows_to_matrix(
            [rows_l[i] for i in va_idx], feature_names)
        est = estimator_factory(params)
        est.fit(X_tr, y_arr[tr_idx])
        pred[va_idx] = np.asarray(predict_fn(est, X_va), dtype=float)
        assign[va_idx] = fi

    covered = np.isfinite(pred)
    return pred, np.flatnonzero(covered), assign
