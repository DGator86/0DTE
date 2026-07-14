"""
tests/test_crossfit.py
======================
Cross-fitting leakage, determinism, and selection-integrity tests
(Prediction Engine V3 Part 1 §4 / §12).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from prediction.crossfit import (
    NestedCrossFitConfig,
    build_nested_session_folds,
    crossfit_classifier,
    crossfit_regressor,
)


def _make_clf_data(n_sessions: int = 40, rows_per: int = 4, seed: int = 0):
    rng = np.random.RandomState(seed)
    rows, y, sessions, groups = [], [], [], []
    for i in range(n_sessions):
        s = f"2026-02-{i + 1:02d}"
        for j in range(rows_per):
            x1 = float(rng.randn())
            x2 = float(rng.randn())
            # weakly informative label
            p = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.3 * x2)))
            rows.append({"x1": x1, "x2": x2, "noise": float(rng.randn())})
            y.append(int(rng.rand() < p))
            sessions.append(s)
            groups.append(f"{s}-snap-{j // 2}")  # 2 candidates per snapshot
    return rows, y, sessions, groups


def _clf_factory(params: dict):
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        solver="lbfgs",
        max_iter=500,
        random_state=0,
    )


def _predict_raw(est, X):
    return est.predict_proba(X)[:, 1]


def _reg_factory(params: dict):
    return Ridge(alpha=float(params.get("alpha", 1.0)))


CFG = NestedCrossFitConfig(
    outer_folds=3,
    inner_folds=2,
    embargo_sessions=1,
    min_train_sessions=8,
    min_validation_sessions=3,
    retain_fold_models=True,
    random_state=42,
)


def test_classifier_determinism():
    rows, y, sessions, _ = _make_clf_data()
    grid = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    a = crossfit_classifier(rows, y, sessions, grid, _clf_factory, _predict_raw, CFG)
    b = crossfit_classifier(rows, y, sessions, grid, _clf_factory, _predict_raw, CFG)
    assert a.selected_params == b.selected_params
    np.testing.assert_allclose(a.oof_raw_predictions, b.oof_raw_predictions)
    np.testing.assert_array_equal(a.oof_row_indices, b.oof_row_indices)
    np.testing.assert_array_equal(a.fold_assignments, b.fold_assignments)


def test_regressor_determinism():
    rows, y, sessions, groups = _make_clf_data()
    y_f = [float(v) + 0.1 * r["x1"] for v, r in zip(y, rows)]
    grid = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]
    a = crossfit_regressor(rows, y_f, sessions, grid, _reg_factory, CFG, groups)
    b = crossfit_regressor(rows, y_f, sessions, grid, _reg_factory, CFG, groups)
    assert a.selected_params == b.selected_params
    np.testing.assert_allclose(a.oof_raw_predictions, b.oof_raw_predictions)


def test_outer_test_sessions_never_in_train_or_cal():
    rows, y, sessions, _ = _make_clf_data()
    grid = [{"C": 1.0}]
    result = crossfit_classifier(
        rows, y, sessions, grid, _clf_factory, _predict_raw, CFG)
    for fd in result.diagnostics["fold_definitions"]:
        tr = set(fd["train_sessions"])
        cal = set(fd["calibration_sessions"])
        va = set(fd["validation_sessions"])
        emb = set(fd["embargoed_sessions"])
        assert not (tr & va)
        assert not (cal & va)
        assert not (emb & tr)
        assert not (emb & va)


def test_snapshot_groups_never_divided():
    rows, y, sessions, groups = _make_clf_data()
    y_f = [float(v) for v in y]
    grid = [{"alpha": 1.0}, {"alpha": 10.0}]
    result = crossfit_regressor(
        rows, y_f, sessions, grid, _reg_factory, CFG, group_ids=groups)
    # For every outer fold, check groups of train vs val rows
    for fi, fd_meta in enumerate(result.diagnostics["fold_definitions"]):
        tr_set = set(fd_meta["train_sessions"])
        va_set = set(fd_meta["validation_sessions"])
        tr_idx = [i for i, s in enumerate(sessions) if s in tr_set]
        va_idx = [i for i, s in enumerate(sessions) if s in va_set]
        tr_g = {groups[i] for i in tr_idx}
        va_g = {groups[i] for i in va_idx}
        assert not (tr_g & va_g)


def test_mutating_test_labels_cannot_change_selected_hyperparameters():
    rows, y, sessions, _ = _make_clf_data(seed=1)
    grid = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    base = crossfit_classifier(
        rows, y, sessions, grid, _clf_factory, _predict_raw, CFG)
    # Identify outer validation row indices and flip their labels
    folds = build_nested_session_folds(sessions, CFG)
    val_sessions = set()
    for fd in folds:
        val_sessions.update(fd.validation_sessions)
    y_mut = list(y)
    for i, s in enumerate(sessions):
        if s in val_sessions:
            y_mut[i] = 1 - y_mut[i]
    mutated = crossfit_classifier(
        rows, y_mut, sessions, grid, _clf_factory, _predict_raw, CFG)
    assert base.selected_params == mutated.selected_params


def test_oof_row_indices_match_predictions_length():
    rows, y, sessions, _ = _make_clf_data()
    result = crossfit_classifier(
        rows, y, sessions, [{"C": 1.0}], _clf_factory, _predict_raw, CFG)
    assert len(result.oof_raw_predictions) == len(result.oof_row_indices)
    assert len(result.oof_row_indices) > 0
    assert set(result.oof_row_indices.tolist()).issubset(set(range(len(rows))))


def test_classifier_primary_metric_is_log_loss():
    rows, y, sessions, _ = _make_clf_data()
    result = crossfit_classifier(
        rows, y, sessions, [{"C": 0.5}, {"C": 2.0}],
        _clf_factory, _predict_raw, CFG)
    assert result.diagnostics["inner_selection"]["selection_metric"] == "log_loss"


def test_regressor_selection_not_sole_mse():
    rows, y, sessions, groups = _make_clf_data()
    y_f = [10.0 * float(v) + r["x1"] for v, r in zip(y, rows)]
    result = crossfit_regressor(
        rows, y_f, sessions, [{"alpha": 0.5}, {"alpha": 5.0}],
        _reg_factory, CFG, group_ids=groups)
    assert result.diagnostics["inner_selection"]["selection_metric"] == (
        "huber_bias_downside")
    for m in result.fold_metrics:
        assert "huber" in m
        assert "selection_score" in m
        assert "mse" in m  # reported but not sole criterion


def test_fold_assignments_cover_validation_rows():
    rows, y, sessions, _ = _make_clf_data()
    result = crossfit_classifier(
        rows, y, sessions, [{"C": 1.0}], _clf_factory, _predict_raw, CFG)
    assigned = result.fold_assignments
    assert (assigned >= 0).any()
    # Every OOF row has a non-negative fold assignment
    for idx in result.oof_row_indices:
        assert assigned[idx] >= 0
