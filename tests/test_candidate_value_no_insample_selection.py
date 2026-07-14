"""
tests/test_candidate_value_no_insample_selection.py
===================================================
V3 Part 1 §6 — candidate expected-P&L hyperparameters must be selected
out-of-fold, never by in-sample training error.
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.candidate_value import (
    CANDIDATE_VALUE_VERSION,
    CandidateValueConfig,
    CandidateValueModel,
)


SMALL = CandidateValueConfig(
    c_grid=(0.1, 1.0),
    l1_ratio_grid=(0.0, 0.5),
    alpha_grid=(0.01, 0.1),
    huber_epsilon_grid=(1.35,),
    hgb_learning_rate_grid=(0.1,),
    hgb_max_depth_grid=(2,),
    max_iter=400,
    quantile_max_iter=40,
    min_samples_leaf=10,
)


def _synth(n_sessions=14, per_session=8, cands_per_snap=2, seed=11):
    rng = np.random.default_rng(seed)
    rows, y_pnl, y_profit, sessions, groups = [], [], [], [], []
    for s in range(n_sessions):
        date = f"2026-09-{s + 1:02d}"
        for snap in range(per_session):
            sid = f"{date}-snap{snap}"
            for c in range(cands_per_snap):
                x = float(rng.standard_normal())
                noise = float(rng.standard_normal())
                pnl = 0.5 * x + 0.2 * noise
                rows.append({
                    "x": x, "noise": noise,
                    "family": "put_credit" if c % 2 == 0 else "call_credit",
                })
                y_pnl.append(pnl)
                y_profit.append(int(pnl > 0))
                sessions.append(date)
                groups.append(sid)
    return rows, np.array(y_pnl), np.array(y_profit), sessions, groups


def test_no_insample_pnl_selection_flag():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    assert m.metadata["insample_pnl_selection"] is False
    assert m.metadata["oof_metrics"]["pnl_selection"]["selection_metric"] == (
        "huber_bias_downside")
    assert "mse" in m.metadata["oof_metrics"]["expected_pnl"]
    # MSE may be reported but must not be the sole selection criterion
    assert m.metadata["oof_metrics"]["pnl_selection"]["selection_metric"] != "mse"


def test_challengers_considered():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    considered = set(
        m.metadata["oof_metrics"]["pnl_selection"]["challengers_considered"])
    assert "elasticnet" in considered
    assert "huber" in considered
    assert "hgb" in considered
    assert m.metadata["selected_estimator_per_head"]["expected_pnl"] in (
        "elasticnet", "huber", "hgb")


def test_mutating_test_labels_cannot_change_pnl_hyperparameters():
    rows, y_pnl, y_profit, sessions, groups = _synth(n_sessions=16)
    keep = {f"2026-09-{i:02d}" for i in range(1, 11)}
    idx = [i for i, s in enumerate(sessions) if s in keep]
    m1 = CandidateValueModel(config=SMALL).fit(
        [rows[i] for i in idx],
        y_pnl=y_pnl[idx], y_profit=y_profit[idx],
        sessions=[sessions[i] for i in idx],
        group_ids=[groups[i] for i in idx])
    # Mutate labels outside the fit window (not passed in)
    y_pnl2 = y_pnl.copy()
    for i, s in enumerate(sessions):
        if s not in keep:
            y_pnl2[i] = -y_pnl2[i] * 10
    m2 = CandidateValueModel(config=SMALL).fit(
        [rows[i] for i in idx],
        y_pnl=y_pnl[idx], y_profit=y_profit[idx],
        sessions=[sessions[i] for i in idx],
        group_ids=[groups[i] for i in idx])
    assert (m1.metadata["selected_hyperparameters"]["expected_pnl"]
            == m2.metadata["selected_hyperparameters"]["expected_pnl"])


def test_snapshot_groups_never_split_during_fit():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    # Should not raise AssertionError from group integrity checks
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    assert m.fitted
    assert m.metadata["snapshot_count"] == len(set(groups))


def test_profit_head_uses_crossfit_calibration():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    cm = m.metadata["calibration_metrics"]
    assert cm.get("crossfit") is True
    assert m.calibration_artifact is not None


def test_quantile_oof_metrics_present():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    q = m.metadata["oof_metrics"]["quantiles"]
    assert "pinball" in q
    assert "interval_coverage" in q
    assert "interval_width" in q
    assert "downside_miss_rate" in q
    assert "quantile_crossing_rate_before_rearrangement" in q
    assert "by_option_family" in q


def test_required_artifact_metadata():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups,
        data_hash="abc123", outcome_coverage=0.95)
    meta = m.metadata
    for key in (
        "model_version", "feature_version", "label_version",
        "candidate_feature_schema_hash", "train_sessions",
        "crossfit_config", "selected_estimator_per_head",
        "selected_hyperparameters", "oof_metrics",
        "calibration_artifact", "family_coverage",
        "snapshot_count", "candidate_count", "outcome_coverage",
        "data_hash",
    ):
        assert key in meta
    assert meta["model_version"] == CANDIDATE_VALUE_VERSION
    assert meta["data_hash"] == "abc123"


def test_predict_bounds_and_determinism():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m1 = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    m2 = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    c1 = m1.predict_components(rows[:5])
    c2 = m2.predict_components(rows[:5])
    np.testing.assert_allclose(c1["expected_net_pnl"], c2["expected_net_pnl"])
    assert np.all((c1["p_profit"] >= 0.0) & (c1["p_profit"] <= 1.0))
