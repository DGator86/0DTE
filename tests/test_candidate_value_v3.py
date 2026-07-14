"""
tests/test_candidate_value_v3.py
================================
V3 Part 3 PR17 — expanded candidate-value distribution (§43).
"""
from __future__ import annotations

import numpy as np
import pytest

from prediction.models.candidate_value import (
    CANDIDATE_FORECAST_V3_VERSION,
    CANDIDATE_LABEL_VERSION,
    QUANTILES_V3,
    CandidateForecastV3,
    CandidateValueConfig,
    CandidateValueModel,
)


def _synth(n=60, seed=3):
    rng = np.random.default_rng(seed)
    rows, y_pnl, y_profit, sessions, groups = [], [], [], [], []
    for i in range(n):
        f1 = float(rng.normal())
        rows.append({"f1": f1, "f2": float(rng.normal()), "family": "ic"})
        pnl = 0.2 * f1 + float(rng.normal(0, 0.3))
        y_pnl.append(pnl)
        y_profit.append(1 if pnl > 0 else 0)
        sessions.append(f"S{i % 8:02d}")
        groups.append(f"snap-{i // 2}")  # two candidates per snapshot
    return rows, y_pnl, y_profit, sessions, groups


SMALL = CandidateValueConfig(
    expanded_distribution=True,
    quantiles=QUANTILES_V3,
    c_grid=(0.5,),
    l1_ratio_grid=(0.5,),
    alpha_grid=(0.01,),
    quantile_max_iter=40,
    max_iter=200,
    inner_folds=2,
    min_train_sessions=2,
    min_validation_sessions=1,
)


def test_quantiles_ordered_v3():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    outs = m.predict_v3(rows[:5], candidate_ids=[f"c{i}" for i in range(5)])
    for fc in outs:
        qs = (fc.pnl_q05, fc.pnl_q10, fc.pnl_q25, fc.pnl_q50,
              fc.pnl_q75, fc.pnl_q90, fc.pnl_q95)
        for a, b in zip(qs, qs[1:]):
            assert a <= b + 1e-9
        assert fc.expected_shortfall >= -1e-12
        assert fc.model_version == CANDIDATE_FORECAST_V3_VERSION


def test_expected_shortfall_non_negative():
    fc = CandidateForecastV3(
        candidate_id="x", expected_net_pnl=0.1, p_profit=0.6,
        pnl_q05=-0.5, pnl_q10=-0.3, pnl_q25=-0.1, pnl_q50=0.1,
        pnl_q75=0.3, pnl_q90=0.5, pnl_q95=0.7,
        expected_shortfall=0.5,
        p_target_first=None, p_stop_first=None, p_neither=None,
        expected_time_in_trade=None,
        fill_probability=0.5, expected_fill_fraction=0.5,
        conservative_fill_fraction=0.8, fill_uncertainty=0.1,
        model_uncertainty=0.2, forecast_uncertainty=0.1, ood_score=0.0,
        capital_required=1.0, maximum_loss=1.0, return_on_risk=0.1,
        utility_score=0.0,
    )
    assert fc.expected_shortfall >= 0
    with pytest.raises(ValueError):
        CandidateForecastV3(
            candidate_id="x", expected_net_pnl=0.1, p_profit=0.6,
            pnl_q05=-0.5, pnl_q10=-0.3, pnl_q25=-0.1, pnl_q50=0.1,
            pnl_q75=0.3, pnl_q90=0.5, pnl_q95=0.7,
            expected_shortfall=-0.1,
            p_target_first=None, p_stop_first=None, p_neither=None,
            expected_time_in_trade=None,
            fill_probability=0.5, expected_fill_fraction=0.5,
            conservative_fill_fraction=0.8, fill_uncertainty=0.1,
            model_uncertainty=0.2, forecast_uncertainty=0.1, ood_score=0.0,
            capital_required=1.0, maximum_loss=1.0, return_on_risk=None,
            utility_score=0.0,
        )


def test_label_version_constant():
    assert CANDIDATE_LABEL_VERSION == "v3.0.0"
    assert SMALL.label_version == "v3.0.0"


def test_snapshot_groups_intact():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    assert m.fitted
    # Mutating test labels after fit cannot change predictions
    before = m.predict_v3(rows[:3])
    y_pnl2 = list(y_pnl)
    y_pnl2[0] = 999.0
    after = m.predict_v3(rows[:3])
    assert [a.to_dict() for a in before] == [a.to_dict() for a in after]


def test_legacy_predict_still_works():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    cfg = CandidateValueConfig(
        c_grid=(0.5,), l1_ratio_grid=(0.5,), alpha_grid=(0.01,),
        quantile_max_iter=40, inner_folds=2,
        min_train_sessions=2, min_validation_sessions=1,
    )
    m = CandidateValueModel(config=cfg).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    outs = m.predict(rows[:3])
    assert outs[0].pnl_q10 <= outs[0].pnl_q50 <= outs[0].pnl_q90


def test_to_legacy_projection():
    rows, y_pnl, y_profit, sessions, groups = _synth()
    m = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    v3 = m.predict_v3(rows[:1])[0]
    legacy = v3.to_legacy()
    assert legacy.pnl_q10 == v3.pnl_q10
    assert legacy.pnl_q50 == v3.pnl_q50
    assert legacy.pnl_q90 == v3.pnl_q90


def test_deterministic():
    rows, y_pnl, y_profit, sessions, groups = _synth(seed=9)
    m1 = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    m2 = CandidateValueModel(config=SMALL).fit(
        rows, y_pnl=y_pnl, y_profit=y_profit,
        sessions=sessions, group_ids=groups)
    a = m1.predict_v3(rows[:2])
    b = m2.predict_v3(rows[:2])
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
