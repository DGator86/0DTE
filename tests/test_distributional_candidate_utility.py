"""
tests/test_distributional_candidate_utility.py
==============================================
V3 Part 3 PR18 — distributional utility monotonicity (§7 / §43).
"""
from __future__ import annotations

import pytest

from prediction.candidate_ranker import CandidateUtilityConfig, candidate_utility
from prediction.models.candidate_value import CandidateForecastV3


def _fc(**overrides) -> CandidateForecastV3:
    base = dict(
        candidate_id="c1",
        expected_net_pnl=0.40,
        p_profit=0.60,
        pnl_q05=-0.80,
        pnl_q10=-0.50,
        pnl_q25=-0.20,
        pnl_q50=0.20,
        pnl_q75=0.50,
        pnl_q90=0.80,
        pnl_q95=1.00,
        expected_shortfall=0.80,
        p_target_first=0.4,
        p_stop_first=0.3,
        p_neither=0.3,
        expected_time_in_trade=15.0,
        fill_probability=0.6,
        expected_fill_fraction=0.5,
        conservative_fill_fraction=0.8,
        fill_uncertainty=0.20,
        model_uncertainty=0.30,
        forecast_uncertainty=0.10,
        ood_score=0.05,
        capital_required=1.0,
        maximum_loss=1.0,
        return_on_risk=0.4,
        utility_score=0.0,
    )
    base.update(overrides)
    return CandidateForecastV3(**base)


def _cfg(**kw):
    zeros = dict(
        lambda_shortfall=0.0, lambda_tail=0.0, lambda_fill=0.0,
        lambda_model=0.0, lambda_forecast=0.0, lambda_ood=0.0,
        lambda_capital=0.0,
    )
    zeros.update(kw)
    return CandidateUtilityConfig(**zeros)


def test_higher_expected_pnl_does_not_reduce_utility():
    cfg = _cfg()
    low = candidate_utility(_fc(expected_net_pnl=0.1), cfg=cfg)
    high = candidate_utility(_fc(expected_net_pnl=0.5), cfg=cfg)
    assert high >= low


def test_higher_shortfall_does_not_increase_utility():
    cfg = _cfg(lambda_shortfall=0.5)
    low = candidate_utility(_fc(expected_shortfall=0.2), cfg=cfg)
    high = candidate_utility(_fc(expected_shortfall=0.9), cfg=cfg)
    assert high <= low


def test_worse_q05_does_not_increase_utility():
    cfg = _cfg(lambda_tail=0.25)
    mild = candidate_utility(_fc(
        pnl_q05=-0.2, pnl_q10=-0.15, pnl_q25=-0.05,
    ), cfg=cfg)
    bad = candidate_utility(_fc(
        pnl_q05=-1.0, pnl_q10=-0.80, pnl_q25=-0.40,
    ), cfg=cfg)
    assert bad <= mild


def test_fill_uncertainty_does_not_increase_utility():
    cfg = _cfg(lambda_fill=0.25)
    assert candidate_utility(_fc(fill_uncertainty=0.9), cfg=cfg) <= \
        candidate_utility(_fc(fill_uncertainty=0.1), cfg=cfg)


def test_model_uncertainty_does_not_increase_utility():
    cfg = _cfg(lambda_model=0.25)
    assert candidate_utility(_fc(model_uncertainty=0.9), cfg=cfg) <= \
        candidate_utility(_fc(model_uncertainty=0.1), cfg=cfg)


def test_forecast_uncertainty_does_not_increase_utility():
    cfg = _cfg(lambda_forecast=0.2)
    assert candidate_utility(_fc(forecast_uncertainty=0.9), cfg=cfg) <= \
        candidate_utility(_fc(forecast_uncertainty=0.1), cfg=cfg)


def test_ood_does_not_increase_utility():
    cfg = _cfg(lambda_ood=0.2)
    assert candidate_utility(_fc(ood_score=0.9), cfg=cfg) <= \
        candidate_utility(_fc(ood_score=0.1), cfg=cfg)


def test_capital_does_not_increase_utility():
    cfg = _cfg(lambda_capital=0.1, portfolio_risk_budget=1.0)
    assert candidate_utility(_fc(), capital=2.0, cfg=cfg) <= \
        candidate_utility(_fc(), capital=0.5, cfg=cfg)


def test_full_formula():
    cfg = CandidateUtilityConfig(
        lambda_shortfall=0.5, lambda_tail=0.25, lambda_fill=0.25,
        lambda_model=0.25, lambda_forecast=0.2, lambda_ood=0.2,
        lambda_capital=0.1, portfolio_risk_budget=2.0,
    )
    fc = _fc(expected_net_pnl=1.0, expected_shortfall=0.4, pnl_q05=-0.8,
             fill_uncertainty=0.2, model_uncertainty=0.1,
             forecast_uncertainty=0.05, ood_score=0.1, capital_required=1.0)
    got = candidate_utility(fc, cfg=cfg)
    expected = (
        1.0 - 0.5 * 0.4 - 0.25 * 0.8 - 0.25 * 0.2 - 0.25 * 0.1
        - 0.2 * 0.05 - 0.2 * 0.1 - 0.1 * (1.0 / 2.0)
    )
    assert got == pytest.approx(expected)
