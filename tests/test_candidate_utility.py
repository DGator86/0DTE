"""
tests/test_candidate_utility.py
===============================
PR 8 acceptance — V2 utility monotonicity (§14.2):
  * utility decreases when expected_shortfall increases;
  * utility decreases when fill_uncertainty increases;
  * utility decreases when model_uncertainty increases;
  * capital penalty reduces utility for larger capital.
"""
from __future__ import annotations

import pytest

from prediction.candidate_ranker import UtilityConfig, candidate_utility
from prediction.models.candidate_value import CandidateForecast


def _fc(**overrides) -> CandidateForecast:
    base = dict(
        candidate_id="c1",
        expected_net_pnl=0.40,
        p_profit=0.60,
        pnl_q10=-0.50,
        pnl_q50=0.20,
        pnl_q90=0.80,
        expected_shortfall=0.50,
        fill_uncertainty=0.20,
        model_uncertainty=0.30,
        utility_score=0.0,
    )
    base.update(overrides)
    return CandidateForecast(**base)


class TestUtilityMonotonicity:
    def test_shortfall_hurts(self):
        cfg = UtilityConfig(lambda_shortfall=0.5, lambda_fill=0.0,
                            lambda_model=0.0, lambda_capital=0.0)
        low = candidate_utility(_fc(expected_shortfall=0.2), capital=0, cfg=cfg)
        high = candidate_utility(_fc(expected_shortfall=0.8), capital=0, cfg=cfg)
        assert high < low

    def test_fill_uncertainty_hurts(self):
        cfg = UtilityConfig(lambda_shortfall=0.0, lambda_fill=0.25,
                            lambda_model=0.0, lambda_capital=0.0)
        low = candidate_utility(_fc(fill_uncertainty=0.1), capital=0, cfg=cfg)
        high = candidate_utility(_fc(fill_uncertainty=0.9), capital=0, cfg=cfg)
        assert high < low

    def test_model_uncertainty_hurts(self):
        cfg = UtilityConfig(lambda_shortfall=0.0, lambda_fill=0.0,
                            lambda_model=0.25, lambda_capital=0.0)
        low = candidate_utility(_fc(model_uncertainty=0.1), capital=0, cfg=cfg)
        high = candidate_utility(_fc(model_uncertainty=0.9), capital=0, cfg=cfg)
        assert high < low

    def test_capital_penalty(self):
        cfg = UtilityConfig(lambda_shortfall=0.0, lambda_fill=0.0,
                            lambda_model=0.0, lambda_capital=0.10,
                            portfolio_risk_budget=1.0)
        small = candidate_utility(_fc(), capital=0.5, cfg=cfg)
        large = candidate_utility(_fc(), capital=2.0, cfg=cfg)
        assert large < small

    def test_formula_matches_spec(self):
        cfg = UtilityConfig(lambda_shortfall=0.5, lambda_fill=0.25,
                            lambda_model=0.25, lambda_capital=0.10,
                            portfolio_risk_budget=2.0)
        fc = _fc(expected_net_pnl=1.0, expected_shortfall=0.4,
                 fill_uncertainty=0.2, model_uncertainty=0.1)
        got = candidate_utility(fc, capital=1.0, cfg=cfg)
        expected = 1.0 - 0.5 * 0.4 - 0.25 * 0.2 - 0.25 * 0.1 - 0.10 * (1.0 / 2.0)
        assert got == pytest.approx(expected)
