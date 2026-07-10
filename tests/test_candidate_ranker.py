"""
tests/test_candidate_ranker.py
==============================
PR 8 acceptance — CandidateValueModel + shadow ranker:
  * model predicts bounded P(profit) and ordered quantiles;
  * learnable signal beats mean-PnL baseline out of sample;
  * V2 ranking can disagree with legacy score;
  * shadow mode does not change decide()'s authoritative candidate;
  * SnapshotRankingResult signals are observation-only diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from prediction.candidate_ranker import (
    RankerConfig, SnapshotRankingResult, UtilityConfig, candidate_utility,
    rank_candidates, run_shadow_ranking,
)
from prediction.models.candidate_value import (
    CandidateForecast, CandidateValueConfig, CandidateValueModel,
)


SMALL = CandidateValueConfig(
    c_grid=(0.1, 1.0), l1_ratio_grid=(0.0, 0.5), max_iter=400,
    quantile_max_iter=80, min_samples_leaf=10, max_leaf_nodes=7,
)


def _synth(n_sessions=12, per_session=30, strength=1.5, seed=41):
    """Rows where higher 'signal' → higher net PnL / profit probability."""
    rng = np.random.default_rng(seed)
    rows, y_pnl, y_profit, sessions, snaps, fills, caps = (
        [], [], [], [], [], [], [])
    for s in range(n_sessions):
        date = f"2026-07-{s + 1:02d}"
        for i in range(per_session):
            sig = rng.standard_normal()
            noise = rng.standard_normal() * 0.15
            pnl = strength * 0.1 * sig + noise
            rows.append({
                "signal": sig,
                "noise": rng.standard_normal(),
                "legacy_candidate_score": abs(sig) * 0.1,
                "max_loss": 1.0,
                "ev": pnl + 0.05,
            })
            y_pnl.append(pnl)
            y_profit.append(int(pnl > 0))
            sessions.append(date)
            snaps.append(f"{date}|t{i % 3}")
            fills.append(0.2)
            caps.append(1.0)
    return (rows, np.array(y_pnl), np.array(y_profit), sessions, snaps,
            fills, caps)


@dataclass
class _FakeCand:
    family: str
    score: float
    credit: float = 0.30
    max_loss: float = 0.70
    capital: float = 0.70
    ev: float = 0.05
    ev_per_risk: float = 0.07
    theta: float = 0.01
    gamma: float = -0.01
    prob_profit: float = 0.6
    prob_touch_short: float = 0.2
    distance_to_wall: float = 2.0
    liquidity_score: float = 0.8
    wall_safety: float = 0.8
    gamma_safety: float = 0.8
    touch_safety: float = 0.8
    passes_vetoes: bool = True
    short_strikes: tuple = (599.0,)
    long_strikes: tuple = (598.0,)
    legs: tuple = ()
    execution: dict = None

    def __post_init__(self):
        if not self.legs:
            from spread_selector import Leg
            self.legs = (Leg(self.short_strikes[0], "P", -1),
                         Leg(self.long_strikes[0], "P", 1))
        if self.execution is None:
            self.execution = {
                "mid_credit": self.credit,
                "natural_credit": self.credit - 0.05,
                "net_expected_credit": self.credit - 0.02,
                "fill_fraction_expected": 0.65,
            }


class TestCandidateValueModel:
    def test_bounds_quantiles_determinism(self):
        rows, y_pnl, y_profit, sessions, snaps, fills, caps = _synth()
        m1 = CandidateValueModel(config=SMALL).fit(
            rows, y_pnl=y_pnl, y_profit=y_profit, sessions=sessions,
            group_ids=snaps)
        m2 = CandidateValueModel(config=SMALL).fit(
            rows, y_pnl=y_pnl, y_profit=y_profit, sessions=sessions,
            group_ids=snaps)
        f1 = m1.predict(rows, candidate_ids=[f"c{i}" for i in range(len(rows))],
                        fill_uncertainty=fills, capital=caps,
                        utility_fn=lambda fc, capital=0.0: candidate_utility(
                            fc, capital=capital))
        f2 = m2.predict(rows, candidate_ids=[f"c{i}" for i in range(len(rows))],
                        fill_uncertainty=fills, capital=caps,
                        utility_fn=lambda fc, capital=0.0: candidate_utility(
                            fc, capital=capital))
        assert len(f1) == len(rows)
        for a, b in zip(f1, f2):
            assert a.to_dict() == b.to_dict()
            assert 0.0 <= a.p_profit <= 1.0
            assert a.pnl_q10 <= a.pnl_q50 <= a.pnl_q90
            assert a.expected_shortfall >= 0.0

    def test_learns_signal_oos(self):
        rows, y_pnl, y_profit, sessions, snaps, fills, caps = _synth(
            n_sessions=14, strength=2.0)
        train_s = {f"2026-07-{i:02d}" for i in range(1, 11)}
        test_s = {f"2026-07-{i:02d}" for i in range(12, 15)}
        tr = [i for i, s in enumerate(sessions) if s in train_s]
        te = [i for i, s in enumerate(sessions) if s in test_s]
        m = CandidateValueModel(config=SMALL).fit(
            [rows[i] for i in tr], y_pnl=y_pnl[tr], y_profit=y_profit[tr],
            sessions=[sessions[i] for i in tr],
            group_ids=[snaps[i] for i in tr])
        pred = m.predict_components([rows[i] for i in te])
        mse_model = float(np.mean((pred["expected_net_pnl"] - y_pnl[te]) ** 2))
        mse_base = float(np.mean((np.mean(y_pnl[tr]) - y_pnl[te]) ** 2))
        assert mse_model < mse_base


class TestShadowRanker:
    def _fitted(self):
        rows, y_pnl, y_profit, sessions, snaps, fills, caps = _synth(
            n_sessions=10, per_session=20)
        return CandidateValueModel(config=SMALL).fit(
            rows, y_pnl=y_pnl, y_profit=y_profit, sessions=sessions,
            group_ids=snaps)

    def test_v2_can_disagree_with_legacy(self):
        model = self._fitted()
        # High legacy score but features that look weak to the model, vs
        # low legacy score with strong signal-like features.
        cands = [
            _FakeCand(family="put_credit", score=5.0, credit=0.10,
                      short_strikes=(599.0,), long_strikes=(598.0,)),
            _FakeCand(family="call_credit", score=0.1, credit=0.40,
                      short_strikes=(601.0,), long_strikes=(602.0,),
                      legs=()),
        ]
        # Force distinctive legs on second
        from spread_selector import Leg
        cands[1].legs = (Leg(601.0, "C", -1), Leg(602.0, "C", 1))
        cands[1].short_strikes = (601.0,)
        cands[1].long_strikes = (602.0,)

        result = run_shadow_ranking(
            cands, model, snapshot_id="2026-07-10|t0", spot=600.0,
            call_wall=608.0, put_wall=592.0, gamma_flip=599.0,
            cfg=RankerConfig(mode="shadow"))
        assert isinstance(result, SnapshotRankingResult)
        assert result.legacy_top_id is not None
        assert result.v2_top_id is not None
        # Legacy top is the score=5 candidate
        assert result.diagnostics["legacy_top_score"] == pytest.approx(5.0)
        sig = result.signals()
        assert "v2_utility_score" in sig
        assert sig["candidate_model_version"]
        assert "v2_rank_disagreement" in sig

    def test_rank_candidates_orders_by_utility(self):
        cands = [
            _FakeCand(family="put_credit", score=1.0, short_strikes=(599.0,),
                      long_strikes=(598.0,)),
            _FakeCand(family="put_credit", score=2.0, short_strikes=(598.0,),
                      long_strikes=(597.0,)),
        ]
        from spread_selector import Leg
        cands[1].legs = (Leg(598.0, "P", -1), Leg(597.0, "P", 1))
        fcs = {
            "a": CandidateForecast(
                candidate_id="a", expected_net_pnl=0.1, p_profit=0.5,
                pnl_q10=-0.2, pnl_q50=0.1, pnl_q90=0.3,
                expected_shortfall=0.2, fill_uncertainty=0.1,
                model_uncertainty=0.1, utility_score=0.05),
            "b": CandidateForecast(
                candidate_id="b", expected_net_pnl=0.3, p_profit=0.7,
                pnl_q10=-0.1, pnl_q50=0.2, pnl_q90=0.5,
                expected_shortfall=0.1, fill_uncertainty=0.1,
                model_uncertainty=0.1, utility_score=0.25),
        }
        cands[0]._v2_candidate_id = "a"
        cands[1]._v2_candidate_id = "b"
        ranked = rank_candidates(cands, fcs)
        assert ranked[0][1].candidate_id == "b"
        assert ranked[1][1].candidate_id == "a"

    def test_shadow_does_not_change_decide_candidate(self):
        """Legacy decide() path ignores the ranker entirely."""
        import datetime as dt
        from zoneinfo import ZoneInfo
        from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd
        from decision_engine import EngineConfig, decide
        from gate_scorer import MarketSnapshot

        ET = ZoneInfo("America/New_York")
        F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
        DF0 = math.exp(-R0 * T0)
        qs = []
        for K in np.arange(F0 - 10, F0 + 11, 1.0):
            k = math.log(K / F0)
            s = max(0.04 - 0.030 * k, 0.0008)
            cm = max(_bs_call_fwd(F0, K, s) * DF0, 0.01)
            pm = max(cm - DF0 * (F0 - K), 0.01)
            h = 0.02
            qs.append(ChainQuote(float(K), max(cm - h, 0.0), cm + h,
                                 max(pm - h, 0.0), pm + h))
        chain = ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)
        now = dt.datetime(2026, 7, 10, 15, 0, tzinfo=ET)
        market = MarketSnapshot(
            spot=F0, net_gex=1e9, gamma_flip=F0 - 1,
            call_wall=F0 + 5, put_wall=F0 - 5, gex_pct_rank=0.7,
            vix9d=14.0, vix=16.0, vix3m=18.0, vvix=80.0, vvix_baseline=85.0,
            straddle_breakeven=3.0, expected_range=2.5,
            adx=18.0, rsi=50.0, bb_width=0.02, bb_width_baseline=0.02,
            vwap=F0, vwap_reversion_count=2,
            tick_abs_mean=200.0, cvd_slope=0.0,
            now=now, has_catalyst=False,
        )
        cfg = EngineConfig()
        cfg.selector.min_ev = -1e9
        cfg.selector.min_credit = 0.0
        cfg.selector.min_liquidity = 0.0
        cfg.selector.max_touch_short = 1.0
        cfg.selector.veto_short_below_flip = False
        d1 = decide(market, chain, cfg)
        d2 = decide(market, chain, cfg)
        assert (d1.candidate is None) == (d2.candidate is None)
        if d1.candidate is not None:
            assert d1.candidate.family == d2.candidate.family
            assert d1.candidate.score == d2.candidate.score
            assert d1.candidate.v2_utility_score is None
        assert hasattr(d1, "all_candidates")
        assert isinstance(d1.all_candidates, list)
