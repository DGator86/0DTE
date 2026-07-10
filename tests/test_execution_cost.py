"""
tests/test_execution_cost.py
============================
PR 6 acceptance — execution cost model:
  * mid / natural / expected / conservative credits for multi-leg structures;
  * expected credit never exceeds midpoint credit;
  * expected debit is never cheaper than midpoint debit;
  * conservative is never better than expected;
  * fees and exit drag are non-negative;
  * paper/manual FillRecord captures realized fill fraction;
  * selector attaches an execution panel; journal settles expected-fill P&L;
  * TearSheet / economic_pnl prefer expected-fill when present.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from execution_cost import (
    ExecutionCostConfig, FillRecord, LegQuote, estimate_execution,
    fill_credit, half_spread_cost, make_fill_record, mid_credit,
    natural_credit, net_pnl, quotes_from_chain,
)
from journal import Journal, economic_pnl, realized_pnl
from prediction.models.fill import (FillPriorConfig, base_fill_fraction,
                                    fill_fraction_for)
from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd
from spread_selector import (
    Leg, SelectorConfig, _evaluate, _credit, DEBIT_FAMILIES,
)

ET = ZoneInfo("America/New_York")
F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
DF0 = math.exp(-R0 * T0)


def _chain(atm_s: float = 0.04, spread: float = 0.04) -> ChainSnapshot:
    """Synthetic chain with a controllable bid-ask width around Black mids."""
    qs = []
    for K in np.arange(F0 - 10, F0 + 11, 1.0):
        k = math.log(K / F0)
        s = max(atm_s - 0.030 * k, 0.0008)
        cm = max(_bs_call_fwd(F0, K, s) * DF0, 0.01)
        pm = max(cm - DF0 * (F0 - K), 0.01)
        h = spread / 2.0
        qs.append(ChainQuote(float(K), max(cm - h, 0.0), cm + h,
                             max(pm - h, 0.0), pm + h))
    return ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)


def _pcs_legs():
    return (Leg(599.0, "P", -1), Leg(598.0, "P", 1))


def _lcs_legs():
    return (Leg(601.0, "C", 1), Leg(603.0, "C", -1))


def _ic_legs():
    return (Leg(597.0, "P", 1), Leg(598.0, "P", -1),
            Leg(602.0, "C", -1), Leg(603.0, "C", 1))


class TestStrategyPrices:
    def test_mid_and_natural_credit_spread(self):
        chain = _chain(spread=0.10)
        q = quotes_from_chain(chain)
        legs = _pcs_legs()
        mid = mid_credit(legs, q)
        nat = natural_credit(legs, q)
        assert mid is not None and nat is not None
        assert mid > nat                              # natural is worse
        assert half_spread_cost(legs, q) == pytest.approx(mid - nat)

    def test_mid_and_natural_debit_spread(self):
        chain = _chain(spread=0.10)
        q = quotes_from_chain(chain)
        legs = _lcs_legs()
        mid = mid_credit(legs, q)
        nat = natural_credit(legs, q)
        assert mid is not None and nat is not None
        assert mid < 0 and nat < mid                  # natural debit is larger
        assert half_spread_cost(legs, q) == pytest.approx(mid - nat)

    def test_fill_credit_endpoints(self):
        assert fill_credit(0.50, 0.40, 0.0) == pytest.approx(0.50)
        assert fill_credit(0.50, 0.40, 1.0) == pytest.approx(0.40)
        assert fill_credit(0.50, 0.40, 0.5) == pytest.approx(0.45)


class TestEstimateExecution:
    def test_credit_monotonicity(self):
        est = estimate_execution(_pcs_legs(), quotes_from_chain(_chain()),
                                 "put_credit")
        assert est is not None
        assert est.expected_credit <= est.mid_credit + 1e-12
        assert est.conservative_credit <= est.expected_credit + 1e-12
        assert est.natural_credit <= est.conservative_credit + 1e-12

    def test_debit_monotonicity(self):
        est = estimate_execution(_lcs_legs(), quotes_from_chain(_chain()),
                                 "long_call_spread")
        assert est is not None
        # All credits are negative; "never better" means more negative or equal
        assert est.expected_credit <= est.mid_credit + 1e-12
        assert est.conservative_credit <= est.expected_credit + 1e-12

    def test_fees_and_round_trip_nonneg(self):
        est = estimate_execution(_ic_legs(), quotes_from_chain(_chain()),
                                 "iron_condor")
        assert est.entry_fees > 0
        assert est.exit_fees_expected > 0
        assert est.round_trip_cost_expected >= 0
        assert est.net_expected_credit <= est.expected_credit
        assert est.n_legs == 4
        assert est.fill_fraction_expected == pytest.approx(0.65, abs=0.01)

    def test_net_pnl_matches_structure_math(self):
        legs = _pcs_legs()
        pnl = net_pnl(legs, 0.30, 602.0, entry_fees=0.01,
                      exit_fees=0.01, exit_slippage=0.02)
        assert pnl == pytest.approx(0.30 - 0.04)


class TestFillRecord:
    def test_realized_fill_fraction(self):
        chain = _chain(spread=0.10)
        q = quotes_from_chain(chain)
        legs = _pcs_legs()
        mid = mid_credit(legs, q)
        nat = natural_credit(legs, q)
        # fill halfway between mid and natural
        fill = 0.5 * (mid + nat)
        rec = make_fill_record(
            candidate_id="c1", snapshot_id="s1", family="put_credit",
            decision_ts="2026-07-10T15:00:00-04:00", legs=legs, quotes=q,
            fill_price=fill)
        assert rec.realized_fill_fraction() == pytest.approx(0.5, abs=0.01)
        assert rec.to_dict()["realized_fill_fraction"] == pytest.approx(0.5, abs=0.01)


class TestSelectorAttachment:
    def test_candidate_carries_execution_panel(self):
        from rnd_extractor import extract_rnd
        from spread_selector import GammaContext, _chain_maps
        chain = _chain()
        rnd = extract_rnd(chain)
        dx = rnd.grid[1] - rnd.grid[0]
        phys = rnd.pdf / max(np.sum(rnd.pdf) * dx, 1e-12)
        ctx = GammaContext(spot=F0, call_wall=F0 + 5, put_wall=F0 - 5,
                           gamma_flip=F0 - 1, net_gex=1e9, gex_pct_rank=0.7)
        cand = _evaluate(
            "put_credit", _pcs_legs(), chain, rnd, phys, ctx,
            SelectorConfig(min_ev=-1e9, min_credit=0.0, min_liquidity=0.0,
                           max_touch_short=1.0, veto_short_below_flip=False),
            {})
        assert cand is not None
        assert cand.execution is not None
        assert cand.execution["expected_credit"] <= cand.execution["mid_credit"]
        assert cand.execution["conservative_credit"] <= cand.execution["expected_credit"]
        # mid credit on the candidate matches the diagnostic mid path
        cmid, pmid, _ = _chain_maps(chain)
        assert cand.credit == pytest.approx(_credit(_pcs_legs(), cmid, pmid),
                                            abs=1e-4)


class TestJournalSettlement:
    def test_settles_expected_fill_pnl(self, tmp_path):
        j = Journal(db_path=str(tmp_path / "j.db"))
        legs = [{"strike": 599.0, "kind": "P", "qty": -1},
                {"strike": 598.0, "kind": "P", "qty": 1}]
        execution = {
            "net_expected_credit": 0.25,
            "net_conservative_credit": 0.20,
            "exit_slippage_expected": 0.02,
            "exit_fees_expected": 0.007,
            "exit_slippage_stop": 0.04,
        }
        row = {
            "session_date": "2026-07-10", "ts": "2026-07-10T15:00:00-04:00",
            "spot": 600.0, "net_gex": 0.0, "gex_regime": "flat",
            "gex_pct_rank": 0.5, "zero_gamma_dist": 0.0,
            "zero_gamma_dist_pct": 0.0, "adx": 15.0,
            "call_wall": 605.0, "put_wall": 595.0,
            "selected_family": "put_credit",
            "short_strikes": "[599.0]", "long_strikes": "[598.0]",
            "legs_json": json.dumps(legs), "credit": 0.30,
            "candidate_score": 0.1, "ev": 0.05, "max_loss": 0.70,
            "ev_per_risk": 0.07, "theta": 0.0, "gamma": 0.0,
            "prob_profit": 0.6, "prob_touch_short": 0.2,
            "liquidity_score": 0.8, "wall_safety": 0.8,
            "gamma_safety": 0.8, "touch_safety": 0.8,
            "gate_pass": 1, "gate_score": 70.0, "gate_failed": "[]",
            "veto_reasons": "[]", "decision": "TRADE", "no_trade_reason": "",
            "was_traded": 1, "candidate_present": 1, "regime_direction": "none",
            "signals_json": None,
            "execution_json": json.dumps(execution),
            "credit_expected": 0.25, "credit_conservative": 0.20,
        }
        j.log(row)
        assert j.settle_session("2026-07-10", 602.0) == 1
        settled = j.fetch(settled_only=True)[0]
        assert settled["realized_pnl"] == pytest.approx(0.30)
        assert settled["realized_pnl_expected"] == pytest.approx(
            0.25 - 0.02 - 0.007)
        assert settled["realized_pnl_conservative"] == pytest.approx(
            0.20 - 0.04 - 0.007)
        assert economic_pnl(settled) == pytest.approx(
            settled["realized_pnl_expected"])

    def test_economic_pnl_falls_back_to_mid(self):
        assert economic_pnl({"realized_pnl": 0.12}) == pytest.approx(0.12)
        assert economic_pnl({}) is None
