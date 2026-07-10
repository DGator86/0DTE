"""
tests/test_fill_monotonicity.py
===============================
PR 6 acceptance — fill-fraction priors never improve under adverse conditions:
  * wider relative spreads produce a higher (worse) fill_fraction;
  * older quotes produce a higher fill_fraction;
  * late-day trading produces a higher fill_fraction;
  * higher realized vol produces a higher fill_fraction;
  * more legs produce a weakly higher base prior;
  * expected/conservative credits worsen as spreads widen.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from execution_cost import (
    ExecutionCostConfig, estimate_execution, quotes_from_chain,
)
from prediction.models.fill import (
    DEFAULT_FILL_BY_N_LEGS, FillPriorConfig, base_fill_fraction,
    fill_fraction_for,
)
from rnd_extractor import ChainQuote, ChainSnapshot, _bs_call_fwd
from spread_selector import Leg

F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
DF0 = math.exp(-R0 * T0)


def _chain(spread: float) -> ChainSnapshot:
    qs = []
    for K in np.arange(F0 - 8, F0 + 9, 1.0):
        k = math.log(K / F0)
        s = max(0.04 - 0.030 * k, 0.0008)
        cm = max(_bs_call_fwd(F0, K, s) * DF0, 0.05)
        pm = max(cm - DF0 * (F0 - K), 0.05)
        h = spread / 2.0
        qs.append(ChainQuote(float(K), max(cm - h, 0.01), cm + h,
                             max(pm - h, 0.01), pm + h))
    return ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)


PCS = (Leg(599.0, "P", -1), Leg(598.0, "P", 1))


class TestFillPriorMonotonicity:
    def test_more_legs_worse_or_equal_base(self):
        assert (base_fill_fraction("x", 1)
                <= base_fill_fraction("x", 2)
                <= base_fill_fraction("x", 4))
        assert DEFAULT_FILL_BY_N_LEGS[1] < DEFAULT_FILL_BY_N_LEGS[2]
        assert DEFAULT_FILL_BY_N_LEGS[2] < DEFAULT_FILL_BY_N_LEGS[4]

    def test_family_priors_match_spec(self):
        assert base_fill_fraction("long_call", 1) == pytest.approx(0.35)
        assert base_fill_fraction("put_credit", 2) == pytest.approx(0.50)
        assert base_fill_fraction("iron_condor", 4) == pytest.approx(0.65)

    def test_stale_quote_worsens(self):
        fresh, _ = fill_fraction_for("put_credit", n_legs=2,
                                     quote_age_seconds=0.0)
        stale, diag = fill_fraction_for("put_credit", n_legs=2,
                                        quote_age_seconds=30.0)
        assert stale > fresh
        assert "stale_quote" in diag["penalties"]

    def test_late_day_worsens(self):
        early, _ = fill_fraction_for("put_credit", n_legs=2,
                                     minutes_to_close=180.0)
        late, diag = fill_fraction_for("put_credit", n_legs=2,
                                       minutes_to_close=30.0)
        assert late > early
        assert "late_day" in diag["penalties"]

    def test_wide_spread_worsens(self):
        tight, _ = fill_fraction_for("put_credit", n_legs=2,
                                     relative_spread=0.02)
        wide, diag = fill_fraction_for("put_credit", n_legs=2,
                                       relative_spread=0.25)
        assert wide > tight
        assert "wide_spread" in diag["penalties"]

    def test_high_vol_worsens(self):
        calm, _ = fill_fraction_for("put_credit", n_legs=2, realized_vol=0.10)
        hot, diag = fill_fraction_for("put_credit", n_legs=2, realized_vol=0.40)
        assert hot > calm
        assert "high_vol" in diag["penalties"]

    def test_penalties_never_improve(self):
        """Stacking every adverse condition must not beat the base prior."""
        base, _ = fill_fraction_for("put_credit", n_legs=2)
        worse, _ = fill_fraction_for(
            "put_credit", n_legs=2,
            quote_age_seconds=60.0, minutes_to_close=15.0,
            relative_spread=0.30, realized_vol=0.50)
        assert worse >= base
        assert 0.0 <= worse <= 1.0

    def test_clipped_to_unit_interval(self):
        cfg = FillPriorConfig(stale_quote_penalty=0.9, late_day_penalty=0.9,
                              wide_spread_penalty=0.9, high_vol_penalty=0.9)
        frac, _ = fill_fraction_for(
            "iron_condor", n_legs=4, cfg=cfg,
            quote_age_seconds=60, minutes_to_close=10,
            relative_spread=0.5, realized_vol=0.5)
        assert frac == pytest.approx(1.0)


class TestWiderSpreadsWorseFills:
    def test_expected_credit_falls_as_spread_widens(self):
        tight = estimate_execution(PCS, quotes_from_chain(_chain(0.02)),
                                   "put_credit")
        wide = estimate_execution(PCS, quotes_from_chain(_chain(0.20)),
                                  "put_credit")
        assert tight is not None and wide is not None
        # Same mid (Black mid unchanged); wider book → larger concession
        assert wide.half_spread_cost > tight.half_spread_cost
        assert wide.expected_credit < tight.expected_credit
        assert wide.conservative_credit < tight.conservative_credit
        assert wide.round_trip_cost_expected > tight.round_trip_cost_expected

    def test_older_quotes_do_not_improve_expected_fill(self):
        q = quotes_from_chain(_chain(0.08))
        fresh = estimate_execution(PCS, q, "put_credit",
                                   quote_age_seconds=0.0)
        stale = estimate_execution(PCS, q, "put_credit",
                                   quote_age_seconds=60.0)
        assert stale.fill_fraction_expected >= fresh.fill_fraction_expected
        assert stale.expected_credit <= fresh.expected_credit + 1e-12
