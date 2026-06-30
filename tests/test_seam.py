"""
tests/test_seam.py
==================
End-to-end seam tests for the Track B → Track A unified pipeline.

Covers:
  - Premium path: IC structure → iron_condor candidate with all journal fields
  - Directional path: forced LCS/LP structure → long_call_spread / long_put_spread
  - NT path: stand-down produces a journal row with candidate_present=0
  - regime_direction field propagated for all paths
  - No-chain path: logs stub row with regime_direction set
"""
from __future__ import annotations

import datetime as dt
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gate_scorer import MarketSnapshot
from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
from decision_engine import decide, EngineConfig, TradeDecision
from unified_loop import UnifiedOrchestrator, SyntheticUnifiedFeed, TickSnapshot, _no_trade_row
from decision_matrix import TradeIntent, Decision as MatrixDecision
from regime_classifier import RegimeState
from journal import Journal, COLUMNS

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _make_chain(spot: float, t_hours: float = 4.0, n_strikes: int = 20,
                r: float = 0.05) -> ChainSnapshot:
    T = t_hours / (24 * 365)
    DF = math.exp(-r * T)
    F = spot * math.exp(r * T)
    qs = []
    for K in np.arange(spot - n_strikes, spot + n_strikes + 1, 1.0):
        k = math.log(K / F)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F, K, s) * DF
        pm = max(cm - DF * (F - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=spot, t_years=T, r=r)


def _make_market(spot: float = 600.0, now: dt.datetime = None) -> MarketSnapshot:
    if now is None:
        now = dt.datetime(2026, 6, 27, 10, 0, tzinfo=ET)
    return MarketSnapshot(
        spot=spot, net_gex=4.0e9, gamma_flip=spot - 6.0,
        call_wall=spot + 15.0, put_wall=spot - 15.0, gex_pct_rank=0.80,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=90.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=12.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=spot, vwap_reversion_count=3,
        tick_abs_mean=450.0, cvd_slope=0.01,
        now=now, has_catalyst=False,
    )


# --------------------------------------------------------------------------- #
# decide() unit tests                                                           #
# --------------------------------------------------------------------------- #

class TestDecideDirectionField:
    def test_direction_propagates_to_row(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, direction="call")
        assert d.direction == "call"
        row = d.as_row()
        assert row["regime_direction"] == "call"

    def test_direction_default_empty(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain)
        assert d.direction == ""
        row = d.as_row()
        assert row["regime_direction"] == ""

    def test_row_has_all_journal_columns(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, direction="both")
        row = d.as_row()
        missing = [c for c in COLUMNS if c not in row]
        assert missing == [], f"as_row() missing columns: {missing}"

    def test_target_structure_ic(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, target_structure="IC", direction="both")
        assert d.direction == "both"
        if d.candidate:
            assert d.candidate.family in ("iron_condor",)

    def test_target_structure_lcs(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, target_structure="LCS", direction="call")
        assert d.direction == "call"
        if d.candidate:
            assert d.candidate.family == "long_call_spread"

    def test_target_structure_lps(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, target_structure="LPS", direction="put")
        assert d.direction == "put"
        if d.candidate:
            assert d.candidate.family == "long_put_spread"


# --------------------------------------------------------------------------- #
# _no_trade_row unit tests                                                      #
# --------------------------------------------------------------------------- #

class TestNoTradeRow:
    def _make_intent(self, structure: str, direction: str = "both") -> TradeIntent:
        return TradeIntent(
            exec_regime="range", context_regime="range",
            direction_bias=direction, bias_value=0.0,
            decision=MatrixDecision(structure=structure, direction=direction,
                                    conviction="HIGH", capture="test",
                                    strike_rule="test", anchor_tf="—"),
            size_mult=1.0, vetoes=[], note="",
        )

    def _make_regime(self) -> RegimeState:
        return RegimeState(
            confidences={}, reliabilities={},
            dominant_regime="range", permitted_engine="premium",
            vetoes=[], global_information_gain=0.0,
            standardized={}, stand_down=False,
        )

    def test_regime_direction_in_row(self):
        market = _make_market()
        intent = self._make_intent("NT", direction="none")
        regime = self._make_regime()
        row = _no_trade_row(market, intent, regime, direction="none")
        assert row["regime_direction"] == "none"
        assert "regime_direction" in row

    def test_direction_kwarg_overrides_intent(self):
        market = _make_market()
        intent = self._make_intent("IC", direction="both")
        regime = self._make_regime()
        row = _no_trade_row(market, intent, regime, reason="no_chain", direction="call")
        assert row["regime_direction"] == "call"

    def test_fallback_to_intent_direction(self):
        market = _make_market()
        intent = self._make_intent("IC", direction="put")
        regime = self._make_regime()
        row = _no_trade_row(market, intent, regime)
        assert row["regime_direction"] == "put"

    def test_no_trade_row_has_all_columns(self):
        market = _make_market()
        intent = self._make_intent("NT", direction="none")
        regime = self._make_regime()
        row = _no_trade_row(market, intent, regime)
        missing = [c for c in COLUMNS if c not in row]
        assert missing == [], f"_no_trade_row() missing columns: {missing}"


# --------------------------------------------------------------------------- #
# Journal schema test                                                           #
# --------------------------------------------------------------------------- #

class TestJournalSchema:
    def test_regime_direction_column_present(self):
        assert "regime_direction" in COLUMNS

    def test_journal_accepts_row_with_direction(self):
        market = _make_market()
        chain = _make_chain(market.spot)
        d = decide(market, chain, direction="call")
        row = d.as_row()
        jrn = Journal(":memory:")
        row_id = jrn.log(row)
        assert row_id > 0
        fetched = jrn.fetch()
        assert fetched[0]["regime_direction"] == "call"
        jrn.close()


# --------------------------------------------------------------------------- #
# UnifiedOrchestrator end-to-end tests                                          #
# --------------------------------------------------------------------------- #

class TestUnifiedOrchestratorSeam:
    def _run_ticks(self, chain=None, n_ticks=15):
        feed = SyntheticUnifiedFeed(days=5, chain=chain)
        jrn = Journal(":memory:")
        orch = UnifiedOrchestrator(feed=feed, journal=jrn)
        start = dt.datetime(2026, 6, 27, 9, 30, tzinfo=ET)
        ticks = [start + dt.timedelta(minutes=i) for i in range(n_ticks)]
        results = orch.run_replay(ticks)
        return results, jrn

    def test_no_chain_regime_direction_logged(self):
        results, jrn = self._run_ticks(chain=None, n_ticks=10)
        rows = jrn.fetch()
        assert len(rows) > 0
        # All rows must have regime_direction (may be empty string for some)
        for r in rows:
            assert "regime_direction" in r

    def test_with_chain_trade_row_has_direction(self):
        chain = _make_chain(600.0)
        results, jrn = self._run_ticks(chain=chain, n_ticks=20)
        rows = jrn.fetch()
        assert len(rows) > 0
        # Every row in the DB must have the column (value may be empty/None)
        for r in rows:
            assert "regime_direction" in r

    def test_nt_path_candidate_present_zero(self):
        """Stand-down / NT rows must have candidate_present=0."""
        results, jrn = self._run_ticks(chain=None, n_ticks=10)
        rows = jrn.fetch()
        for r in rows:
            if r["decision"] == "NO_TRADE" and r.get("legs_json") is None:
                assert r["candidate_present"] == 0

    def test_with_chain_was_traded_flag(self):
        chain = _make_chain(600.0)
        results, jrn = self._run_ticks(chain=chain, n_ticks=20)
        rows = jrn.fetch()
        for r in rows:
            if r["decision"] == "TRADE":
                assert r["was_traded"] == 1
            else:
                assert r["was_traded"] == 0
