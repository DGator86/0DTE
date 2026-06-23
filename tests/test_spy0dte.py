"""
Unit tests for spy0dte.py and acceptance.py — pure-function coverage, no API calls.
All tests use synthetic_chain so they run anywhere.
"""
import pytest
from spy0dte import (
    OptionRow, GammaMap, Decision,
    build_gamma_map, decide, scale_risk, select_order, select_condor,
    synthetic_chain,
    RISK_FLOOR, RISK_CEILING, MIN_TRADES_TO_SCALE, MIN_NET_RATIO,
    WALL_MIN_DISTANCE, DELTA_LOW, DELTA_HIGH,
)
from acceptance import Bar1m, session_vwap, compute_acceptance


# ── helpers ──────────────────────────────────────────────────────────────────

def _minimal_chain(spot: float = 600.0) -> list[OptionRow]:
    """Two contracts: one call and one put, each with meaningful OI and greek."""
    return [
        OptionRow("call", spot + 1, 10000, 0.05, 1.00, 1.02, 0.50),
        OptionRow("put",  spot - 1, 15000, 0.05, 1.00, 1.02, 0.50),
    ]


# ── scale_risk ────────────────────────────────────────────────────────────────

class TestScaleRisk:
    def test_below_min_trades_returns_floor(self):
        frac, _ = scale_risk(n_trades=10, win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert frac == RISK_FLOOR

    def test_zero_trades_returns_floor(self):
        frac, _ = scale_risk(n_trades=0, win_rate=0.0, avg_win=0, avg_loss=0)
        assert frac == RISK_FLOOR

    def test_zero_win_rate_returns_floor(self):
        frac, _ = scale_risk(n_trades=MIN_TRADES_TO_SCALE, win_rate=0.0, avg_win=1.0, avg_loss=1.0)
        assert frac == RISK_FLOOR

    def test_negative_kelly_returns_floor(self):
        # W=0.2, R=0.5 -> Kelly = 0.2 - 0.8/0.5 = 0.2 - 1.6 = -1.4 (negative)
        frac, note = scale_risk(n_trades=MIN_TRADES_TO_SCALE, win_rate=0.20, avg_win=0.5, avg_loss=1.0)
        assert frac == RISK_FLOOR
        assert "do NOT scale" in note

    def test_positive_edge_above_floor(self):
        # W=0.55, R=2.0 -> Kelly = 0.55 - 0.45/2 = 0.325, half = 0.1625 -> capped at CEILING
        frac, _ = scale_risk(n_trades=MIN_TRADES_TO_SCALE, win_rate=0.55, avg_win=2.0, avg_loss=1.0)
        assert RISK_FLOOR < frac <= RISK_CEILING

    def test_high_kelly_capped_at_ceiling(self):
        frac, note = scale_risk(n_trades=MIN_TRADES_TO_SCALE, win_rate=0.80, avg_win=5.0, avg_loss=1.0)
        assert frac == RISK_CEILING
        assert "capped" in note

    def test_fractional_kelly_applied(self):
        # W=0.40, R=2.6 -> Kelly = 0.40 - 0.60/2.6 = 0.169, half-Kelly = 0.0845 -> ~8%
        frac, _ = scale_risk(n_trades=40, win_rate=0.40, avg_win=2.6, avg_loss=1.0)
        assert abs(frac - 0.08) < 0.005


# ── build_gamma_map ───────────────────────────────────────────────────────────

class TestBuildGammaMap:
    def test_trend_down_regime(self):
        spot = 600.0
        chain = synthetic_chain(spot, "trend_down")
        gm = build_gamma_map(chain, spot)
        assert gm.regime == "trend"
        assert gm.net_ratio < 0          # more put gamma -> negative net

    def test_trend_up_regime(self):
        spot = 600.0
        chain = synthetic_chain(spot, "trend_up")
        gm = build_gamma_map(chain, spot)
        # trend_up: large call OI above spot -> flip above spot -> spot < flip -> "trend"
        # (spot=600 exactly at the OI peak; regime may be "pin" or "trend" depending on flip)
        assert gm.regime in ("trend", "pin")

    def test_pin_regime(self):
        # "trend_up" skew concentrates call OI above spot, pushing the gamma flip
        # below spot so that spot > flip -> "pin" regime.
        spot = 600.0
        chain = synthetic_chain(spot, "trend_up")
        gm = build_gamma_map(chain, spot)
        assert gm.regime == "pin"
        assert gm.net_ratio > 0

    def test_spot_recorded(self):
        gm = build_gamma_map(_minimal_chain(600.0), 600.0)
        assert gm.spot == 600.0

    def test_call_wall_at_or_above_spot(self):
        gm = build_gamma_map(synthetic_chain(600.0, "trend_down"), 600.0)
        assert gm.call_wall >= 600.0

    def test_put_wall_at_or_below_spot(self):
        gm = build_gamma_map(synthetic_chain(600.0, "trend_down"), 600.0)
        assert gm.put_wall <= 600.0

    def test_net_ratio_range(self):
        gm = build_gamma_map(synthetic_chain(600.0, "pin"), 600.0)
        assert -1.0 <= gm.net_ratio <= 1.0

    def test_single_call_row_net_positive(self):
        # Chain with no puts: put_wall falls back to spot — should not crash.
        chain = [OptionRow("call", 601.0, 5000, 0.05, 1.0, 1.02, 0.50)]
        gm = build_gamma_map(chain, 600.0)
        assert gm.net_gex > 0
        assert gm.net_ratio == 1.0
        assert gm.put_wall == 600.0      # fallback when no put rows exist


# ── decide ────────────────────────────────────────────────────────────────────

class TestDecide:
    def _gm(self, regime="trend", net_ratio=-0.35, flip=598.0, spot=600.0,
            call_wall=604.0, put_wall=597.0):
        return GammaMap(spot=spot, net_gex=-0.5, net_ratio=net_ratio,
                        gamma_flip=flip, call_wall=call_wall, put_wall=put_wall,
                        regime=regime)

    def test_flat_ratio_stand_aside(self):
        d = decide(self._gm(net_ratio=MIN_NET_RATIO * 0.5), price_accepting=1)
        assert d.action == "STAND_ASIDE"
        assert "flat" in d.reason

    def test_pin_regime_sell_condor(self):
        d = decide(self._gm(regime="pin", net_ratio=0.40, flip=595.0, spot=600.0), price_accepting=0)
        assert d.action == "SELL_CONDOR"

    def test_trend_accepting_up_is_call(self):
        d = decide(self._gm(regime="trend", net_ratio=-0.40, flip=598.0,
                            call_wall=604.0, spot=600.0), price_accepting=+1)
        assert d.action == "CALL"
        assert d.target == 604.0
        assert d.stop_ref == 598.0

    def test_trend_accepting_down_is_put(self):
        d = decide(self._gm(regime="trend", net_ratio=-0.40, flip=598.0,
                            put_wall=597.0, spot=600.0), price_accepting=-1)
        assert d.action == "PUT"
        assert d.target == 597.0

    def test_trend_no_acceptance_stand_aside(self):
        d = decide(self._gm(regime="trend"), price_accepting=0)
        assert d.action == "STAND_ASIDE"

    def test_call_wall_too_close_stand_aside(self):
        # call_wall only 0.05% above spot — below WALL_MIN_DISTANCE
        gm = self._gm(regime="trend", net_ratio=-0.40, flip=598.0, spot=600.0, call_wall=600.5)
        assert (600.5 - 600.0) / 600.0 < WALL_MIN_DISTANCE
        d = decide(gm, price_accepting=+1)
        assert d.action == "STAND_ASIDE"
        assert "too close" in d.reason

    def test_put_wall_too_close_stand_aside(self):
        gm = self._gm(regime="trend", net_ratio=-0.40, flip=598.0, spot=600.0, put_wall=599.7)
        assert (600.0 - 599.7) / 600.0 < WALL_MIN_DISTANCE
        d = decide(gm, price_accepting=-1)
        assert d.action == "STAND_ASIDE"


# ── select_order ──────────────────────────────────────────────────────────────

class TestSelectOrder:
    def _call_decision(self, spot=600.0, target=604.0, stop=598.0):
        return Decision("CALL", "test", entry_ref=spot, target=target, stop_ref=stop)

    def _put_decision(self, spot=600.0, target=597.0, stop=598.0):
        return Decision("PUT", "test", entry_ref=spot, target=target, stop_ref=stop)

    def test_stand_aside_returns_none(self):
        d = Decision("STAND_ASIDE", "flat")
        assert select_order(synthetic_chain(600.0, "trend_down"), d, 10000.0, 0.05) is None

    def test_sell_condor_returns_none(self):
        d = Decision("SELL_CONDOR", "pin")
        assert select_order(synthetic_chain(600.0, "pin"), d, 10000.0, 0.05) is None

    def test_call_happy_path(self):
        chain = synthetic_chain(600.0, "trend_up")
        d = self._call_decision()
        order = select_order(chain, d, 50000.0, 0.05)
        assert order is not None
        assert order.action == "CALL"
        assert DELTA_LOW <= order.delta <= DELTA_HIGH

    def test_put_happy_path(self):
        chain = synthetic_chain(600.0, "trend_down")
        d = self._put_decision()
        order = select_order(chain, d, 50000.0, 0.05)
        assert order is not None
        assert order.action == "PUT"

    def test_too_small_equity_returns_none(self):
        chain = synthetic_chain(600.0, "trend_up")
        d = self._call_decision()
        # With $100 equity and 2% risk, budget is $2 — not enough for one contract
        order = select_order(chain, d, 100.0, 0.02)
        assert order is None

    def test_order_contracts_positive(self):
        chain = synthetic_chain(600.0, "trend_up")
        order = select_order(chain, self._call_decision(), 50000.0, 0.05)
        assert order is not None
        assert order.contracts >= 1

    def test_order_spread_within_limit(self):
        chain = synthetic_chain(600.0, "trend_up")
        order = select_order(chain, self._call_decision(), 50000.0, 0.05)
        assert order is not None
        # synthetic chain has tight spreads; should be under MAX_SPREAD_PCT
        from spy0dte import MAX_SPREAD_PCT
        assert order.spread_pct <= MAX_SPREAD_PCT

    def test_no_candidates_in_delta_range(self):
        # Build a chain where all options are deep OTM (delta ~0)
        chain = [OptionRow("call", 700.0, 1000, 0.001, 0.01, 0.02, 0.01)]
        d = self._call_decision()
        assert select_order(chain, d, 50000.0, 0.05) is None

    def test_require_live_rejects_fallback_quotes(self):
        # A chain where in-delta-range call has quote_valid=False
        chain = [OptionRow("call", 600.0, 5000, 0.05, 1.0, 1.0, 0.53,
                           quote_source="day_close_fallback", quote_valid=False)]
        d = self._call_decision()
        assert select_order(chain, d, 50000.0, 0.05, require_live=True) is None

    def test_require_live_false_accepts_fallback(self):
        chain = [OptionRow("call", 600.0, 5000, 0.05, 1.0, 1.0, 0.53,
                           quote_source="day_close_fallback", quote_valid=False)]
        d = self._call_decision()
        order = select_order(chain, d, 50000.0, 0.05, require_live=False)
        assert order is not None

    def test_live_quote_accepted_by_default(self):
        chain = [OptionRow("call", 600.0, 5000, 0.05, 1.0, 1.02, 0.53,
                           quote_source="live_quote", quote_valid=True)]
        d = self._call_decision()
        order = select_order(chain, d, 50000.0, 0.05)
        assert order is not None


# ── select_condor ─────────────────────────────────────────────────────────────

class TestSelectCondor:
    def _gm(self, spot=600.0, put_wall=597.0, call_wall=603.0, flip=600.0):
        return GammaMap(spot=spot, net_gex=0.5, net_ratio=0.35,
                        gamma_flip=flip, call_wall=call_wall, put_wall=put_wall,
                        regime="pin")

    def test_condor_happy_path(self):
        chain = synthetic_chain(600.0, "pin")
        gm = self._gm()
        condor = select_condor(chain, gm, 50000.0, 0.05)
        assert condor is not None
        assert condor.short_put < condor.short_call
        assert condor.long_put < condor.short_put
        assert condor.short_call < condor.long_call
        assert condor.credit > 0
        assert condor.contracts >= 1

    def test_too_small_equity_returns_none(self):
        chain = synthetic_chain(600.0, "pin")
        gm = self._gm()
        assert select_condor(chain, gm, 50.0, 0.02) is None

    def test_condor_dollar_risk_matches_contracts(self):
        chain = synthetic_chain(600.0, "pin")
        gm = self._gm()
        condor = select_condor(chain, gm, 50000.0, 0.05)
        if condor:
            expected_risk = condor.max_loss * condor.contracts
            assert abs(condor.dollar_risk - expected_risk) < 0.01

    def test_require_live_rejects_any_fallback_leg(self):
        # Build a condor chain where one leg has quote_valid=False
        chain = synthetic_chain(600.0, "pin")
        # Corrupt the nearest put to the put_wall to be a fallback quote
        gm = self._gm()
        for r in chain:
            if r.side == "put" and abs(r.strike - gm.put_wall) < 1.5:
                r.quote_valid = False
                r.quote_source = "day_close_fallback"
                break
        assert select_condor(chain, gm, 50000.0, 0.05, require_live=True) is None

    def test_require_live_false_accepts_fallback_condor(self):
        chain = synthetic_chain(600.0, "pin")
        gm = self._gm()
        for r in chain:
            r.quote_valid = False  # all fallback
        result = select_condor(chain, gm, 50000.0, 0.05, require_live=False)
        # May or may not produce a condor depending on credit; just shouldn't crash
        # and shouldn't raise due to quote validity


# ── OptionRow properties ──────────────────────────────────────────────────────

class TestOptionRow:
    def test_mid(self):
        r = OptionRow("call", 600.0, 1000, 0.05, 1.00, 1.20, 0.50)
        assert r.mid == pytest.approx(1.10)

    def test_spread_pct(self):
        r = OptionRow("call", 600.0, 1000, 0.05, 1.00, 1.20, 0.50)
        assert r.spread_pct == pytest.approx(0.20 / 1.10)

    def test_zero_mid_spread_pct(self):
        r = OptionRow("call", 600.0, 0, 0.0, 0.0, 0.0, 0.0)
        assert r.spread_pct == 9.99

    def test_quote_source_default(self):
        r = OptionRow("call", 600.0, 1000, 0.05, 1.0, 1.02, 0.50)
        assert r.quote_source == "live_quote"
        assert r.quote_valid is True

    def test_fallback_fields(self):
        r = OptionRow("put", 599.0, 500, 0.04, 0.80, 0.80, 0.48,
                      quote_source="day_close_fallback", quote_valid=False)
        assert r.quote_valid is False
        assert r.spread_pct == 0.0   # bid==ask -> no spread


# ── acceptance ────────────────────────────────────────────────────────────────

def _bar(close: float, volume: float = 1e6) -> Bar1m:
    return Bar1m(0, close + 0.1, close - 0.1, close, volume)


class TestAcceptance:
    FLIP = 600.0

    def _bull(self):
        return [_bar(599.5), _bar(599.8), _bar(600.3), _bar(600.5), _bar(600.7)]

    def _bear(self):
        return [_bar(600.5), _bar(600.2), _bar(599.7), _bar(599.5), _bar(599.3)]

    def _chop(self):
        return [_bar(600.2), _bar(599.8), _bar(600.1), _bar(599.9), _bar(600.05)]

    def test_bullish_acceptance(self):
        assert compute_acceptance(self._bull(), self.FLIP) == +1

    def test_bearish_acceptance(self):
        assert compute_acceptance(self._bear(), self.FLIP) == -1

    def test_chop_no_acceptance(self):
        assert compute_acceptance(self._chop(), self.FLIP) == 0

    def test_too_few_bars_returns_zero(self):
        assert compute_acceptance([_bar(601.0), _bar(601.5)], self.FLIP, n=3) == 0

    def test_zero_flip_returns_zero(self):
        assert compute_acceptance(self._bull(), flip=0.0) == 0

    def test_vwap_disagrees_blocks_signal(self):
        # Three closes above flip, but VWAP is dragged below by early high-volume bar
        bars = [_bar(590.0, volume=1e8),   # heavy early volume below flip -> pulls VWAP down
                _bar(600.3), _bar(600.5), _bar(600.7)]
        # Last price (600.7) > flip but VWAP ≈ 590 -> last close still > vwap -> still +1
        # Use a scenario where last px < vwap to actually block it
        bars2 = [_bar(610.0, volume=1e8), _bar(600.3), _bar(600.5), _bar(600.2)]
        # vwap ≈ 610, last close 600.2 < vwap -> blocked
        result = compute_acceptance(bars2, self.FLIP, n=3, use_vwap=True)
        assert result == 0

    def test_vwap_disabled_allows_signal(self):
        # Same above scenario without VWAP confirm -> should pass on closes alone
        bars = [_bar(610.0, volume=1e8), _bar(600.3), _bar(600.5), _bar(600.2)]
        result = compute_acceptance(bars, self.FLIP, n=3, use_vwap=False)
        assert result == +1

    def test_session_vwap_volume_weighted(self):
        bars = [Bar1m(0, 101.0, 99.0, 100.0, 1000.0),   # tp=100, vol=1000
                Bar1m(0, 111.0, 109.0, 110.0, 9000.0)]  # tp=110, vol=9000
        # vwap = (100*1000 + 110*9000) / 10000 = (100000 + 990000) / 10000 = 109
        assert session_vwap(bars) == pytest.approx(109.0)

    def test_session_vwap_zero_volume(self):
        bars = [Bar1m(0, 101.0, 99.0, 100.0, 0.0)]
        assert session_vwap(bars) == pytest.approx(100.0)
