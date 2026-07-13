"""
tests/test_pin_premium.py
=========================
Pin-at-flip should sell premium (IC/IF), not buy breakout debit — even under
negative GEX / elevated ADX. Covers detector, matrix soft-exempt, gate,
selector, and PredictionPolicy parity.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gate_scorer import MarketSnapshot, GateConfig, Decision, evaluate
from pin_regime import assess_pin, PinConfig
from decision_matrix import decide_from_matrix, PREMIUM_STRUCTURES
from mtf_matrix import build_matrix, demo_input, regime_rows
from policy.prediction_policy import PredictionPolicy
from policy.contracts import PolicyInput, StructuralState
from prediction.inference import heuristic_bundle_from_tick
from unified_loop import TickSnapshot

ET = ZoneInfo("America/New_York")


def _now():
    return dt.datetime(2026, 7, 13, 13, 30, tzinfo=ET)


def _pin_market(**kw):
    """Dashboard-like tape: spot glued to flip, put wall $1 under, short GEX."""
    base = dict(
        spot=749.96, net_gex=-2.24e9, gamma_flip=749.96,
        call_wall=752.0, put_wall=749.0, gex_pct_rank=0.4,
        gex_rank_warm=True,
        vix9d=13.5, vix=14.29, vix3m=16.0, vvix=90.0, vvix_baseline=95.0,
        straddle_breakeven=2.5, expected_range=2.0,
        adx=22.0, rsi=48.0, bb_width=0.8, bb_width_baseline=1.0,
        vwap=750.5, vwap_reversion_count=1, tick_abs_mean=400.0,
        cvd_slope=-0.05, now=_now(), has_catalyst=False,
    )
    base.update(kw)
    return MarketSnapshot(**base)


class TestAssessPin:
    def test_flip_pin_detected_under_short_gex(self):
        pin = assess_pin(_pin_market())
        assert pin.is_pin is True
        assert abs(pin.zg_pct) < 1e-9
        assert pin.inside_walls is True

    def test_wide_of_flip_not_pin(self):
        pin = assess_pin(_pin_market(spot=755.0, gamma_flip=749.96))
        assert pin.is_pin is False

    def test_prefer_fly_when_very_tight(self):
        pin = assess_pin(_pin_market(), cfg=PinConfig(fly_zg_frac=0.001))
        assert pin.prefer_fly is True


class TestMatrixPinOverride:
    def _rows_regimes(self):
        rows = build_matrix(demo_input())
        return rows, regime_rows(rows)

    def test_short_gamma_no_longer_flips_credit_under_pin(self):
        rows, regimes = self._rows_regimes()
        pin = assess_pin(_pin_market())
        assert pin.is_pin
        # Force a premium cell via pin remap path: even with short_gamma veto,
        # credit must survive.
        intent = decide_from_matrix(
            rows, regimes,
            vetoes=["short_gamma_regime", "trending"],
            pin=pin,
        )
        assert intent.decision.structure in PREMIUM_STRUCTURES | {"NT"}
        # If the table landed on premium, soft-exempt must keep it credit.
        if intent.decision.structure in PREMIUM_STRUCTURES:
            assert "premium veto" not in intent.note
            assert "pin soft-exempt" in intent.note

    def test_breakout_remapped_toward_compression(self):
        rows, regimes = self._rows_regimes()
        pin = assess_pin(_pin_market())
        raw = decide_from_matrix(rows, regimes, vetoes=[], pin=None)
        pinned = decide_from_matrix(rows, regimes, vetoes=[], pin=pin)
        assert pinned.exec_regime == "compression"
        assert pinned.context_regime == "compression"
        assert pinned.decision.structure in PREMIUM_STRUCTURES
        if (raw.exec_regime, raw.context_regime) != ("compression", "compression"):
            assert "pin force" in pinned.note

    def test_trend_compression_bear_becomes_credit_under_pin(self):
        """Dashboard failure mode: trend×compression×bear → LPS while pinned."""
        rows, regimes = self._rows_regimes()
        pin = assess_pin(_pin_market())
        assert pin.is_pin
        pinned = decide_from_matrix(
            rows, regimes,
            vetoes=["short_gamma_regime"],
            pin=pin,
        )
        assert pinned.exec_regime == "compression"
        assert pinned.context_regime == "compression"
        assert pinned.decision.structure in PREMIUM_STRUCTURES
        assert pinned.decision.structure != "LPS"


class TestGatePinExempt:
    def test_premium_gate_passes_short_gex_when_pinned(self):
        m = _pin_market()
        blocked = evaluate(m, GateConfig(), structure_class="premium",
                           pin_active=False)
        assert blocked.decision is Decision.NO_GO
        assert any("GEX_SHORT" in g for g in blocked.failed_gates)

        allowed = evaluate(m, GateConfig(), structure_class="premium",
                           pin_active=True)
        assert allowed.decision is Decision.GO
        assert not any("GEX_SHORT" in g for g in allowed.failed_gates)
        assert not any("BELOW_FLIP" in g for g in allowed.failed_gates)
        assert not any("TRENDING" in g for g in allowed.failed_gates)

    def test_term_inverted_soft_exempt_under_pin(self):
        m = _pin_market(vix9d=18.0, vix=16.0, vix3m=15.0)  # backwardation
        blocked = evaluate(m, GateConfig(), structure_class="premium",
                           pin_active=False)
        assert any("TERM_INVERTED" in g for g in blocked.failed_gates)
        allowed = evaluate(m, GateConfig(), structure_class="premium",
                           pin_active=True)
        assert not any("TERM_INVERTED" in g for g in allowed.failed_gates)
        assert allowed.decision is Decision.GO


class TestSelectorPinExempt:
    def test_credit_survives_short_gamma_when_pin_active(self):
        from spread_selector import GammaContext, SelectorConfig, _evaluate, Leg
        from rnd_extractor import (
            ChainQuote, ChainSnapshot, extract_rnd, RNDConfig, _bs_call_fwd,
        )
        import math

        spot = 750.0
        T0, r0 = 4.0 / (24 * 365), 0.05
        DF0 = math.exp(-r0 * T0)
        F0 = spot * math.exp(r0 * T0)
        qs = []
        for K in np.arange(spot - 10, spot + 11, 1.0):
            k = math.log(K / F0)
            s = max(0.006 - 0.03 * k, 0.0008)
            cm = _bs_call_fwd(F0, K, s) * DF0
            pm = max(cm - DF0 * (F0 - K), 0.0)
            h = 0.01 + 0.002 * max(cm, pm)
            qs.append(ChainQuote(
                float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))
        chain = ChainSnapshot(qs, spot=spot, t_years=T0, r=r0)
        rnd = extract_rnd(chain, RNDConfig())
        phys = np.ones_like(rnd.grid)
        phys = phys / (np.sum(phys) * (rnd.grid[1] - rnd.grid[0]))
        ctx_blocked = GammaContext(
            spot=spot, call_wall=752.0, put_wall=749.0,
            gamma_flip=750.0, net_gex=-2e9, pin_active=False,
        )
        ctx_pin = GammaContext(
            spot=spot, call_wall=752.0, put_wall=749.0,
            gamma_flip=750.0, net_gex=-2e9, pin_active=True,
        )
        legs_pcs = (Leg(spot - 1, "P", -1), Leg(spot - 3, "P", 1))
        cfg = SelectorConfig(
            veto_short_below_flip=False, min_ev=-10.0, max_touch_short=1.0)
        blocked = _evaluate(
            "put_credit", legs_pcs, chain, rnd, phys, ctx_blocked, cfg, {})
        pinned = _evaluate(
            "put_credit", legs_pcs, chain, rnd, phys, ctx_pin, cfg, {})
        assert blocked is not None
        assert "short_gamma_regime" in (blocked.veto_reasons or ())
        assert pinned is not None
        assert "short_gamma_regime" not in (pinned.veto_reasons or ())
        assert pinned.passes_vetoes is True


class TestPolicyPinPremium:
    def test_v2_prefers_premium_under_pin(self):
        m = _pin_market()
        snap = TickSnapshot(market=m, bars=None, chain=None)
        bundle = heuristic_bundle_from_tick(
            snap, {"regime_bias_value": 50.0}, snapshot_id="pin-test")
        # Force usable range survival for eligibility
        from dataclasses import replace
        bundle = replace(
            bundle,
            p_range_survive_30m=0.70,
            uncertainty=0.30,
            data_quality=0.9,
            feature_coverage=0.9,
        )
        pin = PolicyInput(
            predictions=bundle,
            structural_state=StructuralState.from_market(m),
            operational_risk_state={
                "hard_vetoes": ["short_gamma_regime", "trending"],
                "stand_down": False,
                "implied_remaining_move": 0.003,
                "pin_active": True,
            },
        )
        dec = PredictionPolicy().decide(pin)
        assert dec.action == "TRADE"
        assert dec.structure_code in PREMIUM_STRUCTURES
