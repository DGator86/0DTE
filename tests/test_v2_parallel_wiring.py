"""
tests/test_v2_parallel_wiring.py
================================
Acceptance for V2 parallel shadow path + review-fix hardening:
  * heuristic bundle is usable by PredictionPolicy
  * feeds / TickSnapshot option_rows drive GEX variants
  * RAS directional veto_escalation is direction-aware
  * implied_remaining_move never falls back to chan_bb_width
  * live_state carries parallel + v2_signals
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gate_scorer import MarketSnapshot
from prediction.inference import (
    HeuristicCandidateValueModel, heuristic_bundle_from_tick,
    make_bundle_provider, make_physical_forecast_provider,
)
from prediction.contracts import PredictionBundle
from policy.prediction_policy import PredictionPolicy, bundle_is_usable
from policy.contracts import PolicyInput, StructuralState
from regime_alignment import (
    EntrySnapshot, _score_veto_escalation, _veto_undermines,
)
from resample import RawBars
from spy0dte import OptionRow
from unified_loop import TickSnapshot, UnifiedOrchestrator

ET = ZoneInfo("America/New_York")


def _now():
    return dt.datetime(2026, 7, 10, 11, 30, tzinfo=ET)


def _market(**kw):
    base = dict(
        spot=600.0, net_gex=1e9, gamma_flip=598.0,
        call_wall=605.0, put_wall=595.0, gex_pct_rank=0.7,
        gex_rank_warm=True,
        vix9d=14.0, vix=15.0, vix3m=17.0, vvix=90.0, vvix_baseline=95.0,
        straddle_breakeven=3.0, expected_range=2.4,
        adx=16.0, rsi=50.0, bb_width=1.2, bb_width_baseline=1.0,
        vwap=600.0, vwap_reversion_count=0, tick_abs_mean=400.0,
        cvd_slope=0.0, now=_now(), has_catalyst=False,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def _bars(n=60):
    ts = np.array([_now() - dt.timedelta(minutes=n - i)
                   for i in range(n)], dtype="datetime64[ns]")
    px = np.linspace(599, 601, n)
    return RawBars(ts=ts, open=px, high=px + 0.2, low=px - 0.2,
                   close=px, volume=np.ones(n) * 1000)


def _rows(spot=600.0):
    rows = []
    for k in range(int(spot) - 5, int(spot) + 6):
        rows.append(OptionRow("put", float(k), 1000, 0.05, 1.0, 1.1, 0.3,
                              volume=200))
        rows.append(OptionRow("call", float(k), 1000, 0.05, 1.0, 1.1, 0.3,
                              volume=200))
    return rows


@dataclass
class _Feed:
    snap: TickSnapshot

    def snapshot(self, now):
        return self.snap

    def settlement_price(self, session_date):
        return 600.0


class TestHeuristicBundle:
    def test_usable_and_has_range_survive(self):
        snap = TickSnapshot(market=_market(), bars=_bars(), chain=None)
        bundle = heuristic_bundle_from_tick(
            snap, {"regime_bias_value": 62.0},
            snapshot_id="abc", symbol="SPY")
        assert bundle_is_usable(bundle)
        assert bundle.p_range_survive_30m is not None
        assert bundle.p_up_30m is not None
        assert 0.0 <= bundle.uncertainty <= 1.0

    def test_policy_accepts_heuristic_bundle(self):
        snap = TickSnapshot(market=_market(), bars=_bars())
        bundle = heuristic_bundle_from_tick(
            snap, {"regime_bias_value": 70.0}, snapshot_id="x")
        pin = PolicyInput(
            predictions=bundle,
            structural_state=StructuralState.from_market(snap.market),
            operational_risk_state={
                "hard_vetoes": [],
                "stand_down": False,
                "implied_remaining_move": 0.004,
            },
        )
        dec = PredictionPolicy().decide(pin)
        assert dec.action in ("TRADE", "NO_TRADE")
        assert dec.source == "v2"


class TestDirectionalVetoEscalation:
    def _entry(self, bias="bull", structure="LCS", vetoes=()):
        return EntrySnapshot(
            dominant_regime="trend",
            permitted_engine="directional",
            exec_regime="trend",
            context_regime="trend",
            direction_bias=bias,
            bias_value=70.0 if bias == "bull" else 30.0,
            vetoes=list(vetoes),
            net_gex=1e9,
            gamma_flip=598.0,
            flip_cushion=0.005,
            spot=600.0,
            structure=structure,
            structure_class="directional",
            dominant_confidence=70.0,
        )

    def test_below_flip_hostile_to_bullish(self):
        entry = self._entry(bias="bull")
        assert _veto_undermines("below_gamma_flip", entry) is True
        assert _veto_undermines("short_gamma", entry) is False

    def test_below_flip_benign_to_bearish(self):
        entry = self._entry(bias="bear", structure="LPS")
        assert _veto_undermines("below_gamma_flip", entry) is False

    def test_score_penalizes_bullish_below_flip(self):
        from regime_classifier import RegimeState
        entry = self._entry(bias="bull")
        regime = RegimeState(
            confidences={"trend": 70.0, "directional_confidence": 70.0},
            reliabilities={"trend": 1.0},
            dominant_regime="trend", permitted_engine="directional",
            vetoes=["below_gamma_flip"],
            global_information_gain=10.0,
            standardized={},
            stand_down=False,
        )
        score, note = _score_veto_escalation(regime, entry)
        assert score < 0
        assert "hostile" in note


class TestImpliedMoveFallback:
    def test_dual_run_journals_policy_keys(self):
        from rnd_extractor import ChainQuote, ChainSnapshot
        chain = ChainSnapshot(
            quotes=[ChainQuote(600.0, 1.0, 1.1, 1.0, 1.1)],
            spot=600.0, t_years=0.001, r=0.05,
        )
        snap = TickSnapshot(
            market=_market(expected_range=2.4, straddle_breakeven=3.0),
            bars=_bars(200),
            chain=chain,
            option_rows=_rows(),
            gex_feed_source="test",
        )
        orch = UnifiedOrchestrator(
            feed=_Feed(snap),
            policy_mode="shadow",
            prediction_bundle_provider=make_bundle_provider(symbol="SPY"),
        )
        result = orch.tick(_now())
        assert result is not None
        assert result.signals.get("policy_mode") == "shadow"
        assert "policy_fallback_used" in result.signals or \
            "v2_policy_action" in result.signals


class TestOptionRowsGex:
    def test_gex_variants_journaled_when_rows_attached(self):
        from rnd_extractor import ChainQuote, ChainSnapshot
        chain = ChainSnapshot(
            quotes=[ChainQuote(k, 1.0, 1.1, 1.0, 1.1)
                    for k in range(595, 606)],
            spot=600.0, t_years=0.001, r=0.05,
        )
        snap = TickSnapshot(
            market=_market(), bars=_bars(200), chain=chain,
            option_rows=_rows(), gex_feed_source="test",
        )
        orch = UnifiedOrchestrator(
            feed=_Feed(snap),
            policy_mode="shadow",
            prediction_bundle_provider=make_bundle_provider(symbol="SPY"),
        )
        result = orch.tick(_now())
        assert result is not None
        assert "gex_oi_net_gex" in result.signals
        assert "gex_rank_warm" in result.signals


class TestLiveStateParallel:
    def test_serialize_includes_parallel_and_v2(self):
        from dashboard.state import serialize_tick_result
        from decision_matrix import Decision, TradeIntent
        from regime_classifier import RegimeState

        intent = TradeIntent(
            exec_regime="compression", context_regime="compression",
            direction_bias="neutral", bias_value=50.0,
            decision=Decision(
                structure="IC", direction="both", conviction="MED",
                capture="theta", strike_rule="", anchor_tf="15m"),
            size_mult=1.0, vetoes=[], note="",
        )
        regime = RegimeState(
            confidences={"compression": 70.0},
            reliabilities={"compression": 1.0},
            dominant_regime="compression", permitted_engine="premium_selling",
            vetoes=[],
            global_information_gain=10.0,
            standardized={},
            stand_down=False,
        )

        @dataclass
        class _R:
            ts: object
            regime: object
            intent: object
            decision: object
            final_size_mult: float
            vetoes: list
            snapshot: object
            ras_results: list
            signals: dict

        snap = TickSnapshot(market=_market(), bars=_bars())
        result = _R(
            ts=_now(), regime=regime, intent=intent, decision=None,
            final_size_mult=0.0, vetoes=[], snapshot=snap, ras_results=[],
            signals={
                "policy_mode": "shadow",
                "v2_policy_structure": "IC",
                "v2_policy_action": "TRADE",
                "policy_disagreement": 0.0,
                "phys_v2_mean": 0.0,
                "gex_oi_net_gex": 1.2,
            },
        )
        payload = serialize_tick_result(result)
        assert "parallel" not in payload
        assert payload["legacy"]["parallel"]["structure"] == "IC"
        assert payload["forecast"]["parallel"]["structure"] == "IC"
        assert "v2_signals" not in payload
        assert "phys_v2_mean" in payload["forecast"]["v2_signals"]
        # live.v1 forecast section is always present; summary stays empty
        # when no v2_fc_* keys were journaled on this fixture.
        assert payload["forecast"]["source_version"] == "v2"
        assert payload["forecast"]["summary"] is None
        assert payload["system"]["compat_flat_keys"] is False


class TestV2ObservationOnStandDown:
    """V2 panels must populate even when legacy stands down / NT."""

    def _chain_snap(self, market):
        from rnd_extractor import ChainQuote, ChainSnapshot
        chain = ChainSnapshot(
            quotes=[ChainQuote(float(k), 2.0, 2.2, 2.0, 2.2)
                    for k in range(590, 611)],
            spot=600.0, t_years=0.001, r=0.05,
        )
        rows = []
        for k in range(590, 611):
            rows.append(OptionRow("put", float(k), 1000, 0.05, 2.0, 2.2, 0.5,
                                  volume=200))
            rows.append(OptionRow("call", float(k), 1000, 0.05, 2.0, 2.2, 0.5,
                                  volume=200))
        return TickSnapshot(
            market=market, bars=_bars(200), chain=chain,
            option_rows=rows, gex_feed_source="test",
        )

    def test_stand_down_journals_forecast_phys_ranker(self):
        # Hostile / high-vol snapshot that forces regime stand_down + NT.
        market = _market(
            net_gex=-5e9, gamma_flip=610.0, gex_pct_rank=0.1,
            vix9d=40.0, vix=35.0, vix3m=30.0, vvix=140.0,
            straddle_breakeven=8.0, expected_range=6.0,
            adx=40.0, rsi=20.0, bb_width=5.0, vwap=620.0,
            vwap_reversion_count=5, tick_abs_mean=2000.0,
            cvd_slope=-1.0, has_catalyst=True,
        )
        snap = self._chain_snap(market)
        bp = make_bundle_provider(symbol="SPY")
        orch = UnifiedOrchestrator(
            feed=_Feed(snap),
            policy_mode="shadow",
            prediction_bundle_provider=bp,
            physical_forecast_provider=make_physical_forecast_provider(bp),
            candidate_value_model=HeuristicCandidateValueModel(),
        )
        result = orch.tick(_now())
        assert result is not None
        assert result.decision is None  # stand-down / NT early return
        assert result.signals.get("v2_fc_p_up_30m") is not None
        assert result.signals.get("phys_v2_mean") is not None
        assert result.signals.get("phys_density_mode") is not None
        assert result.signals.get("v2_top_candidate_id") is not None

        from dashboard.state import serialize_tick_result
        payload = serialize_tick_result(result)
        assert payload["forecast"] is not None
        assert "p_up_30m" in payload["forecast"]
        assert "phys_v2_mean" in payload["forecast"]["v2_signals"]


class TestHeuristicRanker:
    def test_predict_returns_forecasts(self):
        model = HeuristicCandidateValueModel()
        rows = [
            {"ev": 0.15, "max_loss": 1.0, "prob_profit": 0.6,
             "prob_touch_short": 0.2, "credit": 0.3},
            {"ev": 0.05, "max_loss": 1.0, "prob_profit": 0.4,
             "prob_touch_short": 0.5, "credit": 0.1},
        ]
        out = model.predict(rows, candidate_ids=["a", "b"],
                            fill_uncertainty=[0.2, 0.4],
                            capital=[1.0, 1.0])
        assert len(out) == 2
        assert out[0].utility_score >= out[1].utility_score


class TestChainStoreWarmup:
    def test_old_recording_infers_cold_at_neutral_rank(self):
        from chain_store import _market_from_dict
        d = {
            "spot": 600.0, "net_gex": 0.0, "gamma_flip": 600.0,
            "call_wall": 605.0, "put_wall": 595.0, "gex_pct_rank": 0.5,
            "vix9d": 14, "vix": 15, "vix3m": 17, "vvix": 90,
            "vvix_baseline": 95, "straddle_breakeven": 3, "expected_range": 2,
            "adx": 15, "rsi": 50, "bb_width": 1, "bb_width_baseline": 1,
            "vwap": 600, "vwap_reversion_count": 0, "tick_abs_mean": 400,
            "cvd_slope": 0, "now": _now().isoformat(), "has_catalyst": False,
        }
        m = _market_from_dict(d)
        assert m.gex_rank_warm is False
