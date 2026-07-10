"""
tests/test_gex_variants.py
==========================
PR 9 acceptance — GEX measurement research program:
  * all variants share the GEXSnapshot contract;
  * OI provider matches spy0dte.build_gamma_map;
  * missing volume does not contaminate OI;
  * disagreement is journaled;
  * no variant affects gate / MarketSnapshot policy fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gex.base import compute_all_variants, compute_disagreement
from gex.contracts import GEXSnapshot, GexAssumption, GexVariantId
from gex.oi import OiGexProvider
from gex.volume_proxy import VolumeProxyProvider
from gex.weekly import WeeklyGexProvider, decay_weight
from spy0dte import OptionRow, build_gamma_map


def _rows(spot=600.0, with_volume=True):
    """Synthetic chain with asymmetric OI so flip/walls are well-defined."""
    rows = []
    for k in range(int(spot) - 10, int(spot) + 11):
        # more put OI below, more call OI above
        put_oi = 5000 if k <= spot else 800
        call_oi = 5000 if k >= spot else 800
        put_vol = (2000 if k <= spot else 100) if with_volume else 0
        call_vol = (2000 if k >= spot else 100) if with_volume else 0
        g = 0.05
        rows.append(OptionRow("put", float(k), put_oi, g, 1.0, 1.1, 0.3,
                              volume=put_vol))
        rows.append(OptionRow("call", float(k), call_oi, g, 1.0, 1.1, 0.3,
                              volume=call_vol))
    return rows


@dataclass
class _WeeklyRow:
    side: str
    strike: float
    oi: int
    gamma: float
    volume: int = 0
    dte_days: float = 0.0
    bid: float = 1.0
    ask: float = 1.1
    delta: float = 0.3


class TestContract:
    def test_all_variants_same_fields(self):
        rows = _rows()
        bundle = compute_all_variants(spot=600.0, rows_0dte=rows)
        for snap in bundle.snapshots():
            assert isinstance(snap, GEXSnapshot)
            d = snap.to_dict()
            for key in ("net_gex", "gamma_flip", "call_wall", "put_wall",
                        "gex_concentration", "wall_concentration",
                        "quality_score", "assumption_set", "source_age"):
                assert key in d
        sig = bundle.to_signals_json()
        assert "gex_oi_net_gex" in sig
        assert "gex_volume_net_gex" in sig
        assert "gex_hybrid_net_gex" in sig
        assert "gex_disagree_n_variants" in sig


class TestOiMatchesBaseline:
    def test_matches_build_gamma_map(self):
        rows = _rows()
        gm = build_gamma_map(rows, 600.0)
        oi = OiGexProvider().compute(spot=600.0, rows=rows)
        assert oi.variant == GexVariantId.OI
        assert oi.net_gex == pytest.approx(gm.net_gex, abs=1e-3)
        assert oi.gamma_flip == pytest.approx(gm.gamma_flip, abs=0.05)
        assert oi.call_wall == pytest.approx(gm.call_wall)
        assert oi.put_wall == pytest.approx(gm.put_wall)
        assert oi.quality_score > 0


class TestVolumeIsolation:
    def test_missing_volume_does_not_change_oi(self):
        rows_vol = _rows(with_volume=True)
        rows_novol = _rows(with_volume=False)
        oi_a = OiGexProvider().compute(spot=600.0, rows=rows_vol)
        oi_b = OiGexProvider().compute(spot=600.0, rows=rows_novol)
        assert oi_a.net_gex == pytest.approx(oi_b.net_gex)
        assert oi_a.gamma_flip == pytest.approx(oi_b.gamma_flip)

        vol = VolumeProxyProvider().compute(spot=600.0, rows=rows_novol)
        assert vol.missing_volume is True
        assert vol.quality_score == 0.0
        assert not vol.is_finite  # nan net_gex

    def test_volume_uses_volume_not_oi(self):
        # Extreme: put volume huge below, call volume tiny — should differ from OI
        rows = []
        for k in range(590, 611):
            rows.append(OptionRow("put", float(k), oi=100, gamma=0.05,
                                  bid=1, ask=1.1, delta=0.3,
                                  volume=10000 if k < 600 else 10))
            rows.append(OptionRow("call", float(k), oi=10000, gamma=0.05,
                                  bid=1, ask=1.1, delta=0.3,
                                  volume=10 if k > 600 else 10))
        oi = OiGexProvider().compute(spot=600.0, rows=rows)
        vol = VolumeProxyProvider().compute(spot=600.0, rows=rows)
        assert vol.missing_volume is False
        assert vol.is_finite
        # Different weighting → levels need not match
        assert (oi.net_gex != pytest.approx(vol.net_gex, abs=1e-6)
                or oi.put_wall != vol.put_wall)


class TestWeeklyDecay:
    def test_decay_weights(self):
        assert decay_weight(0.0, "sqrt_dte") == pytest.approx(1.0)
        assert decay_weight(3.0, "sqrt_dte") < 1.0
        assert decay_weight(0.0, "linear_dte") == pytest.approx(1.0)
        assert decay_weight(7.0, "linear_dte") < decay_weight(1.0, "linear_dte")

    def test_weekly_includes_dte(self):
        base = _rows()
        weekly_rows = [
            _WeeklyRow(r.side, r.strike, r.oi, r.gamma, r.volume, dte_days=0.0)
            for r in base
        ] + [
            _WeeklyRow(r.side, r.strike, r.oi // 2, r.gamma, 0, dte_days=7.0)
            for r in base
        ]
        snap = WeeklyGexProvider().compute(spot=600.0, rows=weekly_rows)
        assert snap.n_expirations >= 2
        assert snap.quality_score > 0
        assert snap.is_finite


class TestHybridAndDisagreement:
    def test_hybrid_prefers_oi_when_volume_missing(self):
        rows = _rows(with_volume=False)
        bundle = compute_all_variants(spot=600.0, rows_0dte=rows,
                                      minute_of_session=10.0)
        assert bundle.volume.missing_volume
        assert bundle.hybrid.is_finite
        # Early session + no volume → hybrid close to OI
        assert bundle.hybrid.net_gex == pytest.approx(bundle.oi.net_gex, abs=0.5)

    def test_disagreement_when_signs_differ(self):
        pos = GEXSnapshot(
            net_gex=1.0, gamma_flip=599.0, call_wall=605.0, put_wall=595.0,
            gex_concentration=0.2, wall_concentration=0.3, quality_score=1.0,
            assumption_set=GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS,
            variant=GexVariantId.OI)
        neg = GEXSnapshot(
            net_gex=-1.0, gamma_flip=601.0, call_wall=608.0, put_wall=592.0,
            gex_concentration=0.2, wall_concentration=0.3, quality_score=1.0,
            assumption_set=GexAssumption.SAME_DAY_PUT_FLOW_ALT,
            variant=GexVariantId.VOLUME)
        d = compute_disagreement([pos, neg])
        assert d.net_gex_sign_disagree is True
        assert d.flip_spread == pytest.approx(2.0)
        assert d.to_signals()["gex_disagree_sign"] == 1.0


class TestPolicyUnaffected:
    def test_gates_still_use_market_snapshot_oi(self):
        """Variant hybrid flipping sign must not change evaluate_gates input."""
        from gate_scorer import MarketSnapshot, GateConfig, evaluate as gate_evaluate
        import datetime as dt
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        now = dt.datetime(2026, 7, 10, 11, 0, tzinfo=ET)
        # Long-gamma market (positive net_gex) — premium gate should not hit GEX_SHORT
        market = MarketSnapshot(
            spot=600.0, net_gex=2.0, gamma_flip=595.0,
            call_wall=610.0, put_wall=590.0, gex_pct_rank=0.8,
            vix9d=12, vix=13, vix3m=15, vvix=90, vvix_baseline=95,
            straddle_breakeven=3.0, expected_range=2.5,
            adx=12, rsi=50, bb_width=1.0, bb_width_baseline=1.2,
            vwap=600, vwap_reversion_count=2,
            tick_abs_mean=400, cvd_slope=0.0,
            now=now, has_catalyst=False, gex_rank_warm=True,
        )
        # Even if hybrid would be short-gamma, gates read market.net_gex
        rows = _rows(with_volume=True)
        bundle = compute_all_variants(spot=600.0, rows_0dte=rows)
        assert market.net_gex > 0  # policy input unchanged by bundle
        result = gate_evaluate(market, GateConfig())
        assert "GEX_SHORT" not in result.failed_gates
        # Bundle exists independently
        assert bundle.oi.is_finite

    def test_unified_loop_journals_variants_without_changing_decision(self, tmp_path):
        import datetime as dt
        import json
        from zoneinfo import ZoneInfo
        from journal import Journal
        from unified_loop import TickSnapshot, UnifiedOrchestrator
        from tests.test_physical_distribution_independence import (
            _market, _bars, _chain, NOW,
        )
        from resample import RawBars

        class _Feed:
            def __init__(self):
                self._rows = _rows(with_volume=True)

            def snapshot(self, now):
                return TickSnapshot(
                    market=_market(), bars=_bars(), chain=_chain(),
                    option_rows=self._rows, gex_feed_source="TestFeed")

            def settlement_price(self, session_date):
                return 600.0

        j = Journal(db_path=str(tmp_path / "j.db"))
        orch = UnifiedOrchestrator(feed=_Feed(), journal=j)
        result = orch.tick(NOW)
        assert result is not None
        row = j.conn.execute(
            "SELECT net_gex, signals_json FROM evaluations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        # Policy column is still the MarketSnapshot OI value
        assert row[0] == pytest.approx(_market().net_gex)
        sig = json.loads(row[1])
        assert "gex_oi_net_gex" in sig
        assert "gex_disagree_n_variants" in sig
        # Authoritative marker present
        assert sig.get("gex_authoritative") == "oi"
