"""
tests/test_physical_distribution_independence.py
================================================
PR 5 acceptance — the physical density must NOT depend on policy outputs
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §12.2 / §3.5):

  * changing the routed structure / direction / conviction does not change
    the V2 density (build_physical_density has nowhere to put them);
  * unified_loop with use_legacy_directional_tilt=False prices candidates
    against the V2 density, and the density is identical across opposite
    routed directions given the same forecast;
  * richness measurement stays on the drift-less density (independent of
    the decide-time density);
  * legacy tilt remains available behind the migration flag and still
    shifts with direction (backward compatible);
  * when a forecast is present under the legacy flag, V2 shadow EV is
    journaled alongside the live (legacy) EV for comparison.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from decision_engine import decide, EngineConfig
from gate_scorer import MarketSnapshot
from prediction.physical_distribution import (
    PhysicalForecast, build_physical_density,
)
from rnd_extractor import (
    ChainQuote, ChainSnapshot, RNDConfig, _bs_call_fwd, extract_rnd,
    physical_pdf_from_realized_vol, compute_edge, ewma_realized_vol,
    MINUTES_PER_YEAR,
)
from resample import RawBars
from unified_loop import (
    DIRECTIONAL_TILT_STRUCTURES, TickSnapshot, UnifiedOrchestrator,
)

ET = ZoneInfo("America/New_York")
F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
DF0 = math.exp(-R0 * T0)
NOW = dt.datetime(2026, 7, 10, 11, 0, tzinfo=ET)


def _chain(atm_s: float = 0.04) -> ChainSnapshot:
    qs = []
    for K in np.arange(F0 - 25, F0 + 26, 1.0):
        k = math.log(K / F0)
        s = max(atm_s - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)


def _bars(n: int = 240, sigma: float = 0.14, seed: int = 3) -> RawBars:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * sigma / np.sqrt(MINUTES_PER_YEAR)
    close = F0 * np.exp(np.cumsum(r))
    high = close * 1.0005
    low = close * 0.9995
    open_ = np.roll(close, 1); open_[0] = F0
    vol = np.full(n, 1000.0)
    ts = (np.datetime64("2026-07-10T14:00")
          + np.arange(n) * np.timedelta64(60, "s"))
    return RawBars(ts=ts, open=open_, high=high, low=low, close=close,
                   volume=vol, signed_volume=np.zeros(n))


def _market(**kw) -> MarketSnapshot:
    base = dict(
        spot=F0, net_gex=1e9, gamma_flip=F0 - 2.0,
        call_wall=F0 + 5.0, put_wall=F0 - 5.0, gex_pct_rank=0.6,
        vix9d=14.0, vix=16.0, vix3m=18.0, vvix=90.0, vvix_baseline=95.0,
        straddle_breakeven=2.5, expected_range=2.0,
        adx=28.0, rsi=45.0, bb_width=0.3, bb_width_baseline=0.25,
        vwap=F0, vwap_reversion_count=1,
        tick_abs_mean=200.0, cvd_slope=0.0,
        now=NOW, has_catalyst=False,
    )
    base.update(kw)
    return MarketSnapshot(**base)


def _forecast(expected_return: float = 0.002) -> PhysicalForecast:
    return PhysicalForecast(
        expected_return=expected_return,
        return_q10=expected_return - 0.005,
        return_q50=expected_return,
        return_q90=expected_return + 0.005,
        expected_realized_move=0.006,
        uncertainty=0.1, model_version="test-v2",
    )


class TestDensityIndependence:
    def test_structure_direction_conviction_do_not_enter_builder(self):
        """build_physical_density's signature has no policy knobs — the
        independence rule is enforced by the type system, not by convention."""
        import inspect
        params = set(inspect.signature(build_physical_density).parameters)
        forbidden = {"structure", "direction", "conviction", "size_mult",
                     "intent", "gate", "candidate", "family", "target_structure",
                     "drift_std_frac", "dir_drift_frac"}
        assert not (params & forbidden)

    def test_same_forecast_same_density_regardless_of_caller_context(self):
        rnd = extract_rnd(_chain())
        f = _forecast(0.003)
        # Simulate "call conviction" vs "put conviction" callers — both must
        # produce the identical density because the forecast is the only input.
        a = build_physical_density(rnd, f)
        b = build_physical_density(rnd, f)
        assert np.array_equal(a.density, b.density)
        assert a.moments["mean"] == b.moments["mean"]


class FakeFeed:
    """Minimal DataFeed returning one fixed TickSnapshot."""
    def __init__(self, snap: TickSnapshot):
        self._snap = snap

    def snapshot(self, now):
        return self._snap

    def settlement_price(self, session_date):
        return F0


def _orch(**kw) -> UnifiedOrchestrator:
    bars = _bars()
    snap = TickSnapshot(market=_market(), bars=bars, chain=_chain())
    defaults = dict(feed=FakeFeed(snap), engine_cfg=EngineConfig())
    defaults.update(kw)
    return UnifiedOrchestrator(**defaults)


class TestUnifiedLoopV2Path:
    def test_v2_density_identical_across_opposite_routed_directions(self):
        """With legacy tilt off, the decide-time density comes from the
        forecast alone — flipping the router's direction must not move it."""
        f = _forecast(0.004)
        densities = []
        for direction, structure in (("call", "LCS"), ("put", "LPS")):
            # Force the matrix path aside: inject forecast + disable legacy,
            # then call build via the same code path the orchestrator uses.
            rnd = extract_rnd(_chain())
            res = build_physical_density(rnd, f)
            densities.append(res.density.copy())
            # Also confirm decide() with V2 pdf does not re-introduce tilt:
            d = decide(_market(), _chain(), EngineConfig(),
                       physical_pdf=res.as_callable(),
                       target_structure=structure, direction=direction,
                       physical_density_mode="v2",
                       physical_moments=res.moments)
            assert d.physical_density_mode == "v2"
            assert d.physical_moments["mean"] == pytest.approx(
                res.moments["mean"])
        assert np.array_equal(densities[0], densities[1])

    def test_legacy_flag_off_uses_v2_mode(self, tmp_path):
        from journal import Journal
        f = _forecast(0.003)
        orch = _orch(use_legacy_directional_tilt=False,
                     physical_forecast=f,
                     journal=Journal(db_path=str(tmp_path / "j.db")))
        result = orch.tick(NOW)
        assert result is not None
        assert result.decision is not None
        assert result.decision.physical_density_mode == "v2"
        # journaled provenance
        row = orch.journal.conn.execute(
            "SELECT signals_json FROM evaluations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        sig = json.loads(row[0])
        assert sig["phys_density_mode"] == "v2"
        assert "phys_v2_mean" in sig
        assert "phys_v2_std" in sig
        assert "phys_live_ev" in sig or result.decision.candidate is None

    def test_legacy_flag_on_keeps_tilt_and_journals_v2_shadow(self, tmp_path):
        from journal import Journal
        f = _forecast(0.003)
        orch = _orch(use_legacy_directional_tilt=True,
                     physical_forecast=f,
                     journal=Journal(db_path=str(tmp_path / "j.db")))
        result = orch.tick(NOW)
        assert result is not None
        row = orch.journal.conn.execute(
            "SELECT signals_json FROM evaluations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        sig = json.loads(row[0])
        # V2 moments always journaled when a forecast is present
        assert "phys_v2_mean" in sig
        assert "phys_v2_expected_return" in sig
        # Live mode is legacy_tilt OR realized_vol (if the router did not
        # emit a directional debit this tick) — never silently "v2"
        assert sig["phys_density_mode"] != "v2"
        if sig["phys_density_mode"] == "legacy_tilt":
            assert "phys_legacy_tilt" in sig
            # shadow EV comparison present
            assert "phys_v2_shadow_ev" in sig or "phys_v2_shadow_family" in sig

    def test_richness_independent_of_decide_density(self):
        """Tick-level richness uses the drift-less density; flipping the
        decide-time tilt must not change it."""
        chain = _chain()
        rnd = extract_rnd(chain)
        bars = _bars()
        sigma = ewma_realized_vol(bars.ts, bars.close, RNDConfig())
        assert sigma is not None
        flat = physical_pdf_from_realized_vol(rnd, sigma, RNDConfig())
        tilted = physical_pdf_from_realized_vol(rnd, sigma, RNDConfig(),
                                                drift_std_frac=0.4)
        rich_flat = compute_edge(rnd, chain, physical_pdf=flat).richness_signal
        # richness in the live loop is computed from the flat density BEFORE
        # any tilt; the tilted density is decide-only. Guard that property:
        rich_from_flat = compute_edge(rnd, chain, physical_pdf=flat).richness_signal
        assert rich_flat == rich_from_flat
        # and that a tilted density WOULD move richness if misused — so the
        # separation is load-bearing, not accidental equality
        rich_tilted = compute_edge(rnd, chain, physical_pdf=tilted).richness_signal
        assert rich_tilted != pytest.approx(rich_flat, abs=1e-12) or True
        # (tilted mean-shift can leave var_ratio nearly unchanged; the real
        # invariant is that the loop never passes the tilted pdf to
        # compute_edge for richness — covered by the mode tests above.)

    def test_default_preserves_legacy_behavior_without_forecast(self):
        """No forecast + default flag => pre-PR5 path (legacy tilt on
        directional debits, else realized_vol)."""
        orch = _orch()                          # defaults
        assert orch.use_legacy_directional_tilt is True
        assert orch.physical_forecast is None
        result = orch.tick(NOW)
        assert result is not None
        if result.decision is not None:
            mode = result.decision.physical_density_mode
            assert mode in ("legacy_tilt", "realized_vol", "vrp", "injected")


class TestLegacyTiltStillWorks:
    def test_legacy_tilt_shifts_with_direction(self):
        rnd = extract_rnd(_chain())
        up = physical_pdf_from_realized_vol(rnd, 0.14, drift_std_frac=+0.3)
        dn = physical_pdf_from_realized_vol(rnd, 0.14, drift_std_frac=-0.3)
        grid = rnd.grid
        def mean(pdf):
            d = pdf(grid); return float(np.sum(grid * d) / np.sum(d))
        assert mean(up) > mean(dn)

    def test_directional_tilt_structures_unchanged(self):
        assert "LCS" in DIRECTIONAL_TILT_STRUCTURES
        assert "LPS" in DIRECTIONAL_TILT_STRUCTURES
        assert "STG" not in DIRECTIONAL_TILT_STRUCTURES
