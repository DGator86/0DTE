"""
Regression tests for the realized-vol physical density.

The bug these guard against: with no physical_pdf injected, compute_edge fell
back to the static VRP haircut, which *defines* var_phys = (1-vrp)*var_RN — so
the variance ratio (and the `richness` signal injected into the MTF matrix)
was a constant on every tick, in every regime. The realized-vol path sets the
physical variance from bar data the RND doesn't already contain, so richness
must now move with the implied-vs-realized gap.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from rnd_extractor import (
    ChainQuote, ChainSnapshot, RNDConfig, _bs_call_fwd,
    extract_rnd, compute_edge, ewma_realized_vol, physical_pdf_from_realized_vol,
    MINUTES_PER_YEAR,
)
from resample import RawBars
from unified_loop import _realized_vol_pdf

F0, R0, T0 = 600.0, 0.05, 4.0 / (24 * 365)
DF0 = math.exp(-R0 * T0)


def _chain(atm_s: float) -> ChainSnapshot:
    qs = []
    for K in np.arange(F0 - 25, F0 + 26, 1.0):
        k = math.log(K / F0)
        s = max(atm_s - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=F0, t_years=T0, r=R0)


def _minute_series(sigma_annual: float, n: int = 240, seed: int = 3):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * sigma_annual / np.sqrt(MINUTES_PER_YEAR)
    close = 600.0 * np.exp(np.cumsum(r))
    ts = np.datetime64("2026-07-01T13:30") + np.arange(n) * np.timedelta64(60, "s")
    return ts, close


# --------------------------------------------------------------------------- #
# ewma_realized_vol                                                            #
# --------------------------------------------------------------------------- #
def test_ewma_recovers_constant_vol():
    ts, close = _minute_series(0.12, n=390)
    sigma = ewma_realized_vol(ts, close)
    assert sigma == pytest.approx(0.12, rel=0.25)


def test_ewma_none_on_thin_history():
    ts, close = _minute_series(0.12, n=30)   # below rv_min_returns=60
    assert ewma_realized_vol(ts, close) is None
    assert ewma_realized_vol(ts[:1], close[:1]) is None


def test_ewma_ignores_overnight_gap():
    ts, close = _minute_series(0.12, n=240)
    base = ewma_realized_vol(ts, close)
    close_g = close.copy()
    close_g[120:] *= 1.03                          # 3% gap...
    ts_g = ts.copy()
    ts_g[120:] += np.timedelta64(17 * 3600, "s")   # ...across 17 hours
    gapped = ewma_realized_vol(ts_g, close_g)
    assert gapped == pytest.approx(base, rel=0.05)


def test_ewma_weights_recent_vol_more():
    ts, _ = _minute_series(0.10, n=240)
    _, calm = _minute_series(0.08, n=120, seed=1)
    _, wild = _minute_series(0.40, n=120, seed=2)
    calm_then_wild = np.concatenate([calm, wild * (calm[-1] / wild[0])])
    wild_then_calm = np.concatenate([wild, calm * (wild[-1] / calm[0])])
    assert ewma_realized_vol(ts, calm_then_wild) > ewma_realized_vol(ts, wild_then_calm)


# --------------------------------------------------------------------------- #
# physical_pdf_from_realized_vol -> richness actually varies                   #
# --------------------------------------------------------------------------- #
def test_richness_no_longer_constant_across_regimes():
    ts, close = _minute_series(0.12)
    sigma = ewma_realized_vol(ts, close)

    old, new = [], []
    for atm_s in (0.0035, 0.0050, 0.0105):        # calm / normal / panic chains
        snap = _chain(atm_s)
        rnd = extract_rnd(snap)
        old.append(compute_edge(rnd, snap).richness_signal)          # VRP fallback
        pdf = physical_pdf_from_realized_vol(rnd, sigma)
        new.append(compute_edge(rnd, snap, physical_pdf=pdf).richness_signal)

    # the old fallback is a constant by construction...
    assert max(old) - min(old) < 0.01
    # ...the realized-vol path is not, and is monotone in RN vol for fixed RV
    assert new[0] < new[1] <= new[2]
    assert new[1] - new[0] > 0.05


def test_richness_reads_cheap_when_realized_exceeds_implied():
    snap = _chain(0.0035)
    rnd = extract_rnd(snap)
    pdf = physical_pdf_from_realized_vol(rnd, 0.36)   # tape 3x the implied
    edge = compute_edge(rnd, snap, physical_pdf=pdf)
    assert edge.richness_signal < 0.5
    assert edge.variance_ratio < 1.0


def test_squeeze_is_clipped_on_degenerate_vol_prints():
    snap = _chain(0.0050)
    rnd = extract_rnd(snap)
    cfg = RNDConfig()
    tiny = physical_pdf_from_realized_vol(rnd, 1e-6, cfg)
    huge = physical_pdf_from_realized_vol(rnd, 50.0, cfg)
    dx = rnd.grid[1] - rnd.grid[0]
    for pdf, scale in ((tiny, cfg.rv_scale_min), (huge, cfg.rv_scale_max)):
        dens = pdf(rnd.grid)
        m = float(np.sum(rnd.grid * dens) * dx / (np.sum(dens) * dx))
        sd = float(np.sqrt(np.sum((rnd.grid - m) ** 2 * dens) * dx / (np.sum(dens) * dx)))
        assert sd == pytest.approx(scale * rnd.std(), rel=0.05)


def test_factory_returns_none_on_bad_inputs():
    snap = _chain(0.0050)
    rnd = extract_rnd(snap)
    assert physical_pdf_from_realized_vol(rnd, 0.0) is None
    assert physical_pdf_from_realized_vol(rnd, float("nan")) is None


def test_pdf_interpolates_onto_foreign_grids():
    snap = _chain(0.0050)
    rnd = extract_rnd(snap)
    pdf = physical_pdf_from_realized_vol(rnd, 0.12)
    other = np.arange(rnd.grid[0] - 5.0, rnd.grid[-1] + 5.0, 0.25)
    dens = pdf(other)
    assert dens.shape == other.shape
    assert np.all(dens >= 0.0)
    assert dens[0] == 0.0 and dens[-1] == 0.0      # zero outside source support


# --------------------------------------------------------------------------- #
# unified_loop wiring                                                          #
# --------------------------------------------------------------------------- #
def _raw_bars(n: int = 240, sigma: float = 0.12) -> RawBars:
    ts, close = _minute_series(sigma, n=n)
    return RawBars(ts=ts, open=close, high=close * 1.0001,
                   low=close * 0.9999, close=close,
                   volume=np.full(n, 1000.0))


def test_realized_vol_pdf_used_when_bars_sufficient():
    snap = _chain(0.0050)
    rnd = extract_rnd(snap)
    pdf = _realized_vol_pdf(rnd, _raw_bars(), RNDConfig())
    assert pdf is not None
    edge = compute_edge(rnd, snap, physical_pdf=pdf)
    vrp_edge = compute_edge(rnd, snap)
    assert edge.richness_signal != pytest.approx(vrp_edge.richness_signal, abs=0.01)


def test_realized_vol_pdf_falls_back_on_thin_bars_without_raising():
    snap = _chain(0.0050)
    rnd = extract_rnd(snap)
    assert _realized_vol_pdf(rnd, _raw_bars(n=10), RNDConfig()) is None
    assert _realized_vol_pdf(rnd, None, RNDConfig()) is None   # never raises
