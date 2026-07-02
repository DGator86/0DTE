"""
The directional engine must be able to fire.

Three defects kept it structurally dead in the live loop:
  1. decide() ANDed the premium-selling gate against debit structures — the
     tape a debit trade wants (trend, below flip, short gamma) is exactly the
     tape that gate forbids.
  2. regime_classifier veto names ("short_gamma_regime"/"below_gamma_flip")
     never matched decision_matrix's premium-veto set ("short_gamma"/
     "below_flip"), so the credit->debit flip was dead code.
  3. A drift-less physical density prices every debit at EV<=0 by
     construction, so the selector's min_ev veto blocked all fills.
"""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gate_scorer import (
    MarketSnapshot, GateConfig, Decision, evaluate,
    evaluate_directional_gates, score_directional,
)
from rnd_extractor import (
    ChainQuote, ChainSnapshot, _bs_call_fwd, extract_rnd,
    physical_pdf_from_realized_vol,
)
from decision_engine import decide, EngineConfig
from decision_matrix import decide_from_matrix, NO_PREMIUM_VETOES
from mtf_matrix import build_matrix, demo_input, regime_rows

ET = ZoneInfo("America/New_York")


def _chain(spot: float = 745.0, atm_s: float = 0.0060) -> ChainSnapshot:
    T0, r0 = 4.0 / (24 * 365), 0.05
    DF0 = math.exp(-r0 * T0)
    F0 = spot * math.exp(r0 * T0)
    qs = []
    for K in np.arange(spot - 25, spot + 26, 1.0):
        k = math.log(K / F0)
        s = max(atm_s - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))
    return ChainSnapshot(qs, spot=spot, t_years=T0, r=r0)


def _trend_down_market(**overrides) -> MarketSnapshot:
    """The dashboard-print tape: trending down, below flip, weak GEX rank."""
    kw = dict(
        spot=745.0, net_gex=2.2e9, gamma_flip=749.2,
        call_wall=750.0, put_wall=743.0, gex_pct_rank=0.0,
        vix9d=13.69, vix=16.67, vix3m=19.2, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=3.485, expected_range=2.788,
        adx=34.9, rsi=31.4, bb_width=0.46, bb_width_baseline=0.10,
        vwap=748.4, vwap_reversion_count=2,
        tick_abs_mean=480.0, cvd_slope=-0.25,
        now=dt.datetime(2026, 7, 2, 11, 17, tzinfo=ET),
        has_catalyst=False,
    )
    kw.update(overrides)
    return MarketSnapshot(**kw)


# --------------------------------------------------------------------------- #
# gate modes                                                                   #
# --------------------------------------------------------------------------- #
def test_premium_gate_still_blocks_trend_day():
    r = evaluate(_trend_down_market(), GateConfig())
    assert r.decision is Decision.NO_GO
    assert any("TRENDING" in g for g in r.failed_gates)


def test_directional_gate_passes_the_same_tape():
    r = evaluate(_trend_down_market(), GateConfig(),
                 structure_class="directional", direction="put")
    assert r.decision is Decision.GO
    assert r.score > 50
    assert r.kelly_fraction > 0


def test_directional_gate_universal_stops_still_apply():
    m = _trend_down_market(has_catalyst=True, catalyst_label="FOMC")
    r = evaluate(m, GateConfig(), structure_class="directional", direction="put")
    assert r.decision is Decision.NO_GO
    assert any("CATALYST" in g for g in r.failed_gates)

    late = _trend_down_market(now=dt.datetime(2026, 7, 2, 15, 45, tzinfo=ET))
    r2 = evaluate(late, GateConfig(), structure_class="directional", direction="put")
    assert r2.decision is Decision.NO_GO
    assert any("LATE" in g for g in r2.failed_gates)


def test_directional_flow_component_requires_sign_agreement():
    cfg = GateConfig()
    m = _trend_down_market()                      # cvd_slope negative (selling)
    with_put = score_directional(m, cfg, "put")
    with_call = score_directional(m, cfg, "call")
    assert with_put["dir_flow"] > 0
    assert with_call["dir_flow"] == 0.0           # flow disagrees with a long call


# --------------------------------------------------------------------------- #
# veto-name normalization                                                      #
# --------------------------------------------------------------------------- #
def test_live_loop_veto_names_flip_credit_to_debit():
    rows = build_matrix(demo_input())
    regimes = regime_rows(rows)
    # the names regime_classifier actually emits (the live loop's convention)
    intent = decide_from_matrix(rows, regimes,
                                vetoes=["below_gamma_flip", "short_gamma_regime"])
    assert intent.decision.structure in ("LCS", "LPS", "NT")
    assert intent.decision.structure != "PCS"     # premium must not survive
    assert "premium veto" in intent.note or intent.decision.structure == "NT"


def test_both_veto_conventions_are_recognized():
    assert {"short_gamma", "short_gamma_regime",
            "below_flip", "below_gamma_flip"} <= NO_PREMIUM_VETOES


# --------------------------------------------------------------------------- #
# end to end: LPS fills on the trend-down tape                                 #
# --------------------------------------------------------------------------- #
def test_decide_fills_lps_on_trend_day_with_tilted_density():
    chain = _chain()
    rnd = extract_rnd(chain)
    pdf = physical_pdf_from_realized_vol(rnd, 0.14, drift_std_frac=-0.30)
    d = decide(_trend_down_market(), chain, EngineConfig(), physical_pdf=pdf,
               target_structure="LPS", direction="put")
    assert d.decision == "TRADE"
    assert d.candidate is not None
    assert d.candidate.family == "long_put_spread"
    assert d.candidate.credit < 0                 # debit paid
    assert d.candidate.max_loss > 0
    assert d.candidate.passes_vetoes


def test_premium_target_still_faces_premium_gate():
    chain = _chain()
    d = decide(_trend_down_market(), chain, EngineConfig(),
               target_structure="IC", direction="both")
    assert d.decision == "NO_TRADE"
    assert "gate:" in d.no_trade_reason


# --------------------------------------------------------------------------- #
# drift tilt                                                                   #
# --------------------------------------------------------------------------- #
def test_drift_tilt_shifts_density_mean():
    chain = _chain()
    rnd = extract_rnd(chain)
    dx = 0.05
    grid = np.arange(rnd.grid[0] - 3, rnd.grid[-1] + 3, dx)

    def mean_of(pdf):
        dens = pdf(grid)
        return float(np.sum(grid * dens) / np.sum(dens))

    flat = physical_pdf_from_realized_vol(rnd, 0.14)
    up = physical_pdf_from_realized_vol(rnd, 0.14, drift_std_frac=+0.5)
    down = physical_pdf_from_realized_vol(rnd, 0.14, drift_std_frac=-0.5)
    m0, mu, md = mean_of(flat), mean_of(up), mean_of(down)
    assert mu > m0 > md
    # shift magnitude ~ drift_frac * phys_std
    phys_std = 0.14 * rnd.forward * math.sqrt(rnd.t_years)
    assert mu - m0 == pytest.approx(0.5 * phys_std, rel=0.35)
