"""
mtf_matrix.py
=============
Multi-timeframe standardized feature matrix. For each variable, emits a 0..100
score per timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d), plus per-timeframe regime
confidence rows so you can read confluence vs divergence across the term
structure of the signal.

Two variable kinds, handled honestly:
  * NATIVE   -- computed from bars at that resolution; genuinely differs by TF
               (ADX, RSI, realized vol, BB width, VWAP slope, CVD, ATR ...).
  * SNAPSHOT -- a point-in-time state (net GEX, gamma flip, walls, richness,
               RND skew/kurtosis). One value "now"; broadcast across all
               columns and flagged, because it has no per-TF resolution.

The decision-relevant payoff is the bottom block: compression / trend / breakout
confidence per timeframe. Compression high on 1m-15m while trend builds on 1h-4h
is a coiling setup -- premium-harvestable now, with a higher-TF break to respect.

NOT financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def clip100(x):
    return min(100.0, max(0.0, x))


def P(p):                      # already 0..1
    return clip100(100.0 * p)


def S(x, scale):               # signed -> 50 neutral
    return clip100(50.0 + 50.0 * math.tanh(x / scale)) if scale > 0 else 50.0


def N(x, scale):               # near-level -> 100 at zero
    return clip100(100.0 * math.exp(-abs(x) / scale)) if scale > 0 else 0.0


# --------------------------------------------------------------------------- #
# Variable registry                                                           #
# --------------------------------------------------------------------------- #
NATIVE, SNAPSHOT = "native", "snapshot"


@dataclass
class MTFVar:
    domain: str
    name: str
    kind: str
    std: Callable[[float], float]                              # raw -> 0..100 (fixed prior)
    adapt_fn: Optional[Callable[[float, float], float]] = None # (x, scale) -> 0..100
    prior_scale: float = 1.0                                   # prior std used until ScaleBook warms up


# adapt_fn=S or N means the ScaleBook's empirical std will replace prior_scale at runtime.
# Variables using P() or bounded [0..1] inputs don't benefit from scale adaptation.
VARS: list[MTFVar] = [
    # Price geometry
    MTFVar("price", "dist_to_vwap", NATIVE,    lambda x: N(x, 0.20),           N,    0.20),
    MTFVar("price", "vwap_slope", NATIVE,      lambda x: S(x, 0.05),           S,    0.05),
    MTFVar("price", "range_position", NATIVE,  lambda x: clip100(100 * x)),
    # Dealer (SNAPSHOT)
    MTFVar("dealer", "gamma_sign", SNAPSHOT,        lambda x: S(x, 2e9),       S,    2e9),
    MTFVar("dealer", "gamma_magnitude", SNAPSHOT,   lambda x: P(x)),
    MTFVar("dealer", "flip_cushion", SNAPSHOT,      lambda x: S(x, 0.004),     S,    0.004),
    MTFVar("dealer", "channel_tightness", SNAPSHOT, lambda x: N(x, 0.012),     N,    0.012),
    MTFVar("dealer", "wall_proximity", SNAPSHOT,    lambda x: N(x, 0.003),     N,    0.003),
    # Volatility
    MTFVar("vol", "realized_vol", NATIVE,      lambda x: P(x)),
    MTFVar("vol", "rv_expansion", NATIVE,      lambda x: S(x, 0.25),           S,    0.25),
    MTFVar("vol", "term_structure", SNAPSHOT,  lambda x: S(x, 0.08),           S,    0.08),
    MTFVar("vol", "vvix_elevation", SNAPSHOT,  lambda x: S(x, 0.10),           S,    0.10),
    MTFVar("vol", "richness", SNAPSHOT,        lambda x: P(x)),
    # Distribution shape (SNAPSHOT, from RND)
    MTFVar("shape", "skew_dir", SNAPSHOT,      lambda x: S(x, 0.20),           S,    0.20),
    MTFVar("shape", "tail_heaviness", SNAPSHOT, lambda x: clip100(100 * x / 3.0)),
    # Trend (NATIVE -- the multi-TF core)
    MTFVar("trend", "adx_strength", NATIVE,    lambda x: clip100(100 * x / 50.0)),
    MTFVar("trend", "di_spread", NATIVE,       lambda x: S(x, 20.0),           S,   20.0),
    MTFVar("trend", "ema_slope", NATIVE,       lambda x: S(x, 0.05),           S,    0.05),
    MTFVar("trend", "rsi", NATIVE,             lambda x: clip100(x)),
    MTFVar("trend", "bb_compression", NATIVE,  lambda x: clip100(100 * (1 - x))),
    MTFVar("trend", "trend_cleanliness", NATIVE, lambda x: P(x)),
    # Order flow (NATIVE, windowed)
    MTFVar("flow", "cvd_persistence", NATIVE,  lambda x: S(x, 0.4),            S,    0.40),
    MTFVar("flow", "tick_two_sided", NATIVE,   lambda x: N(x, 600.0),          N,  600.0),
    # ------------------------------------------------------------------
    # OBSERVATION-ONLY signal domains (see journal.py's admission rule).
    # These rows render in the matrix and journal via signals_json, but no
    # regime blend, decision cell, gate, or veto consumes them until
    # component_correlations() on settled ticks earns them that power.
    # ------------------------------------------------------------------
    # Dealer dynamics (market_dynamics.DynamicsWindow)
    MTFVar("dealer", "flip_velocity", SNAPSHOT,     lambda x: S(x, 1.0),       S,    1.0),
    MTFVar("dealer", "flip_chase", SNAPSHOT,        lambda x: S(x, 1.0),       S,    1.0),
    MTFVar("dealer", "call_wall_velocity", SNAPSHOT, lambda x: S(x, 1.0),      S,    1.0),
    MTFVar("dealer", "put_wall_velocity", SNAPSHOT, lambda x: S(x, 1.0),       S,    1.0),
    MTFVar("dealer", "gex_velocity_bn", SNAPSHOT,   lambda x: S(x, 0.5),       S,    0.5),
    MTFVar("dealer", "wall_rupture", SNAPSHOT,      lambda x: S(x, 0.5),       S,    0.5),
    # Vol-state dynamics
    MTFVar("vol", "straddle_ramp", SNAPSHOT,        lambda x: S(x, 0.05),      S,    0.05),
    MTFVar("vol", "expected_move_consumed", SNAPSHOT, lambda x: clip100(100 * x / 1.5)),
    # Options-flow lite (from the live chain the RND already uses).
    # pcr is centered at 1.0, so no ScaleBook adapt_fn (S adapts around 0).
    MTFVar("flow", "pcr_volume", SNAPSHOT,          lambda x: S(x - 1.0, 0.5)),
    MTFVar("flow", "volume_oi_ratio", SNAPSHOT,     lambda x: clip100(100 * x / 5.0)),
    # Breadth lite (batched quotes: RSP, sector ETFs, top-10 names)
    MTFVar("breadth", "rsp_spy_div", SNAPSHOT,      lambda x: S(x, 0.004),     S,    0.004),
    MTFVar("breadth", "sector_align", SNAPSHOT,     lambda x: P(x)),
    MTFVar("breadth", "top10_pressure", SNAPSHOT,   lambda x: S(x, 0.006),     S,    0.006),
]


# --------------------------------------------------------------------------- #
# Input + matrix build                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MTFInput:
    # native[var_name][tf] -> raw value
    native: dict
    # snapshot[var_name] -> raw value (single, broadcast)
    snapshot: dict

    def raw(self, var: MTFVar, tf: str):
        if var.kind == SNAPSHOT:
            return self.snapshot.get(var.name)
        return self.native.get(var.name, {}).get(tf)


@dataclass
class MatrixRow:
    domain: str
    variable: str
    kind: str
    scores: dict          # tf -> 0..100 or None


def build_matrix(inp: MTFInput, scale_book=None) -> list[MatrixRow]:
    """
    Build the scored matrix from inp.

    scale_book: optional ScaleBook (from regime_classifier).  When provided,
    raw values are fed to Welford online estimation; the learned std replaces
    the fixed prior_scale for S/N variables once enough samples accumulate.
    Snapshot variables are updated once per tick (not once per TF) to avoid
    inflating the sample count.
    """
    rows = []
    for v in VARS:
        scores = {}
        sb_updated = False   # prevent 7x updates for broadcast snapshot values
        for tf in TIMEFRAMES:
            r = inp.raw(v, tf)
            if r is None:
                scores[tf] = None
                continue
            if scale_book is not None and v.adapt_fn is not None:
                if v.kind == NATIVE or not sb_updated:
                    scale_book.update(v.name, r)
                    sb_updated = True
                scale = scale_book.std(v.name, v.prior_scale)
                scores[tf] = round(v.adapt_fn(r, scale), 1)
            else:
                scores[tf] = round(v.std(r), 1)
        rows.append(MatrixRow(v.domain, v.name, v.kind, scores))
    return rows


# --------------------------------------------------------------------------- #
# Per-timeframe regime confidence (the decision rows)                          #
# --------------------------------------------------------------------------- #
# (variable, weight, invert) — reuse the standardized cells
_REGIME_DEF = {
    "compression": [("adx_strength", 1.3, True), ("bb_compression", 1.0, False),
                    ("rv_expansion", 1.0, True), ("tick_two_sided", 0.8, False),
                    ("gamma_sign", 1.2, False), ("channel_tightness", 1.0, False)],
    "trend": [("adx_strength", 1.5, False), ("di_spread", 0.8, False),
              ("ema_slope", 0.8, False), ("rv_expansion", 0.8, False),
              ("bb_compression", 1.0, True), ("trend_cleanliness", 0.8, False)],
    "breakout": [("adx_strength", 1.0, False), ("rv_expansion", 1.0, False),
                 ("tick_two_sided", 0.8, True), ("gamma_sign", 1.2, True),
                 ("vvix_elevation", 0.8, False)],
}


def regime_rows(rows: list[MatrixRow]) -> dict:
    by_name = {r.variable: r for r in rows}
    out = {}
    for regime, weights in _REGIME_DEF.items():
        tf_scores = {}
        for tf in TIMEFRAMES:
            num = den = 0.0
            for vname, w, invert in weights:
                r = by_name.get(vname)
                if not r:
                    continue
                val = r.scores.get(tf)
                if val is None:
                    continue
                vv = (100.0 - val) if invert else val
                num += w * vv
                den += w
            tf_scores[tf] = round(num / den, 1) if den > 0 else None
        out[regime] = tf_scores
    return out


# --------------------------------------------------------------------------- #
# Text renderer                                                                #
# --------------------------------------------------------------------------- #
def render_text(rows: list[MatrixRow], regimes: dict) -> str:
    hdr = f"{'domain':<8}{'variable':<20}{'kind':<5}" + "".join(f"{tf:>6}" for tf in TIMEFRAMES)
    lines = [hdr, "-" * len(hdr)]
    cur = None
    for r in rows:
        if r.domain != cur:
            cur = r.domain
        k = "S" if r.kind == SNAPSHOT else "N"
        cells = "".join((f"{r.scores[tf]:>6.0f}" if r.scores[tf] is not None else f"{'·':>6}")
                        for tf in TIMEFRAMES)
        lines.append(f"{r.domain:<8}{r.variable:<20}{k:<5}{cells}")
    lines.append("=" * len(hdr))
    for regime, tfs in regimes.items():
        cells = "".join((f"{tfs[tf]:>6.0f}" if tfs[tf] is not None else f"{'·':>6}")
                        for tf in TIMEFRAMES)
        lines.append(f"{'REGIME':<8}{regime:<20}{'':<5}{cells}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo: a "coiling" day -- compressed on low TFs, trend building on high TFs    #
# --------------------------------------------------------------------------- #
def demo_input() -> MTFInput:
    def ramp(lo, hi):
        return {tf: lo + (hi - lo) * i / (len(TIMEFRAMES) - 1)
                for i, tf in enumerate(TIMEFRAMES)}
    native = {
        "dist_to_vwap": ramp(0.02, 0.30),
        "vwap_slope": ramp(0.005, 0.09),
        "range_position": ramp(0.50, 0.82),
        "realized_vol": ramp(0.20, 0.70),
        "rv_expansion": ramp(-0.25, 0.45),
        "adx_strength": {"1m": 11, "5m": 13, "15m": 16, "30m": 19, "1h": 24, "4h": 28, "1d": 23},
        "di_spread": ramp(3, 22),
        "ema_slope": ramp(0.004, 0.085),
        "rsi": {"1m": 52, "5m": 55, "15m": 58, "30m": 61, "1h": 65, "4h": 69, "1d": 63},
        "bb_compression": {"1m": 0.65, "5m": 0.72, "15m": 0.85, "30m": 0.98,
                           "1h": 1.15, "4h": 1.35, "1d": 1.10},   # raw bbw ratio
        "trend_cleanliness": ramp(0.25, 0.80),
        "cvd_persistence": ramp(0.05, 0.7),
        "tick_two_sided": {"1m": 430, "5m": 470, "15m": 540, "30m": 640,
                           "1h": 760, "4h": 880, "1d": 700},
    }
    snapshot = {
        "gamma_sign": 4.0e9, "gamma_magnitude": 0.86, "flip_cushion": 0.006,
        "channel_tightness": 0.010, "wall_proximity": 0.0025,
        "term_structure": 0.16, "vvix_elevation": -0.03, "richness": 0.66,
        "skew_dir": -0.17, "tail_heaviness": 0.30,
    }
    return MTFInput(native=native, snapshot=snapshot)


if __name__ == "__main__":
    inp = demo_input()
    rows = build_matrix(inp)
    regimes = regime_rows(rows)
    print(render_text(rows, regimes))
    print("\nN = native (varies by timeframe) · S = snapshot (point-in-time, broadcast) · '·' = no data")
