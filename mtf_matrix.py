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
    std: Callable[[float], float]   # raw -> 0..100


# std_fn per variable (scales are reasonable fixed priors here; wire to a
# ScaleBook for adaptive behavior in production)
VARS: list[MTFVar] = [
    # Price geometry
    MTFVar("price", "dist_to_vwap", NATIVE, lambda x: N(x, 0.20)),          # x = % from vwap
    MTFVar("price", "vwap_slope", NATIVE, lambda x: S(x, 0.05)),           # %/bar
    MTFVar("price", "range_position", NATIVE, lambda x: clip100(100 * x)),  # 0..1 within TF range
    # Dealer (SNAPSHOT)
    MTFVar("dealer", "gamma_sign", SNAPSHOT, lambda x: S(x, 2e9)),         # net GEX $
    MTFVar("dealer", "gamma_magnitude", SNAPSHOT, lambda x: P(x)),         # pct rank 0..1
    MTFVar("dealer", "flip_cushion", SNAPSHOT, lambda x: S(x, 0.004)),     # % above flip
    MTFVar("dealer", "channel_tightness", SNAPSHOT, lambda x: clip100(100 * math.exp(-x / 0.012))),  # wall width %
    MTFVar("dealer", "wall_proximity", SNAPSHOT, lambda x: N(x, 0.003)),   # % to nearest wall
    # Volatility
    MTFVar("vol", "realized_vol", NATIVE, lambda x: P(x)),                 # pctile 0..1, per TF
    MTFVar("vol", "rv_expansion", NATIVE, lambda x: S(x, 0.25)),          # bbw ratio - 1
    MTFVar("vol", "term_structure", SNAPSHOT, lambda x: S(x, 0.08)),      # (vix3m-vix)/vix
    MTFVar("vol", "vvix_elevation", SNAPSHOT, lambda x: S(x, 0.10)),      # (vvix/base - 1)
    MTFVar("vol", "richness", SNAPSHOT, lambda x: P(x)),                  # variance ratio signal 0..1
    # Distribution shape (SNAPSHOT, from RND)
    MTFVar("shape", "skew_dir", SNAPSHOT, lambda x: S(x, 0.20)),
    MTFVar("shape", "tail_heaviness", SNAPSHOT, lambda x: clip100(100 * x / 3.0)),  # excess kurt
    # Trend (NATIVE -- the multi-TF core)
    MTFVar("trend", "adx_strength", NATIVE, lambda x: clip100(100 * x / 50.0)),
    MTFVar("trend", "di_spread", NATIVE, lambda x: S(x, 20.0)),
    MTFVar("trend", "ema_slope", NATIVE, lambda x: S(x, 0.05)),
    MTFVar("trend", "rsi", NATIVE, lambda x: clip100(x)),
    MTFVar("trend", "bb_compression", NATIVE, lambda x: clip100(100 * (1 - x))),    # x = bbw ratio
    MTFVar("trend", "trend_cleanliness", NATIVE, lambda x: P(x)),         # R^2 0..1
    # Order flow (NATIVE, windowed)
    MTFVar("flow", "cvd_persistence", NATIVE, lambda x: S(x, 0.4)),
    MTFVar("flow", "tick_two_sided", NATIVE, lambda x: clip100(100 * math.exp(-x / 600.0))),
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


def build_matrix(inp: MTFInput) -> list[MatrixRow]:
    rows = []
    for v in VARS:
        scores = {}
        for tf in TIMEFRAMES:
            r = inp.raw(v, tf)
            scores[tf] = round(v.std(r), 1) if r is not None else None
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
