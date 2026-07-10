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
    # Volatility channels (NATIVE; raw values from resample._channel_features,
    # which documents all periods/thresholds). Width ranks and squeeze are
    # already 0..1 percentiles/grades -> P(). Positions are 0..1 in-channel
    # -> clip100. Signed rates use S(); breakouts are 0-when-inside ATR
    # penetrations with a fixed prior so 50 = "no breakout" is deterministic
    # for downstream threshold logic (no ScaleBook adaptation).
    MTFVar("channel", "bb_width", NATIVE,           lambda x: P(x)),
    MTFVar("channel", "bb_position", NATIVE,        lambda x: clip100(100 * x)),
    MTFVar("channel", "bb_squeeze", NATIVE,         lambda x: P(x)),
    MTFVar("channel", "bb_expansion", NATIVE,       lambda x: S(x, 0.25),      S,    0.25),
    MTFVar("channel", "keltner_width", NATIVE,      lambda x: P(x)),
    MTFVar("channel", "keltner_position", NATIVE,   lambda x: clip100(100 * x)),
    MTFVar("channel", "keltner_trend_strength", NATIVE, lambda x: S(x, 0.25),  S,    0.25),
    MTFVar("channel", "donchian_width", NATIVE,     lambda x: P(x)),
    MTFVar("channel", "donchian_position", NATIVE,  lambda x: clip100(100 * x)),
    MTFVar("channel", "donchian_breakout_up", NATIVE,   lambda x: S(x, 0.5)),
    MTFVar("channel", "donchian_breakout_down", NATIVE, lambda x: S(x, 0.5)),
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


# --------------------------------------------------------------------------- #
# Feature toggles (the feature-impact workflow's ON/OFF switch)                #
# --------------------------------------------------------------------------- #
# Names in this set are excluded from the matrix AND from every _REGIME_DEF
# blend, giving a clean controlled comparison without touching the registry.
# Set process-wide via set_disabled_vars() (scripts/feature_impact.py does
# this from a config's mtf.disabled_vars) or per call via the disabled_vars
# parameter on build_matrix()/regime_rows().
_DISABLED_VARS: frozenset[str] = frozenset()


def set_disabled_vars(names) -> None:
    """Disable matrix variables process-wide (None/empty re-enables all)."""
    global _DISABLED_VARS
    _DISABLED_VARS = frozenset(names or ())


def get_disabled_vars() -> frozenset[str]:
    return _DISABLED_VARS


def build_matrix(inp: MTFInput, scale_book=None,
                 disabled_vars: Optional[set] = None) -> list[MatrixRow]:
    """
    Build the scored matrix from inp.

    scale_book: optional adaptive scale book. Two protocols are supported,
    selected by the book's SCORE_BEFORE_UPDATE class marker:

      * V2 (prediction.scalers.RobustScaleBook, SCORE_BEFORE_UPDATE=True):
        NATIVE variables are keyed "{name}:{tf}" — the 1m distribution of a
        feature can never pool with its 1d distribution — SNAPSHOT variables
        by name; and every observation is scored against the EXISTING state
        before it updates that state, so a value never influences its own
        standardized score.

      * legacy (regime_classifier.ScaleBook): name-only Welford keying with
        update-before-score, preserved verbatim behind configuration
        (UnifiedOrchestrator.use_legacy_scaler) for transition comparisons.

    Snapshot variables are updated once per tick (not once per TF) to avoid
    inflating the sample count.

    disabled_vars: variable names to exclude (feature-impact ON/OFF testing);
    defaults to the process-wide set from set_disabled_vars().
    """
    off = _DISABLED_VARS if disabled_vars is None else frozenset(disabled_vars)
    lagged = bool(getattr(scale_book, "SCORE_BEFORE_UPDATE", False))
    rows = []
    for v in VARS:
        if v.name in off:
            continue
        scores = {}
        sb_updated = False   # prevent 7x updates for broadcast snapshot values
        snapshot_update = None   # (key, raw): applied AFTER all TFs are scored
        for tf in TIMEFRAMES:
            r = inp.raw(v, tf)
            if r is None:
                scores[tf] = None
                continue
            if scale_book is not None and v.adapt_fn is not None:
                if lagged:
                    key = v.name if v.kind == SNAPSHOT else f"{v.name}:{tf}"
                    scale = scale_book.std(key, v.prior_scale)
                    scores[tf] = round(v.adapt_fn(r, scale), 1)
                    # Score first, update after: native keys update right
                    # away (no other TF shares the key); the broadcast
                    # snapshot key defers until every column is scored.
                    if v.kind == NATIVE:
                        scale_book.update(key, r)
                    elif snapshot_update is None:
                        snapshot_update = (key, r)
                else:
                    if v.kind == NATIVE or not sb_updated:
                        scale_book.update(v.name, r)
                        sb_updated = True
                    scale = scale_book.std(v.name, v.prior_scale)
                    scores[tf] = round(v.adapt_fn(r, scale), 1)
            else:
                scores[tf] = round(v.std(r), 1)
        if snapshot_update is not None:
            scale_book.update(*snapshot_update)
        rows.append(MatrixRow(v.domain, v.name, v.kind, scores))
    return rows


# --------------------------------------------------------------------------- #
# Per-timeframe regime confidence (the decision rows)                          #
# --------------------------------------------------------------------------- #
# (variable, weight, invert[, "fold"]) — reuse the standardized cells.
# "fold" scores the magnitude of deviation from the 50-neutral (|v-50|*2),
# for signed variables where either direction is evidence (e.g. a strong
# Keltner ride up OR down both mean "trending").
# Channel contributions enter at modest weights (0.6-0.8, below the
# incumbents) to limit blast radius while they prove out:
#   compression: TTM squeeze on + narrow Donchian range = coiling
#   trend:       persistent ride of the Keltner upper/lower half (folded,
#                direction-agnostic); squeeze off
#   breakout:    Donchian penetration (either leg lifts off its 50-neutral) +
#                Bollinger width expanding
_REGIME_DEF = {
    "compression": [("adx_strength", 1.3, True), ("bb_compression", 1.0, False),
                    ("rv_expansion", 1.0, True), ("tick_two_sided", 0.8, False),
                    ("gamma_sign", 1.2, False), ("channel_tightness", 1.0, False),
                    ("bb_squeeze", 0.8, False), ("donchian_width", 0.8, True)],
    "trend": [("adx_strength", 1.5, False), ("di_spread", 0.8, False),
              ("ema_slope", 0.8, False), ("rv_expansion", 0.8, False),
              ("bb_compression", 1.0, True), ("trend_cleanliness", 0.8, False),
              ("keltner_trend_strength", 0.8, False, "fold"),
              ("bb_squeeze", 0.6, True)],
    "breakout": [("adx_strength", 1.0, False), ("rv_expansion", 1.0, False),
                 ("tick_two_sided", 0.8, True), ("gamma_sign", 1.2, True),
                 ("vvix_elevation", 0.8, False),
                 ("donchian_breakout_up", 0.6, False),
                 ("donchian_breakout_down", 0.6, False),
                 ("bb_expansion", 0.8, False)],
}


def regime_rows(rows: list[MatrixRow], disabled_vars: Optional[set] = None) -> dict:
    """Blend matrix rows into per-TF regime confidences. Disabled variables
    (parameter, or the process-wide set) contribute nothing: rows filtered out
    of build_matrix() are already absent, and the blend skips them explicitly
    in case a caller passes unfiltered rows."""
    off = _DISABLED_VARS if disabled_vars is None else frozenset(disabled_vars)
    by_name = {r.variable: r for r in rows}
    out = {}
    for regime, weights in _REGIME_DEF.items():
        tf_scores = {}
        for tf in TIMEFRAMES:
            num = den = 0.0
            for spec in weights:
                vname, w, invert = spec[0], spec[1], spec[2]
                fold = len(spec) > 3 and spec[3] == "fold"
                if vname in off:
                    continue
                r = by_name.get(vname)
                if not r:
                    continue
                val = r.scores.get(tf)
                if val is None:
                    continue
                if fold:
                    vv = clip100(abs(val - 50.0) * 2.0)
                elif invert:
                    vv = 100.0 - val
                else:
                    vv = val
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
