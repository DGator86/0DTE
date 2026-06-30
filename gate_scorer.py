"""
gate_scorer.py
==============
Pre-trade GO/NO-GO gate + confidence scorer for the GEX-based 0DTE SPY/XSP
premium-selling system (notification-and-manual-execution model).

Design philosophy
-----------------
Two layers, deliberately separated:

  1. HARD GATES  - binary kill switches. ANY failure => NO_GO, no alert fires,
                   regardless of how attractive everything else looks. These
                   encode the regime requirements that, when violated, flip the
                   dealer-hedging tailwind into a headwind.

  2. WEIGHTED SCORE (0-100) - only computed once every gate passes. Measures how
                   *clean* the ranging setup is, and maps to a fraction of your
                   Kelly stake. A passing-but-marginal day sizes small; a
                   textbook day sizes up.

The gate layer protects you from ruin (short-gamma / catalyst days).
The score layer optimizes sizing on the days you're allowed to play.

Thresholds are adaptive where possible (percentiles, ratios) and every constant
lives in GateConfig so nothing is buried. Consumes outputs from your existing
GEX, regime-detection, and Monte-Carlo modules.

NOT financial advice - this is decision-support tooling for your own system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import datetime as dt
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class MarketSnapshot:
    """A point-in-time view assembled by the upstream modules."""

    # --- price / gamma structure (from GEX module) ---
    spot: float
    net_gex: float                 # signed dollar gamma (notional). + = dealers long gamma
    gamma_flip: float              # zero-gamma price level
    call_wall: float               # strike of largest positive call gamma (resistance)
    put_wall: float                # strike of largest positive put gamma (support)
    gex_pct_rank: float            # 0..1 percentile of |net_gex| vs trailing window (adaptive)

    # --- volatility pricing ---
    vix9d: float
    vix: float
    vix3m: float
    vvix: float
    vvix_baseline: float           # e.g. trailing 20d median VVIX
    straddle_breakeven: float      # ATM straddle price as a +/- move in $ (implied day range)
    expected_range: float          # conditional realized day range from MC/history, same units

    # --- range technicals ---
    adx: float
    rsi: float
    bb_width: float                # current Bollinger band width
    bb_width_baseline: float       # trailing median band width
    vwap: float
    vwap_reversion_count: int      # # of times price has crossed back to VWAP this session

    # --- order flow ---
    tick_abs_mean: float           # rolling mean of |$TICK| over last N mins
    cvd_slope: float               # signed slope of cumulative volume delta (normalized -1..1)

    # --- timing / catalyst ---
    now: dt.datetime               # tz-aware; will be coerced to ET
    has_catalyst: bool             # FOMC/CPI/PCE/NFP/mega-cap earnings in the danger window
    catalyst_label: str = ""

    def et_time(self) -> dt.time:
        t = self.now if self.now.tzinfo else self.now.replace(tzinfo=ET)
        return t.astimezone(ET).time()

    def mtf_snapshot(self) -> dict:
        """Snapshot dict for mtf_matrix SNAPSHOT-kind variables (pre-standardized inputs)."""
        s = self.spot
        return {
            "gamma_sign":        self.net_gex,
            "gamma_magnitude":   self.gex_pct_rank,
            "flip_cushion":      (s - self.gamma_flip) / s,
            "channel_tightness": (self.call_wall - self.put_wall) / s,
            "wall_proximity":    min(abs(self.call_wall - s), abs(s - self.put_wall)) / s,
            "term_structure":    (self.vix3m - self.vix) / self.vix if self.vix else 0.0,
            "vvix_elevation":    self.vvix / self.vvix_baseline - 1.0 if self.vvix_baseline else 0.0,
            # richness / skew_dir / tail_heaviness injected from RND after extraction
        }

    def dealer_vetoes(self) -> list:
        """Hard vetoes for the regime classifier and Track B routing."""
        v = []
        if self.net_gex <= 0:
            v.append("short_gamma")
        if self.spot < self.gamma_flip:
            v.append("below_flip")
        if self.vix >= self.vix3m:
            v.append("term_backwardation")
        if self.has_catalyst:
            v.append(f"catalyst:{self.catalyst_label or 'event'}")
        return v


@dataclass
class GateConfig:
    # --- hard gate thresholds ---
    min_gex_pct_rank: float = 0.60     # |GEX| must be in top 40% of its trailing range
    flip_buffer_frac: float = 0.0015   # spot must sit >0.15% above flip (clear of the knife's edge)
    max_adx: float = 20.0              # >=20 => trend present => no premium selling
    contango_ratio_max: float = 1.00   # vix9d/vix must be < this (term structure not inverted)
    require_no_catalyst: bool = True

    # --- timing ---
    morning_resolve_time: dt.time = dt.time(10, 30)   # before this, discovery unresolved
    late_lockout_time: dt.time = dt.time(15, 30)      # after this, close-gamma too hot to initiate

    # --- scoring weights (sum to 100) ---
    w_gex_magnitude: float = 20.0
    w_flip_distance: float = 15.0
    w_straddle_rich: float = 18.0
    w_vvix_calm: float = 7.0
    w_adx_depth: float = 8.0
    w_bb_contract: float = 6.0
    w_rsi_center: float = 6.0
    w_tick_osc: float = 6.0
    w_cvd_flat: float = 6.0
    w_wall_prox: float = 5.0
    w_timing: float = 3.0

    # --- scoring shape params ---
    tick_calm_ref: float = 600.0       # |TICK| mean at/below ref = calm; ~1000 = thrust
    wall_prox_frac: float = 0.0025     # within 0.25% of a wall = "at the edge", ideal entry

    # --- sizing map (score -> fraction of Kelly stake) ---
    score_floor: float = 50.0          # below this (but gates passed) => minimum size
    kelly_frac_min: float = 0.15
    kelly_frac_max: float = 1.00


class Decision(Enum):
    GO = "GO"
    NO_GO = "NO_GO"


class Side(Enum):
    SELL_CALL = "SELL_CALL"   # price at/near call wall -> fade resistance
    SELL_PUT = "SELL_PUT"     # price at/near put wall  -> fade support
    WAIT_FOR_EDGE = "WAIT"    # mid-range; no wall to lean on yet


@dataclass
class GateResult:
    decision: Decision
    score: float                       # 0..100, only meaningful when decision == GO
    failed_gates: list[str]
    subscores: dict[str, float]
    side: Side
    nearer_wall: str
    wall_distance_frac: float
    kelly_fraction: float              # 0 when NO_GO
    rationale: str


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear-ramp value in [lo, hi] -> [0, 1], clipped. Handles inverted ranges."""
    if hi == lo:
        return 0.0
    return _clip01((value - lo) / (hi - lo))


# --------------------------------------------------------------------------- #
# Hard gates                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_gates(s: MarketSnapshot, cfg: GateConfig) -> list[str]:
    """Return list of failed-gate reasons. Empty list => all gates pass."""
    failed: list[str] = []

    # 1. Long-gamma regime: net GEX positive AND meaningfully large
    if s.net_gex <= 0:
        failed.append("GEX_SHORT: net gamma <= 0 (dealers short gamma; moves amplify)")
    elif s.gex_pct_rank < cfg.min_gex_pct_rank:
        failed.append(
            f"GEX_WEAK: |GEX| rank {s.gex_pct_rank:.2f} < {cfg.min_gex_pct_rank:.2f} "
            "(positive but too thin to pin)"
        )

    # 2. Above the gamma flip, with buffer
    flip_min = s.gamma_flip * (1 + cfg.flip_buffer_frac)
    if s.spot < flip_min:
        failed.append(
            f"BELOW_FLIP: spot {s.spot:.2f} not clear of flip {s.gamma_flip:.2f} "
            f"(+{cfg.flip_buffer_frac:.2%} buffer)"
        )

    # 3. Term structure not inverted (contango)
    ratio = s.vix9d / s.vix if s.vix else float("inf")
    if ratio >= cfg.contango_ratio_max or s.vix >= s.vix3m:
        failed.append(
            f"TERM_INVERTED: VIX9D/VIX={ratio:.3f}, VIX={s.vix:.2f} vs VIX3M={s.vix3m:.2f} "
            "(stress/backwardation = breakout risk)"
        )

    # 4. No trend present
    if s.adx >= cfg.max_adx:
        failed.append(f"TRENDING: ADX {s.adx:.1f} >= {cfg.max_adx:.0f}")

    # 5. No catalyst in the danger window
    if cfg.require_no_catalyst and s.has_catalyst:
        label = s.catalyst_label or "scheduled event"
        failed.append(f"CATALYST: {label} in window (range-breaking)")

    # 6. Timing lockout (initiating too late into close-gamma)
    if s.et_time() >= cfg.late_lockout_time:
        failed.append(
            f"LATE: {s.et_time():%H:%M} ET past lockout {cfg.late_lockout_time:%H:%M} "
            "(close-gamma too hot to initiate)"
        )

    return failed


# --------------------------------------------------------------------------- #
# Weighted score                                                               #
# --------------------------------------------------------------------------- #
def score_setup(s: MarketSnapshot, cfg: GateConfig) -> dict[str, float]:
    """Per-component scores (already weighted). Sum = total 0..100."""
    sub: dict[str, float] = {}

    # GEX magnitude: percentile rank above the gate floor, ramped to 1.0
    gex_q = _scale(s.gex_pct_rank, cfg.min_gex_pct_rank, 1.0)
    sub["gex_magnitude"] = cfg.w_gex_magnitude * gex_q

    # Distance above flip: more cushion = safer, saturating ~1.5% above
    flip_dist = (s.spot - s.gamma_flip) / s.spot
    sub["flip_distance"] = cfg.w_flip_distance * _scale(flip_dist, 0.0, 0.015)

    # Straddle richness: implied day move vs expected realized. >1 means selling rich.
    rich = (s.straddle_breakeven / s.expected_range) if s.expected_range else 0.0
    sub["straddle_rich"] = cfg.w_straddle_rich * _scale(rich, 0.90, 1.40)

    # VVIX calm: at/below baseline = full marks; spiking = penalized
    vvix_q = 1.0 - _scale(s.vvix, s.vvix_baseline, s.vvix_baseline * 1.30)
    sub["vvix_calm"] = cfg.w_vvix_calm * vvix_q

    # ADX depth: deeper below the gate cutoff = cleaner range (best near ~12)
    sub["adx_depth"] = cfg.w_adx_depth * _scale(cfg.max_adx - s.adx, 0.0, 8.0)

    # Bollinger contraction: width below baseline = compression
    contract = 1.0 - _scale(s.bb_width / s.bb_width_baseline, 0.6, 1.1) if s.bb_width_baseline else 0.0
    sub["bb_contract"] = cfg.w_bb_contract * _clip01(contract)

    # RSI centering: peak at 50, falls off toward 30/70
    rsi_center = 1.0 - _clip01(abs(s.rsi - 50.0) / 20.0)
    sub["rsi_center"] = cfg.w_rsi_center * rsi_center

    # TICK oscillation: |TICK| mean near/below calm ref = two-sided auction
    tick_q = 1.0 - _scale(s.tick_abs_mean, cfg.tick_calm_ref, 1000.0)
    sub["tick_osc"] = cfg.w_tick_osc * tick_q

    # CVD flatness: |slope| near 0 = no persistent aggressor
    sub["cvd_flat"] = cfg.w_cvd_flat * (1.0 - _clip01(abs(s.cvd_slope)))

    # Wall proximity: closer to a wall = a real edge to lean on for entry
    d_call = abs(s.call_wall - s.spot) / s.spot
    d_put = abs(s.spot - s.put_wall) / s.spot
    nearest = min(d_call, d_put)
    sub["wall_prox"] = cfg.w_wall_prox * (1.0 - _scale(nearest, 0.0, cfg.wall_prox_frac * 4))

    # Timing: best in the post-discovery / pre-close-gamma window
    t = s.et_time()
    if t < cfg.morning_resolve_time:
        timing_q = 0.3                      # discovery unresolved
    elif t < dt.time(14, 0):
        timing_q = 1.0                      # prime decay grind
    else:
        timing_q = 0.7                      # still ok, charm building
    sub["timing"] = cfg.w_timing * timing_q

    return sub


def pick_side(s: MarketSnapshot, cfg: GateConfig) -> tuple[Side, str, float]:
    d_call = abs(s.call_wall - s.spot) / s.spot
    d_put = abs(s.spot - s.put_wall) / s.spot
    if d_call <= cfg.wall_prox_frac and d_call <= d_put:
        return Side.SELL_CALL, "call_wall", d_call
    if d_put <= cfg.wall_prox_frac and d_put < d_call:
        return Side.SELL_PUT, "put_wall", d_put
    # mid-range: report the nearer wall but advise waiting for a tag
    if d_call <= d_put:
        return Side.WAIT_FOR_EDGE, "call_wall", d_call
    return Side.WAIT_FOR_EDGE, "put_wall", d_put


def score_to_kelly(score: float, cfg: GateConfig) -> float:
    """Map a passing score to a fraction of the Kelly stake."""
    q = _scale(score, cfg.score_floor, 100.0)
    return cfg.kelly_frac_min + (cfg.kelly_frac_max - cfg.kelly_frac_min) * q


# --------------------------------------------------------------------------- #
# Top-level entry point                                                        #
# --------------------------------------------------------------------------- #
def evaluate(s: MarketSnapshot, cfg: Optional[GateConfig] = None) -> GateResult:
    cfg = cfg or GateConfig()
    failed = evaluate_gates(s, cfg)

    side, nearer_wall, wall_dist = pick_side(s, cfg)

    if failed:
        return GateResult(
            decision=Decision.NO_GO,
            score=0.0,
            failed_gates=failed,
            subscores={},
            side=side,
            nearer_wall=nearer_wall,
            wall_distance_frac=wall_dist,
            kelly_fraction=0.0,
            rationale="NO-GO. " + " | ".join(failed),
        )

    sub = score_setup(s, cfg)
    total = round(sum(sub.values()), 1)
    kelly = round(score_to_kelly(total, cfg), 3)

    if side is Side.WAIT_FOR_EDGE:
        action = (
            f"Gates clear (score {total}). Mid-range: hold for a tag of "
            f"{nearer_wall} ({wall_dist:.2%} away) before selling."
        )
    else:
        action = (
            f"GO (score {total}). {side.value} into {nearer_wall} "
            f"({wall_dist:.2%} away). Size {kelly:.0%} of Kelly. "
            f"Invalidate on {nearer_wall} break w/ expanding volume or GEX flip."
        )

    return GateResult(
        decision=Decision.GO,
        score=total,
        failed_gates=[],
        subscores={k: round(v, 1) for k, v in sub.items()},
        side=side,
        nearer_wall=nearer_wall,
        wall_distance_frac=wall_dist,
        kelly_fraction=kelly,
        rationale=action,
    )


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = GateConfig()

    # ---- A: textbook ranging day, price parked on the call wall ----
    good = MarketSnapshot(
        spot=602.50, net_gex=4.2e9, gamma_flip=596.0,
        call_wall=603.0, put_wall=598.0, gex_pct_rank=0.88,
        vix9d=12.1, vix=13.0, vix3m=15.2, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.10, expected_range=3.20,
        adx=12.5, rsi=52.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=601.9, vwap_reversion_count=5,
        tick_abs_mean=480.0, cvd_slope=0.05,
        now=dt.datetime(2026, 6, 25, 11, 20, tzinfo=ET),
        has_catalyst=False,
    )

    # ---- B: below flip + CPI morning => must hard-fail ----
    bad = MarketSnapshot(
        spot=588.0, net_gex=-1.1e9, gamma_flip=593.0,
        call_wall=596.0, put_wall=585.0, gex_pct_rank=0.40,
        vix9d=19.5, vix=18.0, vix3m=17.0, vvix=120.0, vvix_baseline=95.0,
        straddle_breakeven=6.0, expected_range=6.5,
        adx=28.0, rsi=38.0, bb_width=3.1, bb_width_baseline=2.0,
        vwap=590.0, vwap_reversion_count=1,
        tick_abs_mean=910.0, cvd_slope=-0.7,
        now=dt.datetime(2026, 6, 25, 9, 5, tzinfo=ET),
        has_catalyst=True, catalyst_label="CPI 08:30",
    )

    for tag, snap in (("A (clean range)", good), ("B (short-gamma + CPI)", bad)):
        r = evaluate(snap, cfg)
        print(f"\n=== {tag} ===")
        print(f"Decision : {r.decision.value}")
        print(f"Score    : {r.score}")
        if r.decision is Decision.GO:
            print(f"Side     : {r.side.value}  Kelly: {r.kelly_fraction:.0%}")
            print("Subscores:", r.subscores)
        else:
            for g in r.failed_gates:
                print("  x", g)
        print("Rationale:", r.rationale)
