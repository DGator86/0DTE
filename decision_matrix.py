"""
decision_matrix.py
==================
Turns the multi-timeframe matrix (mtf_matrix.py) into a concrete trade intent:
direction, structure family, strike rule, conviction/size, and invalidation.

Three decision axes, collapsed from the matrix:
  EXECUTION regime = blend of fast TFs (1m/5m/15m) -- what's harvestable NOW
  CONTEXT  regime  = blend of slow TFs (1h/4h/1d)  -- the bigger move / the threat
  DIRECTION bias   = directional cells, context-weighted -> bull / neutral / bear

The full 3x3x3 = 27-cell DECISION_TABLE below enumerates every combination and
the structure it implies. Dealer-state vetoes (catalyst, short gamma, below
flip) override the premium-selling cells, because those are regime FACTS that
no timeframe blend should be allowed to outvote.

Structure families
  IC  iron condor (neutral credit)        LCS long call spread (bull debit)
  PCS put credit spread (bull credit)     LPS long put spread (bear debit)
  CCS call credit spread (bear credit)    LC  long call (convex bull)
  IF  iron fly (pinned credit)            LP  long put (convex bear)
  STG long strangle (vol expansion)       BKS ratio backspread (convex dir.)
  NT  no trade / stand down

NOT financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mtf_matrix import build_matrix, regime_rows, MatrixRow, TIMEFRAMES

FAST = ["1m", "5m", "15m"]
SLOW = ["1h", "4h", "1d"]
DIR_VARS = ["di_spread", "ema_slope", "cvd_persistence", "vwap_slope", "rsi"]


# --------------------------------------------------------------------------- #
# Decision spec                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    structure: str          # family code
    direction: str          # "call" | "put" | "both" | "none"
    conviction: str         # HIGH | MED | LOW | NONE
    capture: str            # what edge this harvests
    strike_rule: str        # how to place strikes
    anchor_tf: str          # timeframe whose level defines invalidation


SIZE = {"HIGH": 1.0, "MED": 0.6, "LOW": 0.3, "NONE": 0.0}


# --------------------------------------------------------------------------- #
# The 27-cell table: (exec_regime, context_regime, direction) -> Decision      #
# --------------------------------------------------------------------------- #
def _D(s, d, c, cap, rule, anchor):
    return Decision(s, d, c, cap, rule, anchor)

DECISION_TABLE: dict[tuple, Decision] = {
    # ===== EXEC = COMPRESSION (fast tape ranging -> harvest theta) =====
    ("compression", "compression", "bull"):
        _D("PCS", "put", "HIGH", "theta + mild up-drift", "short put ~0.30Δ below spot / under put wall; long 1-2$ below", "15m"),
    ("compression", "compression", "neutral"):
        _D("IC", "both", "HIGH", "pure theta, two-sided decay", "shorts ~0.18-0.22Δ inside the put/call wall channel", "15m"),
    ("compression", "compression", "bear"):
        _D("CCS", "call", "HIGH", "theta + mild down-drift", "short call ~0.30Δ above spot / over call wall; long 1-2$ above", "15m"),

    # the COIL: fast range, slow trend -> sell premium but lean with the trend
    ("compression", "trend", "bull"):
        _D("PCS", "put", "MED", "theta while leaning with slow uptrend", "short put just under fast support; size down; respect slow trend break", "1h"),
    ("compression", "trend", "neutral"):
        _D("IC", "both", "MED", "theta, but skew condor toward trend side", "widen the side the slow trend points to; tighten the other", "1h"),
    ("compression", "trend", "bear"):
        _D("CCS", "call", "MED", "theta while leaning with slow downtrend", "short call just over fast resistance; size down", "1h"),

    # fast range but slow EXPANDING -> pre-break compression, danger
    ("compression", "breakout", "bull"):
        _D("LCS", "call", "LOW", "anticipate the break up; minimal theta", "small long call spread in trend dir; or stand aside", "1h"),
    ("compression", "breakout", "neutral"):
        _D("STG", "both", "LOW", "buy cheap vol before expansion (IV still low here)", "long ATM strangle ~0.30Δ; only if IV rank low", "1h"),
    ("compression", "breakout", "bear"):
        _D("LPS", "put", "LOW", "anticipate the break down; minimal theta", "small long put spread in trend dir; or stand aside", "1h"),

    # ===== EXEC = TREND (fast tape trending -> directional debit) =====
    # fast trend INTO slow range -> likely stall/revert at range edge
    ("trend", "compression", "bull"):
        _D("LCS", "call", "LOW", "ride momentum but expect slow-range cap", "tight long call spread; take profit fast at slow resistance", "5m"),
    ("trend", "compression", "neutral"):
        _D("NT", "none", "NONE", "trend without direction into a range = no edge", "stand down", "—"),
    ("trend", "compression", "bear"):
        _D("LPS", "put", "LOW", "ride momentum but expect slow-range floor", "tight long put spread; take profit fast at slow support", "5m"),

    # ALIGNED TREND -> cleanest directional; premium selling stands down
    ("trend", "trend", "bull"):
        _D("LCS", "call", "HIGH", "directional delta + gamma, trend continuation", "long ATM/ITM call spread, width = 1 fast expected move", "1h"),
    ("trend", "trend", "neutral"):
        _D("NT", "none", "NONE", "aligned trend needs a direction; none present", "stand down until DI/EMA resolve", "—"),
    ("trend", "trend", "bear"):
        _D("LPS", "put", "HIGH", "directional delta + gamma, trend continuation", "long ATM/ITM put spread, width = 1 fast expected move", "1h"),

    # trend + slow expansion -> strong momentum, go convex
    ("trend", "breakout", "bull"):
        _D("LC", "call", "HIGH", "convex breakout continuation up", "long ~0.40-0.50Δ call; or call backspread for convexity", "1h"),
    ("trend", "breakout", "neutral"):
        _D("STG", "both", "MED", "something's expanding, direction unclear", "long straddle; manage as a gamma scalp", "1h"),
    ("trend", "breakout", "bear"):
        _D("LP", "put", "HIGH", "convex breakout continuation down", "long ~0.40-0.50Δ put; or put backspread for convexity", "1h"),

    # ===== EXEC = BREAKOUT (fast tape expanding -> long vol / directional) =====
    # fast spike into slow range -> fakeout / liquidity grab risk
    ("breakout", "compression", "bull"):
        _D("LC", "call", "LOW", "breakout against a range = fakeout risk", "small long call, quick exit; or NT", "5m"),
    ("breakout", "compression", "neutral"):
        _D("NT", "none", "NONE", "vol spike into a range, no direction = noise", "stand down", "—"),
    ("breakout", "compression", "bear"):
        _D("LP", "put", "LOW", "breakdown against a range = fakeout risk", "small long put, quick exit; or NT", "5m"),

    # fast breakout WITH slow trend -> continuation, highest convex edge
    ("breakout", "trend", "bull"):
        _D("LC", "call", "HIGH", "convex continuation, fast + slow aligned up", "long ~0.45Δ call or bull backspread", "30m"),
    ("breakout", "trend", "neutral"):
        _D("STG", "both", "MED", "expansion in a trending context", "long straddle, lean toward trend side", "30m"),
    ("breakout", "trend", "bear"):
        _D("LP", "put", "HIGH", "convex continuation, fast + slow aligned down", "long ~0.45Δ put or bear backspread", "30m"),

    # full expansion both TFs -> vol event; vol already rich, temper size
    ("breakout", "breakout", "bull"):
        _D("LC", "call", "MED", "directional, but buying expensive vol late", "long call; smaller size, IV is rich", "30m"),
    ("breakout", "breakout", "neutral"):
        _D("STG", "both", "MED", "vol event, no direction; IV rich", "long straddle only if you expect RV>IV to persist", "30m"),
    ("breakout", "breakout", "bear"):
        _D("LP", "put", "MED", "directional, but buying expensive vol late", "long put; smaller size, IV is rich", "30m"),
}

PREMIUM_STRUCTURES = {"PCS", "CCS", "IC", "IF"}

# ---- channel-based conviction adjustment (post-table size multiplier) ------
# Fast-TF Bollinger/Keltner/Donchian cells nudge the table-defined size.
# Scores are the standardized 0..100 matrix cells (donchian breakouts read 50
# when price is inside the prior channel).
CH_SQUEEZE_BOOST_MIN = 60.0   # avg fast-TF bb_squeeze at/above this = strong squeeze
CH_BREAKOUT_MIN = 65.0        # avg fast-TF donchian breakout leg at/above this = active break
CH_BOOST = 1.15               # credit-structure boost in squeeze-with-no-breakout
CH_TRIM = 0.75                # trim when a breakout opposes the trade

# Veto names that forbid premium selling. Both naming conventions are accepted:
# gate_scorer.dealer_vetoes emits "short_gamma"/"below_flip" while
# regime_classifier._vetoes (the one wired into the live loop) emits
# "short_gamma_regime"/"below_gamma_flip". Matching only one set silently
# disabled the credit->debit flip below for the live path.
NO_PREMIUM_VETOES = {
    "short_gamma", "short_gamma_regime",
    "below_flip", "below_gamma_flip",
    "term_backwardation",
}


# --------------------------------------------------------------------------- #
# Collapse the matrix into the three axes                                      #
# --------------------------------------------------------------------------- #
def _regime_blend(regimes: dict, tfs: list) -> dict:
    """Average available regime cells across a timeframe basket.

    Real feeds commonly have incomplete higher-timeframe indicators early in a
    session/history window. Missing cells must degrade gracefully instead of
    crashing or being treated as zero.
    """
    out = {}
    for regime, cells in regimes.items():
        vals = [cells.get(tf) for tf in tfs if cells.get(tf) is not None]
        out[regime] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def _dominant(blend: dict) -> tuple:
    valid = {k: v for k, v in blend.items() if v is not None}
    if not valid:
        return "none", None
    name = max(valid, key=valid.get)
    return name, valid[name]


def _direction_bias(rows: list, tfs_fast, tfs_slow) -> tuple:
    by = {r.variable: r for r in rows}
    def composite(tfs):
        vals = []
        for v in DIR_VARS:
            r = by.get(v)
            if not r:
                continue
            cell = [r.scores[t] for t in tfs if r.scores.get(t) is not None]
            if cell:
                vals.append(sum(cell) / len(cell))
        return sum(vals) / len(vals) if vals else 50.0
    fast = composite(tfs_fast)
    slow = composite(tfs_slow)
    bias = 0.4 * fast + 0.6 * slow            # context-weighted
    label = "bull" if bias >= 58 else ("bear" if bias <= 42 else "neutral")
    return label, round(bias, 1)


def _avg_cell(by: dict, var: str, tfs: list) -> Optional[float]:
    r = by.get(var)
    if not r:
        return None
    vals = [r.scores[t] for t in tfs if r.scores.get(t) is not None]
    return (sum(vals) / len(vals)) if vals else None


# Structures hurt by a downside / upside Donchian break, respectively.
# STG/BKS want expansion and NT has no size, so none of them are trimmed.
_BULL_EXPOSED = {"PCS", "LCS", "LC"}
_BEAR_EXPOSED = {"CCS", "LPS", "LP"}


def _channel_size_adjust(rows: list, structure: str) -> tuple[float, str]:
    """Post-table conviction multiplier from fast-TF channel cells.

    Returns (multiplier in [CH_TRIM, CH_BOOST], reason). Neutral 1.0 whenever
    channel cells are unavailable (short history) so behavior is unchanged.

      * Boost credit structures when the fast tape shows a strong TTM squeeze
        (avg bb_squeeze >= CH_SQUEEZE_BOOST_MIN) with no active Donchian
        breakout -- the highest-quality theta-harvest environment.
      * Trim any directionally exposed structure when an active Donchian
        breakout (avg leg >= CH_BREAKOUT_MIN) opposes it; neutral premium
        (IC/IF) is trimmed on a strong break in either direction.
    """
    by = {r.variable: r for r in rows}
    squeeze = _avg_cell(by, "bb_squeeze", FAST)
    brk_up = _avg_cell(by, "donchian_breakout_up", FAST)
    brk_dn = _avg_cell(by, "donchian_breakout_down", FAST)
    brk_max = max(brk_up or 50.0, brk_dn or 50.0)

    hostile_break = (
        (structure in _BULL_EXPOSED and (brk_dn or 50.0) >= CH_BREAKOUT_MIN)
        or (structure in _BEAR_EXPOSED and (brk_up or 50.0) >= CH_BREAKOUT_MIN)
        or (structure in {"IC", "IF"} and brk_max >= CH_BREAKOUT_MIN)
    )
    if hostile_break:
        side = "down" if (brk_dn or 50.0) >= (brk_up or 50.0) else "up"
        return CH_TRIM, f"channel trim: donchian breakout {side} opposes {structure}"

    if (structure in PREMIUM_STRUCTURES and squeeze is not None
            and squeeze >= CH_SQUEEZE_BOOST_MIN and brk_max < CH_BREAKOUT_MIN):
        return CH_BOOST, f"channel boost: fast-TF squeeze {squeeze:.0f}, no breakout"

    return 1.0, ""


# --------------------------------------------------------------------------- #
# Live decision                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class TradeIntent:
    exec_regime: str
    context_regime: str
    direction_bias: str
    bias_value: float
    decision: Decision
    size_mult: float
    vetoes: list
    note: str


def decide_from_matrix(rows: list, regimes: dict,
                       vetoes: Optional[list] = None) -> TradeIntent:
    vetoes = vetoes or []
    fast_blend = _regime_blend(regimes, FAST)
    slow_blend = _regime_blend(regimes, SLOW)
    exec_r, _ = _dominant(fast_blend)
    ctx_r, _ = _dominant(slow_blend)
    dir_label, dir_val = _direction_bias(rows, FAST, SLOW)

    # insufficient history on a basket => regime undefined => stand down cleanly
    if exec_r == "none" or ctx_r == "none":
        return TradeIntent(
            exec_regime=exec_r, context_regime=ctx_r,
            direction_bias=dir_label, bias_value=dir_val,
            decision=Decision("NT", "none", "NONE",
                              "insufficient timeframe history; regime undefined",
                              "stand down", "—"),
            size_mult=0.0, vetoes=vetoes,
            note="insufficient history: a timeframe basket had no computable regime",
        )

    decision = DECISION_TABLE[(exec_r, ctx_r, dir_label)]
    size = SIZE[decision.conviction]
    note = ""

    # dealer-state vetoes override premium-selling cells
    hard_stop = any(v.startswith("catalyst") for v in vetoes)
    no_premium = any(v in NO_PREMIUM_VETOES for v in vetoes)

    if hard_stop:
        decision = DECISION_TABLE[("trend", "trend", "neutral")]  # the NT cell
        decision = Decision("NT", "none", "NONE", "catalyst hard stop", "stand down", "—")
        size = 0.0
        note = "catalyst veto: all engines blocked"
    elif no_premium and decision.structure in PREMIUM_STRUCTURES:
        # flip a credit structure to its directional debit cousin or stand down
        flip = {"PCS": "LCS", "CCS": "LPS", "IC": "NT", "IF": "NT"}[decision.structure]
        if flip == "NT":
            decision = Decision("NT", "none", "NONE",
                                "premium forbidden (short gamma) and no clean direction",
                                "stand down", "—")
            size = 0.0
        else:
            d = "call" if flip == "LCS" else "put"
            decision = Decision(flip, d, "LOW",
                                "premium forbidden by dealer state; express bias as debit",
                                "small directional spread in bias direction", decision.anchor_tf)
            size = SIZE["LOW"]
        note = "premium veto: short-gamma/below-flip forces directional or stand-down"

    # channel-based conviction adjustment (post-table, post-veto): fast-TF
    # squeeze/breakout cells nudge size, never resurrect a vetoed trade
    if size > 0:
        ch_mult, ch_note = _channel_size_adjust(rows, decision.structure)
        if ch_mult != 1.0:
            size *= ch_mult
            note = f"{note}; {ch_note}" if note else ch_note

    return TradeIntent(
        exec_regime=exec_r, context_regime=ctx_r,
        direction_bias=dir_label, bias_value=dir_val,
        decision=decision, size_mult=round(size, 2), vetoes=vetoes, note=note,
    )


# --------------------------------------------------------------------------- #
# Renderers                                                                    #
# --------------------------------------------------------------------------- #
def render_full_table() -> str:
    h = f"{'exec':<12}{'context':<12}{'dir':<9}{'struct':<6}{'side':<6}{'conv':<6}{'capture'}"
    lines = [h, "-" * 96]
    order = ["compression", "trend", "breakout"]
    dirs = ["bull", "neutral", "bear"]
    for e in order:
        for c in order:
            for d in dirs:
                dec = DECISION_TABLE[(e, c, d)]
                lines.append(f"{e:<12}{c:<12}{d:<9}{dec.structure:<6}{dec.direction:<6}"
                             f"{dec.conviction:<6}{dec.capture}")
    return "\n".join(lines)


if __name__ == "__main__":
    from mtf_matrix import demo_input

    rows = build_matrix(demo_input())
    regimes = regime_rows(rows)

    print("FULL 27-CELL DECISION TABLE\n")
    print(render_full_table())

    print("\n\nLIVE DECISION on the coiling-day matrix:")
    intent = decide_from_matrix(rows, regimes, vetoes=[])
    d = intent.decision
    print(f"  exec={intent.exec_regime}  context={intent.context_regime}  "
          f"bias={intent.direction_bias} ({intent.bias_value})")
    print(f"  -> {d.structure} ({d.direction}), conviction {d.conviction}, size x{intent.size_mult}")
    print(f"     capture : {d.capture}")
    print(f"     strikes : {d.strike_rule}")
    print(f"     invalidate on {d.anchor_tf} structure break")

    print("\n  Same matrix, but dealer state flips to SHORT GAMMA:")
    intent2 = decide_from_matrix(rows, regimes, vetoes=["short_gamma", "below_flip"])
    d2 = intent2.decision
    print(f"  -> {d2.structure} ({d2.direction}), conviction {d2.conviction}, size x{intent2.size_mult}")
    print(f"     {intent2.note}")
