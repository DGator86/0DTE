"""
decision_matrix.py
==================
27-cell (dealer_regime × vol_regime × momentum_regime) -> TradeIntent.

The table encodes the premium-selling bias of this system: in a long-gamma,
low-volatility, ranging market, premium selling is the structural trade.
In a short-gamma trending market, directional long-premium is the alternative.
High-vol / at-flip / catalyst -> stand aside.

Dealer vetoes override the table: short-gamma or backwardation always blocks
premium engines regardless of the other two axes.

Bug fixed (and guarded): _dominant() crashed when a timeframe basket had no
computable regime (all None). Now: ignores None regimes, and decide_from_matrix
stands down cleanly when a basket is undefined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# TradeIntent                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TradeIntent:
    structure: str            # "put_credit" | "call_credit" | "iron_condor" |
                              # "LCS" | "LP" | "LC" | "STG" | "BKS" | "none"
    direction: str            # "put_side" | "call_side" | "neutral" | "none"
    conviction: str           # "HIGH" | "MEDIUM" | "LOW" | "NONE"
    size_mult: float          # fraction of normal size (0.0 = no trade)
    engine: str               # "premium_selector" | "directional_selector" | "none"
    rationale: str

    @property
    def is_trade(self) -> bool:
        return self.size_mult > 0 and self.structure != "none"


# --------------------------------------------------------------------------- #
# The 27-cell table                                                            #
# --------------------------------------------------------------------------- #
# Key: (dealer_regime, vol_regime, momentum_regime)
# Value: (structure, direction, conviction, size_mult, engine)

_CELL = tuple[str, str, str, float, str]

DECISION_TABLE: dict[tuple[str, str, str], _CELL] = {
    # ---- LONG_GAMMA ----
    # Low vol: premium selling at its cleanest
    ("long_gamma", "low_vol",    "neutral"): ("iron_condor",  "neutral",   "HIGH",   1.00, "premium_selector"),
    ("long_gamma", "low_vol",    "bull"):    ("call_credit",  "call_side", "MEDIUM", 0.80, "premium_selector"),
    ("long_gamma", "low_vol",    "bear"):    ("put_credit",   "put_side",  "MEDIUM", 0.80, "premium_selector"),
    # Normal vol: same families, smaller size
    ("long_gamma", "normal_vol", "neutral"): ("iron_condor",  "neutral",   "MEDIUM", 0.70, "premium_selector"),
    ("long_gamma", "normal_vol", "bull"):    ("call_credit",  "call_side", "LOW",    0.55, "premium_selector"),
    ("long_gamma", "normal_vol", "bear"):    ("put_credit",   "put_side",  "LOW",    0.55, "premium_selector"),
    # High vol in long-gamma: vol spike under long-gamma = potential breakout. Stand aside.
    ("long_gamma", "high_vol",   "neutral"): ("none",         "none",      "NONE",   0.00, "none"),
    ("long_gamma", "high_vol",   "bull"):    ("none",         "none",      "NONE",   0.00, "none"),
    ("long_gamma", "high_vol",   "bear"):    ("none",         "none",      "NONE",   0.00, "none"),

    # ---- SHORT_GAMMA ----
    # Short gamma + trend: directional
    ("short_gamma", "normal_vol", "bull"):   ("LCS",  "call_side", "HIGH",   1.00, "directional_selector"),
    ("short_gamma", "normal_vol", "bear"):   ("LP",   "put_side",  "HIGH",   1.00, "directional_selector"),
    ("short_gamma", "normal_vol", "neutral"):("none", "none",      "NONE",   0.00, "none"),
    ("short_gamma", "low_vol",    "bull"):   ("LCS",  "call_side", "MEDIUM", 0.70, "directional_selector"),
    ("short_gamma", "low_vol",    "bear"):   ("LP",   "put_side",  "MEDIUM", 0.70, "directional_selector"),
    ("short_gamma", "low_vol",    "neutral"):("none", "none",      "NONE",   0.00, "none"),
    # Short gamma + high vol: convexity long
    ("short_gamma", "high_vol",   "bull"):   ("LC",   "call_side", "MEDIUM", 0.50, "directional_selector"),
    ("short_gamma", "high_vol",   "bear"):   ("LP",   "put_side",  "MEDIUM", 0.50, "directional_selector"),
    ("short_gamma", "high_vol",   "neutral"):("STG",  "neutral",   "LOW",    0.30, "directional_selector"),

    # ---- AT_FLIP ---- (transitional, discovery unresolved)
    ("at_flip", "low_vol",    "bull"):    ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "low_vol",    "bear"):    ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "low_vol",    "neutral"): ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "normal_vol", "bull"):    ("LCS",  "call_side", "LOW", 0.40, "directional_selector"),
    ("at_flip", "normal_vol", "bear"):    ("LP",   "put_side",  "LOW", 0.40, "directional_selector"),
    ("at_flip", "normal_vol", "neutral"): ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "high_vol",   "bull"):    ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "high_vol",   "bear"):    ("none", "none", "NONE", 0.00, "none"),
    ("at_flip", "high_vol",   "neutral"): ("none", "none", "NONE", 0.00, "none"),
}


# --------------------------------------------------------------------------- #
# Regime helpers                                                               #
# --------------------------------------------------------------------------- #
def _dominant(values: list) -> Optional[str]:
    """
    Most common non-None value in a list. Returns None if list is empty
    or has only None entries. Guarded against the crash on short history.
    """
    counts: dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def _basket_regime(rows: list[dict]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (dealer, vol, momentum) by majority vote across TF rows."""
    dealer = _dominant([r.get("dealer_regime") for r in rows])
    vol = _dominant([r.get("vol_regime") for r in rows])
    momentum = _dominant([r.get("momentum_regime") for r in rows])
    return dealer, vol, momentum


# --------------------------------------------------------------------------- #
# Top-level                                                                    #
# --------------------------------------------------------------------------- #
def decide_from_matrix(
    matrix: dict[str, float],
    regime_rows_list: list[dict],
    has_catalyst: bool = False,
) -> TradeIntent:
    """
    Look up the DECISION_TABLE cell for the consensus regime.
    Dealer vetoes (short-gamma regime, catalyst, undefined basket) override.
    """
    if has_catalyst:
        return TradeIntent("none", "none", "NONE", 0.0, "none",
                           "CATALYST: hard stop — all engines blocked")

    dealer, vol, momentum = _basket_regime(regime_rows_list)

    if dealer is None or vol is None or momentum is None:
        return TradeIntent("none", "none", "NONE", 0.0, "none",
                           f"basket_undefined: dealer={dealer} vol={vol} mom={momentum}")

    key = (dealer, vol, momentum)
    cell = DECISION_TABLE.get(key)

    if cell is None:
        return TradeIntent("none", "none", "NONE", 0.0, "none",
                           f"no_table_entry: {key}")

    structure, direction, conviction, size_mult, engine = cell

    rationale = (f"dealer={dealer} vol={vol} momentum={momentum} "
                 f"-> {structure} ({conviction} conv, {size_mult:.0%} size)")

    return TradeIntent(
        structure=structure,
        direction=direction,
        conviction=conviction,
        size_mult=size_mult,
        engine=engine,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from mtf_matrix import build_matrix, regime_rows, MTFInput, TIMEFRAMES

    # Simulate a long-gamma ranging day
    native_clean = {}
    for tf in TIMEFRAMES:
        native_clean[tf] = {
            "adx": 12.0, "rsi": 51.0, "ema_dist": 0.001,
            "bb_width": 0.011, "rv": 0.018, "cvd": 0.05,
            "vwap_dist": 0.0003, "tick_abs": 450.0,
        }
    snap_clean = {
        "spot": 602.5, "net_gex": 4.2e9, "gamma_flip": 596.0,
        "call_wall": 603.0, "put_wall": 598.0, "gex_pct_rank": 0.88,
        "vix": 13.0, "vix9d": 12.1, "vix3m": 15.2,
        "vvix": 92.0, "vvix_baseline": 95.0,
        "straddle_breakeven": 4.1, "expected_range": 3.2,
        "adx": 12.0, "rsi": 51.0,
        "bb_width": 1.5, "bb_width_baseline": 2.1,
        "cvd_slope": 0.03, "tick_abs_mean": 450.0,
    }

    # Simulate a short-gamma trending day
    native_trend = {}
    for tf in TIMEFRAMES:
        native_trend[tf] = {
            "adx": 26.0, "rsi": 36.0, "ema_dist": -0.006,
            "bb_width": 0.035, "rv": 0.05, "cvd": -0.5,
            "vwap_dist": -0.004, "tick_abs": 850.0,
        }
    snap_trend = {
        "spot": 588.0, "net_gex": -1.1e9, "gamma_flip": 593.0,
        "call_wall": 596.0, "put_wall": 585.0, "gex_pct_rank": 0.40,
        "vix": 18.0, "vix9d": 19.5, "vix3m": 17.0,
        "vvix": 120.0, "vvix_baseline": 95.0,
        "straddle_breakeven": 6.0, "expected_range": 6.5,
        "adx": 28.0, "rsi": 38.0,
        "bb_width": 3.1, "bb_width_baseline": 2.0,
        "cvd_slope": -0.7, "tick_abs_mean": 910.0,
    }

    print("=== DECISION_TABLE (27 cells) ===")
    for (d, v, m), (s, dr, cv, sz, eng) in sorted(DECISION_TABLE.items()):
        print(f"  {d:12} {v:12} {m:8} -> {s:15} {cv:6} {sz:.2f}")

    for label, native, snap in [
        ("clean_range", native_clean, snap_clean),
        ("short_gamma_trend", native_trend, snap_trend),
    ]:
        inp = MTFInput(native=native, snapshot=snap)
        mat = build_matrix(inp)
        rows = regime_rows(inp, mat)
        intent = decide_from_matrix(mat, rows)
        print(f"\n=== {label} ===")
        for r in rows:
            print(f"  [{r['tf']}] dealer={r['dealer_regime']}  vol={r['vol_regime']}  "
                  f"mom={r['momentum_regime']}")
        print(f"  INTENT: {intent}")
