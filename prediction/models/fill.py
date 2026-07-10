"""
prediction/models/fill.py
=========================
Fill-fraction priors for the V2 execution-cost model
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §13.2–13.3).

fill_fraction ∈ [0, 1] interpolates strategy price from midpoint (0) to
natural (1). Before empirical fill records exist, structure-specific priors
are used:

  single-leg liquid option : 0.35
  two-leg vertical         : 0.50
  four-leg structure       : 0.65

Penalties stack on top for late-day trading, stale quotes, wide relative
spreads, and elevated realized vol. The empirical model
(fill_fraction ~ structure + spread + price + time + vol + quote age) is
trained later from paper/manual FillRecord rows — this module only supplies
the prior and the feature vector shape that trainer will consume.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


# Spec §13.2 defaults
DEFAULT_FILL_BY_N_LEGS = {
    1: 0.35,
    2: 0.50,
    3: 0.55,          # e.g. broken-wing / backspread-ish
    4: 0.65,
}

# Family overrides when leg count alone is ambiguous (e.g. long_strangle = 2
# legs but typically wider than a tight vertical).
FAMILY_FILL_PRIOR: dict[str, float] = {
    "long_call": 0.35,
    "long_put": 0.35,
    "naked_defended_call": 0.40,
    "cash_secured_put": 0.40,
    "put_credit": 0.50,
    "call_credit": 0.50,
    "long_call_spread": 0.50,
    "long_put_spread": 0.50,
    "long_strangle": 0.55,
    "broken_wing": 0.55,
    "backspread_call": 0.55,
    "backspread_put": 0.55,
    "backspread": 0.55,
    "iron_fly": 0.65,
    "iron_condor": 0.65,
}


@dataclass
class FillPriorConfig:
    by_n_legs: dict = field(default_factory=lambda: dict(DEFAULT_FILL_BY_N_LEGS))
    by_family: dict = field(default_factory=lambda: dict(FAMILY_FILL_PRIOR))
    # Additive penalties (clipped so the final fraction stays in [0, 1]).
    late_day_penalty: float = 0.10          # inside late_day_minutes_to_close
    late_day_minutes_to_close: float = 60.0
    stale_quote_penalty: float = 0.10
    stale_quote_seconds: float = 5.0
    wide_spread_penalty: float = 0.10       # when relative_spread > threshold
    wide_spread_rel: float = 0.15           # half-spread / mid across legs
    high_vol_penalty: float = 0.05
    high_vol_threshold: float = 0.25        # annualized realized vol
    # Floor / ceiling
    min_fraction: float = 0.0
    max_fraction: float = 1.0


def n_legs(legs: Sequence) -> int:
    return len(list(legs))


def base_fill_fraction(family: str, n_legs_: int,
                       cfg: Optional[FillPriorConfig] = None) -> float:
    """Structure prior before situational penalties."""
    cfg = cfg or FillPriorConfig()
    if family in cfg.by_family:
        return float(cfg.by_family[family])
    # nearest configured leg-count bucket
    if n_legs_ in cfg.by_n_legs:
        return float(cfg.by_n_legs[n_legs_])
    # clamp to [1, 4] range of known priors
    key = min(max(int(n_legs_), 1), 4)
    return float(cfg.by_n_legs.get(key, 0.50))


def fill_fraction_for(
    family: str,
    *,
    n_legs: Optional[int] = None,
    quote_age_seconds: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    relative_spread: Optional[float] = None,
    option_price: Optional[float] = None,       # reserved for empirical model
    realized_vol: Optional[float] = None,
    cfg: Optional[FillPriorConfig] = None,
) -> tuple[float, dict]:
    """
    Return (fill_fraction, diagnostics).

    Penalties only ever WORSEN the fill (raise the fraction toward natural).
    Older quotes, later sessions, wider spreads, and higher vol never improve
    the expected fill — an acceptance criterion of PR 6.
    """
    cfg = cfg or FillPriorConfig()
    n = int(n_legs) if n_legs is not None else 2
    base = base_fill_fraction(family, n, cfg)
    penalties = {}
    total_pen = 0.0

    if (minutes_to_close is not None
            and minutes_to_close <= cfg.late_day_minutes_to_close):
        penalties["late_day"] = cfg.late_day_penalty
        total_pen += cfg.late_day_penalty

    if (quote_age_seconds is not None
            and quote_age_seconds >= cfg.stale_quote_seconds):
        penalties["stale_quote"] = cfg.stale_quote_penalty
        total_pen += cfg.stale_quote_penalty

    if (relative_spread is not None
            and relative_spread >= cfg.wide_spread_rel):
        penalties["wide_spread"] = cfg.wide_spread_penalty
        total_pen += cfg.wide_spread_penalty

    if (realized_vol is not None
            and realized_vol >= cfg.high_vol_threshold):
        penalties["high_vol"] = cfg.high_vol_penalty
        total_pen += cfg.high_vol_penalty

    frac = min(max(base + total_pen, cfg.min_fraction), cfg.max_fraction)
    diag = {
        "base": base,
        "n_legs": n,
        "family": family,
        "penalties": penalties,
        "total_penalty": total_pen,
        "fill_fraction": frac,
        "option_price": option_price,
    }
    return frac, diag


def fill_feature_vector(family: str, *, n_legs: int,
                        relative_spread: float,
                        option_price: float,
                        minutes_to_close: float,
                        quote_age_seconds: float,
                        realized_vol: float) -> dict:
    """Feature dict for the eventual empirical fill model (§13.3)."""
    return {
        "family": family,
        "n_legs": int(n_legs),
        "relative_spread": float(relative_spread),
        "option_price": float(option_price),
        "minutes_to_close": float(minutes_to_close),
        "quote_age_seconds": float(quote_age_seconds),
        "realized_vol": float(realized_vol),
    }
