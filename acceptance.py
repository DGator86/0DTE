"""
acceptance.py  —  compute the engine's `price_accepting` input. ONE primitive.

This deliberately does NOT build a technical-indicator engine. The edge here is
the leading gamma signal; gating it behind a stack of lagging indicators (full
EMA/ATR/relative-volume confluence) dilutes the signal and starves the sample.

Acceptance answers one question: has price *committed* through the gamma flip,
or is this a wick? That requires exactly two things:
  1. N consecutive completed 1-minute closes on one side of the flip
  2. VWAP agrees (one confirm line — the intraday fair-value reference)

Returns +1 (bullish acceptance), -1 (bearish), 0 (none / chop / at the flip).
Feed the result straight into spy0dte.decide(gm, price_accepting).

Pass COMPLETED bars only — never the in-progress current minute, or you reintroduce
the wick-chasing this is meant to prevent.
"""
from __future__ import annotations
from dataclasses import dataclass

ACCEPT_BARS = 3          # consecutive closes required beyond the flip
USE_VWAP_CONFIRM = True  # the single confirm; set False to use flip-closes alone


@dataclass
class Bar1m:
    ts_ms: int
    high: float
    low: float
    close: float
    volume: float


def session_vwap(bars: list[Bar1m]) -> float:
    """Volume-weighted average of typical price over the session so far."""
    num = den = 0.0
    for b in bars:
        tp = (b.high + b.low + b.close) / 3.0
        num += tp * b.volume
        den += b.volume
    return (num / den) if den > 0 else (bars[-1].close if bars else 0.0)


def compute_acceptance(bars: list[Bar1m], flip: float,
                       n: int = ACCEPT_BARS, use_vwap: bool = USE_VWAP_CONFIRM) -> int:
    """+1 bullish / -1 bearish / 0 none. `bars` are completed 1m bars, oldest→newest."""
    if len(bars) < n or flip <= 0:
        return 0
    last = bars[-n:]
    closes = [b.close for b in last]
    vwap = session_vwap(bars)
    px = closes[-1]

    above = all(c > flip for c in closes)
    below = all(c < flip for c in closes)

    if above and (not use_vwap or px > vwap):
        return +1
    if below and (not use_vwap or px < vwap):
        return -1
    return 0


if __name__ == "__main__":
    def mk(c, v=1e6):
        return Bar1m(0, c + 0.1, c - 0.1, c, v)

    flip = 600.0
    # three closes above flip, price above vwap -> +1
    bull = [mk(599.5), mk(599.8), mk(600.3), mk(600.5), mk(600.7)]
    # three closes below flip -> -1
    bear = [mk(600.5), mk(600.2), mk(599.7), mk(599.5), mk(599.3)]
    # straddling the flip (chop) -> 0
    chop = [mk(600.2), mk(599.8), mk(600.1), mk(599.9), mk(600.05)]
    print("bullish:", compute_acceptance(bull, flip))
    print("bearish:", compute_acceptance(bear, flip))
    print("chop:   ", compute_acceptance(chop, flip))
