"""
live_feed_adapter.py
====================
Vendor-agnostic feed adapter + PipelineOrchestrator for Track B (MTF regime routing).

Seam closed: FeedSnapshot now carries an optional ChainSnapshot and
PipelineOrchestrator.build_ticket() calls spread_selector.select_spreads()
using the STRUCTURE_TO_FAMILIES mapping when the regime produces a known
structure family. RND-derived signals (richness, skew, kurtosis) are also
injected into the MTF snapshot when a chain is present, so the matrix sees
a richer signal set.

MarketSnapshot is imported from gate_scorer — single source of truth for
all market state fields (spot, walls, flip, GEX, vol surface, etc.).

route_ticket() is kept for backward compatibility; it now delegates to
build_ticket() and returns the same rich dict.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from gate_scorer import MarketSnapshot
from resample import RawBars, build_mtf_input
from mtf_matrix import build_matrix, regime_rows, render_text
from decision_matrix import TradeIntent, decide_from_matrix
from rnd_extractor import (
    ChainSnapshot, RiskNeutralDensity, extract_rnd, compute_edge, RNDConfig,
)
from spread_selector import (
    GammaContext, SelectorConfig, SelectionResult, select_spreads,
    STRUCTURE_TO_FAMILIES,
)


# --------------------------------------------------------------------------- #
# Snapshot bundle                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class FeedSnapshot:
    raw: RawBars
    market: MarketSnapshot
    chain: Optional[ChainSnapshot] = None   # seam: carries optional option chain


# --------------------------------------------------------------------------- #
# Protocol                                                                      #
# --------------------------------------------------------------------------- #
class DataFeed(Protocol):
    def snapshot(self, symbol: str, lookback_minutes: int) -> FeedSnapshot: ...


# --------------------------------------------------------------------------- #
# Pipeline types                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    rows: list
    regimes: dict
    intent: TradeIntent
    snapshot: MarketSnapshot
    matrix_text: str
    rnd: Optional[RiskNeutralDensity] = None   # available when chain was present


# --------------------------------------------------------------------------- #
# Orchestrator                                                                  #
# --------------------------------------------------------------------------- #
class PipelineOrchestrator:
    def __init__(self, feed: DataFeed, lookback_minutes: int = 20_000):
        self.feed = feed
        self.lookback_minutes = int(lookback_minutes)

    def run_once(
        self,
        symbol: str = "SPY",
        rnd_cfg: Optional[RNDConfig] = None,
    ) -> PipelineResult:
        """
        Run one MTF+regime cycle.  When a ChainSnapshot is present the RND is
        extracted first; richness / skew / kurtosis are injected into the MTF
        snapshot so that the matrix sees the full signal set.
        """
        fs = self.feed.snapshot(symbol=symbol, lookback_minutes=self.lookback_minutes)

        # ---- Track A RND (optional: enriches MTF snapshot) ----
        rnd = edge = None
        if fs.chain is not None:
            try:
                rnd = extract_rnd(fs.chain, rnd_cfg or RNDConfig())
                edge = compute_edge(rnd, fs.chain, rnd_cfg or RNDConfig())
            except Exception:
                pass

        snap_dict = fs.market.mtf_snapshot()
        if edge is not None:
            snap_dict["richness"] = edge.richness_signal
        if rnd is not None:
            try:
                snap_dict["skew_dir"] = rnd.skew()
                snap_dict["tail_heaviness"] = rnd.excess_kurtosis()
            except Exception:
                pass

        inp = build_mtf_input(fs.raw, snap_dict)
        rows = build_matrix(inp)
        regimes = regime_rows(rows)
        vetoes = fs.market.dealer_vetoes()
        intent = decide_from_matrix(rows, regimes, vetoes=vetoes)

        return PipelineResult(
            rows=rows, regimes=regimes, intent=intent,
            snapshot=fs.market, matrix_text=render_text(rows, regimes),
            rnd=rnd,
        )

    def build_ticket(
        self,
        result: PipelineResult,
        selector_cfg: Optional[SelectorConfig] = None,
        rnd_cfg: Optional[RNDConfig] = None,
        symbol: str = "SPY",
    ) -> dict[str, Any]:
        """
        Produce a concrete ticket.  For known structure families this calls
        spread_selector.select_spreads() with the family set implied by the
        regime decision; for NT or unknown families it returns a stub.

        Always returns a dict; 'candidate' is None when no chain is available
        or when selection finds nothing passing vetoes.
        """
        intent = result.intent
        d = intent.decision
        structure = d.structure

        base: dict[str, Any] = {
            "structure": structure,
            "direction": d.direction,
            "conviction": d.conviction,
            "size_mult": intent.size_mult,
            "intent": intent,
            "vetoes": intent.vetoes,
            "note": intent.note,
        }

        if structure == "NT":
            return {**base, "engine": "none", "candidate": None,
                    "note": "Stand-down / NT cell — no trade"}

        target_families = STRUCTURE_TO_FAMILIES.get(structure)
        if target_families is None:
            return {**base, "engine": "unknown", "candidate": None,
                    "note": f"No family mapping for structure {structure!r}"}

        # Fresh snapshot so we always have an up-to-date chain reference
        fs = self.feed.snapshot(symbol, self.lookback_minutes)
        chain = fs.chain
        if chain is None:
            return {**base, "engine": "no_chain", "candidate": None,
                    "note": "No ChainSnapshot in FeedSnapshot — inject chain to enable selection"}

        ctx = GammaContext.from_market_snapshot(fs.market)
        try:
            rnd = extract_rnd(chain, rnd_cfg or RNDConfig())
            edge = compute_edge(rnd, chain, rnd_cfg or RNDConfig())
            sel: SelectionResult = select_spreads(
                chain, rnd, edge, ctx,
                cfg=selector_cfg or SelectorConfig(),
                target_families=target_families,
            )
        except Exception as exc:
            return {**base, "engine": "error", "candidate": None,
                    "error": str(exc), "note": f"Selection failed: {exc}"}

        candidate = sel.best
        if candidate is not None:
            # Scale the position cap by Track B size multiplier
            candidate.size_cap = round(intent.size_mult * candidate.size_cap, 3)

        return {
            **base,
            "engine": "spread_selector",
            "candidate": candidate,
            "ranked": sel.ranked[:6],
            "no_trade_reason": sel.no_trade_reason,
        }

    def route_ticket(self, result: PipelineResult) -> dict[str, Any]:
        """Backward-compatible wrapper: delegates to build_ticket()."""
        return self.build_ticket(result)


# --------------------------------------------------------------------------- #
# Synthetic feed (testing + demo)                                               #
# --------------------------------------------------------------------------- #
class SyntheticFeed:
    """
    Replay feed using synthetic bars + a static gate_scorer.MarketSnapshot.
    Inject a ChainSnapshot via the chain= argument to enable Track A selection.
    """

    def __init__(
        self,
        days: int = 30,
        seed: int = 7,
        market: Optional[MarketSnapshot] = None,
        chain: Optional[ChainSnapshot] = None,
    ):
        from resample import _synth_bars
        self._raw = _synth_bars(days=days, seed=seed)
        spot = float(self._raw.close[-1])
        self._market = market or MarketSnapshot(
            spot=spot, net_gex=4.0e9, gamma_flip=spot - 6.0,
            call_wall=spot + 5.0, put_wall=spot - 5.0, gex_pct_rank=0.86,
            vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
            straddle_breakeven=4.0, expected_range=3.2,
            adx=13.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
            vwap=spot, vwap_reversion_count=3,
            tick_abs_mean=480.0, cvd_slope=0.02,
            now=dt.datetime.now(dt.timezone.utc),
            has_catalyst=False,
        )
        self._chain = chain   # optional; enables Track A if provided

    def snapshot(self, symbol: str, lookback_minutes: int) -> FeedSnapshot:
        n = min(int(lookback_minutes), len(self._raw.close))
        raw = RawBars(
            ts=self._raw.ts[-n:], open=self._raw.open[-n:], high=self._raw.high[-n:],
            low=self._raw.low[-n:], close=self._raw.close[-n:], volume=self._raw.volume[-n:],
            signed_volume=(None if self._raw.signed_volume is None
                           else self._raw.signed_volume[-n:]),
            tick=None if self._raw.tick is None else self._raw.tick[-n:],
        )
        return FeedSnapshot(raw=raw, market=self._market, chain=self._chain)


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    from rnd_extractor import ChainQuote, _bs_call_fwd

    # Build a synthetic 0DTE option chain (±20 strikes, 4 h to expiry)
    spot0 = 600.0
    T0, r0 = 4.0 / (24 * 365), 0.05
    DF0 = math.exp(-r0 * T0)
    F0 = spot0 * math.exp(r0 * T0)

    qs = []
    for K in np.arange(spot0 - 20, spot0 + 21, 1.0):
        k = math.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    chain = ChainSnapshot(qs, spot=spot0, t_years=T0, r=r0)

    feed = SyntheticFeed(days=30, chain=chain)
    orch = PipelineOrchestrator(feed, lookback_minutes=30 * 390)

    print("=" * 60)
    print("  live_feed_adapter — seam-closed demo")
    print("=" * 60)

    result = orch.run_once("SPY")
    print(result.matrix_text)
    print()

    ticket = orch.build_ticket(result)
    print(f"Structure:  {ticket['structure']}")
    print(f"Direction:  {ticket['direction']}")
    print(f"Conviction: {ticket['conviction']}")
    print(f"Size mult:  {ticket['size_mult']}")
    print(f"Engine:     {ticket['engine']}")
    print(f"Vetoes:     {ticket['vetoes']}")

    c = ticket.get("candidate")
    if c is not None:
        print(f"\nBest candidate: {c.family}")
        print(f"  Shorts:     {c.short_strikes}")
        print(f"  Longs:      {c.long_strikes}")
        print(f"  Credit:     {c.credit}")
        print(f"  Max loss:   {c.max_loss}")
        print(f"  EV:         {c.ev}")
        print(f"  EV/risk:    {c.ev_per_risk}")
        print(f"  P(profit):  {c.prob_profit}")
        print(f"  Score:      {c.score}")
        print(f"  Size cap:   {c.size_cap}  (after Track B mult)")
    else:
        reason = ticket.get("no_trade_reason") or ticket.get("note", "")
        print(f"\nNo candidate: {reason}")

    ranked = ticket.get("ranked", [])
    if ranked:
        print(f"\nTop {len(ranked)} ranked candidates:")
        for r in ranked:
            mark = " ← best" if c is not None and r is c else ""
            print(f"  {r.family:<20} EV/risk={r.ev_per_risk:+.4f}  "
                  f"score={r.score:.4f}  vetoed={'no' if r.passes_vetoes else 'YES'}{mark}")

    print("=" * 60)
