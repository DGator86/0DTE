from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np
from gate_scorer import MarketSnapshot  # single source of truth for market state
from resample import RawBars, build_mtf_input
from mtf_matrix import build_matrix, regime_rows, render_text
from decision_matrix import TradeIntent, decide_from_matrix

@dataclass
class FeedSnapshot:
    raw: RawBars
    market: MarketSnapshot

class DataFeed(Protocol):
    def snapshot(self, symbol: str, lookback_minutes: int) -> FeedSnapshot: ...

@dataclass
class PipelineResult:
    rows: list; regimes: dict; intent: TradeIntent; snapshot: MarketSnapshot; matrix_text: str

class PipelineOrchestrator:
    def __init__(self, feed, lookback_minutes=20_000):
        self.feed=feed; self.lookback_minutes=int(lookback_minutes)
    def run_once(self, symbol="SPY"):
        fs=self.feed.snapshot(symbol=symbol, lookback_minutes=self.lookback_minutes)
        inp=build_mtf_input(fs.raw, fs.market.mtf_snapshot())
        rows=build_matrix(inp); regimes=regime_rows(rows)
        vetoes=fs.market.dealer_vetoes()
        intent=decide_from_matrix(rows, regimes, vetoes=vetoes)
        return PipelineResult(rows=rows, regimes=regimes, intent=intent,
            snapshot=fs.market, matrix_text=render_text(rows, regimes))
    def route_ticket(self, result):
        d=result.intent.decision
        premium={"PCS","CCS","IC","IF"}
        engine="premium_selector" if d.structure in premium else "directional_selector"
        if d.structure=="NT": engine="none"
        return {"engine":engine,"structure":d.structure,"direction":d.direction,
            "conviction":d.conviction,"size_mult":result.intent.size_mult,
            "anchor_tf":d.anchor_tf,"strike_rule":d.strike_rule,
            "vetoes":result.intent.vetoes,"note":result.intent.note}

class SyntheticFeed:
    """Replay feed using synthetic bars + a static gate_scorer.MarketSnapshot."""

    def __init__(self, days=30, seed=7, market: Optional[MarketSnapshot] = None):
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

    def snapshot(self, symbol: str, lookback_minutes: int) -> FeedSnapshot:
        n = min(int(lookback_minutes), len(self._raw.close))
        raw = RawBars(
            ts=self._raw.ts[-n:], open=self._raw.open[-n:], high=self._raw.high[-n:],
            low=self._raw.low[-n:], close=self._raw.close[-n:], volume=self._raw.volume[-n:],
            signed_volume=(None if self._raw.signed_volume is None
                           else self._raw.signed_volume[-n:]),
            tick=None if self._raw.tick is None else self._raw.tick[-n:],
        )
        return FeedSnapshot(raw=raw, market=self._market)

if __name__ == "__main__":
    orch=PipelineOrchestrator(SyntheticFeed(days=30), lookback_minutes=30*390)
    result=orch.run_once("SPY")
    print(result.matrix_text); print()
    print(orch.route_ticket(result))
