from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Any
import numpy as np
import pandas as pd
from resample import RawBars, build_mtf_input
from mtf_matrix import build_matrix, regime_rows, render_text
from decision_matrix import TradeIntent, decide_from_matrix

@dataclass
class MarketSnapshot:
    gamma_sign: float; gamma_magnitude: float; flip_cushion: float
    channel_tightness: float; wall_proximity: float
    term_structure: float; vvix_elevation: float; richness: float
    skew_dir: float; tail_heaviness: float
    spot: Optional[float] = None; call_wall: Optional[float] = None
    put_wall: Optional[float] = None; gamma_flip: Optional[float] = None
    net_gex: Optional[float] = None; gex_pct_rank: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None; extra: dict = field(default_factory=dict)
    def mtf_snapshot(self):
        return {"gamma_sign": self.gamma_sign,"gamma_magnitude": self.gamma_magnitude,
            "flip_cushion": self.flip_cushion,"channel_tightness": self.channel_tightness,
            "wall_proximity": self.wall_proximity,"term_structure": self.term_structure,
            "vvix_elevation": self.vvix_elevation,"richness": self.richness,
            "skew_dir": self.skew_dir,"tail_heaviness": self.tail_heaviness}
    def dealer_vetoes(self):
        v=[]
        if self.gamma_sign < 0: v.append("short_gamma")
        if self.flip_cushion < 0: v.append("below_flip")
        if self.term_structure < 0: v.append("term_backwardation")
        if bool(self.extra.get("catalyst_now", False)): v.append("catalyst_now")
        return v

@dataclass
class FeedSnapshot:
    raw: RawBars; market: MarketSnapshot

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
    def __init__(self, days=30, seed=7, market=None):
        from resample import _synth_bars
        self.raw=_synth_bars(days=days, seed=seed)
        self.market=market or MarketSnapshot(
            gamma_sign=4.0e9,gamma_magnitude=0.86,flip_cushion=0.006,
            channel_tightness=0.010,wall_proximity=0.0025,term_structure=0.16,
            vvix_elevation=-0.03,richness=0.66,skew_dir=-0.17,tail_heaviness=0.30,
            spot=float(self.raw.close[-1]),call_wall=float(self.raw.close[-1]+5),
            put_wall=float(self.raw.close[-1]-5),gamma_flip=float(self.raw.close[-1]-2),
            net_gex=4.0e9,gex_pct_rank=0.86)
    def snapshot(self, symbol, lookback_minutes):
        n=min(int(lookback_minutes), len(self.raw.close))
        raw=RawBars(ts=self.raw.ts[-n:],open=self.raw.open[-n:],high=self.raw.high[-n:],
            low=self.raw.low[-n:],close=self.raw.close[-n:],volume=self.raw.volume[-n:],
            signed_volume=None if self.raw.signed_volume is None else self.raw.signed_volume[-n:],
            tick=None if self.raw.tick is None else self.raw.tick[-n:])
        return FeedSnapshot(raw=raw, market=self.market)

if __name__ == "__main__":
    orch=PipelineOrchestrator(SyntheticFeed(days=30), lookback_minutes=30*390)
    result=orch.run_once("SPY")
    print(result.matrix_text); print()
    print(orch.route_ticket(result))
