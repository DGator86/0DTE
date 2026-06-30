"""
orchestrator.py
==============
The tick loop. Deliberately dumb: it owns the clock and the data feed, and on
every tick it calls the pure decision_engine and writes the result to the
journal. All intelligence lives downstream; all measurement lives in the journal.

    feed.snapshot(now) -> (MarketSnapshot, ChainSnapshot)
        -> decision_engine.decide(...) -> TradeDecision
        -> journal.log(decision.as_row())

After the close, settle(session_date) pulls the settlement price from the feed
and fills realized P&L for every logged candidate that day (hypothetical for
no-trades), making the gate measurable.

Bind it to production by implementing the DataFeed protocol against your VPS /
options data / GEX module. A SyntheticFeed is included for replay tests.

NOT financial advice.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

from gate_scorer import MarketSnapshot
from rnd_extractor import ChainSnapshot
from decision_engine import decide, EngineConfig, TradeDecision
from journal import Journal

ET = ZoneInfo("America/New_York")


class DataFeed(Protocol):
    """Implement this against your real data sources."""
    def snapshot(self, now: dt.datetime) -> Optional[tuple[MarketSnapshot, ChainSnapshot]]:
        ...
    def settlement_price(self, session_date: str) -> Optional[float]:
        ...


@dataclass
class Orchestrator:
    feed: DataFeed
    journal: Journal
    cfg: Optional[EngineConfig] = None
    physical_pdf: Optional[object] = None      # callable(grid)->density; single source of truth

    def tick(self, now: dt.datetime) -> Optional[TradeDecision]:
        snap = self.feed.snapshot(now)
        if snap is None:
            return None
        market, chain = snap
        decision = decide(market, chain, self.cfg, physical_pdf=self.physical_pdf)
        self.journal.log(decision.as_row())
        return decision

    def run_replay(self, timestamps: Sequence[dt.datetime]) -> list[TradeDecision]:
        """Drive the loop over an explicit list of tick times (backtest/replay)."""
        out = []
        for t in timestamps:
            d = self.tick(t)
            if d is not None:
                out.append(d)
        return out

    def run_live(self, interval_seconds: int, until: dt.datetime,
                 clock=lambda: dt.datetime.now(ET)) -> list[TradeDecision]:
        """Thin live loop; sleeps between ticks until `until`. Clock injectable."""
        out = []
        while clock() < until:
            d = self.tick(clock())
            if d is not None:
                out.append(d)
            time.sleep(interval_seconds)
        return out

    def settle(self, session_date: str) -> int:
        price = self.feed.settlement_price(session_date)
        if price is None:
            return 0
        return self.journal.settle_session(session_date, price)


# --------------------------------------------------------------------------- #
# Synthetic feed for replay tests                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticFeed:
    """Generates evolving market+chain snapshots from a scripted scenario list.

    Each scenario is a dict overriding the defaults below; the feed builds a
    consistent option chain around the given spot/vol so the engine has real
    quotes to work with.
    """
    scenarios: dict                      # ts -> overrides
    settle: float
    base_spot: float = 600.0

    def _build_chain(self, spot, t_years, s_atm, skew_s) -> ChainSnapshot:
        import numpy as np
        from rnd_extractor import ChainQuote, _bs_call_fwd
        F = spot
        DF = 1.0
        qs = []
        for K in np.arange(spot - 20, spot + 21, 1.0):
            k = np.log(K / F)
            s = max(s_atm + skew_s * k, 0.0008)
            cm = _bs_call_fwd(F, K, s) * DF
            pm = max(cm - DF * (F - K), 0.0)
            cm = max(cm, 0.0)
            h = 0.01 + 0.002 * max(cm, pm)
            qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h, max(pm - h, 0), pm + h))
        return ChainSnapshot(qs, spot=spot, t_years=t_years, r=0.05)

    def snapshot(self, now: dt.datetime):
        key = now.isoformat()
        if key not in self.scenarios:
            return None
        o = self.scenarios[key]
        spot = o.get("spot", self.base_spot)
        market = MarketSnapshot(
            spot=spot,
            net_gex=o.get("net_gex", 3.5e9),
            gamma_flip=o.get("gamma_flip", spot - 7),
            call_wall=o.get("call_wall", spot + 6),
            put_wall=o.get("put_wall", spot - 5),
            gex_pct_rank=o.get("gex_pct_rank", 0.85),
            vix9d=o.get("vix9d", 12.0), vix=o.get("vix", 13.0), vix3m=o.get("vix3m", 15.0),
            vvix=o.get("vvix", 92.0), vvix_baseline=o.get("vvix_baseline", 95.0),
            straddle_breakeven=o.get("straddle_breakeven", 4.0),
            expected_range=o.get("expected_range", 3.2),
            adx=o.get("adx", 13.0), rsi=o.get("rsi", 51.0),
            bb_width=o.get("bb_width", 1.4), bb_width_baseline=o.get("bb_width_baseline", 2.0),
            vwap=o.get("vwap", spot), vwap_reversion_count=o.get("vwap_reversion_count", 4),
            tick_abs_mean=o.get("tick_abs_mean", 480.0), cvd_slope=o.get("cvd_slope", 0.05),
            now=now, has_catalyst=o.get("has_catalyst", False),
            catalyst_label=o.get("catalyst_label", ""),
        )
        chain = self._build_chain(
            spot, o.get("t_years", 5.0 / (24 * 365)),
            o.get("s_atm", 0.0050), o.get("skew_s", -0.030),
        )
        return market, chain

    def settlement_price(self, session_date: str):
        return self.settle
