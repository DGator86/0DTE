"""
orchestrator.py
===============
Track-A tick loop: DataFeed -> decision_engine.decide -> journal.log;
settle() post-close fills realized P&L.

DataFeed is a Protocol: implement snapshot() to return the current
(MarketSnapshot, ChainSnapshot) pair for any data source.

SyntheticFeed generates a self-contained demo session that needs no
network, no API key, and no external data files.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Optional, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from gate_scorer import MarketSnapshot, GateConfig
from rnd_extractor import ChainSnapshot, ChainQuote, RNDConfig, _bs_call_fwd
from decision_engine import decide, EngineConfig
from journal import Journal, SettlementResult

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# DataFeed protocol                                                            #
# --------------------------------------------------------------------------- #
class DataFeed(Protocol):
    def snapshot(self) -> tuple[MarketSnapshot, ChainSnapshot]:
        """Return current (market, chain) pair. Blocking until data is ready."""
        ...

    def has_next(self) -> bool:
        """False when the session ends (used by the tick loop)."""
        ...


# --------------------------------------------------------------------------- #
# Synthetic feed (demo / backtest stub)                                       #
# --------------------------------------------------------------------------- #
@dataclass
class _SyntheticTick:
    now: dt.datetime
    spot: float
    net_gex: float
    gex_pct_rank: float
    gamma_flip: float
    call_wall: float
    put_wall: float
    vix9d: float
    vix: float
    vix3m: float
    vvix: float
    adx: float
    rsi: float
    has_catalyst: bool


class SyntheticFeed:
    """
    Replays a fixed sequence of synthetic ticks for a demo session.
    Tick 0-2: ranging clean day (should GO).
    Tick 3: simulates late-session (post-lockout, should NO_GO on timing).
    Tick 4: simulated catalyst (NO_GO).
    """

    def __init__(self, date: Optional[dt.date] = None):
        d = date or dt.date(2026, 6, 26)
        self._ticks: list[_SyntheticTick] = [
            _SyntheticTick(
                now=dt.datetime(d.year, d.month, d.day, 10, 45, tzinfo=ET),
                spot=602.50, net_gex=4.2e9, gex_pct_rank=0.88,
                gamma_flip=596.0, call_wall=603.0, put_wall=598.0,
                vix9d=12.1, vix=13.0, vix3m=15.2, vvix=92.0,
                adx=12.5, rsi=52.0, has_catalyst=False,
            ),
            _SyntheticTick(
                now=dt.datetime(d.year, d.month, d.day, 11, 30, tzinfo=ET),
                spot=603.10, net_gex=4.5e9, gex_pct_rank=0.91,
                gamma_flip=596.0, call_wall=603.0, put_wall=598.0,
                vix9d=11.8, vix=12.8, vix3m=15.0, vvix=90.0,
                adx=11.0, rsi=54.0, has_catalyst=False,
            ),
            _SyntheticTick(
                now=dt.datetime(d.year, d.month, d.day, 13, 0, tzinfo=ET),
                spot=602.80, net_gex=4.3e9, gex_pct_rank=0.89,
                gamma_flip=596.0, call_wall=603.0, put_wall=598.0,
                vix9d=12.0, vix=12.9, vix3m=15.1, vvix=91.0,
                adx=12.0, rsi=51.0, has_catalyst=False,
            ),
            _SyntheticTick(
                now=dt.datetime(d.year, d.month, d.day, 15, 35, tzinfo=ET),
                spot=603.0, net_gex=4.0e9, gex_pct_rank=0.85,
                gamma_flip=596.0, call_wall=603.0, put_wall=598.0,
                vix9d=12.2, vix=13.1, vix3m=15.3, vvix=93.0,
                adx=13.0, rsi=50.0, has_catalyst=False,
            ),
            _SyntheticTick(
                now=dt.datetime(d.year, d.month, d.day, 9, 5, tzinfo=ET),
                spot=588.0, net_gex=-1.1e9, gex_pct_rank=0.40,
                gamma_flip=593.0, call_wall=596.0, put_wall=585.0,
                vix9d=19.5, vix=18.0, vix3m=17.0, vvix=120.0,
                adx=28.0, rsi=38.0, has_catalyst=True,
            ),
        ]
        self._idx = 0

    def has_next(self) -> bool:
        return self._idx < len(self._ticks)

    def snapshot(self) -> tuple[MarketSnapshot, ChainSnapshot]:
        t = self._ticks[self._idx]
        self._idx += 1

        ms = MarketSnapshot(
            spot=t.spot,
            net_gex=t.net_gex,
            gamma_flip=t.gamma_flip,
            call_wall=t.call_wall,
            put_wall=t.put_wall,
            gex_pct_rank=t.gex_pct_rank,
            vix9d=t.vix9d, vix=t.vix, vix3m=t.vix3m,
            vvix=t.vvix, vvix_baseline=95.0,
            straddle_breakeven=3.8, expected_range=3.0,
            adx=t.adx, rsi=t.rsi,
            bb_width=1.5, bb_width_baseline=2.1,
            vwap=t.spot - 0.2, vwap_reversion_count=4,
            tick_abs_mean=520.0, cvd_slope=0.03,
            now=t.now, has_catalyst=t.has_catalyst,
            catalyst_label="CPI 08:30" if t.has_catalyst else "",
        )
        chain = _synthetic_chain(t.spot, r=0.05, t_years=4.0 / (24 * 365))
        return ms, chain


def _synthetic_chain(spot: float, r: float, t_years: float) -> ChainSnapshot:
    """Build a realistic-ish synthetic 0DTE chain around spot."""
    F = spot * np.exp(r * t_years)
    DF = np.exp(-r * t_years)
    quotes = []
    for K in np.arange(spot - 15, spot + 16, 1.0):
        k = np.log(K / F)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F, K, s) * DF
        pm = max(cm - DF * (F - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        quotes.append(ChainQuote(
            float(K),
            max(cm - h, 0.0), cm + h,
            max(pm - h, 0.0), pm + h,
        ))
    return ChainSnapshot(quotes, spot=spot, t_years=t_years, r=r)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
class Orchestrator:
    """
    Drives the Track-A tick loop: feed -> decide -> journal.
    Call run_session() during market hours.
    Call settle() after close with the final underlying price.
    """

    def __init__(
        self,
        feed: DataFeed,
        journal: Journal,
        cfg: Optional[EngineConfig] = None,
        tick_interval_s: float = 0.0,   # seconds between ticks (0 = as fast as feed)
        verbose: bool = True,
    ):
        self.feed = feed
        self.journal = journal
        self.cfg = cfg or EngineConfig()
        self.tick_interval_s = tick_interval_s
        self.verbose = verbose
        self._session_date: Optional[str] = None

    def run_session(self) -> int:
        """Run until feed.has_next() returns False. Returns number of ticks processed."""
        n = 0
        while self.feed.has_next():
            market, chain = self.feed.snapshot()
            self._session_date = market.now.astimezone(ET).date().isoformat()

            td = decide(market, chain, self.cfg)
            row = td.as_row()
            self.journal.log(row)
            n += 1

            if self.verbose:
                _print_tick(td)

            if self.tick_interval_s > 0:
                time.sleep(self.tick_interval_s)

        return n

    def settle(self, close_price: float,
               date: Optional[str] = None) -> Optional[SettlementResult]:
        """Post-close settlement pass. Uses session_date from the most recent tick."""
        d = date or self._session_date
        if d is None:
            return None
        result = self.journal.settle_session(d, close_price)
        if self.verbose:
            print(f"\n[settle] {d}  close={close_price:.2f}  "
                  f"trades={result.n_trades}  no-trades={result.n_no_trades}  "
                  f"trade_pnl={result.trade_pnl:+.3f}  "
                  f"blocked_pnl={result.blocked_pnl:+.3f}  "
                  f"gate_helped={result.gate_helped}")
        return result


def _print_tick(td) -> None:
    cand = td.candidate
    tag = f"{td.decision:<8}"
    gate = f"gate={td.gate_score:.0f}" if td.gate_pass else f"gate=FAIL({','.join(g.split(':')[0] for g in td.gate_failed)})"
    sel = f"{cand.family} cr={cand.credit:.2f} ml={cand.max_loss:.2f} ev={cand.ev:.3f}" if cand else "no_candidate"
    print(f"  {td.ts[11:16]}  spot={td.spot:.2f}  {tag}  {gate}  {sel}")


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os, tempfile
    db = os.path.join(tempfile.mkdtemp(), "orch_demo.db")

    feed = SyntheticFeed()
    jnl = Journal(db)
    orch = Orchestrator(feed, jnl, verbose=True)

    print("=== Track-A tick loop demo ===")
    n = orch.run_session()
    print(f"\nProcessed {n} ticks.")

    orch.settle(close_price=602.50)

    eff = jnl.gate_effectiveness(lookback_days=1)
    print(f"\nGate effectiveness: {eff}")
