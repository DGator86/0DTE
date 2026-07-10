"""
backtest.py
===========
Replay the full 0DTE pipeline over a historical tick sequence and produce a
tearsheet: cumulative P&L, Sharpe, max drawdown, gate hit rate, EV accuracy.

Design
------
  run_backtest(feed, timestamps, engine_cfg, classifier_cfg) -> TearSheet

  * feed must satisfy the DataFeed protocol from unified_loop.py:
        snapshot(now)           -> Optional[TickSnapshot]
        settlement_price(date)  -> Optional[float]
  * An in-memory Journal is created; every tick is logged (trade AND no-trade).
  * After replay, each unique session date is settled via the feed's
    settlement_price(); rows without a price are left unsettled.
  * TearSheet queries the journal for settled rows and computes metrics.

The synthetic demo at __main__ runs 20 days × 390 minutes with the same chain
fixture used in unified_loop.py.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from journal import Journal
from unified_loop import UnifiedOrchestrator, TickSnapshot
from decision_engine import EngineConfig
from regime_classifier import ClassifierConfig
from risk_manager import RiskConfig, RiskManager

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Metrics helpers                                                               #
# --------------------------------------------------------------------------- #
def _daily_pnl(rows: list[dict]) -> dict[str, float]:
    by_date: dict[str, float] = {}
    for r in rows:
        if r["was_traded"] == 1 and r["realized_pnl"] is not None:
            d = r["session_date"]
            by_date[d] = by_date.get(d, 0.0) + r["realized_pnl"]
    return by_date


def _sharpe(daily: dict[str, float]) -> Optional[float]:
    vals = list(daily.values())
    n = len(vals)
    if n < 2:
        return None
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    sd = var ** 0.5
    return round((mu / sd) * math.sqrt(252), 3) if sd > 0 else None


def _max_drawdown(daily: dict[str, float]) -> float:
    peak = cum = 0.0
    dd = 0.0
    for d in sorted(daily):
        cum += daily[d]
        if cum > peak:
            peak = cum
        dd = max(dd, peak - cum)
    return round(dd, 6)


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov  = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    varx = sum((a - mx) ** 2 for a in xs)
    vary = sum((b - my) ** 2 for b in ys)
    return round(cov / math.sqrt(varx * vary), 3) if varx > 0 and vary > 0 else None


# --------------------------------------------------------------------------- #
# TearSheet                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class TearSheet:
    # tick-level counts
    total_ticks: int
    trade_ticks: int         # was_traded == 1
    no_trade_ticks: int
    gate_pass_rate: float    # gate_pass / rows_with_candidate
    win_rate: Optional[float]           # trades with realized_pnl > 0

    # P&L (per-contract, not scaled by size_mult — the journal stores raw credit)
    total_pnl: float
    mean_pnl_per_trade: Optional[float]
    daily_pnl: dict[str, float]         # session_date -> sum of realized_pnl

    # risk metrics
    sharpe: Optional[float]             # annualised (×√252), using daily P&L
    max_drawdown: float                 # peak-to-trough on cumulative P&L

    # model quality
    mean_ev: Optional[float]            # average EV across traded rows
    ev_accuracy: Optional[float]        # Pearson(ev, realized_pnl) on settled trades

    # gate quality
    gate_effectiveness: dict = field(default_factory=dict)
    component_correlations: dict = field(default_factory=dict)

    # probability calibration + regime coverage (used by the optimizer's
    # composite metric; None/{} when the window is too thin to judge)
    brier_skill: Optional[float] = None
    regime_counts: dict = field(default_factory=dict)

    def print(self) -> None:
        w = 56
        pct = lambda x: f"{x * 100:.1f}%" if x is not None else "n/a"
        flt = lambda x, d=3: f"{x:.{d}f}" if x is not None else "n/a"

        print("=" * w)
        print("  Backtest Tearsheet")
        print("=" * w)
        print(f"  Total ticks:       {self.total_ticks:>8,}")
        print(f"  Trades taken:      {self.trade_ticks:>8,}")
        print(f"  No-trade ticks:    {self.no_trade_ticks:>8,}")
        print(f"  Gate pass rate:    {pct(self.gate_pass_rate):>8}")
        print("-" * w)
        print(f"  Win rate:          {pct(self.win_rate):>8}")
        print(f"  Total P&L:         {flt(self.total_pnl, 4):>8}")
        print(f"  Mean P&L / trade:  {flt(self.mean_pnl_per_trade, 4):>8}")
        print(f"  Sharpe (ann.):     {flt(self.sharpe, 3):>8}")
        print(f"  Max drawdown:      {flt(self.max_drawdown, 4):>8}")
        print("-" * w)
        print(f"  Mean EV:           {flt(self.mean_ev, 4):>8}")
        print(f"  EV accuracy (r):   {flt(self.ev_accuracy, 3):>8}")
        print("-" * w)
        eff = self.gate_effectiveness
        t = eff.get("trades_taken", {})
        b = eff.get("blocked_by_gate", {})
        print(f"  Gate — taken:      n={t.get('n', 0):>4}  mean_pnl={flt(t.get('mean'), 4)}")
        print(f"  Gate — blocked:    n={b.get('n', 0):>4}  mean_pnl={flt(b.get('mean'), 4)}")
        print(f"  Verdict: {eff.get('verdict', 'n/a')}")
        print("-" * w)

        if self.daily_pnl:
            print("  Daily P&L:")
            cum = 0.0
            for date in sorted(self.daily_pnl):
                pnl = self.daily_pnl[date]
                cum += pnl
                bar_len = min(20, int(abs(pnl) * 40))
                sign = "+" if pnl >= 0 else "-"
                bar = ("█" * bar_len) if pnl >= 0 else ("░" * bar_len)
                print(f"    {date}  {sign}{abs(pnl):.4f}  cum={cum:.4f}  {bar}")

        if isinstance(self.component_correlations, dict) and "n" not in list(self.component_correlations.keys())[:1]:
            print("-" * w)
            print("  Component correlations (vs realized P&L):")
            for k, v in sorted(self.component_correlations.items(),
                                key=lambda kv: -abs(kv[1] or 0)):
                if k == "n":
                    continue
                bar = "█" * min(20, int(abs(v or 0) * 20))
                sign = "+" if (v or 0) >= 0 else ""
                print(f"    {k:<30}  r={sign}{flt(v, 3)}  {bar}")

        print("=" * w)

    def to_dict(self) -> dict:
        return {
            "total_ticks": self.total_ticks,
            "trade_ticks": self.trade_ticks,
            "no_trade_ticks": self.no_trade_ticks,
            "gate_pass_rate": self.gate_pass_rate,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "mean_pnl_per_trade": self.mean_pnl_per_trade,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "mean_ev": self.mean_ev,
            "ev_accuracy": self.ev_accuracy,
            "gate_verdict": self.gate_effectiveness.get("verdict"),
            "daily_pnl": self.daily_pnl,
        }


# --------------------------------------------------------------------------- #
# Core runner                                                                   #
# --------------------------------------------------------------------------- #
def run_backtest(
    feed,
    timestamps: list[dt.datetime],
    engine_cfg: Optional[EngineConfig] = None,
    classifier_cfg: Optional[ClassifierConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
    journal: Optional[Journal] = None,
) -> TearSheet:
    """
    Replay `timestamps` through the full pipeline and return a TearSheet.

    feed      — satisfies DataFeed protocol (snapshot + settlement_price)
    risk_cfg  — optional intraday risk guards applied during replay
    journal   — pass an external Journal to inspect it afterwards; defaults to :memory:
    """
    jrn = journal or Journal(":memory:")
    orch = UnifiedOrchestrator(
        feed=feed, journal=jrn,
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        risk_manager=RiskManager(risk_cfg) if risk_cfg else None,
    )

    # -- replay --
    results = orch.run_replay(timestamps)

    # -- settle every session date we saw --
    seen_dates: set[str] = set()
    for r in results:
        seen_dates.add(r.ts.astimezone(ET).date().isoformat())
    for d in sorted(seen_dates):
        price = feed.settlement_price(d)
        if price is not None:
            orch.settle(d)

    # -- pull settled rows --
    settled_rows = jrn.fetch(settled_only=True)
    all_rows = jrn.fetch()

    # -- tick counts --
    total = len(all_rows)
    traded = [r for r in settled_rows if r["was_traded"] == 1]
    no_trade = total - sum(1 for r in all_rows if r["was_traded"] == 1)

    # -- gate pass rate (over rows that had a candidate) --
    with_cand = [r for r in all_rows if r["candidate_present"] == 1]
    gate_pass_rate = (
        round(sum(1 for r in with_cand if r["gate_pass"] == 1) / len(with_cand), 4)
        if with_cand else 0.0
    )

    # -- P&L --
    pnls = [r["realized_pnl"] for r in traded if r["realized_pnl"] is not None]
    total_pnl = round(sum(pnls), 6) if pnls else 0.0
    mean_pnl = round(sum(pnls) / len(pnls), 6) if pnls else None
    win_rate = round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else None

    # -- daily P&L + risk --
    daily = _daily_pnl(settled_rows)
    sharpe = _sharpe(daily)
    mdd = _max_drawdown(daily)

    # -- EV accuracy --
    ev_pairs = [(r["ev"], r["realized_pnl"])
                for r in traded
                if r["ev"] is not None and r["realized_pnl"] is not None]
    mean_ev = round(sum(e for e, _ in ev_pairs) / len(ev_pairs), 6) if ev_pairs else None
    ev_acc = _pearson([e for e, _ in ev_pairs], [p for _, p in ev_pairs]) if ev_pairs else None

    return TearSheet(
        total_ticks=total,
        trade_ticks=len(traded),
        no_trade_ticks=no_trade,
        gate_pass_rate=gate_pass_rate,
        win_rate=win_rate,
        total_pnl=total_pnl,
        mean_pnl_per_trade=mean_pnl,
        daily_pnl=daily,
        sharpe=sharpe,
        max_drawdown=mdd,
        mean_ev=mean_ev,
        ev_accuracy=ev_acc,
        gate_effectiveness=jrn.gate_effectiveness(),
        component_correlations=jrn.component_correlations(),
        brier_skill=jrn.prob_calibration().get("brier_skill"),
        regime_counts=jrn.regime_diversity().get("regimes", {}),
    )


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    import numpy as np
    from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
    from unified_loop import SyntheticUnifiedFeed

    DAYS = 20
    spot0 = 600.0
    T0, r0 = 4.0 / (24 * 365), 0.05
    DF0 = math.exp(-r0 * T0)
    F0  = spot0 * math.exp(r0 * T0)

    qs = []
    for K in np.arange(spot0 - 15, spot0 + 16, 1.0):
        k = math.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h  = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    chain = ChainSnapshot(qs, spot=spot0, t_years=T0, r=r0)

    feed = SyntheticUnifiedFeed(days=DAYS, chain=chain, settle=600.0)
    start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=ET)
    ticks = [start + dt.timedelta(minutes=i) for i in range(DAYS * 390)]

    print(f"Running backtest: {DAYS} days × 390 min = {len(ticks):,} ticks …\n")
    ts = run_backtest(feed, ticks)
    ts.print()
