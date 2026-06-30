"""
walk_forward.py
===============
Out-of-sample validation for the 0DTE pipeline via walk-forward analysis.

Two modes
---------
  expanding  — each fold's warm-up starts at tick 0 and grows; test window
               slides forward with a fixed size.
  rolling    — both warm-up and test windows are fixed size and slide forward.

Per fold
--------
  1. Create a fresh feed via feed_factory().
  2. Run the full UnifiedOrchestrator over warm-up ticks with no journal
     attached — this populates the MTF matrix and regime classifier without
     polluting the measurement window.
  3. Attach a fresh in-memory Journal; run the test ticks.
  4. Settle test-period session dates only.
  5. Compute a TearSheet from the test-period rows.

The per-fold TearSheets and the aggregate summary are returned in a
WalkForwardResult, which has a .print() method for human-readable output.

Usage
-----
    from walk_forward import run_walk_forward, WalkForwardConfig

    result = run_walk_forward(
        feed_factory=lambda: SyntheticUnifiedFeed(days=30, chain=chain),
        timestamps=ticks,
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=3),
    )
    result.print()

feed_factory must return a fresh DataFeed instance each call so that each fold
gets its own independent bar stream.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from journal import Journal
from unified_loop import UnifiedOrchestrator
from decision_engine import EngineConfig
from regime_classifier import ClassifierConfig
from risk_manager import RiskConfig, RiskManager
from backtest import TearSheet, _daily_pnl, _sharpe, _max_drawdown, _pearson

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Config & result types                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    mode: str = "expanding"     # "expanding" | "rolling"
    n_folds: int = 5
    train_frac: float = 0.6    # warm-up as fraction of each fold's total window


@dataclass
class FoldResult:
    fold: int                   # 1-based
    mode: str
    warm_start: dt.datetime
    test_start: dt.datetime
    test_end: dt.datetime
    n_warm_ticks: int
    n_test_ticks: int
    tearsheet: TearSheet


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: list[FoldResult]

    # -- aggregate stats -----------------------------------------------------

    def _trade_counts(self) -> list[int]:
        return [f.tearsheet.trade_ticks for f in self.folds]

    def _win_rates(self) -> list[float]:
        return [f.tearsheet.win_rate for f in self.folds if f.tearsheet.win_rate is not None]

    def _pnls(self) -> list[float]:
        return [f.tearsheet.total_pnl for f in self.folds]

    def _sharpes(self) -> list[float]:
        return [f.tearsheet.sharpe for f in self.folds if f.tearsheet.sharpe is not None]

    def n_profitable(self) -> int:
        return sum(1 for f in self.folds if f.tearsheet.total_pnl > 0)

    # -- output --------------------------------------------------------------

    def print(self) -> None:
        cfg = self.config
        w = 72
        print("=" * w)
        print(f"  Walk-Forward Result  mode={cfg.mode}  "
              f"folds={cfg.n_folds}  train_frac={cfg.train_frac:.0%}")
        print("=" * w)
        hdr = (f"  {'Fold':>4}  {'Test window':<23}  "
               f"{'Warm':>6}  {'Test':>6}  "
               f"{'Trades':>7}  {'Win%':>5}  {'PnL':>8}  {'Sharpe':>7}")
        print(hdr)
        print("-" * w)

        for fr in self.folds:
            ts = fr.tearsheet
            win_s   = f"{ts.win_rate * 100:.0f}%" if ts.win_rate is not None else "n/a"
            sharpe_s = f"{ts.sharpe:+.2f}" if ts.sharpe is not None else "n/a"
            window = (f"{fr.test_start.strftime('%m-%d')} "
                      f"→ {fr.test_end.strftime('%m-%d %H:%M')}")
            print(f"  {fr.fold:>4}  {window:<23}  "
                  f"{fr.n_warm_ticks:>6,}  {fr.n_test_ticks:>6,}  "
                  f"{ts.trade_ticks:>7,}  {win_s:>5}  "
                  f"{ts.total_pnl:>+8.4f}  {sharpe_s:>7}")

        print("-" * w)

        # aggregate row
        pnls = self._pnls()
        wrs  = self._win_rates()
        shs  = self._sharpes()
        tcs  = self._trade_counts()

        def _fmt_stat(vals, fmt):
            if not vals:
                return "n/a", "n/a"
            mu = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mu:{fmt}}", f"{sd:{fmt}}"

        mu_tc, sd_tc   = _fmt_stat(tcs,  ">7.0f")
        mu_wr, sd_wr   = _fmt_stat([w * 100 for w in wrs], ">5.0f")
        mu_pnl, sd_pnl = _fmt_stat(pnls, ">+8.4f")
        mu_sh, sd_sh   = _fmt_stat(shs,  ">+7.2f")

        print(f"  {'Mean':>4}  {'':23}  {'':>6}  {'':>6}  "
              f"{mu_tc}  {mu_wr}%  {mu_pnl}  {mu_sh}")
        print(f"  {'Std':>4}  {'':23}  {'':>6}  {'':>6}  "
              f"{sd_tc}  {sd_wr}%  {sd_pnl}  {sd_sh}")
        print("=" * w)
        print(f"  Consistency: {self.n_profitable()}/{len(self.folds)} folds profitable")
        print("=" * w)

    def to_dict(self) -> dict:
        pnls = self._pnls()
        shs  = self._sharpes()
        wrs  = self._win_rates()
        return {
            "mode": self.config.mode,
            "n_folds": self.config.n_folds,
            "train_frac": self.config.train_frac,
            "n_profitable": self.n_profitable(),
            "mean_pnl": round(sum(pnls) / len(pnls), 6) if pnls else None,
            "mean_sharpe": round(sum(shs) / len(shs), 3) if shs else None,
            "mean_win_rate": round(sum(wrs) / len(wrs), 4) if wrs else None,
            "folds": [
                {
                    "fold": f.fold,
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "total_pnl": f.tearsheet.total_pnl,
                    "sharpe": f.tearsheet.sharpe,
                    "win_rate": f.tearsheet.win_rate,
                    "trades": f.tearsheet.trade_ticks,
                }
                for f in self.folds
            ],
        }


# --------------------------------------------------------------------------- #
# Fold index builder                                                             #
# --------------------------------------------------------------------------- #
def _make_fold_indices(
    n: int, cfg: WalkForwardConfig
) -> list[tuple[int, int, int]]:
    """
    Returns list of (warm_start, test_start, test_end) index triples.
    Indices are into the `timestamps` list passed to run_walk_forward.
    """
    if cfg.mode == "expanding":
        # Initial warm-up = train_frac * total; remainder split into n_folds
        initial_warm = int(n * cfg.train_frac)
        remaining = n - initial_warm
        fold_size = max(1, remaining // cfg.n_folds)
        triples = []
        for i in range(cfg.n_folds):
            test_start = initial_warm + i * fold_size
            test_end   = (test_start + fold_size) if i < cfg.n_folds - 1 else n
            if test_start >= n:
                break
            triples.append((0, test_start, min(test_end, n)))
        return triples
    else:  # rolling
        fold_size = max(1, n // cfg.n_folds)
        warm_size = max(1, int(fold_size * cfg.train_frac / max(1e-9, 1.0 - cfg.train_frac)))
        triples = []
        for i in range(cfg.n_folds):
            test_start = i * fold_size
            test_end   = (test_start + fold_size) if i < cfg.n_folds - 1 else n
            warm_start = max(0, test_start - warm_size)
            if test_start >= n:
                break
            triples.append((warm_start, test_start, min(test_end, n)))
        return triples


# --------------------------------------------------------------------------- #
# Single-fold runner                                                             #
# --------------------------------------------------------------------------- #
def _run_fold(
    feed,
    warm_ticks: list[dt.datetime],
    test_ticks: list[dt.datetime],
    engine_cfg: Optional[EngineConfig],
    classifier_cfg: Optional[ClassifierConfig],
    risk_cfg: Optional[RiskConfig],
) -> TearSheet:
    jrn = Journal(":memory:")

    orch = UnifiedOrchestrator(
        feed=feed,
        journal=None,   # no logging during warm-up
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        risk_manager=RiskManager(risk_cfg) if risk_cfg else None,
    )

    # Warm-up: advance feed + classifier state without logging
    for t in warm_ticks:
        try:
            orch.tick(t)
        except Exception:
            pass

    # Test: attach journal; log every tick
    orch.journal = jrn
    for t in test_ticks:
        try:
            orch.tick(t)
        except Exception:
            pass

    # Settle test-period session dates only
    test_dates: set[str] = {t.astimezone(ET).date().isoformat() for t in test_ticks}
    for d in sorted(test_dates):
        price = feed.settlement_price(d)
        if price is not None:
            jrn.settle_session(d, price)

    # Build TearSheet from the test-window journal rows
    settled = jrn.fetch(settled_only=True)
    all_rows = jrn.fetch()

    total = len(all_rows)
    traded = [r for r in settled if r["was_traded"] == 1]
    no_trade = total - sum(1 for r in all_rows if r["was_traded"] == 1)

    with_cand = [r for r in all_rows if r["candidate_present"] == 1]
    gate_pass_rate = (
        round(sum(1 for r in with_cand if r["gate_pass"] == 1) / len(with_cand), 4)
        if with_cand else 0.0
    )

    pnls = [r["realized_pnl"] for r in traded if r["realized_pnl"] is not None]
    total_pnl   = round(sum(pnls), 6) if pnls else 0.0
    mean_pnl    = round(sum(pnls) / len(pnls), 6) if pnls else None
    win_rate    = round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else None

    daily   = _daily_pnl(settled)
    sharpe  = _sharpe(daily)
    mdd     = _max_drawdown(daily)

    ev_pairs = [(r["ev"], r["realized_pnl"])
                for r in traded if r["ev"] is not None and r["realized_pnl"] is not None]
    mean_ev = round(sum(e for e, _ in ev_pairs) / len(ev_pairs), 6) if ev_pairs else None
    ev_acc  = _pearson([e for e, _ in ev_pairs], [p for _, p in ev_pairs]) if ev_pairs else None

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
    )


# --------------------------------------------------------------------------- #
# Main entry point                                                               #
# --------------------------------------------------------------------------- #
def run_walk_forward(
    feed_factory: Callable,
    timestamps: list[dt.datetime],
    wf_cfg: Optional[WalkForwardConfig] = None,
    engine_cfg: Optional[EngineConfig] = None,
    classifier_cfg: Optional[ClassifierConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
) -> WalkForwardResult:
    """
    Run walk-forward validation over `timestamps`.

    feed_factory  — callable() that returns a fresh DataFeed for each fold.
    timestamps    — the complete tick sequence (training + test combined).
    wf_cfg        — walk-forward configuration (mode, n_folds, train_frac).
    """
    cfg = wf_cfg or WalkForwardConfig()
    n   = len(timestamps)
    triples = _make_fold_indices(n, cfg)

    fold_results: list[FoldResult] = []
    for fold_idx, (warm_start, test_start, test_end) in enumerate(triples, start=1):
        warm_ticks = timestamps[warm_start:test_start]
        test_ticks = timestamps[test_start:test_end]

        print(f"  Fold {fold_idx}/{len(triples)}: "
              f"warm={len(warm_ticks):,} ticks  "
              f"test={len(test_ticks):,} ticks "
              f"({test_ticks[0].strftime('%m-%d')} → {test_ticks[-1].strftime('%m-%d %H:%M') if test_ticks else '?'})"
              , flush=True)

        feed = feed_factory()
        ts   = _run_fold(feed, warm_ticks, test_ticks, engine_cfg, classifier_cfg, risk_cfg)

        fold_results.append(FoldResult(
            fold=fold_idx,
            mode=cfg.mode,
            warm_start=timestamps[warm_start],
            test_start=test_ticks[0] if test_ticks else timestamps[test_start],
            test_end=test_ticks[-1] if test_ticks else timestamps[test_end - 1],
            n_warm_ticks=len(warm_ticks),
            n_test_ticks=len(test_ticks),
            tearsheet=ts,
        ))

    return WalkForwardResult(config=cfg, folds=fold_results)


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np
    from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
    from unified_loop import SyntheticUnifiedFeed

    DAYS = 30
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

    start = dt.datetime(2026, 6, 1, 9, 30, tzinfo=ET)
    ticks = [start + dt.timedelta(minutes=i) for i in range(DAYS * 390)]

    def make_feed():
        return SyntheticUnifiedFeed(days=DAYS, chain=chain, settle=spot0)

    for mode in ("expanding", "rolling"):
        print(f"\n{'='*72}")
        print(f"  Walk-Forward demo — mode={mode}, {DAYS} days, 3 folds")
        print(f"{'='*72}\n")
        result = run_walk_forward(
            feed_factory=make_feed,
            timestamps=ticks,
            wf_cfg=WalkForwardConfig(mode=mode, n_folds=3, train_frac=0.6),
        )
        print()
        result.print()
