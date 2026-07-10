"""
walk_forward.py
===============
Out-of-sample validation for the 0DTE pipeline via walk-forward analysis.

Fold unit (Prediction Engine V2, PR 1)
--------------------------------------
Folds are built from COMPLETE trading sessions by default
(``fold_unit="session"``): no session is ever split between warm-up and test,
no session appears on both sides, and a configurable EMBARGO of whole
sessions (default 1) separates the end of warm-up from the start of test.
The legacy tick-index folding remains available via ``fold_unit="tick"`` for
A/B comparison only — it can put the morning of a session in warm-up and the
afternoon of the SAME session in test, which leaks state and shared
settlement labels across the boundary.

Two modes
---------
  expanding  — each fold's warm-up starts at the first session and grows;
               the test window slides forward.
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

Failure accounting
------------------
Exceptions during warm-up or test ticks are no longer silently swallowed:
every failure becomes a structured TickFailure record on the fold, and a fold
whose TEST failure fraction exceeds ``max_failed_tick_frac`` (default 1%) is
marked invalid and excluded from aggregate statistics (it is still listed in
the output so nothing disappears).

Session-level statistics
------------------------
Because intraday ticks are heavily correlated, the effective sample size is
the number of independent SESSIONS. The result reports the independent test
session count prominently and computes a session-bootstrap confidence
interval for per-session P&L (1,000 replications, 95% by default).

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
import hashlib
import statistics
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from journal import Journal
from unified_loop import UnifiedOrchestrator
from decision_engine import EngineConfig
from regime_classifier import ClassifierConfig
from risk_manager import RiskConfig, RiskManager
from backtest import TearSheet, _daily_pnl, _sharpe, _max_drawdown, _pearson
from validation.session_folds import make_session_folds, session_date
from validation.bootstrap import session_bootstrap

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Config & result types                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    mode: str = "expanding"     # "expanding" | "rolling"
    n_folds: int = 5
    train_frac: float = 0.6    # warm-up as fraction of each fold's total window
    fold_unit: str = "session"  # "session" (default) | "tick" (legacy, leaky)
    embargo_sessions: int = 1   # whole sessions skipped between warm-up and test
    max_failed_tick_frac: float = 0.01   # invalidate fold above this TEST failure rate
    # Pin the first test session exactly (expanding + session unit only) —
    # used by the session-based holdout evaluation.
    initial_warm_sessions: Optional[int] = None


@dataclass(frozen=True)
class TickFailure:
    """One tick that raised during a fold run. Nothing is silently dropped."""
    ts: str
    session_date: str
    stage: str                  # "warm" | "test"
    exception_type: str
    message: str
    traceback_hash: str

    def to_dict(self) -> dict:
        return {"ts": self.ts, "session_date": self.session_date,
                "stage": self.stage, "exception_type": self.exception_type,
                "message": self.message, "traceback_hash": self.traceback_hash}


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
    # session provenance
    warm_sessions: tuple[str, ...] = ()
    embargoed_sessions: tuple[str, ...] = ()
    test_sessions: tuple[str, ...] = ()
    # failure accounting
    failures: list[TickFailure] = field(default_factory=list)
    n_failed_warm: int = 0
    n_failed_test: int = 0
    valid: bool = True
    invalid_reason: Optional[str] = None

    @property
    def n_test_sessions(self) -> int:
        return len(self.test_sessions)


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: list[FoldResult]

    # -- aggregate stats -----------------------------------------------------
    # Aggregates use only VALID folds: a fold that dropped >1% of its test
    # ticks on exceptions is survivorship-biased by construction and must not
    # flatter the summary numbers. Invalid folds remain listed in the output.

    @property
    def valid_folds(self) -> list[FoldResult]:
        return [f for f in self.folds if f.valid]

    def _trade_counts(self) -> list[int]:
        return [f.tearsheet.trade_ticks for f in self.valid_folds]

    def _win_rates(self) -> list[float]:
        return [f.tearsheet.win_rate for f in self.valid_folds
                if f.tearsheet.win_rate is not None]

    def _pnls(self) -> list[float]:
        return [f.tearsheet.total_pnl for f in self.valid_folds]

    def _sharpes(self) -> list[float]:
        return [f.tearsheet.sharpe for f in self.valid_folds
                if f.tearsheet.sharpe is not None]

    def n_profitable(self) -> int:
        return sum(1 for f in self.valid_folds if f.tearsheet.total_pnl > 0)

    # -- session-level statistics ---------------------------------------------

    def n_test_sessions(self) -> int:
        """Independent test sessions across valid folds — the honest sample
        size of this walk-forward, NOT the tick count."""
        return len({d for f in self.valid_folds for d in f.test_sessions})

    def session_pnls(self) -> dict[str, float]:
        """Per-session realized P&L over valid test windows. Sessions with no
        trades count as 0.0 — choosing not to trade is still one independent
        observation of the strategy."""
        out: dict[str, float] = {}
        for f in self.valid_folds:
            for d in f.test_sessions:
                out.setdefault(d, 0.0)
            for d, pnl in (f.tearsheet.daily_pnl or {}).items():
                out[d] = out.get(d, 0.0) + pnl
        return out

    def session_pnl_bootstrap(self, n_boot: int = 1000, ci: float = 0.95,
                              seed: int = 20260710) -> dict:
        """Session-bootstrap CI for mean per-session P&L (complete sessions
        resampled with replacement)."""
        return session_bootstrap(self.session_pnls(), n_boot=n_boot, ci=ci,
                                 seed=seed)

    def failure_summary(self) -> dict:
        by_exc: dict[str, int] = {}
        n_warm = n_test = 0
        for f in self.folds:
            n_warm += f.n_failed_warm
            n_test += f.n_failed_test
            for fl in f.failures:
                by_exc[fl.exception_type] = by_exc.get(fl.exception_type, 0) + 1
        return {"n_failed_warm": n_warm, "n_failed_test": n_test,
                "by_exception": by_exc,
                "n_invalid_folds": sum(1 for f in self.folds if not f.valid)}

    # -- output --------------------------------------------------------------

    def print(self) -> None:
        cfg = self.config
        w = 78
        print("=" * w)
        print(f"  Walk-Forward Result  mode={cfg.mode}  unit={cfg.fold_unit}  "
              f"folds={cfg.n_folds}  train_frac={cfg.train_frac:.0%}  "
              f"embargo={cfg.embargo_sessions}")
        print("=" * w)
        hdr = (f"  {'Fold':>4}  {'Test window':<23}  "
               f"{'Warm':>6}  {'Test':>6}  {'Sess':>4}  "
               f"{'Trades':>7}  {'Win%':>5}  {'PnL':>8}  {'Sharpe':>7}")
        print(hdr)
        print("-" * w)

        for fr in self.folds:
            ts = fr.tearsheet
            win_s   = f"{ts.win_rate * 100:.0f}%" if ts.win_rate is not None else "n/a"
            sharpe_s = f"{ts.sharpe:+.2f}" if ts.sharpe is not None else "n/a"
            window = (f"{fr.test_start.strftime('%m-%d')} "
                      f"→ {fr.test_end.strftime('%m-%d %H:%M')}")
            invalid = "" if fr.valid else "  ← INVALID"
            print(f"  {fr.fold:>4}  {window:<23}  "
                  f"{fr.n_warm_ticks:>6,}  {fr.n_test_ticks:>6,}  "
                  f"{fr.n_test_sessions:>4}  "
                  f"{ts.trade_ticks:>7,}  {win_s:>5}  "
                  f"{ts.total_pnl:>+8.4f}  {sharpe_s:>7}{invalid}")

        print("-" * w)

        # aggregate row (valid folds only)
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

        print(f"  {'Mean':>4}  {'':23}  {'':>6}  {'':>6}  {'':>4}  "
              f"{mu_tc}  {mu_wr}%  {mu_pnl}  {mu_sh}")
        print(f"  {'Std':>4}  {'':23}  {'':>6}  {'':>6}  {'':>4}  "
              f"{sd_tc}  {sd_wr}%  {sd_pnl}  {sd_sh}")
        print("=" * w)
        print(f"  Consistency: {self.n_profitable()}/{len(self.valid_folds)} "
              f"valid folds profitable"
              + (f"  ({len(self.folds) - len(self.valid_folds)} invalid)"
                 if len(self.valid_folds) < len(self.folds) else ""))
        boot = self.session_pnl_bootstrap()
        n_sess = self.n_test_sessions()
        print(f"  Independent test sessions: {n_sess} "
              f"(the honest sample size — not the tick count)")
        if boot.get("stat") is not None:
            print(f"  Mean session P&L: {boot['stat']:+.4f}  "
                  f"95% session-bootstrap CI [{boot['ci_low']:+.4f}, "
                  f"{boot['ci_high']:+.4f}]")
        fs = self.failure_summary()
        if fs["n_failed_warm"] or fs["n_failed_test"]:
            print(f"  Tick failures: warm={fs['n_failed_warm']} "
                  f"test={fs['n_failed_test']}  by_exception={fs['by_exception']}")
        print("=" * w)

    def to_dict(self) -> dict:
        pnls = self._pnls()
        shs  = self._sharpes()
        wrs  = self._win_rates()
        return {
            "mode": self.config.mode,
            "fold_unit": self.config.fold_unit,
            "embargo_sessions": self.config.embargo_sessions,
            "n_folds": self.config.n_folds,
            "train_frac": self.config.train_frac,
            "n_valid_folds": len(self.valid_folds),
            "n_profitable": self.n_profitable(),
            "n_test_sessions": self.n_test_sessions(),
            "mean_pnl": round(sum(pnls) / len(pnls), 6) if pnls else None,
            "mean_sharpe": round(sum(shs) / len(shs), 3) if shs else None,
            "mean_win_rate": round(sum(wrs) / len(wrs), 4) if wrs else None,
            "session_pnl_bootstrap": self.session_pnl_bootstrap(),
            "failures": self.failure_summary(),
            "folds": [
                {
                    "fold": f.fold,
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "total_pnl": f.tearsheet.total_pnl,
                    "sharpe": f.tearsheet.sharpe,
                    "win_rate": f.tearsheet.win_rate,
                    "trades": f.tearsheet.trade_ticks,
                    "n_test_sessions": f.n_test_sessions,
                    "n_failed_warm": f.n_failed_warm,
                    "n_failed_test": f.n_failed_test,
                    "valid": f.valid,
                    "invalid_reason": f.invalid_reason,
                }
                for f in self.folds
            ],
        }


# --------------------------------------------------------------------------- #
# Fold index builders                                                           #
# --------------------------------------------------------------------------- #
def _make_fold_indices(
    n: int, cfg: WalkForwardConfig
) -> list[tuple[int, int, int]]:
    """
    LEGACY tick-index folds: (warm_start, test_start, test_end) triples into
    the `timestamps` list. Kept only for fold_unit="tick" A/B comparison —
    these boundaries can split a session between warm-up and test.
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


@dataclass(frozen=True)
class _FoldWindow:
    """Internal: one fold's tick windows plus session provenance."""
    warm_start: int
    warm_end: int
    test_start: int
    test_end: int
    warm_sessions: tuple[str, ...]
    embargoed_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def _fold_windows(timestamps: list[dt.datetime],
                  cfg: WalkForwardConfig) -> list[_FoldWindow]:
    if cfg.fold_unit == "session":
        return [
            _FoldWindow(f.warm_start, f.warm_end, f.test_start, f.test_end,
                        f.warm_sessions, f.embargoed_sessions, f.test_sessions)
            for f in make_session_folds(
                timestamps, mode=cfg.mode, n_folds=cfg.n_folds,
                train_frac=cfg.train_frac,
                embargo_sessions=cfg.embargo_sessions,
                initial_warm_sessions=cfg.initial_warm_sessions)
        ]
    if cfg.fold_unit == "tick":
        def _dates(lo: int, hi: int) -> tuple[str, ...]:
            seen: dict[str, None] = {}
            for t in timestamps[lo:hi]:
                seen.setdefault(session_date(t))
            return tuple(seen)
        return [
            _FoldWindow(w, t, t, e, _dates(w, t), (), _dates(t, e))
            for w, t, e in _make_fold_indices(len(timestamps), cfg)
        ]
    raise ValueError(f"unknown fold_unit: {cfg.fold_unit!r}")


# --------------------------------------------------------------------------- #
# Single-fold runner                                                             #
# --------------------------------------------------------------------------- #
def _record_failure(t: dt.datetime, stage: str,
                    exc: BaseException) -> TickFailure:
    tb = traceback.format_exc()
    return TickFailure(
        ts=t.isoformat(),
        session_date=session_date(t),
        stage=stage,
        exception_type=type(exc).__name__,
        message=str(exc)[:500],
        traceback_hash=hashlib.sha256(tb.encode()).hexdigest()[:16],
    )


def _run_fold(
    feed,
    warm_ticks: list[dt.datetime],
    test_ticks: list[dt.datetime],
    engine_cfg: Optional[EngineConfig],
    classifier_cfg: Optional[ClassifierConfig],
    risk_cfg: Optional[RiskConfig],
) -> tuple[TearSheet, list[TickFailure]]:
    jrn = Journal(":memory:")
    failures: list[TickFailure] = []

    orch = UnifiedOrchestrator(
        feed=feed,
        journal=None,   # no logging during warm-up
        engine_cfg=engine_cfg,
        classifier_cfg=classifier_cfg,
        risk_manager=RiskManager(risk_cfg) if risk_cfg else None,
    )

    # Warm-up: advance feed + classifier state without logging.
    # Failures no longer vanish — each one becomes a TickFailure record.
    for t in warm_ticks:
        try:
            orch.tick(t)
        except Exception as exc:
            failures.append(_record_failure(t, "warm", exc))

    # Test: attach journal; log every tick
    orch.journal = jrn
    for t in test_ticks:
        try:
            orch.tick(t)
        except Exception as exc:
            failures.append(_record_failure(t, "test", exc))

    # Settle test-period session dates only
    test_dates: set[str] = {session_date(t) for t in test_ticks}
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

    ts = TearSheet(
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
    return ts, failures


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
    wf_cfg        — walk-forward configuration (mode, n_folds, train_frac,
                    fold_unit, embargo_sessions, max_failed_tick_frac).
    """
    cfg = wf_cfg or WalkForwardConfig()
    windows = _fold_windows(timestamps, cfg)

    fold_results: list[FoldResult] = []
    for fold_idx, win in enumerate(windows, start=1):
        warm_ticks = timestamps[win.warm_start:win.warm_end]
        test_ticks = timestamps[win.test_start:win.test_end]

        print(f"  Fold {fold_idx}/{len(windows)}: "
              f"warm={len(warm_ticks):,} ticks/{len(win.warm_sessions)} sess  "
              f"embargo={len(win.embargoed_sessions)} sess  "
              f"test={len(test_ticks):,} ticks/{len(win.test_sessions)} sess "
              f"({test_ticks[0].strftime('%m-%d')} → {test_ticks[-1].strftime('%m-%d %H:%M') if test_ticks else '?'})"
              , flush=True)

        feed = feed_factory()
        ts, failures = _run_fold(feed, warm_ticks, test_ticks,
                                 engine_cfg, classifier_cfg, risk_cfg)

        n_failed_warm = sum(1 for f in failures if f.stage == "warm")
        n_failed_test = sum(1 for f in failures if f.stage == "test")
        failed_frac = n_failed_test / max(1, len(test_ticks))
        valid = failed_frac <= cfg.max_failed_tick_frac
        invalid_reason = None
        if not valid:
            invalid_reason = (
                f"{n_failed_test}/{len(test_ticks)} test ticks failed "
                f"({failed_frac:.1%} > {cfg.max_failed_tick_frac:.1%} allowed)")
            print(f"    fold {fold_idx} INVALID: {invalid_reason}", flush=True)

        fold_results.append(FoldResult(
            fold=fold_idx,
            mode=cfg.mode,
            warm_start=timestamps[win.warm_start],
            test_start=test_ticks[0] if test_ticks else timestamps[win.test_start],
            test_end=test_ticks[-1] if test_ticks else timestamps[win.test_end - 1],
            n_warm_ticks=len(warm_ticks),
            n_test_ticks=len(test_ticks),
            tearsheet=ts,
            warm_sessions=win.warm_sessions,
            embargoed_sessions=win.embargoed_sessions,
            test_sessions=win.test_sessions,
            failures=failures,
            n_failed_warm=n_failed_warm,
            n_failed_test=n_failed_test,
            valid=valid,
            invalid_reason=invalid_reason,
        ))

    return WalkForwardResult(config=cfg, folds=fold_results)


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
def _print_calibration(jrn: Journal) -> None:
    cal = jrn.calibration()
    d = cal["directional"]["overall"]
    sess = cal["directional"].get("sessions") or {}
    pp = cal["prob_profit"]
    ev = cal["ev"]
    print("\n  Predictive power over the full window (settled ticks, no-trades incl.):")
    if d["n"]:
        print(f"    direction: n={d['n']}  hit={d['hit_rate']:.1%}  "
              f"signed move={d['avg_fwd_move_pct']:+.3f}%")
    if sess.get("n_sessions"):
        lo, hi = sess.get("hit_rate_ci95") or (None, None)
        ci = (f"  95% CI [{lo:.1%}, {hi:.1%}]"
              if lo is not None and hi is not None else "")
        print(f"    by session: {sess['n_sessions']} independent sessions  "
              f"mean session hit={sess['mean_session_hit_rate']:.1%}{ci}")
    if pp.get("n"):
        print(f"    prob_profit: n={pp['n']}  Brier={pp['brier']:.4f}  "
              f"skill={pp['brier_skill']}  base={pp['base_rate']:.1%}")
    if ev.get("n"):
        print(f"    EV: n={ev['n']}  bias={ev['mean_ev_error']:+.4f}  "
              f"MAE={ev['mae_ev_error']:.4f}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Walk-forward validation. Default: demo on the coupled "
                    "synthetic world. --recorded DIR: out-of-sample test on "
                    "REAL ticks recorded by shadow mode (chain_store).")
    ap.add_argument("--recorded", metavar="DIR",
                    help="directory of ticks_*.jsonl.gz recordings "
                         "(VPS default: /var/lib/zerodte/ticks)")
    ap.add_argument("--mode", default="expanding", choices=["expanding", "rolling"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=0.6)
    ap.add_argument("--fold-unit", dest="fold_unit", default="session",
                    choices=["session", "tick"],
                    help="'session' (default) builds folds from complete "
                         "sessions with an embargo; 'tick' is the legacy "
                         "leaky index split, for comparison only")
    ap.add_argument("--embargo", type=int, default=1,
                    help="whole sessions skipped between warm-up and test")
    args = ap.parse_args()

    wf_cfg = WalkForwardConfig(mode=args.mode, n_folds=args.folds,
                               train_frac=args.train_frac,
                               fold_unit=args.fold_unit,
                               embargo_sessions=args.embargo)

    if args.recorded:
        from backtest import run_backtest
        from chain_store import RecordedFeed

        probe = RecordedFeed(args.recorded)
        ticks = probe.timestamps()
        days = {t.date() for t in ticks}
        print(f"  {len(ticks):,} recorded ticks across {len(days)} sessions "
              f"in {args.recorded!r}")
        if len(ticks) < 100 or len(days) < 3:
            print("  Not enough recorded history yet for a meaningful walk-forward "
                  "(want >= 3 sessions). Let shadow mode record longer.")
            return

        result = run_walk_forward(
            feed_factory=lambda: RecordedFeed(args.recorded),
            timestamps=ticks, wf_cfg=wf_cfg,
        )
        print()
        result.print()

        jrn = Journal(":memory:")
        run_backtest(RecordedFeed(args.recorded), ticks, journal=jrn)
        _print_calibration(jrn)
        return

    # Demo on the COUPLED synthetic world (synthetic_world.py): GEX drives the
    # price dynamics, chains reprice off the live path every tick, and each
    # session settles at its actual close — so EV accuracy, win rates, and the
    # directional readouts measure something real about the pipeline, not just
    # plumbing. (The old static-chain SyntheticUnifiedFeed made prediction
    # unmeasurable by construction.)
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    DAYS = 12
    STRIDE = 5      # every 5th minute keeps the demo fast

    def make_feed():
        return CoupledSyntheticFeed(WorldConfig(days=DAYS, seed=11, tick_stride=STRIDE))

    ticks = make_feed().timestamps()

    for mode in ("expanding", "rolling"):
        print(f"\n{'='*72}")
        print(f"  Walk-Forward demo — mode={mode}, {DAYS} days (coupled world), "
              f"3 session-unit folds, 1-session embargo")
        print(f"{'='*72}\n")
        result = run_walk_forward(
            feed_factory=make_feed,
            timestamps=ticks,
            wf_cfg=WalkForwardConfig(mode=mode, n_folds=3, train_frac=0.6),
        )
        print()
        result.print()


if __name__ == "__main__":
    main()
