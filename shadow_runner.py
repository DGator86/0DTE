"""
shadow_runner.py
================
Shadow mode: run the full 0DTE pipeline live — regime routing, RND extraction,
spread selection, gate scoring — but journal every tick and never route an order.

Run this for 2-4 weeks before going live. The journal accumulates the evidence
base for gate_effectiveness() and component_correlations(), which tell you:
  - whether the gate is filtering losers or blocking winners
  - which score components actually predict realized P&L

Usage:
    python shadow_runner.py [--symbol SPY] [--db shadow.db] [--interval 60]
    python shadow_runner.py --report [--db shadow.db]

Stops cleanly on Ctrl-C. Settlement runs automatically at 4:15 PM ET each day.

SECURITY: credentials from environment only.
    export MASSIVE_API_KEY=...
    export MASSIVE_BASE_URL=https://api.massive.com

NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from zoneinfo import ZoneInfo

from massive_feed import MassiveDataFeed
from journal import Journal
from unified_loop import UnifiedOrchestrator
from notifier import Notifier, Ticket

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shadow] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("shadow")


# --------------------------------------------------------------------------- #
# Market calendar helpers                                                      #
# --------------------------------------------------------------------------- #
def _market_open(now: dt.datetime) -> bool:
    et = now.astimezone(ET)
    if et.weekday() >= 5:          # Sat=5, Sun=6
        return False
    t = et.time()
    return dt.time(9, 30) <= t < dt.time(16, 0)


def _next_open(now: dt.datetime) -> dt.datetime:
    """Return the next market open (9:30 ET on the next/same weekday)."""
    et = now.astimezone(ET)
    open_today = et.replace(hour=9, minute=30, second=0, microsecond=0)
    if et < open_today and et.weekday() < 5:
        return open_today
    candidate = (et + dt.timedelta(days=1)).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    while candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    return candidate


def _settle_eligible(now: dt.datetime) -> bool:
    """4:15 PM ET or later on a weekday — settlement prices are available."""
    et = now.astimezone(ET)
    if et.weekday() >= 5:
        return False
    return et.hour > 16 or (et.hour == 16 and et.minute >= 15)


# --------------------------------------------------------------------------- #
# Shadow runner                                                                #
# --------------------------------------------------------------------------- #
class ShadowRunner:
    """
    Drives UnifiedOrchestrator in shadow (no-order) mode.

    Every tick during market hours:
      - Fetches bars + chain from Massive
      - Runs regime classifier + decision matrix (Track B)
      - Runs RND + gate + spread selector (Track A, when chain available)
      - Journals the result — trade or no-trade, with all scores
      - Prints a one-line summary to stdout

    At 4:15 PM ET each session day:
      - Calls settle() to fetch the EOD price and fill realized P&L + ev_error

    No orders are ever placed. The journal is the only output.
    """

    def __init__(
        self,
        symbol: str = "SPY",
        db_path: str = "shadow.db",
        interval_s: int = 60,
        lookback_minutes: int = 7800,
        vix9d: float = 14.0,
        vix: float = 15.0,
        vix3m: float = 17.0,
        vvix: float = 92.0,
        vvix_baseline: float = 95.0,
    ) -> None:
        self.symbol = symbol
        self.interval_s = interval_s

        self._jrn = Journal(db_path)
        self._feed = MassiveDataFeed(
            underlying=symbol,
            lookback_minutes=lookback_minutes,
            vix9d=vix9d,
            vix=vix,
            vix3m=vix3m,
            vvix=vvix,
            vvix_baseline=vvix_baseline,
        )
        self._orch = UnifiedOrchestrator(feed=self._feed, journal=self._jrn)
        self._notifier = Notifier()
        self._settled: set[str] = set()

        log.info("Initialized. DB=%s symbol=%s interval=%ds", db_path, symbol, interval_s)
        log.info("No orders will be placed — shadow mode only.")

    # -- public API ----------------------------------------------------------

    def run(self) -> None:
        log.info("Starting shadow loop. Ctrl-C to stop.")
        try:
            while True:
                now = dt.datetime.now(ET)
                self._maybe_settle(now)

                if _market_open(now):
                    self._tick(now)
                    time.sleep(self.interval_s)
                else:
                    # off-hours: check every 5 min so we don't miss settle window
                    nxt = _next_open(now)
                    secs_to_open = (nxt - now).total_seconds()
                    log.info(
                        "Market closed. Next open %s ET (%.1fh).",
                        nxt.strftime("%a %Y-%m-%d %H:%M"),
                        secs_to_open / 3600,
                    )
                    time.sleep(min(300, secs_to_open))

        except KeyboardInterrupt:
            log.info("Stopped by user. Run with --report for calibration summary.")

    def report(self) -> None:
        """Print gate effectiveness and score-component correlations to stdout."""
        eff = self._jrn.gate_effectiveness()
        taken = eff["trades_taken"]
        blocked = eff["blocked_by_gate"]

        print("\n" + "=" * 60)
        print("  Shadow Mode Calibration Report")
        print("=" * 60)
        print(f"  Trades taken:    n={taken['n']:5d}  "
              f"mean_ev={taken.get('mean_ev', 0) or 0:.3f}  "
              f"mean_pnl={taken.get('mean_pnl', 0) or 0:.3f}")
        print(f"  Blocked by gate: n={blocked['n']:5d}  "
              f"mean_ev={blocked.get('mean_ev', 0) or 0:.3f}  "
              f"mean_pnl={blocked.get('mean_pnl', 0) or 0:.3f}")
        print(f"  Verdict: {eff['verdict']}")

        corr = self._jrn.component_correlations()
        if corr:
            print("\n  Gate component correlations with realized P&L:")
            for comp, r in sorted(corr.items(), key=lambda kv: -abs(kv[1] or 0)):
                if r is None:
                    print(f"    {comp:<32} n/a")
                else:
                    bar = "█" * min(20, int(abs(r) * 20))
                    sign = "+" if r >= 0 else ""
                    print(f"    {comp:<32} r={sign}{r:.3f}  {bar}")
        else:
            print("\n  No settled rows yet — run longer before interpreting correlations.")

        print("=" * 60)

        # unsettled sessions needing manual attention
        unsettled = self._jrn.unsettled_dates()
        if unsettled:
            print(f"\n  Unsettled sessions ({len(unsettled)}): {', '.join(unsettled)}")
            print("  Re-run with --settle <YYYY-MM-DD> to backfill.")

    def settle_date(self, session_date: str) -> None:
        """Manually settle a specific session (backfill if auto-settle missed it)."""
        n = self._orch.settle(session_date)
        log.info("Settled %s: %d rows updated", session_date, n)

    # -- internals -----------------------------------------------------------

    def _tick(self, now: dt.datetime) -> None:
        try:
            result = self._orch.tick(now)
        except Exception as exc:
            log.warning("tick() error: %s", exc)
            return

        if result is None:
            log.debug("tick() returned None (feed not ready)")
            return

        regime = result.regime.dominant_regime
        struct = result.intent.decision.structure
        mult = result.final_size_mult
        if result.decision is not None:
            dec = result.decision.decision
            gate = "PASS" if result.decision.gate_pass else "FAIL"
        else:
            dec = "NO_TRADE"
            gate = "—"

        log.info(
            "%s  regime=%-20s  %s  gate=%s  decision=%s  x%.2f",
            now.strftime("%H:%M:%S"),
            regime,
            struct,
            gate,
            dec,
            mult,
        )

        if (result.decision is not None
                and result.decision.decision == "TRADE"
                and result.decision.gate_pass):
            ticket = Ticket.from_tick_result(result, self.symbol)
            self._notifier.send(ticket)

    def _maybe_settle(self, now: dt.datetime) -> None:
        if not _settle_eligible(now):
            return
        session_date = now.astimezone(ET).date().isoformat()
        if session_date in self._settled:
            return
        n = self._orch.settle(session_date)
        self._settled.add(session_date)
        if n > 0:
            log.info("Settled %s: %d rows updated with EOD price.", session_date, n)
        else:
            log.info("Settled %s: no rows updated (settlement price may not be available yet).",
                     session_date)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="0DTE shadow mode — journal every tick, never place an order."
    )
    p.add_argument("--symbol",   default="SPY",        help="Underlying ticker (default: SPY)")
    p.add_argument("--db",       default="shadow.db",  help="SQLite journal path (default: shadow.db)")
    p.add_argument("--interval", type=int, default=60, help="Seconds between ticks (default: 60)")
    p.add_argument("--lookback", type=int, default=7800,
                   help="Bar history in minutes (default: 7800 ≈ 20 days)")
    p.add_argument("--vix9d",    type=float, default=14.0)
    p.add_argument("--vix",      type=float, default=15.0)
    p.add_argument("--vix3m",    type=float, default=17.0)
    p.add_argument("--vvix",     type=float, default=92.0)
    p.add_argument("--vvix-baseline", dest="vvix_baseline", type=float, default=95.0)
    p.add_argument("--report",   action="store_true",
                   help="Print calibration report from the journal DB and exit")
    p.add_argument("--settle",   metavar="YYYY-MM-DD",
                   help="Manually settle a specific session date and exit")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    runner = ShadowRunner(
        symbol=args.symbol,
        db_path=args.db,
        interval_s=args.interval,
        lookback_minutes=args.lookback,
        vix9d=args.vix9d,
        vix=args.vix,
        vix3m=args.vix3m,
        vvix=args.vvix,
        vvix_baseline=args.vvix_baseline,
    )

    if args.report:
        runner.report()
    elif args.settle:
        runner.settle_date(args.settle)
    else:
        runner.run()
