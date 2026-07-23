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
import os
import sys
import time
from zoneinfo import ZoneInfo

import dataclasses

from chain_store import ChainRecorder
from composite_feed import build_default_feed
from journal import Journal
from unified_loop import UnifiedOrchestrator
from notifier import Notifier, Ticket
from risk_manager import PositionMonitor, PositionRiskConfig, RiskConfig, RiskManager
from paper_broker import PaperBroker, PaperConfig
from market_calendar import is_market_open, next_market_open, market_status
from dashboard.state import heartbeat_state, serialize_tick_result, write_live_state
from regime_alignment import RASConfig, position_context_from_entry_ctx
from typing import Optional

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shadow] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("shadow")


# --------------------------------------------------------------------------- #
# Settlement helper                                                            #
# --------------------------------------------------------------------------- #
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
        risk_cfg: Optional[RiskConfig] = None,
        paper_db: Optional[str] = None,
        paper_cfg: "Optional[PaperConfig]" = None,
        live_state_path: str = "live_state.json",
        record_dir: Optional[str] = None,   # None = <db_dir>/ticks; "" disables
        ras_exit: bool = True,              # False = RAS observation-only (no auto-exits)
        champion_path: Optional[str] = None,  # None = configs/champion.json; "" disables
        policy_mode: str = "shadow",          # legacy | shadow | champion
        prediction_db: Optional[str] = None,  # None = <state_dir>/prediction_store.sqlite
        use_legacy_directional_tilt: bool = True,
        enable_v2_parallel: bool = True,      # heuristic bundle + ranker always on
    ) -> None:
        self.symbol = symbol
        self.interval_s = interval_s
        self.live_state_path = live_state_path
        self.policy_mode = policy_mode

        self._jrn = Journal(db_path)
        # Adaptive state (GEX percentile window, scale books) lives next to the
        # journal DB so restarts and deploys don't cold-start the gates.
        state_dir = os.path.dirname(os.path.abspath(db_path))
        # Auto-detect credentialed providers (Tradier -> Tastytrade -> Massive)
        # and fail over between them per tick; Yahoo backstops settlement.
        self._feed = build_default_feed(
            symbol=symbol,
            lookback_minutes=lookback_minutes,
            vix9d=vix9d,
            vix=vix,
            vix3m=vix3m,
            vvix=vvix,
            vvix_baseline=vvix_baseline,
            gex_history_path=os.path.join(state_dir, "gex_history.json"),
        )
        self._risk = RiskManager(risk_cfg) if risk_cfg else None
        # One RASConfig shared by the orchestrator (scores + actions), the
        # position monitor (action suppression), and the paper broker (whether
        # an "exit" action actually closes) — a single flag, never three that
        # can drift apart.
        self._ras_cfg = RASConfig(exit_enabled=ras_exit)
        # Champion config: the ONE live configuration, produced by the
        # adaptive-learning promotion flow and installed only via the human
        # approval CLI. Missing file = dataclass defaults (unchanged
        # behaviour); an INVALID file raises — silently trading on defaults
        # when a champion was intended is the worse failure mode.
        engine_cfg = classifier_cfg = None
        regime_overrides = None
        self.champion = None
        if champion_path is None:
            champion_path = os.path.join("configs", "champion.json")
        if champion_path and os.path.isfile(champion_path):
            from adaptive_learning.config_store import load_config
            rec = load_config(champion_path)          # raises on invalid file
            engine_cfg, classifier_cfg = rec.engine_cfg()
            regime_overrides = rec.regime_overrides or None
            self.champion = rec
            log.info("Champion config loaded: %s (id=%s, label=%r, "
                     "%d overrides, %d regime overrides)",
                     champion_path, rec.config_id[:8], rec.label,
                     len(rec.overrides), len(rec.regime_overrides or {}))

        # Prediction Engine V2 parallel path (shadow by default).
        pred_db = prediction_db
        if pred_db is None:
            pred_db = os.path.join(state_dir, "prediction_store.sqlite")
        self._prediction_store = None
        bundle_provider = None
        physical_provider = None
        candidate_model = None
        if enable_v2_parallel and pred_db:
            from prediction.storage import PredictionStore
            from prediction.inference import (
                HeuristicCandidateValueModel,
                make_bundle_provider,
                make_physical_forecast_provider,
            )
            self._prediction_store = PredictionStore(db_path=pred_db)
            bundle_provider = make_bundle_provider(
                symbol=symbol, store=self._prediction_store)
            physical_provider = make_physical_forecast_provider(bundle_provider)
            candidate_model = HeuristicCandidateValueModel()
            log.info("V2 parallel enabled: policy_mode=%s prediction_db=%s "
                     "legacy_tilt=%s",
                     policy_mode, pred_db, use_legacy_directional_tilt)

        self._orch = UnifiedOrchestrator(
            feed=self._feed, journal=self._jrn, risk_manager=self._risk,
            engine_cfg=engine_cfg, classifier_cfg=classifier_cfg,
            state_path=os.path.join(state_dir, "adaptive_state.json"),
            ras_cfg=self._ras_cfg,
            regime_overrides=regime_overrides,
            symbol=symbol,
            paper_db_path=paper_db or "paper.sqlite",
            policy_mode=policy_mode,
            prediction_store=self._prediction_store,
            prediction_bundle_provider=bundle_provider,
            physical_forecast_provider=physical_provider,
            candidate_value_model=candidate_model,
            use_legacy_directional_tilt=use_legacy_directional_tilt,
        )
        # Record every tick (market + chain + incremental bars) so a REAL-data
        # walk-forward becomes possible. ~1 MB/session gzipped; you cannot
        # backfill what you never saved.
        if record_dir is None:
            record_dir = os.path.join(state_dir, "ticks")
        self._recorder = ChainRecorder(record_dir) if record_dir else None
        self._notifier = Notifier()
        self._settled: set[str] = set()
        # Settlement is a required feed source for overall LIVE. Cache the last
        # successful observation so we can report an honest age without hitting
        # the provider on every tick.
        self._settlement_obs_at: Optional[dt.datetime] = None
        self._settlement_obs_date: Optional[str] = None

        # In-house paper trading: auto-executes TRADE tickets on SIMULATED fills
        # over the live chain, with stop-loss / target / trailing / EOD / RAS
        # exits. No real orders are ever placed.
        paper_cfg = dataclasses.replace(paper_cfg or PaperConfig(),
                                        ras_exit_enabled=ras_exit)
        self._paper = PaperBroker(
            db_path=paper_db or "paper.sqlite",
            cfg=paper_cfg, notifier=self._notifier, symbol=symbol,
            position_monitor=PositionMonitor(PositionRiskConfig(ras=self._ras_cfg)),
        )

        log.info("Initialized. DB=%s symbol=%s interval=%ds", db_path, symbol, interval_s)
        log.info("Paper account: $%.0f start (simulated fills, no real orders).",
                 self._paper.cfg.starting_cash)
        log.info("RAS position management: %s",
                 "ACTIVE (warning/tighten/exit on paper positions)" if ras_exit
                 else "observation-only (--no-ras-exit)")
        if self._risk:
            cfg = risk_cfg
            log.info(
                "Risk manager: max_loss=%.2f max_positions=%d max_gamma=%.4f",
                cfg.daily_loss_limit, cfg.max_open_positions, cfg.max_portfolio_gamma,
            )
        log.info("No orders will be placed — shadow mode only.")

    # -- public API ----------------------------------------------------------

    def run(self) -> None:
        log.info("Starting shadow loop. Ctrl-C to stop.")
        try:
            while True:
                now = dt.datetime.now(ET)
                self._maybe_settle(now)

                if is_market_open(now):
                    self._tick(now)
                    time.sleep(self.interval_s)
                else:
                    # off-hours: check every 5 min so we don't miss settle window
                    nxt = next_market_open(now)
                    secs_to_open = (nxt - now).total_seconds()
                    log.info(
                        "Market closed. Next open %s ET (%.1fh).",
                        nxt.strftime("%a %Y-%m-%d %H:%M"),
                        secs_to_open / 3600,
                    )
                    self._heartbeat(
                        now, "market_closed",
                        f"Market closed — next open {nxt.strftime('%a %H:%M')} ET.",
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

        # -- predictive power: does the system call the market forward? --
        cal = self._jrn.calibration()
        d = cal["directional"]["overall"]
        print("\n  Predictive power (all settled ticks, no-trades included):")
        if d["n"]:
            print(f"    Direction bias:  n={d['n']:5d}  hit={d['hit_rate']:.1%}  "
                  f"signed fwd move={d['avg_fwd_move_pct']:+.3f}%")
            for side, s in cal["directional"]["by_direction"].items():
                if s["n"]:
                    print(f"      {side:<5} n={s['n']:5d}  hit={s['hit_rate']:.1%}  "
                          f"move={s['avg_fwd_move_pct']:+.3f}%")
        else:
            print("    Direction bias:  no resolved-bias settled ticks yet")
        pp = cal["prob_profit"]
        if pp.get("n"):
            print(f"    prob_profit:     n={pp['n']:5d}  Brier={pp['brier']:.4f}  "
                  f"skill={pp['brier_skill']}  base_rate={pp['base_rate']:.1%}")
            for b in pp.get("bins", []):
                print(f"      p∈{b['bin']}  n={b['n']:4d}  "
                      f"predicted={b['mean_predicted']:.2f}  realized={b['realized_rate']:.2f}")
        ev = cal["ev"]
        if ev.get("n"):
            print(f"    EV bias:         n={ev['n']:5d}  mean_err={ev['mean_ev_error']:+.4f}  "
                  f"MAE={ev['mae_ev_error']:.4f}  (mean |EV|={ev['mean_abs_ev']:.4f})")

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

        if self._risk:
            st = self._risk.status()
            print(f"\n  Risk manager status:")
            print(f"    Open positions:  {st['open_positions']}")
            print(f"    Daily loss used: {st['daily_loss_committed']:.4f}")
            print(f"    Net gamma:       {st['net_gamma']:.6f}")

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

    def _heartbeat(self, now: dt.datetime, status: str, note: str) -> None:
        """Write a liveness-only live_state so the dashboard can show *why*
        there is no tick (feed down / market closed) instead of blank dashes.
        Never raises into the loop."""
        try:
            write_live_state(
                self.live_state_path,
                heartbeat_state(
                    now,
                    status=status,
                    note=note,
                    feed_source=getattr(self._feed, "last_source", None),
                    paper_summary=self._paper.report(now),
                    market_status=market_status(now),
                ),
            )
        except Exception as exc:
            log.warning("heartbeat write error: %s", exc)

    def _refresh_settlement_obs(self, now: dt.datetime) -> None:
        """Observe settlement at most once per session date."""
        session_date = now.astimezone(ET).date().isoformat()
        if self._settlement_obs_date == session_date:
            return
        px = None
        try:
            px = self._feed.settlement_price(session_date)
        except Exception:
            px = None
        if px is None:
            prior = (now.astimezone(ET).date() - dt.timedelta(days=1)).isoformat()
            try:
                px = self._feed.settlement_price(prior)
            except Exception:
                px = None
        if px is not None:
            self._settlement_obs_date = session_date
            self._settlement_obs_at = now

    def _feed_ages_seconds(self, now: dt.datetime, result) -> dict[str, float]:
        """Per-source ages for live.v1 feed badges on a successful tick.

        A truthy provider name alone must not claim overall LIVE (age unknown
        ⇒ DELAYED). Successful snapshots report age 0 for the sources that
        just produced data; settlement uses the cached observation age.
        """
        ages: dict[str, float] = {"spot": 0.0, "bars": 0.0}
        snap = getattr(result, "snapshot", None)
        if snap is not None and getattr(snap, "chain", None) is not None:
            ages["option_chain"] = 0.0
        self._refresh_settlement_obs(now)
        if self._settlement_obs_at is not None:
            ages["settlement"] = max(
                0.0, (now - self._settlement_obs_at).total_seconds())
        return ages

    def _tick(self, now: dt.datetime) -> None:
        position_contexts = []
        for pos in self._paper.open_positions:
            ctx = position_context_from_entry_ctx(pos.id, pos.entry_ctx)
            if ctx is not None:
                position_contexts.append(ctx)
        try:
            result = self._orch.tick(now, position_contexts=position_contexts or None)
        except Exception as exc:
            log.warning("tick() error: %s", exc)
            self._heartbeat(now, "feed_error", f"Feed error: {exc}")
            return

        if result is None:
            log.debug("tick() returned None (feed not ready)")
            self._heartbeat(
                now, "feed_not_ready",
                "Feed not ready — no market data returned. Check the data feed "
                "credentials/provider (e.g. TRADIER_ACCESS_TOKEN).",
            )
            return

        if self._recorder and result.snapshot is not None:
            self._recorder.record(now, result.snapshot)

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

        # Drive the paper broker: mark open positions and auto-execute exits/
        # entries on simulated fills. Never raises into the tick loop.
        try:
            for ev in self._paper.on_tick(now, result):
                log.info("  %s  (paper equity=$%.2f)", ev, self._paper.cash)
        except Exception as exc:
            log.warning("paper broker error: %s", exc)

        try:
            write_live_state(
                self.live_state_path,
                serialize_tick_result(
                    result,
                    feed_source=getattr(self._feed, "last_source", None),
                    paper_summary=self._paper.report(now),
                    market_status=market_status(now),
                    feed_ages_seconds=self._feed_ages_seconds(now, result),
                ),
            )
        except Exception as exc:
            log.warning("live state write error: %s", exc)

    def _maybe_settle(self, now: dt.datetime) -> None:
        if not _settle_eligible(now):
            return
        session_date = now.astimezone(ET).date().isoformat()
        if session_date in self._settled:
            return
        n = self._orch.settle(session_date)
        self._settled.add(session_date)
        if self._recorder:
            price = self._feed.settlement_price(session_date)
            if price is not None:
                self._recorder.record_settlement(session_date, price)
        if self._risk:
            self._risk.close_positions()
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
    p.add_argument("--paper-report", dest="paper_report", action="store_true",
                   help="Print the paper-trading report (P&L, win rate, exits) and exit")
    p.add_argument("--paper-db", dest="paper_db", default="paper.sqlite",
                   help="SQLite path for paper trades (default: paper.sqlite)")
    p.add_argument("--paper-cash", dest="paper_cash", type=float, default=1000.0,
                   help="Starting virtual cash for paper trading (default: 1000)")
    p.add_argument("--live-state", dest="live_state", default="live_state.json",
                   help="Path for dashboard live_state.json (default: live_state.json)")
    p.add_argument("--record-dir", dest="record_dir", default=None,
                   help="Directory for tick recordings (default: <db dir>/ticks; "
                        "pass an empty string to disable)")
    p.add_argument("--settle",   metavar="YYYY-MM-DD",
                   help="Manually settle a specific session date and exit")
    p.add_argument("--max-loss", dest="max_loss", type=float, default=0.0,
                   help="Daily max_loss budget per contract (0 = disabled)")
    p.add_argument("--max-positions", dest="max_positions", type=int, default=0,
                   help="Max concurrent open positions (0 = unlimited)")
    p.add_argument("--max-gamma", dest="max_gamma", type=float, default=0.0,
                   help="Max portfolio net |gamma| (0 = disabled)")
    p.add_argument("--no-ras-exit", dest="ras_exit", action="store_false",
                   help="RAS observation-only: log scores/actions but never "
                        "auto-close paper positions (default: exits enabled)")
    p.add_argument("--champion", dest="champion_path", default=None,
                   help="Champion config JSON (default: configs/champion.json "
                        "when present; pass an empty string to force defaults)")
    p.add_argument("--policy-mode", dest="policy_mode", default="shadow",
                   choices=["legacy", "shadow", "champion"],
                   help="Policy promotion mode (default: shadow = dual-run, "
                        "legacy authoritative)")
    p.add_argument("--prediction-db", dest="prediction_db", default=None,
                   help="PredictionStore sqlite path "
                        "(default: <db dir>/prediction_store.sqlite)")
    p.add_argument("--no-legacy-directional-tilt",
                   dest="use_legacy_directional_tilt", action="store_false",
                   help="Price candidates with V2 physical density instead of "
                        "legacy directional tilt")
    p.add_argument("--no-v2-parallel", dest="enable_v2_parallel",
                   action="store_false",
                   help="Disable V2 heuristic bundle / ranker / prediction store")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    risk_cfg = None
    if args.max_loss > 0 or args.max_positions > 0 or args.max_gamma > 0:
        risk_cfg = RiskConfig(
            daily_loss_limit=args.max_loss or float("inf"),
            max_open_positions=args.max_positions,
            max_portfolio_gamma=args.max_gamma or float("inf"),
        )

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
        risk_cfg=risk_cfg,
        paper_db=args.paper_db,
        paper_cfg=PaperConfig(starting_cash=args.paper_cash),
        live_state_path=args.live_state,
        record_dir=args.record_dir,
        ras_exit=args.ras_exit,
        champion_path=args.champion_path,
        policy_mode=args.policy_mode,
        prediction_db=args.prediction_db,
        use_legacy_directional_tilt=args.use_legacy_directional_tilt,
        enable_v2_parallel=args.enable_v2_parallel,
    )

    if args.report:
        runner.report()
    elif args.paper_report:
        runner._paper.print_report()
    elif args.settle:
        runner.settle_date(args.settle)
    else:
        runner.run()
