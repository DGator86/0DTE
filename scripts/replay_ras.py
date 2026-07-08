#!/usr/bin/env python3
"""
scripts/replay_ras.py
=====================
Validate the Regime Alignment Score (RAS) against a real paper trade.

Loads a closed trade (with its stored entry snapshot) from the paper-trading
DB, replays the session's recorded ticks through UnifiedOrchestrator, and
prints a tick-by-tick RAS timeline — score, EMA, action, and every component
note — over the trade's holding window. This is the Section-8 test from the
RAS handoff: reconstruct the entry, watch the score as the regime moved, and
confirm the notes are human-readable.

Usage:
    # replay a specific trade against its recorded session ticks
    python3 scripts/replay_ras.py --paper-db paper.sqlite --ticks-dir /var/lib/zerodte/ticks \
        [--trade-id abc123def456]

    # no args: if no paper DB / recordings exist, runs a synthetic
    # deterioration demo (bull entry; regime flips bear + short gamma)
    python3 scripts/replay_ras.py

Read-only: never writes to the paper DB or the journal. No orders, ever.
NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_alignment import (  # noqa: E402
    PositionContext, RASConfig, RASResult,
    compute_regime_alignment, position_context_from_entry_ctx,
)

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def _print_header(label: str) -> None:
    print("\n" + "=" * 78)
    print(f"  RAS replay — {label}")
    print("=" * 78)


def _print_eval(ts_label: str, ras: RASResult, verbose: bool = True) -> None:
    print(f"\n[{ts_label}]  score={ras.score:+7.1f}  ema={ras.ema_score:+7.1f}  "
          f"action={ras.action.upper()}")
    if verbose:
        for c in ras.components:
            print(f"    {c.name:<22} raw={c.raw:+.2f}  w={c.weight:.1f}  "
                  f"contrib={c.contribution:+.2f}  | {c.note}")


# --------------------------------------------------------------------------- #
# Real-trade replay against recorded ticks                                     #
# --------------------------------------------------------------------------- #
def _load_trade(paper_db: str, trade_id: str | None) -> dict | None:
    """Fetch one closed paper trade (default: the worst loser with a stored
    entry snapshot) as a dict, or None when nothing usable exists."""
    if not os.path.exists(paper_db):
        return None
    conn = sqlite3.connect(paper_db)
    conn.row_factory = sqlite3.Row
    try:
        if trade_id:
            row = conn.execute(
                "SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM paper_trades WHERE entry_ctx IS NOT NULL "
                "ORDER BY pnl_dollars ASC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    trade = dict(row)
    try:
        trade["entry_ctx"] = json.loads(trade.get("entry_ctx") or "{}")
    except (json.JSONDecodeError, TypeError):
        trade["entry_ctx"] = {}
    return trade


def replay_trade(paper_db: str, ticks_dir: str, trade_id: str | None,
                 verbose: bool) -> bool:
    """Replay one paper trade's holding window. Returns False when the trade
    or its recorded session cannot be loaded (caller falls back to synthetic)."""
    trade = _load_trade(paper_db, trade_id)
    if trade is None:
        print(f"No usable trade found in {paper_db!r}.")
        return False
    entry_ctx = trade.get("entry_ctx") or {}
    ctx = position_context_from_entry_ctx(trade["id"], entry_ctx)
    if ctx is None:
        print(f"Trade {trade['id']} has no stored entry_snapshot — cannot "
              "reconstruct the entry state. (Snapshots are captured for all "
              "trades opened after RAS integration.)")
        return False

    if not os.path.isdir(ticks_dir):
        print(f"No tick recordings at {ticks_dir!r}.")
        return False

    from chain_store import RecordedFeed
    from journal import Journal
    from unified_loop import UnifiedOrchestrator

    feed = RecordedFeed(ticks_dir)
    if len(feed) == 0:
        print(f"Tick directory {ticks_dir!r} contains no recorded sessions.")
        return False

    opened = dt.datetime.fromisoformat(trade["opened_at"])
    closed = dt.datetime.fromisoformat(trade["closed_at"])

    _print_header(
        f"trade {trade['id']}  {trade['family']} {trade['strikes']}  "
        f"pnl=${trade['pnl_dollars']:+.2f}  exit={trade['exit_reason']}")
    print(f"  opened {opened}  closed {closed}  "
          f"bias={ctx.position_bias}  structure={ctx.entry.structure}")

    # In-memory journal: the replay also exercises the ras_evaluations logging
    # path without touching any on-disk DB.
    orch = UnifiedOrchestrator(feed=feed, journal=Journal(":memory:"))
    n_evals = 0
    for ts in feed.timestamps():
        in_window = opened <= ts <= closed
        # tick() must run for EVERY recorded timestamp (the feed advances one
        # record per snapshot call); the position context only rides along
        # inside the holding window.
        result = orch.tick(ts, position_contexts=[ctx] if in_window else None)
        if result is None or not in_window:
            continue
        for ras in result.ras_results:
            n_evals += 1
            _print_eval(ts.strftime("%Y-%m-%d %H:%M:%S"), ras, verbose)
            ctx.prev_ema_score = ras.ema_score   # EMA continuity across ticks

    if n_evals == 0:
        print("\nNo recorded ticks fell inside the trade's holding window — "
              "check that --ticks-dir covers the trade's session date.")
        return False
    print(f"\n{n_evals} RAS evaluations replayed.")
    return True


# --------------------------------------------------------------------------- #
# Synthetic deterioration demo (no data files needed)                          #
# --------------------------------------------------------------------------- #
def synthetic_demo(verbose: bool) -> None:
    """A bull debit spread entered in a trending regime; the tape then flips
    bear and short-gamma below the flip. The score must fall monotonically
    and end at an actionable level, with readable notes at every step."""
    from decision_matrix import Decision, TradeIntent
    from gate_scorer import MarketSnapshot
    from regime_classifier import RegimeState
    from regime_alignment import build_entry_snapshot

    def market(spot, flip, net_gex):
        return MarketSnapshot(
            spot=spot, net_gex=net_gex, gamma_flip=flip,
            call_wall=spot + 5, put_wall=spot - 5, gex_pct_rank=0.8,
            vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
            straddle_breakeven=4.0, expected_range=3.2,
            adx=14.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
            vwap=spot, vwap_reversion_count=3,
            tick_abs_mean=450.0, cvd_slope=0.05,
            now=dt.datetime(2026, 7, 8, 10, 30, tzinfo=ET),
            has_catalyst=False,
        )

    def regime(dominant, engine, vetoes, confidences, std):
        return RegimeState(
            confidences=confidences,
            reliabilities={k: 0.8 for k in confidences},
            dominant_regime=dominant, permitted_engine=engine,
            vetoes=vetoes, global_information_gain=20.0,
            standardized=std, stand_down=False,
        )

    def intent(exec_r, ctx_r, bias, val, structure="LCS", direction="call"):
        return TradeIntent(
            exec_regime=exec_r, context_regime=ctx_r,
            direction_bias=bias, bias_value=val,
            decision=Decision(structure, direction, "HIGH", "demo", "rule", "15m"),
            size_mult=1.0, vetoes=[], note="",
        )

    std_ok = {"flip_proximity": (30.0, 1.0), "gamma_sign": (65.0, 1.0)}
    std_bad = {"flip_proximity": (85.0, 1.0), "gamma_sign": (25.0, 1.0)}

    # -- entry: bull LCS in a healthy trend, well above the flip, long gamma --
    entry_regime = regime("trend", "directional", [],
                          {"trend": 72.0, "directional_confidence": 70.0,
                           "compression": 25.0, "expansion": 20.0}, std_ok)
    entry_intent = intent("trend", "trend", "bull", 68.0)
    entry_market = market(spot=600.0, flip=594.0, net_gex=4e9)
    snap = build_entry_snapshot(entry_regime, entry_intent, entry_market,
                                "directional", "LCS")
    ctx = PositionContext("demo-lcs", "call", "bull", snap)

    stages = [
        ("T+0  regime unchanged (control)",
         entry_regime, entry_intent, entry_market),
        ("T+1  bias washes out to neutral, spot drifts to the flip",
         regime("trend", "directional", [],
                {"trend": 60.0, "directional_confidence": 58.0,
                 "compression": 32.0, "expansion": 22.0}, std_ok),
         intent("trend", "compression", "neutral", 50.0),
         market(spot=596.5, flip=595.5, net_gex=1e9)),
        ("T+2  bias flips bear, spot below flip, dealers short gamma",
         regime("trend", "directional", ["below_gamma_flip"],
                {"trend": 48.0, "directional_confidence": 44.0,
                 "compression": 30.0, "expansion": 35.0}, std_bad),
         intent("compression", "trend", "bear", 38.0),
         market(spot=594.0, flip=596.0, net_gex=-1.5e9)),
        ("T+3  full deterioration: engine revoked, confidence collapsed",
         regime("breakout", "none",
                ["below_gamma_flip", "short_gamma_regime", "catalyst:CPI"],
                {"trend": 30.0, "directional_confidence": 28.0,
                 "compression": 25.0, "expansion": 60.0}, std_bad),
         intent("breakout", "breakout", "bear", 25.0),
         market(spot=591.0, flip=596.5, net_gex=-4e9)),
    ]

    _print_header("synthetic deterioration demo (bull LCS, regime turns bear)")
    cfg = RASConfig()      # library defaults (exit_enabled=True since activation)
    scores = []
    for label, reg, itt, mkt in stages:
        ras = compute_regime_alignment(reg, itt, mkt, ctx, cfg=cfg)
        _print_eval(label, ras, verbose)
        ctx.prev_ema_score = ras.ema_score
        scores.append(ras.score)

    print("\nScore path:", " -> ".join(f"{s:+.1f}" for s in scores))
    ok = all(b <= a for a, b in zip(scores, scores[1:])) and scores[-1] < -30
    print("PASS: score decays monotonically and reaches an actionable level."
          if ok else
          "FAIL: score did not deteriorate as expected — inspect components.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay RAS over a recorded paper trade (or run a "
                    "synthetic deterioration demo).")
    p.add_argument("--paper-db", default="paper.sqlite",
                   help="Paper-trading SQLite DB (default: paper.sqlite)")
    p.add_argument("--ticks-dir", default="ticks",
                   help="Directory of chain_store recordings (default: ticks)")
    p.add_argument("--trade-id", default=None,
                   help="Specific paper trade id (default: worst loser)")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip the DB and run the synthetic demo directly")
    p.add_argument("--quiet", action="store_true",
                   help="Timeline only, no per-component breakdown")
    args = p.parse_args()

    verbose = not args.quiet
    if not args.synthetic:
        if replay_trade(args.paper_db, args.ticks_dir, args.trade_id, verbose):
            return
        print("Falling back to the synthetic deterioration demo.")
    synthetic_demo(verbose)


if __name__ == "__main__":
    main()
