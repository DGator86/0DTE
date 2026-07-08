#!/usr/bin/env python3
"""
scripts/turn_lag_study.py
=========================
Measure how late each direction signal is at intraday turns, from journal data.

Motivation: the blended direction bias is 60% weighted to session-anchored
slow timeframes, so it is structurally late at V-shaped intraday reversals.
Before re-weighting anything, measure the actual cost: for every mechanical
turn in the recorded spot series, how many minutes later did each candidate
signal flip to agree with the new direction?

Method
------
1. Turn detection (mechanical, no hindsight parameters hidden in the signal):
   a zigzag pivot on the journal's per-tick spot series. A pivot is confirmed
   when price reverses off a running extreme by >= --min-move-pct and the
   reversal holds for >= --hold-min minutes (no new extreme beyond the pivot).
2. For each confirmed turn, scan forward and record the minutes until each
   signal first agrees with the turn direction:
     blend   signals_json.regime_bias_value crossing 50
     fast    signals_json.bias_fast crossing 50
     cross   signals_json.bias_cross event in the turn direction
     dirword journal.regime_direction flipping to call/put
   "never" = the signal did not flip before the next turn (or session end).
3. Print a per-turn table and per-signal summary (median / mean lag, flip
   rate) so the leading channel is chosen from data, not taste.

Usage:
    python3 scripts/turn_lag_study.py --db shadow.db
    python3 scripts/turn_lag_study.py --db shadow.db --session 2026-07-08
    python3 scripts/turn_lag_study.py --db shadow.db --min-move-pct 0.4 --hold-min 30

Read-only. NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Tick:
    ts: dt.datetime
    spot: float
    regime_direction: Optional[str]
    signals: dict


def load_sessions(db_path: str, session: Optional[str] = None) -> dict[str, list[Tick]]:
    """Journal ticks grouped by session_date, time-ascending."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT session_date, ts, spot, regime_direction, signals_json "
               "FROM evaluations WHERE spot IS NOT NULL")
        args: list = []
        if session:
            sql += " AND session_date = ?"
            args.append(session)
        sql += " ORDER BY session_date, ts"
        sessions: dict[str, list[Tick]] = {}
        for r in conn.execute(sql, args):
            try:
                ts = dt.datetime.fromisoformat(r["ts"])
            except (TypeError, ValueError):
                continue
            sig = {}
            if r["signals_json"]:
                try:
                    sig = json.loads(r["signals_json"])
                except json.JSONDecodeError:
                    pass
            sessions.setdefault(r["session_date"], []).append(
                Tick(ts=ts, spot=float(r["spot"]),
                     regime_direction=r["regime_direction"], signals=sig))
        return sessions
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Turn detection: zigzag pivots with a hold requirement                        #
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    idx: int                 # tick index of the pivot (session extreme)
    ts: dt.datetime
    spot: float
    direction: str           # "up" (V-bottom) or "down" (top)
    end_idx: int             # last tick index this turn "owns" (next turn or end)


def detect_turns(ticks: list[Tick], min_move_pct: float,
                 hold_min: float) -> list[Turn]:
    """Zigzag pivot detection on the spot series.

    Track a running extreme; when price reverses off it by >= min_move_pct
    (percent of the extreme) the extreme becomes a candidate pivot. The pivot
    is confirmed only if no new extreme beyond it prints within hold_min
    minutes of the reversal threshold being reached.
    """
    if len(ticks) < 3:
        return []
    turns: list[Turn] = []
    # trend: +1 while tracking a running high (looking for a top),
    #        -1 while tracking a running low (looking for a bottom).
    # Seed from the first move away from tick 0.
    ext_i = 0
    trend = 0
    for i in range(1, len(ticks)):
        px, ext = ticks[i].spot, ticks[ext_i].spot
        if trend == 0:
            if px > ext:
                trend, ext_i = 1, i
            elif px < ext:
                trend, ext_i = -1, i
            continue

        # extend the running extreme
        if (trend == 1 and px >= ext) or (trend == -1 and px <= ext):
            ext_i = i
            continue

        # reversal magnitude off the extreme
        move = abs(px - ext) / ext * 100.0
        if move < min_move_pct:
            continue

        # hold check: no new extreme beyond the pivot for hold_min minutes
        deadline = ticks[i].ts + dt.timedelta(minutes=hold_min)
        held = True
        for j in range(i + 1, len(ticks)):
            if ticks[j].ts > deadline:
                break
            pj = ticks[j].spot
            if (trend == 1 and pj > ext) or (trend == -1 and pj < ext):
                held = False
                break
        if not held:
            continue

        turns.append(Turn(
            idx=ext_i, ts=ticks[ext_i].ts, spot=ext,
            direction="down" if trend == 1 else "up",
            end_idx=len(ticks) - 1,          # fixed up below
        ))
        trend, ext_i = -trend, i             # now track the opposite extreme

    for a, b in zip(turns, turns[1:]):
        a.end_idx = b.idx
    return turns


# --------------------------------------------------------------------------- #
# Lag measurement                                                              #
# --------------------------------------------------------------------------- #
SIGNALS = ["fast", "cross", "dirword", "blend"]


def _agrees(name: str, tick: Tick, direction: str) -> Optional[bool]:
    """Does this signal currently agree with the turn direction?
    None = signal not present on this tick (can't judge)."""
    want_up = direction == "up"
    if name == "blend":
        v = tick.signals.get("regime_bias_value")
        if not isinstance(v, (int, float)):
            return None
        return v > 50.0 if want_up else v < 50.0
    if name == "fast":
        v = tick.signals.get("bias_fast")
        if not isinstance(v, (int, float)):
            return None
        return v > 50.0 if want_up else v < 50.0
    if name == "cross":
        v = tick.signals.get("bias_cross")
        if not isinstance(v, (int, float)):
            return False                     # event signal: absent = no event
        return v > 0 if want_up else v < 0
    if name == "dirword":
        d = tick.regime_direction
        if d not in ("call", "put"):
            return False
        return d == ("call" if want_up else "put")
    return None


@dataclass
class TurnLags:
    turn: Turn
    session: str
    lag_min: dict[str, Optional[float]] = field(default_factory=dict)


def measure_lags(session: str, ticks: list[Tick],
                 turns: list[Turn]) -> list[TurnLags]:
    out = []
    for turn in turns:
        row = TurnLags(turn=turn, session=session)
        for name in SIGNALS:
            lag: Optional[float] = None
            seen_any = False
            for j in range(turn.idx, turn.end_idx + 1):
                a = _agrees(name, ticks[j], turn.direction)
                if a is None:
                    continue
                seen_any = True
                if a:
                    lag = (ticks[j].ts - turn.ts).total_seconds() / 60.0
                    break
            # distinguish "never flipped" (measured, inf) from "no data" (None)
            row.lag_min[name] = lag if lag is not None else (
                math.inf if seen_any else None)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt_lag(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if math.isinf(v):
        return "never"
    return f"{v:5.1f}m"


def print_report(results: list[TurnLags]) -> None:
    if not results:
        print("No confirmed turns found. Try a lower --min-move-pct or "
              "--hold-min, or check the DB has tick history.")
        return

    print(f"\n{'session':<12}{'turn time':<22}{'dir':<6}{'spot':<10}"
          + "".join(f"{s:<9}" for s in SIGNALS))
    print("-" * (50 + 9 * len(SIGNALS)))
    for r in results:
        print(f"{r.session:<12}{r.turn.ts.strftime('%H:%M:%S %Z') or '':<22}"
              f"{r.turn.direction:<6}{r.turn.spot:<10.2f}"
              + "".join(f"{_fmt_lag(r.lag_min[s]):<9}" for s in SIGNALS))

    print("\nPer-signal summary (lag from confirmed turn, minutes):")
    print(f"{'signal':<10}{'n':<5}{'flipped':<9}{'median':<9}{'mean':<9}{'worst':<9}")
    print("-" * 51)
    for name in SIGNALS:
        vals = [r.lag_min[name] for r in results if r.lag_min[name] is not None]
        finite = [v for v in vals if math.isfinite(v)]
        if not vals:
            print(f"{name:<10}{'0':<5}{'—':<9}{'n/a':<9}{'n/a':<9}{'n/a':<9}")
            continue
        rate = f"{len(finite)}/{len(vals)}"
        med = f"{statistics.median(finite):.1f}" if finite else "never"
        mean = f"{statistics.fmean(finite):.1f}" if finite else "never"
        worst = f"{max(finite):.1f}" if finite else "never"
        print(f"{name:<10}{len(vals):<5}{rate:<9}{med:<9}{mean:<9}{worst:<9}")

    print("\nSignals: fast = bias_fast crossing 50 | cross = fast/slow crossover "
          "event\n         dirword = regime_direction word flip | blend = "
          "regime_bias_value crossing 50")
    print("Note: bias_fast / bias_cross only exist in journal rows written "
          "after they were introduced;\nolder sessions report n/a for those "
          "columns.")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run_study(db_path: str, session: Optional[str], min_move_pct: float,
              hold_min: float) -> list[TurnLags]:
    sessions = load_sessions(db_path, session)
    results: list[TurnLags] = []
    for sess, ticks in sorted(sessions.items()):
        turns = detect_turns(ticks, min_move_pct, hold_min)
        results.extend(measure_lags(sess, ticks, turns))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--db", default="shadow.db", help="journal sqlite path")
    p.add_argument("--session", default=None,
                   help="restrict to one session_date (YYYY-MM-DD)")
    p.add_argument("--min-move-pct", type=float, default=0.35,
                   help="reversal off a session extreme to qualify as a turn "
                        "(percent, default 0.35)")
    p.add_argument("--hold-min", type=float, default=30.0,
                   help="minutes the reversal must hold without a new extreme "
                        "(default 30)")
    args = p.parse_args()

    if not os.path.exists(args.db):
        print(f"Journal DB not found: {args.db}")
        return 1

    results = run_study(args.db, args.session, args.min_move_pct, args.hold_min)
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
