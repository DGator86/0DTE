"""
journal.py
==========
Pure persistence layer for the 0DTE system. One row per evaluation tick --
trades AND no-trades, because no-trades are first-class data: they are the only
way to later prove whether a gate veto saved money or filtered a winner.

Responsibilities (and nothing else):
  - own the SQLite schema
  - log(row_dict) one evaluation
  - settle_session(date, settle_price): fill realized P&L for EVERY logged
    candidate that day (hypothetical for no-trades), from stored legs + credit
  - readouts: fetch, and gate_effectiveness() -- the headline measurement

No decision logic, no market math beyond option intrinsic value at settlement.
NOT financial advice.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

# Canonical column order. decision_engine.TradeDecision.as_row() must produce
# exactly these keys (minus the settlement columns, filled later).
COLUMNS = [
    "session_date", "ts", "spot",
    "net_gex", "gex_regime", "gex_pct_rank",
    "zero_gamma_dist", "zero_gamma_dist_pct", "adx",
    "call_wall", "put_wall",
    "selected_family", "short_strikes", "long_strikes", "legs_json", "credit",
    "candidate_score", "ev", "max_loss", "ev_per_risk", "theta", "gamma",
    "prob_profit", "prob_touch_short",
    "liquidity_score", "wall_safety", "gamma_safety", "touch_safety",
    "gate_pass", "gate_score", "gate_failed", "veto_reasons",
    "decision", "no_trade_reason", "was_traded", "candidate_present",
    "regime_direction",     # Track B direction: "call"|"put"|"both"|"none"
]

_SETTLE_COLUMNS = ["settle_price", "realized_pnl", "ev_error", "settled"]


def _coltype(col: str) -> str:
    if col in ("session_date", "ts", "gex_regime", "selected_family",
               "short_strikes", "long_strikes", "legs_json",
               "gate_failed", "veto_reasons", "decision", "no_trade_reason",
               "regime_direction"):
        return "TEXT"
    if col in ("gate_pass", "was_traded", "candidate_present"):
        return "INTEGER"
    return "REAL"


_CREATE = f"""
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(c + " " + _coltype(c) for c in COLUMNS)},
    settle_price REAL, realized_pnl REAL, ev_error REAL,
    settled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_session ON evaluations(session_date);
CREATE INDEX IF NOT EXISTS ix_settled ON evaluations(settled);
"""


def _intrinsic(strike: float, kind: str, S: float) -> float:
    return max(S - strike, 0.0) if kind == "C" else max(strike - S, 0.0)


def realized_pnl(legs: list[dict], credit: float, settle_price: float) -> float:
    """P&L of holding the structure to settlement: credit + sum(qty * intrinsic)."""
    total = float(credit)
    for lg in legs:
        total += lg["qty"] * _intrinsic(lg["strike"], lg["kind"], settle_price)
    return total


@dataclass
class Journal:
    db_path: str = "zerodte_journal.sqlite"

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_CREATE)
        self.conn.commit()

    # ---- write ----
    def log(self, row: dict) -> int:
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"row missing columns: {missing}")
        placeholders = ", ".join("?" for _ in COLUMNS)
        sql = f"INSERT INTO evaluations ({', '.join(COLUMNS)}) VALUES ({placeholders})"
        cur = self.conn.execute(sql, [row[c] for c in COLUMNS])
        self.conn.commit()
        return cur.lastrowid

    # ---- settle ----
    def settle_session(self, session_date: str, settle_price: float) -> int:
        """
        Fill settlement for every unsettled row of the session. Realized P&L is
        computed for ALL candidates that have a structure stored -- the
        hypothetical outcome of no-trades is exactly what makes the gate
        measurable. Returns count settled.
        """
        rows = self.conn.execute(
            "SELECT id, legs_json, credit, ev FROM evaluations "
            "WHERE session_date=? AND settled=0",
            (session_date,),
        ).fetchall()
        n = 0
        for r in rows:
            legs = json.loads(r["legs_json"]) if r["legs_json"] else []
            if legs:
                pnl = realized_pnl(legs, r["credit"], settle_price)
                ev_err = pnl - r["ev"] if r["ev"] is not None else None
            else:
                pnl, ev_err = None, None
            self.conn.execute(
                "UPDATE evaluations SET settle_price=?, realized_pnl=?, ev_error=?, settled=1 "
                "WHERE id=?",
                (settle_price, pnl, ev_err, r["id"]),
            )
            n += 1
        self.conn.commit()
        return n

    # ---- read ----
    def fetch(self, session_date: Optional[str] = None, settled_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM evaluations"
        clauses, args = [], []
        if session_date:
            clauses.append("session_date=?")
            args.append(session_date)
        if settled_only:
            clauses.append("settled=1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def unsettled_dates(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_date FROM evaluations WHERE settled=0 ORDER BY session_date"
        ).fetchall()
        return [r["session_date"] for r in rows]

    # ---- the headline measurement ----
    def gate_effectiveness(self) -> dict:
        """
        Did the gates save money or filter winners? Compares realized P&L of:
          - trades actually taken
          - would-be trades blocked ONLY by the gate (selector had a candidate,
            gate said no): these have a hypothetical realized_pnl.
        If the blocked set has WORSE mean P&L than trades, the gate is earning
        its keep. If BETTER, the gate is costing you edge.
        """
        rows = [r for r in self.fetch(settled_only=True) if r["realized_pnl"] is not None]
        taken = [r["realized_pnl"] for r in rows if r["was_traded"] == 1]
        blocked_by_gate = [
            r["realized_pnl"] for r in rows
            if r["was_traded"] == 0 and r["candidate_present"] == 1 and r["gate_pass"] == 0
        ]

        def stats(xs):
            if not xs:
                return {"n": 0, "mean": None, "total": None, "win_rate": None}
            wins = sum(1 for x in xs if x > 0)
            return {"n": len(xs), "mean": round(sum(xs) / len(xs), 4),
                    "total": round(sum(xs), 4), "win_rate": round(wins / len(xs), 3)}

        t, b = stats(taken), stats(blocked_by_gate)
        verdict = "insufficient data"
        if t["mean"] is not None and b["mean"] is not None:
            verdict = ("gate is EARNING its keep (blocked trades worse than taken)"
                       if b["mean"] < t["mean"]
                       else "gate may be COSTING edge (blocked trades better than taken)")
        return {"trades_taken": t, "blocked_by_gate": b, "verdict": verdict}

    def component_correlations(self) -> dict:
        """Pearson corr of each score component vs realized P&L over settled rows
        that have a structure. Cheap regression-precursor; no pandas needed."""
        rows = [r for r in self.fetch(settled_only=True) if r["realized_pnl"] is not None]
        if len(rows) < 3:
            return {"n": len(rows), "note": "need >=3 settled rows"}
        comps = ["candidate_score", "ev", "ev_per_risk", "wall_safety",
                 "gamma_safety", "touch_safety", "gate_score", "prob_profit"]
        y = [r["realized_pnl"] for r in rows]

        def corr(xs, ys):
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            vx = sum((a - mx) ** 2 for a in xs)
            vy = sum((b - my) ** 2 for b in ys)
            return round(cov / (vx * vy) ** 0.5, 3) if vx > 0 and vy > 0 else None

        out = {"n": len(rows)}
        for c in comps:
            xs = [r[c] for r in rows if r[c] is not None]
            if len(xs) == len(y):
                out[c] = corr(xs, y)
        return out

    def regime_diversity(self) -> dict:
        """
        Distribution of gex_regime across settled trades that were actually
        taken. A track record concentrated in one regime hasn't been tested by
        the conditions it will eventually meet live.
        """
        rows = [
            r for r in self.fetch(settled_only=True)
            if r["was_traded"] == 1 and r["realized_pnl"] is not None
        ]
        counts: dict[str, int] = {}
        for r in rows:
            key = r["gex_regime"] or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return {"n": len(rows), "regimes": counts, "distinct": len(counts)}

    def uptime_gaps(self, gap_threshold_sec: float = 300.0) -> dict:
        """
        Count intraday gaps between consecutive ticks (within the same
        session_date only -- the expected overnight/weekend gap between one
        session's last tick and the next session's first tick is excluded).
        A large intraday gap means the pipeline stalled or crashed mid-session.
        """
        rows = self.fetch()
        by_session: dict[str, list] = {}
        for r in rows:
            by_session.setdefault(r["session_date"], []).append(r["ts"])

        gaps = 0
        max_gap = 0.0
        sessions_with_gaps = 0
        for _, ts_list in by_session.items():
            times = sorted(dt.datetime.fromisoformat(t) for t in ts_list if t)
            session_had_gap = False
            for a, b in zip(times, times[1:]):
                delta = (b - a).total_seconds()
                if delta > gap_threshold_sec:
                    gaps += 1
                    session_had_gap = True
                    max_gap = max(max_gap, delta)
            if session_had_gap:
                sessions_with_gaps += 1

        return {
            "sessions": len(by_session),
            "sessions_with_gaps": sessions_with_gaps,
            "gap_count": gaps,
            "max_gap_sec": round(max_gap, 1),
            "gap_threshold_sec": gap_threshold_sec,
        }

    def close(self):
        self.conn.close()
