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
    "signals_json",         # JSON dict of observation-only signals (see below)
]

# signals_json is the admission channel for NEW signal domains (dealer
# dynamics, options flow, breadth, ...). A candidate signal is journaled here
# on every tick — with NO gate or veto power — until component_correlations()
# and the recorded walk-forward show it predicts realized P&L. Only then does
# it earn a matrix weight or veto. Flexible JSON so adding a signal never
# requires a schema migration.

_SETTLE_COLUMNS = ["settle_price", "realized_pnl", "ev_error", "settled"]


def _coltype(col: str) -> str:
    if col in ("session_date", "ts", "gex_regime", "selected_family",
               "short_strikes", "long_strikes", "legs_json",
               "gate_failed", "veto_reasons", "decision", "no_trade_reason",
               "regime_direction", "signals_json"):
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

CREATE TABLE IF NOT EXISTS ras_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date TEXT, ts TEXT, position_id TEXT,
    score REAL, ema_score REAL, action TEXT,
    components_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_ras_position ON ras_evaluations(position_id);
CREATE INDEX IF NOT EXISTS ix_ras_session ON ras_evaluations(session_date);
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
        # migration: signals_json added after live DBs already existed
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        if "signals_json" not in cols:
            self.conn.execute("ALTER TABLE evaluations ADD COLUMN signals_json TEXT")
        self.conn.commit()

    # ---- write ----
    def log(self, row: dict) -> int:
        row = dict(row)
        row.setdefault("signals_json", None)     # optional; older callers omit it
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"row missing columns: {missing}")
        placeholders = ", ".join("?" for _ in COLUMNS)
        sql = f"INSERT INTO evaluations ({', '.join(COLUMNS)}) VALUES ({placeholders})"
        cur = self.conn.execute(sql, [row[c] for c in COLUMNS])
        self.conn.commit()
        return cur.lastrowid

    def log_ras(self, ts: str, session_date: str, ras) -> int:
        """
        Record one Regime Alignment Score evaluation for an open position.
        `ras` is a regime_alignment.RASResult (duck-typed: position_id, score,
        ema_score, action, components with name/raw/weight/contribution/note).
        Full component breakdown is stored so every score move is explainable
        after the fact — the observability contract of paper-trading RAS.
        """
        components = [
            {"name": c.name, "raw": c.raw, "weight": c.weight,
             "contribution": c.contribution, "note": c.note}
            for c in (ras.components or [])
        ]
        cur = self.conn.execute(
            "INSERT INTO ras_evaluations "
            "(session_date, ts, position_id, score, ema_score, action, components_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_date, ts, ras.position_id, ras.score, ras.ema_score,
             ras.action, json.dumps(components)),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_ras(self, position_id: Optional[str] = None,
                  session_date: Optional[str] = None) -> list[dict]:
        """RAS evaluation history, optionally filtered by position or session.
        components_json is decoded into a `components` list per row."""
        sql = "SELECT * FROM ras_evaluations"
        clauses, args = [], []
        if position_id:
            clauses.append("position_id=?")
            args.append(position_id)
        if session_date:
            clauses.append("session_date=?")
            args.append(session_date)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["components"] = json.loads(row.pop("components_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["components"] = []
            out.append(row)
        return out

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

        # observation-only signals (signals_json): score each numeric key the
        # same way — this is how a candidate signal EARNS gate/veto power
        sig_rows = []
        for r in rows:
            try:
                sig = json.loads(r["signals_json"]) if r.get("signals_json") else None
            except (json.JSONDecodeError, TypeError):
                sig = None
            sig_rows.append(sig if isinstance(sig, dict) else {})
        keys = sorted({k for s in sig_rows for k in s
                       if isinstance(s[k], (int, float))})
        for k in keys:
            pairs = [(s[k], yy) for s, yy in zip(sig_rows, y)
                     if isinstance(s.get(k), (int, float))]
            if len(pairs) >= 3:
                out[f"sig:{k}"] = corr([p for p, _ in pairs], [p for _, p in pairs])
        return out

    # ---- predictive-power readouts -------------------------------------------
    # These answer "does the system predict forward movement?" — distinct from
    # gate_effectiveness ("did the gate filter well?"). All use data the journal
    # already collects on EVERY tick, so the sample builds fast: the direction
    # bias is scored even on no-trade ticks.

    def directional_accuracy(self) -> dict:
        """
        Hit rate of the regime direction bias vs the realized move to settlement,
        over ALL settled ticks whose bias resolved ("call"/"put") — trades and
        no-trades alike. A flat close counts as a miss (you paid theta/spread to
        be there). This is the sample that tells you within days, not weeks,
        whether the directional engine's premise holds.
        """
        rows = [r for r in self.fetch(settled_only=True)
                if r["regime_direction"] in ("call", "put")
                and r["spot"] is not None and r["settle_price"] is not None]

        def stats(rs):
            if not rs:
                return {"n": 0, "hit_rate": None, "avg_fwd_move_pct": None}
            hits = 0
            moves = []
            for r in rs:
                move = (r["settle_price"] - r["spot"]) / r["spot"]
                signed = move if r["regime_direction"] == "call" else -move
                moves.append(signed)
                if signed > 0:
                    hits += 1
            return {"n": len(rs), "hit_rate": round(hits / len(rs), 4),
                    "avg_fwd_move_pct": round(sum(moves) / len(moves) * 100, 4)}

        overall = stats(rows)
        return {
            "overall": overall,
            "by_direction": {
                d: stats([r for r in rows if r["regime_direction"] == d])
                for d in ("call", "put")
            },
            "traded_only": stats([r for r in rows if r["was_traded"] == 1]),
            "note": ("hit = settlement moved in the bias direction; "
                     "avg_fwd_move_pct is the bias-signed mean move (edge in %, "
                     "positive means the bias points the right way on average)"),
        }

    def prob_calibration(self, n_bins: int = 5) -> dict:
        """
        Is prob_profit an honest probability? Brier score over settled rows with
        a candidate, a Brier SKILL score vs always-guessing-the-base-rate
        (positive = the model's probabilities carry information; <= 0 = you
        could do as well quoting one constant), and a reliability table.
        """
        rows = [r for r in self.fetch(settled_only=True)
                if r["prob_profit"] is not None and r["realized_pnl"] is not None]
        if not rows:
            return {"n": 0, "note": "no settled rows with prob_profit"}

        pairs = [(float(r["prob_profit"]), 1.0 if r["realized_pnl"] > 0 else 0.0)
                 for r in rows]
        n = len(pairs)
        base = sum(w for _, w in pairs) / n
        brier = sum((p - w) ** 2 for p, w in pairs) / n
        brier_base = sum((base - w) ** 2 for _, w in pairs) / n
        skill = (1.0 - brier / brier_base) if brier_base > 0 else None

        bins = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            inb = [(p, w) for p, w in pairs
                   if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
            if inb:
                bins.append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": len(inb),
                    "mean_predicted": round(sum(p for p, _ in inb) / len(inb), 4),
                    "realized_rate": round(sum(w for _, w in inb) / len(inb), 4),
                })

        return {"n": n, "base_rate": round(base, 4), "brier": round(brier, 4),
                "brier_skill": round(skill, 4) if skill is not None else None,
                "bins": bins}

    def calibration(self) -> dict:
        """
        The readout mc.py promises: predicted quantities vs realized outcomes.
        Three panels — direction (does the bias point the right way), probability
        (is prob_profit honest), and EV (is the physical-density EV unbiased).
        If MC/selector says one thing and this says another, believe this.
        """
        rows = [r for r in self.fetch(settled_only=True)
                if r["ev"] is not None and r["ev_error"] is not None]
        ev_panel: dict = {"n": len(rows)}
        if rows:
            errs = [r["ev_error"] for r in rows]
            evs = [abs(r["ev"]) for r in rows]
            ev_panel.update({
                "mean_ev_error": round(sum(errs) / len(errs), 4),
                "mae_ev_error": round(sum(abs(e) for e in errs) / len(errs), 4),
                "mean_abs_ev": round(sum(evs) / len(evs), 4),
            })
        return {
            "directional": self.directional_accuracy(),
            "prob_profit": self.prob_calibration(),
            "ev": ev_panel,
        }

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
