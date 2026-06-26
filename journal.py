"""
journal.py
==========
SQLite persistence for the 0DTE system.

Records every evaluation — trades AND no-trades — so settlement can fill
hypothetical P&L for blocked candidates and gate_effectiveness() can later
compare the two populations. That comparison IS the measurement thesis:
if the gate is doing its job, blocked days should have worse hypothetical
outcomes than the days it let through.

Entry points consumed by orchestrator.py:
  log(row: dict) -> int
  settle_session(date, close_price) -> SettlementResult
  gate_effectiveness(lookback_days) -> GateEffectiveness
  component_correlations(lookback_days) -> dict
"""
from __future__ import annotations

import sqlite3
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date    TEXT    NOT NULL,
    ts              TEXT    NOT NULL,
    spot            REAL,
    net_gex         REAL,
    gex_regime      TEXT,
    gex_pct_rank    REAL,
    zero_gamma_dist REAL,
    zero_gamma_dist_pct REAL,
    adx             REAL,
    call_wall       REAL,
    put_wall        REAL,
    selected_family TEXT,
    short_strikes   TEXT,
    long_strikes    TEXT,
    legs_json       TEXT,
    credit          REAL,
    candidate_score REAL,
    ev              REAL,
    max_loss        REAL,
    ev_per_risk     REAL,
    theta           REAL,
    gamma           REAL,
    prob_profit     REAL,
    prob_touch_short REAL,
    liquidity_score REAL,
    wall_safety     REAL,
    gamma_safety    REAL,
    touch_safety    REAL,
    gate_pass       INTEGER,
    gate_score      REAL,
    gate_failed     TEXT,
    veto_reasons    TEXT,
    decision        TEXT,
    no_trade_reason TEXT,
    was_traded      INTEGER DEFAULT 0,
    candidate_present INTEGER DEFAULT 0,
    close_price     REAL,
    realized_pnl    REAL,
    hypothetical_pnl REAL,
    settled         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ev_date     ON evaluations(session_date);
CREATE INDEX IF NOT EXISTS idx_ev_decision ON evaluations(decision);
CREATE INDEX IF NOT EXISTS idx_ev_settled  ON evaluations(settled);
"""


@dataclass
class SettlementResult:
    date: str
    n_trades: int
    n_no_trades: int
    trade_pnl: float
    blocked_pnl: float
    gate_helped: bool


@dataclass
class GateEffectiveness:
    n_trades: int
    n_gate_blocked: int
    n_selector_blocked: int
    avg_trade_pnl: float
    avg_gate_blocked_pnl: float
    avg_selector_blocked_pnl: float
    gate_net_contribution: float


class Journal:
    def __init__(self, db_path: str = "0dte_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(_CREATE_SQL)

    def log(self, row: dict) -> int:
        """Insert one evaluation row (from TradeDecision.as_row()). Returns row id."""
        cols = list(row.keys())
        ph = ", ".join("?" * len(cols))
        sql = f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({ph})"
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(sql, [row[c] for c in cols])
            return cur.lastrowid

    def settle_session(self, date: str, close_price: float) -> SettlementResult:
        """
        Post-close settlement. For each unsettled evaluation on `date`:
        - Traded: fills realized_pnl from spread payoff at close_price.
        - No-trade with candidate: fills hypothetical_pnl (what it would have returned).
        """
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM evaluations WHERE session_date=? AND settled=0", (date,)
            ).fetchall()

            n_trades = n_no_trades = 0
            trade_pnl = blocked_pnl = 0.0

            for r in rows:
                pnl = _settle_legs(r["legs_json"], r["credit"], close_price)
                if r["was_traded"] == 1:
                    con.execute(
                        "UPDATE evaluations SET close_price=?, realized_pnl=?, settled=1 WHERE id=?",
                        (close_price, pnl, r["id"]),
                    )
                    trade_pnl += pnl or 0.0
                    n_trades += 1
                else:
                    hypo = pnl if r["candidate_present"] == 1 else None
                    con.execute(
                        "UPDATE evaluations SET close_price=?, hypothetical_pnl=?, settled=1 WHERE id=?",
                        (close_price, hypo, r["id"]),
                    )
                    if hypo is not None:
                        blocked_pnl += hypo
                    n_no_trades += 1

        gate_helped = n_no_trades > 0 and blocked_pnl < 0.0
        return SettlementResult(date, n_trades, n_no_trades,
                                round(trade_pnl, 4), round(blocked_pnl, 4), gate_helped)

    def gate_effectiveness(self, lookback_days: int = 30) -> GateEffectiveness:
        """
        Compare realized P&L of trades vs hypothetical P&L of gate-blocked days.
        Positive gate_net_contribution means the gate filtered out worse outcomes.
        """
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM evaluations WHERE session_date >= ? AND settled=1", (cutoff,)
            ).fetchall()

        trades = [r for r in rows if r["was_traded"] == 1 and r["realized_pnl"] is not None]
        blocked_gate = [r for r in rows
                        if r["was_traded"] == 0 and r["gate_pass"] == 0
                        and r["hypothetical_pnl"] is not None]
        blocked_sel = [r for r in rows
                       if r["was_traded"] == 0 and r["gate_pass"] == 1
                       and r["hypothetical_pnl"] is not None]

        def _avg(vals: list) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        avg_trade = _avg([r["realized_pnl"] for r in trades])
        avg_gate = _avg([r["hypothetical_pnl"] for r in blocked_gate])
        avg_sel = _avg([r["hypothetical_pnl"] for r in blocked_sel])

        return GateEffectiveness(
            n_trades=len(trades),
            n_gate_blocked=len(blocked_gate),
            n_selector_blocked=len(blocked_sel),
            avg_trade_pnl=round(avg_trade, 4),
            avg_gate_blocked_pnl=round(avg_gate, 4),
            avg_selector_blocked_pnl=round(avg_sel, 4),
            gate_net_contribution=round(avg_trade - avg_gate, 4),
        )

    def component_correlations(self, lookback_days: int = 30) -> dict:
        """Gate score vs realized P&L Pearson correlation on settled trades."""
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT gate_score, realized_pnl FROM evaluations "
                "WHERE session_date >= ? AND settled=1 AND was_traded=1 "
                "AND gate_score IS NOT NULL AND realized_pnl IS NOT NULL",
                (cutoff,),
            ).fetchall()
        if len(rows) < 5:
            return {"error": f"only {len(rows)} settled trades"}
        scores = [r["gate_score"] for r in rows]
        pnls = [r["realized_pnl"] for r in rows]
        return {"gate_score_vs_pnl_pearson": round(_pearson(scores, pnls), 4), "n": len(rows)}


def _settle_legs(legs_json: Optional[str], credit: Optional[float],
                 close: float) -> Optional[float]:
    if not legs_json or credit is None:
        return None
    try:
        legs = json.loads(legs_json)
    except (json.JSONDecodeError, TypeError):
        return None
    pnl = float(credit)
    for lg in legs:
        K = float(lg["strike"])
        qty = int(lg["qty"])
        intrinsic = max(close - K, 0.0) if lg["kind"] == "C" else max(K - close, 0.0)
        pnl += qty * intrinsic
    return round(pnl, 4)


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0.0


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    import tempfile

    db = os.path.join(tempfile.mkdtemp(), "test_journal.db")
    j = Journal(db)
    today = dt.date.today().isoformat()

    j.log({
        "session_date": today, "ts": "2026-06-26T11:20:00-04:00",
        "spot": 602.50, "net_gex": 4.2e9, "gex_regime": "long",
        "gex_pct_rank": 0.88, "zero_gamma_dist": 6.5, "zero_gamma_dist_pct": 0.0108,
        "adx": 12.5, "call_wall": 603.0, "put_wall": 598.0,
        "selected_family": "put_credit",
        "short_strikes": "[598.0]", "long_strikes": "[597.0]",
        "legs_json": json.dumps([{"strike": 598.0, "kind": "P", "qty": -1},
                                  {"strike": 597.0, "kind": "P", "qty": 1}]),
        "credit": 0.30, "candidate_score": 0.012, "ev": 0.18, "max_loss": 0.70,
        "ev_per_risk": 0.257, "theta": 0.35, "gamma": -0.05,
        "prob_profit": 0.72, "prob_touch_short": 0.18,
        "liquidity_score": 0.85, "wall_safety": 0.92, "gamma_safety": 0.88,
        "touch_safety": 0.82,
        "gate_pass": 1, "gate_score": 76.5, "gate_failed": "[]", "veto_reasons": "[]",
        "decision": "TRADE", "no_trade_reason": "", "was_traded": 1, "candidate_present": 1,
    })
    j.log({
        "session_date": today, "ts": "2026-06-26T09:10:00-04:00",
        "spot": 588.0, "net_gex": -1.1e9, "gex_regime": "short",
        "gex_pct_rank": 0.40, "zero_gamma_dist": -5.0, "zero_gamma_dist_pct": -0.0085,
        "adx": 28.0, "call_wall": 596.0, "put_wall": 585.0,
        "selected_family": "put_credit",
        "short_strikes": "[585.0]", "long_strikes": "[584.0]",
        "legs_json": json.dumps([{"strike": 585.0, "kind": "P", "qty": -1},
                                  {"strike": 584.0, "kind": "P", "qty": 1}]),
        "credit": 0.25, "candidate_score": 0.006, "ev": 0.08, "max_loss": 0.75,
        "ev_per_risk": 0.107, "theta": 0.28, "gamma": -0.03,
        "prob_profit": 0.61, "prob_touch_short": 0.42,
        "liquidity_score": 0.70, "wall_safety": 0.55, "gamma_safety": 0.10,
        "touch_safety": 0.58,
        "gate_pass": 0, "gate_score": 0.0,
        "gate_failed": json.dumps(["GEX_SHORT", "TRENDING: ADX 28.0 >= 20"]),
        "veto_reasons": "[]",
        "decision": "NO_TRADE", "no_trade_reason": "gate:GEX_SHORT,TRENDING",
        "was_traded": 0, "candidate_present": 1,
    })

    res = j.settle_session(today, close_price=602.30)
    print(f"Settlement : {res}")
    eff = j.gate_effectiveness(lookback_days=7)
    print(f"Gate eff   : {eff}")
    print("journal OK")
