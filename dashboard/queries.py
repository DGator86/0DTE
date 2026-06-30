"""
dashboard/queries.py
====================
Read-only database queries for the observability dashboard.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from journal import Journal


def journal_fetch(db_path: str, session_date: Optional[str] = None,
                  limit: int = 100, since_id: int = 0) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM evaluations WHERE id > ?"
        args: list = [since_id]
        if session_date:
            sql += " AND session_date = ?"
            args.append(session_date)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            for key in ("gate_failed", "veto_reasons", "short_strikes", "long_strikes", "legs_json"):
                if d.get(key) and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except json.JSONDecodeError:
                        pass
            out.append(d)
        return out
    finally:
        conn.close()


def journal_row(db_path: str, row_id: int) -> Optional[dict]:
    rows = journal_fetch(db_path, limit=1, since_id=row_id - 1)
    for r in rows:
        if r.get("id") == row_id:
            return r
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("gate_failed", "veto_reasons", "short_strikes", "long_strikes", "legs_json"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except json.JSONDecodeError:
                    pass
        return d
    finally:
        conn.close()


def journal_max_id(db_path: str) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(id) FROM evaluations").fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def report_summary(db_path: str) -> dict:
    jrn = Journal(db_path)
    try:
        return {
            "gate_effectiveness": jrn.gate_effectiveness(),
            "component_correlations": jrn.component_correlations(),
            "unsettled_dates": jrn.unsettled_dates(),
        }
    finally:
        jrn.close()


def paper_summary(paper_db_path: str) -> dict:
    if not paper_db_path:
        return {"note": "no paper database configured"}
    try:
        conn = sqlite3.connect(f"file:{paper_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {"note": "paper database unavailable"}

    try:
        rows = conn.execute(
            "SELECT id, opened_at, closed_at, family, contracts, pnl_dollars, "
            "exit_reason, equity_after FROM paper_trades ORDER BY closed_at DESC LIMIT 20"
        ).fetchall()
        all_rows = conn.execute(
            "SELECT pnl_dollars, exit_reason, equity_after FROM paper_trades ORDER BY closed_at"
        ).fetchall()
    except sqlite3.Error:
        return {"note": "paper_trades table not found"}
    finally:
        conn.close()

    pnls = [r[0] for r in all_rows if r[0] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    equity = all_rows[-1][2] if all_rows else None

    by_reason: dict[str, int] = {}
    for r in all_rows:
        reason = r[1] or "unknown"
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "trades": len(all_rows),
        "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
        "total_pnl": round(sum(pnls), 2) if pnls else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "equity": round(equity, 2) if equity is not None else None,
        "by_exit_reason": by_reason,
        "recent_trades": [dict(r) for r in rows],
        "simulated": True,
    }
