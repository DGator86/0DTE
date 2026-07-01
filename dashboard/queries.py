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

    # Max drawdown over the full equity curve (all_rows is chronological by
    # closed_at): largest peak-to-trough drop, in dollars and as a fraction
    # of the peak at that point.
    max_dd_dollars, max_dd_frac, peak = 0.0, 0.0, None
    for r in all_rows:
        e = r[2]
        if e is None:
            continue
        if peak is None or e > peak:
            peak = e
        if peak:
            dd_dollars = peak - e
            dd_frac = dd_dollars / peak if peak else 0.0
            max_dd_dollars = max(max_dd_dollars, dd_dollars)
            max_dd_frac = max(max_dd_frac, dd_frac)

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
        "max_drawdown": round(max_dd_dollars, 2),
        "max_drawdown_pct": round(max_dd_frac, 4),
        "by_exit_reason": by_reason,
        "recent_trades": [dict(r) for r in rows],
        "simulated": True,
    }


# --------------------------------------------------------------------------- #
# Live-readiness checklist -- objective, numbers-based criteria for whether   #
# the paper/shadow track record has earned the right to touch real capital.  #
# Every threshold below is a starting policy, not a guarantee; tune to taste. #
# --------------------------------------------------------------------------- #
READINESS_THRESHOLDS = {
    "min_trades_taken": 30,       # sample size floor for trades_taken.n
    "min_profit_factor": 1.3,     # gross win / gross loss, with margin for real slippage
    "max_drawdown_pct": 0.20,     # largest peak-to-trough drop must stay under this
    "min_distinct_regimes": 2,    # track record must span more than one gex_regime
    "max_gap_sessions_frac": 0.05,  # share of sessions allowed an intraday pipeline gap
}


def readiness_summary(db_path: str, paper_db_path: str,
                       thresholds: Optional[dict] = None) -> dict:
    """Combine gate effectiveness, regime diversity, uptime, and paper P&L
    into a single pass/fail checklist for graduating out of shadow/paper mode."""
    cfg = {**READINESS_THRESHOLDS, **(thresholds or {})}

    jrn = Journal(db_path)
    try:
        gate_eff = jrn.gate_effectiveness()
        corr = jrn.component_correlations()
        regime = jrn.regime_diversity()
        uptime = jrn.uptime_gaps()
    finally:
        jrn.close()

    paper = paper_summary(paper_db_path)

    taken_n = gate_eff["trades_taken"]["n"]
    taken_mean = gate_eff["trades_taken"]["mean"]
    blocked_mean = gate_eff["blocked_by_gate"]["mean"]
    profit_factor = paper.get("profit_factor")
    max_dd_pct = paper.get("max_drawdown_pct")
    distinct_regimes = regime["distinct"]
    gap_frac = (uptime["sessions_with_gaps"] / uptime["sessions"]) if uptime["sessions"] else None

    def check(label, ok, actual, target):
        return {"label": label, "ok": bool(ok), "actual": actual, "target": target}

    checks = [
        check("Sample size", taken_n >= cfg["min_trades_taken"],
              taken_n, f">= {cfg['min_trades_taken']} trades taken"),
        check("Gate adds value",
              taken_mean is not None and blocked_mean is not None and blocked_mean < taken_mean,
              {"taken_mean": taken_mean, "blocked_mean": blocked_mean}, "blocked mean < taken mean"),
        check("Profit factor", profit_factor is not None and profit_factor >= cfg["min_profit_factor"],
              profit_factor, f">= {cfg['min_profit_factor']}"),
        check("Drawdown survivable",
              max_dd_pct is not None and max_dd_pct <= cfg["max_drawdown_pct"],
              max_dd_pct, f"<= {cfg['max_drawdown_pct']:.0%}"),
        check("Regime diversity", distinct_regimes >= cfg["min_distinct_regimes"],
              distinct_regimes, f">= {cfg['min_distinct_regimes']} distinct regimes"),
        check("Infrastructure held up",
              gap_frac is not None and gap_frac <= cfg["max_gap_sessions_frac"],
              uptime, f"<= {cfg['max_gap_sessions_frac']:.0%} of sessions with a gap"),
    ]
    ready = all(c["ok"] for c in checks)

    return {
        "ready": ready,
        "checks": checks,
        "thresholds": cfg,
        "facts": {
            "gate_effectiveness": gate_eff,
            "component_correlations": corr,
            "regime_diversity": regime,
            "uptime": uptime,
            "paper": paper,
        },
    }
