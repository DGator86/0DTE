"""
dashboard/queries.py
====================
Read-only database queries for the observability dashboard.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from dashboard.state import read_live_state
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
            for key in ("gate_failed", "veto_reasons", "short_strikes", "long_strikes",
                        "legs_json", "signals_json"):
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


def ras_history(db_path: str, position_id: Optional[str] = None,
                session_date: Optional[str] = None, limit: int = 500) -> list[dict]:
    """RAS evaluation history (newest last) from journal.ras_evaluations,
    with the component breakdown decoded per row. Read-only."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        sql = "SELECT * FROM ras_evaluations"
        clauses, args = [], []
        if position_id:
            clauses.append("position_id = ?")
            args.append(position_id)
        if session_date:
            clauses.append("session_date = ?")
            args.append(session_date)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []                        # table absent on pre-RAS journals
    finally:
        conn.close()

    out = []
    for r in reversed(rows):
        d = dict(r)
        try:
            d["components"] = json.loads(d.pop("components_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["components"] = []
        out.append(d)
    return out


def validation_reports(db_path: str, report_type: Optional[str] = None,
                       limit: int = 50) -> list[dict]:
    """Validation report history (newest first) from journal.validation_reports,
    with metrics/flags JSON decoded per row. Read-only; degrades to [] on
    legacy databases without the table."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        sql = "SELECT * FROM validation_reports"
        args: list = []
        if report_type:
            sql += " WHERE report_type = ?"
            args.append(report_type)
        sql += " ORDER BY report_date DESC, id DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []                        # table absent on pre-validation journals
    finally:
        conn.close()

    out = []
    for r in rows:
        d = dict(r)
        try:
            d["metrics"] = json.loads(d.pop("metrics_json") or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            d["metrics"] = {}
        try:
            d["flags"] = json.loads(d.pop("flags_json") or "[]") or []
        except (json.JSONDecodeError, TypeError):
            d["flags"] = []
        out.append(d)
    return out


def validation_report_by_id(db_path: str, report_id: int) -> Optional[dict]:
    """Single validation report with full decoded metrics, or None."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT * FROM validation_reports WHERE id = ?", (report_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    d = dict(row)
    try:
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}") or {}
    except (json.JSONDecodeError, TypeError):
        d["metrics"] = {}
    try:
        d["flags"] = json.loads(d.pop("flags_json") or "[]") or []
    except (json.JSONDecodeError, TypeError):
        d["flags"] = []
    return d


# --------------------------------------------------------------------------- #
# Adaptive-learning readouts (Learning tab). All read-only; every function    #
# degrades to [] on legacy databases without the tables.                      #
# --------------------------------------------------------------------------- #
def _ro_rows(db_path: str, sql: str, args: tuple = ()) -> list:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        return conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _decode(d: dict, mapping: dict[str, str]) -> dict:
    for src, dest in mapping.items():
        try:
            d[dest] = json.loads(d.pop(src) or "null")
        except (json.JSONDecodeError, TypeError):
            d[dest] = None
    return d


def learning_runs(db_path: str, limit: int = 50) -> list[dict]:
    """Learning-cycle history (newest first) with diagnostics decoded."""
    rows = _ro_rows(db_path,
                    "SELECT * FROM learning_runs ORDER BY id DESC LIMIT ?",
                    (limit,))
    return [_decode(dict(r), {"diagnostics_json": "diagnostics",
                              "param_space_json": "param_space",
                              "trials_json": "trials"}) for r in rows]


def candidate_configs(db_path: str, status: Optional[str] = None,
                      limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM candidate_configs"
    args: list = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = _ro_rows(db_path, sql, tuple(args))
    return [_decode(dict(r), {"overrides_json": "overrides",
                              "metrics_json": "metrics"}) for r in rows]


def promotions(db_path: str, status: Optional[str] = None,
               limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM promotions"
    args: list = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = _ro_rows(db_path, sql, tuple(args))
    return [_decode(dict(r), {"decision_json": "decision"}) for r in rows]


def feature_scores(db_path: str, latest_only: bool = True,
                   limit: int = 500) -> list[dict]:
    rows = _ro_rows(db_path,
                    "SELECT * FROM feature_scores ORDER BY id DESC LIMIT ?",
                    (limit,))
    out = [_decode(dict(r), {"details_json": "details"}) for r in rows]
    if latest_only:
        seen: set = set()
        latest = []
        for r in out:                        # newest first
            if r["feature"] not in seen:
                seen.add(r["feature"])
                latest.append(r)
        out = latest
    return out


def report_summary(db_path: str) -> dict:
    jrn = Journal(db_path)
    try:
        return {
            "gate_effectiveness": jrn.gate_effectiveness(),
            "component_correlations": jrn.component_correlations(),
            "calibration": jrn.calibration(),
            "decision_funnel": jrn.decision_funnel(),
            "gex_variant_comparison": jrn.gex_variant_comparison(),
            "unsettled_dates": jrn.unsettled_dates(),
        }
    finally:
        jrn.close()


def gex_variant_summary(db_path: str,
                        session_date: Optional[str] = None) -> dict:
    """PR 9 readout — Journal.gex_variant_comparison over HTTP."""
    if not _db_exists(db_path):
        return {"note": "journal database not found"}
    jrn = Journal(db_path)
    try:
        return jrn.gex_variant_comparison(session_date=session_date)
    finally:
        jrn.close()


def _db_exists(path: str) -> bool:
    import os
    return bool(path) and os.path.isfile(path)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def fetch_prediction_for_snapshot(
    snapshot_id: str,
    *,
    prediction_db: str = "",
    journal_db: str = "",
) -> dict:
    """
    Load the latest PredictionBundle row for a journal snapshot_id
    (PR 4+ / prediction_outputs). Tries prediction_db first, then the
    journal DB if it hosts the same table. Read-only; never creates tables.
    """
    if not snapshot_id:
        return {"note": "snapshot_id required", "prediction": None}

    candidates = []
    if prediction_db:
        candidates.append(prediction_db)
    if journal_db and journal_db not in candidates:
        candidates.append(journal_db)

    for path in candidates:
        if not _db_exists(path):
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            continue
        try:
            if not _table_exists(conn, "prediction_outputs"):
                continue
            row = conn.execute(
                "SELECT * FROM prediction_outputs WHERE snapshot_id=? "
                "ORDER BY id DESC LIMIT 1",
                (snapshot_id,),
            ).fetchone()
            if row is None:
                continue
            d = dict(row)
            try:
                preds = json.loads(d.pop("predictions_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                preds = {}
            d["predictions"] = preds if isinstance(preds, dict) else {}
            d["source_db"] = path
            return {"snapshot_id": snapshot_id, "prediction": d}
        finally:
            conn.close()

    return {
        "snapshot_id": snapshot_id,
        "prediction": None,
        "note": "no prediction_outputs row for snapshot",
    }


def fetch_sigma_cone_journal(
    *,
    prediction_db: str = "",
    session_date: Optional[str] = None,
    timeframe: Optional[str] = None,
    settled: Optional[bool] = None,
    limit: int = 200,
) -> dict:
    """
    Read MTF sigma-cone journal + coverage stats from PredictionStore.
    Degrades gracefully when the DB / table is missing.
    """
    if not prediction_db or not _db_exists(prediction_db):
        return {
            "rows": [],
            "coverage": {"n_settled": 0, "hit_rate": None, "by_sigma": {}},
            "note": "prediction database not found",
        }
    try:
        from prediction.storage import PredictionStore
        store = PredictionStore(db_path=prediction_db)
        rows = store.fetch_sigma_cones(
            session_date=session_date, timeframe=timeframe,
            settled=settled, limit=limit)
        coverage = store.sigma_cone_coverage(session_date=session_date)
        store.conn.close()
        return {"rows": rows, "coverage": coverage, "note": None}
    except Exception as exc:
        return {
            "rows": [],
            "coverage": {"n_settled": 0, "hit_rate": None, "by_sigma": {}},
            "note": f"sigma cone journal unavailable: {exc}",
        }


def paper_trades_journal(paper_db_path: str, live_state_path: str = "",
                          limit: int = 200) -> dict:
    """Trade journal: open positions (from live_state, marked every tick) plus
    closed paper trades with entry context (why) and exit reason (how it ended)."""
    open_positions: list = []
    if live_state_path:
        try:
            state = read_live_state(live_state_path) or {}
            open_positions = (state.get("paper") or {}).get("open") or []
        except Exception:
            open_positions = []

    closed: list = []
    try:
        conn = sqlite3.connect(f"file:{paper_db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {"open": open_positions, "closed": [], "note": "paper database unavailable"}
    try:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY closed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    except sqlite3.Error:
        return {"open": open_positions, "closed": [], "note": "paper_trades table not found"}
    finally:
        conn.close()

    for r in rows:
        d = dict(r)
        if d.get("entry_ctx"):
            try:
                d["entry_ctx"] = json.loads(d["entry_ctx"])
            except (json.JSONDecodeError, TypeError):
                d["entry_ctx"] = None
        closed.append(d)
    return {"open": open_positions, "closed": closed}


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
    # -- predictive-power gates: the system must PREDICT before it sizes up --
    "min_directional_n": 100,     # resolved-bias ticks needed before judging direction
    "min_directional_hit": 0.52,  # bias must beat a coin by a spread-covering margin
    "min_calibration_n": 30,      # settled candidates needed before judging probabilities
    "min_brier_skill": 0.0,       # prob_profit must beat always-quoting-the-base-rate
    "max_abs_ev_bias": 0.10,      # |mean ev_error| in $/share; EV must be unbiased-ish
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
        cal = jrn.calibration()
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

    # -- predictive-power checks: profitability without prediction is luck. --
    # These use every settled tick (no-trades included), so they resolve fast.
    dir_all = cal["directional"]["overall"]
    checks.append(check(
        "Directional edge present",
        dir_all["n"] >= cfg["min_directional_n"]
        and dir_all["hit_rate"] is not None
        and dir_all["hit_rate"] >= cfg["min_directional_hit"],
        dir_all,
        f">= {cfg['min_directional_hit']:.0%} hit rate over "
        f">= {cfg['min_directional_n']} resolved-bias ticks",
    ))

    pp = cal["prob_profit"]
    checks.append(check(
        "Probabilities calibrated",
        pp.get("n", 0) >= cfg["min_calibration_n"]
        and pp.get("brier_skill") is not None
        and pp["brier_skill"] >= cfg["min_brier_skill"],
        {k: pp.get(k) for k in ("n", "brier", "brier_skill", "base_rate")},
        f"Brier skill >= {cfg['min_brier_skill']} over >= {cfg['min_calibration_n']} candidates",
    ))

    ev = cal["ev"]
    checks.append(check(
        "EV unbiased",
        ev.get("n", 0) >= cfg["min_calibration_n"]
        and ev.get("mean_ev_error") is not None
        and abs(ev["mean_ev_error"]) <= cfg["max_abs_ev_bias"],
        ev,
        f"|mean ev_error| <= ${cfg['max_abs_ev_bias']:.2f}/share over "
        f">= {cfg['min_calibration_n']} settled candidates",
    ))

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
            "calibration": cal,
        },
    }
