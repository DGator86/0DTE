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

# Optional provenance columns (Prediction Engine V2):
#   snapshot_id (PR 3) — links evaluations to feature_snapshots / prediction_outputs
#   execution_json / credit_expected / credit_conservative (PR 6) — executable
#     entry economics; mid `credit` remains the diagnostic. Settlement fills
#     realized_pnl_expected / realized_pnl_conservative when these are present.
OPTIONAL_COLUMNS = [
    "snapshot_id",
    "execution_json",
    "credit_expected",
    "credit_conservative",
]

# signals_json is the admission channel for NEW signal domains (dealer
# dynamics, options flow, breadth, ...). A candidate signal is journaled here
# on every tick — with NO gate or veto power — until component_correlations()
# and the recorded walk-forward show it predicts realized P&L. Only then does
# it earn a matrix weight or veto. Flexible JSON so adding a signal never
# requires a schema migration.
#
# Prediction Engine V2 / PR 8 shadow ranker keys (observation-only):
#   v2_rank_disagreement, v2_top_candidate_id, legacy_top_candidate_id,
#   v2_utility_score, v2_expected_net_pnl, v2_p_profit,
#   v2_expected_shortfall, candidate_model_version, v2_legacy_spearman,
#   v2_vs_legacy_pnl_delta, legacy_top_score
#
# Prediction Engine V2 / PR 9 GEX variant keys (observation-only, §16.3):
#   gex_oi_*, gex_weekly_*, gex_volume_*, gex_hybrid_*
#     (net_gex, gamma_flip, call_wall, put_wall, gex_concentration,
#      wall_concentration, quality_score, assumption, ...)
#   gex_disagree_flip_spread, gex_disagree_wall_call, gex_disagree_wall_put,
#   gex_disagree_sign, gex_disagree_net_gex_range, gex_disagree_n_variants
#   gex_authoritative, gex_feed_source
# Policy continues to use evaluations.net_gex / call_wall / put_wall (OI).

_SETTLE_COLUMNS = [
    "settle_price", "realized_pnl", "ev_error",
    "realized_pnl_expected", "realized_pnl_conservative", "settled",
]

# --------------------------------------------------------------------------- #
# Structure vocabulary for decision_funnel().                                  #
# selected_family holds spread_selector family names on engine rows but        #
# decision_matrix structure CODES on no-trade stub rows (unified_loop).        #
# These are LOCAL mirrors of spread_selector.STRUCTURE_TO_FAMILIES /           #
# DEBIT_FAMILIES so this module stays stdlib-only (importing spread_selector   #
# would pull scipy into the persistence layer). tests/test_funnel.py asserts   #
# they stay in sync with the source of truth.                                  #
# --------------------------------------------------------------------------- #
STRUCTURE_CODE_TO_FAMILY = {
    "IC": "iron_condor", "PCS": "put_credit", "CCS": "call_credit",
    "IF": "iron_fly", "LCS": "long_call_spread", "LPS": "long_put_spread",
    "LC": "long_call", "LP": "long_put", "STG": "long_strangle",
    "BKS": "backspread",
}
CREDIT_FAMILIES = frozenset({
    "put_credit", "call_credit", "iron_condor", "iron_fly", "broken_wing",
})
DEBIT_FAMILIES = frozenset({
    "long_call_spread", "long_put_spread", "long_call", "long_put",
    "long_strangle", "backspread", "backspread_call", "backspread_put",
})
UNDEFINED_RISK_FAMILIES = frozenset({"naked_defended_call", "cash_secured_put"})


def _coltype(col: str) -> str:
    if col in ("session_date", "ts", "gex_regime", "selected_family",
               "short_strikes", "long_strikes", "legs_json",
               "gate_failed", "veto_reasons", "decision", "no_trade_reason",
               "regime_direction", "signals_json", "snapshot_id",
               "execution_json"):
        return "TEXT"
    if col in ("gate_pass", "was_traded", "candidate_present"):
        return "INTEGER"
    return "REAL"


_CREATE = f"""
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(c + " " + _coltype(c) for c in COLUMNS + OPTIONAL_COLUMNS)},
    settle_price REAL, realized_pnl REAL, ev_error REAL,
    realized_pnl_expected REAL, realized_pnl_conservative REAL,
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

CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT NOT NULL,
    report_type TEXT NOT NULL,           -- 'daily' | 'weekly' | 'feature_impact'
                                         -- | 'drift' | 'promotion_candidate'
    generated_at TEXT,
    metrics_json TEXT,                   -- all key metrics (JSON)
    summary TEXT,                        -- human-readable summary
    flags_json TEXT,                     -- alerts / degradation flags (JSON list)
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_vr_date ON validation_reports(report_date);
CREATE INDEX IF NOT EXISTS ix_vr_type ON validation_reports(report_type);

-- Adaptive Learning Engine (adaptive_learning/): full audit trail of every
-- learning cycle, challenger config, promotion decision, and feature score.
-- All flexible payloads are JSON so new metrics never need a migration.
CREATE TABLE IF NOT EXISTS learning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    mode TEXT,                           -- 'daily' | 'weekly' | 'manual'
    started_at TEXT, finished_at TEXT,
    diagnostics_json TEXT,               -- list of Diagnosis dicts
    param_space_json TEXT,               -- searched space (dot-notation)
    n_trials INTEGER,
    best_score REAL, holdout_score REAL,
    trials_json TEXT,                    -- ranked trial summaries
    outcome TEXT,                        -- 'no_action' | 'candidate_generated'
                                         -- | 'promotion_recommended' | 'rejected'
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_lr_run ON learning_runs(run_id);

CREATE TABLE IF NOT EXISTS candidate_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id TEXT NOT NULL UNIQUE,
    created_at TEXT,
    parent_id TEXT,
    label TEXT,
    overrides_json TEXT,                 -- flat dot-notation overrides
    metrics_json TEXT,                   -- scores at creation time
    status TEXT NOT NULL DEFAULT 'candidate'
                                         -- candidate | pending_review
                                         -- | promoted | rejected | archived
);
CREATE INDEX IF NOT EXISTS ix_cc_status ON candidate_configs(status);

CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id TEXT NOT NULL,
    created_at TEXT,
    decision_json TEXT,                  -- rule-by-rule pass/fail breakdown
    status TEXT NOT NULL DEFAULT 'pending_review',
                                         -- pending_review | approved | rejected
    approved_by TEXT, approved_at TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_pr_config ON promotions(config_id);
CREATE INDEX IF NOT EXISTS ix_pr_status ON promotions(status);

CREATE TABLE IF NOT EXISTS feature_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature TEXT NOT NULL,
    as_of TEXT,
    n INTEGER,
    pearson REAL, spearman REAL, mutual_info REAL, perm_importance REAL,
    stability REAL,
    status TEXT NOT NULL DEFAULT 'observation',
                                         -- observation | experimental
                                         -- | candidate | production
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_fs_feature ON feature_scores(feature);
"""


def _intrinsic(strike: float, kind: str, S: float) -> float:
    return max(S - strike, 0.0) if kind == "C" else max(strike - S, 0.0)


def realized_pnl(legs: list[dict], credit: float, settle_price: float) -> float:
    """P&L of holding the structure to settlement: credit + sum(qty * intrinsic)."""
    total = float(credit)
    for lg in legs:
        total += lg["qty"] * _intrinsic(lg["strike"], lg["kind"], settle_price)
    return total


def economic_pnl(row: dict) -> Optional[float]:
    """
    Primary V2 economic P&L for a settled row (§13.5): prefer net expected-fill
    settlement when present, else fall back to midpoint realized_pnl.
    """
    for key in ("realized_pnl_expected", "realized_pnl"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


@dataclass
class Journal:
    db_path: str = "zerodte_journal.sqlite"

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_CREATE)
        # migrations: columns added after live DBs already existed
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        if "signals_json" not in cols:
            self.conn.execute("ALTER TABLE evaluations ADD COLUMN signals_json TEXT")
        if "snapshot_id" not in cols:
            self.conn.execute("ALTER TABLE evaluations ADD COLUMN snapshot_id TEXT")
        if "execution_json" not in cols:
            self.conn.execute("ALTER TABLE evaluations ADD COLUMN execution_json TEXT")
        if "credit_expected" not in cols:
            self.conn.execute("ALTER TABLE evaluations ADD COLUMN credit_expected REAL")
        if "credit_conservative" not in cols:
            self.conn.execute(
                "ALTER TABLE evaluations ADD COLUMN credit_conservative REAL")
        if "realized_pnl_expected" not in cols:
            self.conn.execute(
                "ALTER TABLE evaluations ADD COLUMN realized_pnl_expected REAL")
        if "realized_pnl_conservative" not in cols:
            self.conn.execute(
                "ALTER TABLE evaluations ADD COLUMN realized_pnl_conservative REAL")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_snapshot ON evaluations(snapshot_id)")
        self.conn.commit()

    # ---- write ----
    def log(self, row: dict) -> int:
        row = dict(row)
        row.setdefault("signals_json", None)     # optional; older callers omit it
        for c in OPTIONAL_COLUMNS:
            row.setdefault(c, None)
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            raise ValueError(f"row missing columns: {missing}")
        cols = COLUMNS + OPTIONAL_COLUMNS
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.execute(sql, [row[c] for c in cols])
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

    # ---- validation reports -------------------------------------------------
    # Persistent record of the scheduled validation pipeline (daily/weekly)
    # and the feature-impact workflow. One row per generated report; metrics
    # and flags are flexible JSON so new metrics never need a migration.
    def log_validation_report(self, report_date: str, report_type: str,
                              metrics: dict, summary: str,
                              flags: Optional[list] = None,
                              notes: Optional[str] = None) -> int:
        """
        Persist one validation report.
          report_date  — session/report date, YYYY-MM-DD
          report_type  — 'daily' | 'weekly' | 'feature_impact'
          metrics      — dict of all key metrics (JSON-serialized)
          summary      — human-readable summary text
          flags        — list of alert/degradation flags (each a dict or str)
          notes        — optional freeform notes
        """
        cur = self.conn.execute(
            "INSERT INTO validation_reports "
            "(report_date, report_type, generated_at, metrics_json, summary, "
            "flags_json, notes) VALUES (?,?,?,?,?,?,?)",
            (report_date, report_type,
             dt.datetime.now(dt.timezone.utc).isoformat(),
             json.dumps(metrics or {}), summary,
             json.dumps(flags or []), notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_validation_reports(self, report_type: Optional[str] = None,
                                 limit: int = 50,
                                 since: Optional[str] = None) -> list[dict]:
        """Validation report history, newest first. metrics_json/flags_json are
        decoded into `metrics` / `flags` per row."""
        sql = "SELECT * FROM validation_reports"
        clauses, args = [], []
        if report_type:
            clauses.append("report_type=?")
            args.append(report_type)
        if since:
            clauses.append("report_date>=?")
            args.append(since)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY report_date DESC, id DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (("metrics_json", "metrics"), ("flags_json", "flags")):
                try:
                    row[dest] = json.loads(row.pop(src) or "null") or ({} if dest == "metrics" else [])
                except (json.JSONDecodeError, TypeError):
                    row[dest] = {} if dest == "metrics" else []
            out.append(row)
        return out

    # ---- adaptive learning (audit trail) --------------------------------------
    # Persistence for the Adaptive Learning Engine: one row per learning cycle,
    # per challenger config, per promotion decision, and per feature score.
    # Same conventions as validation_reports: flexible JSON payloads, decoded
    # on fetch, newest first.
    @staticmethod
    def _decode_json_cols(row: dict, cols: dict[str, str]) -> dict:
        for src, dest in cols.items():
            try:
                row[dest] = json.loads(row.pop(src) or "null")
            except (json.JSONDecodeError, TypeError):
                row[dest] = None
        return row

    def log_learning_run(self, run_id: str, mode: str,
                         started_at: str, finished_at: str,
                         diagnostics: Optional[list] = None,
                         param_space: Optional[dict] = None,
                         n_trials: Optional[int] = None,
                         best_score: Optional[float] = None,
                         holdout_score: Optional[float] = None,
                         trials: Optional[list] = None,
                         outcome: str = "no_action",
                         notes: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO learning_runs (run_id, mode, started_at, finished_at, "
            "diagnostics_json, param_space_json, n_trials, best_score, "
            "holdout_score, trials_json, outcome, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, mode, started_at, finished_at,
             json.dumps(diagnostics or []), json.dumps(param_space or {}),
             n_trials, best_score, holdout_score,
             json.dumps(trials or []), outcome, notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_learning_runs(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM learning_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._decode_json_cols(dict(r), {
            "diagnostics_json": "diagnostics",
            "param_space_json": "param_space",
            "trials_json": "trials",
        }) for r in rows]

    def log_candidate_config(self, config_id: str, created_at: str,
                             overrides: dict,
                             parent_id: Optional[str] = None,
                             label: str = "",
                             metrics: Optional[dict] = None,
                             status: str = "candidate") -> int:
        cur = self.conn.execute(
            "INSERT INTO candidate_configs (config_id, created_at, parent_id, "
            "label, overrides_json, metrics_json, status) VALUES (?,?,?,?,?,?,?)",
            (config_id, created_at, parent_id, label,
             json.dumps(overrides or {}), json.dumps(metrics or {}), status),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_candidate_status(self, config_id: str, status: str) -> int:
        cur = self.conn.execute(
            "UPDATE candidate_configs SET status=? WHERE config_id=?",
            (status, config_id))
        self.conn.commit()
        return cur.rowcount

    def fetch_candidate_configs(self, status: Optional[str] = None,
                                limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM candidate_configs"
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._decode_json_cols(dict(r), {
            "overrides_json": "overrides",
            "metrics_json": "metrics",
        }) for r in rows]

    def log_promotion(self, config_id: str, decision: dict,
                      status: str = "pending_review",
                      notes: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO promotions (config_id, created_at, decision_json, "
            "status, notes) VALUES (?,?,?,?,?)",
            (config_id, dt.datetime.now(dt.timezone.utc).isoformat(),
             json.dumps(decision or {}), status, notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_promotion(self, config_id: str, status: str,
                         approved_by: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "UPDATE promotions SET status=?, approved_by=?, approved_at=? "
            "WHERE config_id=? AND status='pending_review'",
            (status, approved_by,
             dt.datetime.now(dt.timezone.utc).isoformat(), config_id))
        self.conn.commit()
        return cur.rowcount

    def fetch_promotions(self, status: Optional[str] = None,
                         limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM promotions"
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._decode_json_cols(dict(r), {"decision_json": "decision"})
                for r in rows]

    def log_feature_score(self, feature: str, as_of: str, n: int,
                          pearson: Optional[float] = None,
                          spearman: Optional[float] = None,
                          mutual_info: Optional[float] = None,
                          perm_importance: Optional[float] = None,
                          stability: Optional[float] = None,
                          status: str = "observation",
                          details: Optional[dict] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO feature_scores (feature, as_of, n, pearson, spearman, "
            "mutual_info, perm_importance, stability, status, details_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (feature, as_of, n, pearson, spearman, mutual_info,
             perm_importance, stability, status, json.dumps(details or {})),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_feature_scores(self, feature: Optional[str] = None,
                             latest_only: bool = False,
                             limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM feature_scores"
        args: list = []
        if feature:
            sql += " WHERE feature=?"
            args.append(feature)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        out = [self._decode_json_cols(dict(r), {"details_json": "details"})
               for r in rows]
        if latest_only:
            seen: set = set()
            latest = []
            for r in out:                      # newest first
                if r["feature"] not in seen:
                    seen.add(r["feature"])
                    latest.append(r)
            out = latest
        return out

    # ---- settle ----
    def settle_session(self, session_date: str, settle_price: float) -> int:
        """
        Fill settlement for every unsettled row of the session. Realized P&L is
        computed for ALL candidates that have a structure stored -- the
        hypothetical outcome of no-trades is exactly what makes the gate
        measurable.

        Midpoint `realized_pnl` is always filled (diagnostic). When the row
        carries PR 6 executable credits, `realized_pnl_expected` and
        `realized_pnl_conservative` are filled too — those are the primary
        V2 economic metrics (§13.5). Returns count settled.
        """
        rows = self.conn.execute(
            "SELECT id, legs_json, credit, credit_expected, credit_conservative, "
            "execution_json, ev FROM evaluations "
            "WHERE session_date=? AND settled=0",
            (session_date,),
        ).fetchall()
        n = 0
        for r in rows:
            legs = json.loads(r["legs_json"]) if r["legs_json"] else []
            pnl = pnl_exp = pnl_con = ev_err = None
            if legs:
                credit = r["credit"]
                if credit is not None:
                    pnl = realized_pnl(legs, credit, settle_price)
                    ev_err = (pnl - r["ev"]) if r["ev"] is not None else None
                # Executable settlement: apply entry credit + entry fees already
                # netted into credit_expected; subtract expected exit drag from
                # the stored execution panel when present.
                exec_ = {}
                if r["execution_json"]:
                    try:
                        exec_ = json.loads(r["execution_json"]) or {}
                    except (json.JSONDecodeError, TypeError):
                        exec_ = {}
                exit_drag = float(exec_.get("exit_slippage_expected") or 0.0) + float(
                    exec_.get("exit_fees_expected") or 0.0)
                if r["credit_expected"] is not None:
                    pnl_exp = (realized_pnl(legs, r["credit_expected"], settle_price)
                               - exit_drag)
                if r["credit_conservative"] is not None:
                    # Conservative exit: use stop-exit slippage when available.
                    stop_drag = float(exec_.get("exit_slippage_stop") or 0.0) + float(
                        exec_.get("exit_fees_expected") or 0.0)
                    pnl_con = (realized_pnl(legs, r["credit_conservative"],
                                            settle_price) - stop_drag)
            self.conn.execute(
                "UPDATE evaluations SET settle_price=?, realized_pnl=?, ev_error=?, "
                "realized_pnl_expected=?, realized_pnl_conservative=?, settled=1 "
                "WHERE id=?",
                (settle_price, pnl, ev_err, pnl_exp, pnl_con, r["id"]),
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

    def decision_funnel(self, session_date: Optional[str] = None,
                        last_sessions: Optional[int] = None,
                        gate_gex_floor: float = 0.60) -> dict:
        """
        Where do trades come from and where do they die? Aggregates the
        routing/gating funnel over journaled ticks so "why is premium (or
        directional) not trading?" is answerable from data instead of ad-hoc
        SQL:

          routed_structures  what Track B asked for, per tick
                             (signals_json.routed_structure; new rows only)
          structure_mix      final family per row: traded vs blocked
          class_mix          credit / debit / undefined-risk / stand-down
          no_trade_reasons   which layer said no (gate / selector / risk /
                             regime_nt / stand_down / no_chain)
          gate_failures      which hard gate fired (GEX_WEAK, TRENDING, ...)
          selector_vetoes    per-candidate selector vetoes (rows with a candidate)
          regime_vetoes      dealer-state vetoes (stand-down rows' veto_reasons
                             plus signals_json.regime_vetoes on engine rows)
          premium_flips      credit cells forced to a debit cousin by a dealer
                             veto (signals_json.premium_flip; new rows only)
          gex_rank           gex_pct_rank distribution: warm-up-neutral share
                             and share below the premium gate's floor
        """
        rows = self.fetch(session_date=session_date)
        if last_sessions:
            keep = set(sorted({r["session_date"] for r in rows})[-last_sessions:])
            rows = [r for r in rows if r["session_date"] in keep]

        def bump(d: dict, k: str, n: int = 1) -> None:
            d[k] = d.get(k, 0) + n

        def load_list(txt) -> list:
            try:
                v = json.loads(txt) if txt else []
                return v if isinstance(v, list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        structure_mix: dict = {}
        class_mix = {c: {"n": 0, "traded": 0}
                     for c in ("credit", "debit", "undefined_risk",
                               "stand_down", "other")}
        routed: dict = {}
        no_trade_reasons: dict = {}
        gate_failures: dict = {}
        selector_vetoes: dict = {}
        regime_vetoes: dict = {}
        flips = 0
        rows_with_provenance = 0
        gex_vals: list = []
        gex_warmup = 0

        for r in rows:
            fam_raw = r.get("selected_family")
            fam = STRUCTURE_CODE_TO_FAMILY.get(fam_raw, fam_raw)
            if fam is None:
                cls = "stand_down"
            elif fam in CREDIT_FAMILIES:
                cls = "credit"
            elif fam in DEBIT_FAMILIES:
                cls = "debit"
            elif fam in UNDEFINED_RISK_FAMILIES:
                cls = "undefined_risk"
            else:
                cls = "other"
            traded = r.get("was_traded") == 1
            if fam is not None:
                slot = structure_mix.setdefault(fam, {"n": 0, "traded": 0, "blocked": 0})
                slot["n"] += 1
                slot["traded" if traded else "blocked"] += 1
            class_mix[cls]["n"] += 1
            if traded:
                class_mix[cls]["traded"] += 1

            # "gate:GEX_WEAK,TRENDING | selector:no positive-EV structure",
            # "risk:daily_stop", "regime_nt", "stand_down:compression", "no_chain"
            if not traded and r.get("no_trade_reason"):
                for part in str(r["no_trade_reason"]).split(" | "):
                    key = part.split(":", 1)[0].strip()
                    if key:
                        bump(no_trade_reasons, key)

            # Real hard-gate names are UPPERCASE (GEX_WEAK, TRENDING, ...);
            # no-trade stub rows reuse gate_failed for their stand-down marker
            # (regime_nt / stand_down:x / no_chain), which is already counted
            # in no_trade_reasons and must not pollute the gate histogram.
            if not r.get("gate_pass"):
                for g in load_list(r.get("gate_failed")):
                    token = str(g).split(":", 1)[0].strip()
                    if token and token.isupper():
                        bump(gate_failures, token)

            # veto_reasons carries SELECTOR vetoes on rows with a candidate and
            # REGIME vetoes on stand-down stub rows — split them accordingly.
            veto_bucket = (selector_vetoes if r.get("candidate_present") == 1
                           else regime_vetoes)
            for v in load_list(r.get("veto_reasons")):
                token = str(v).split()[0] if str(v).strip() else ""
                if token:
                    bump(veto_bucket, token.split(":", 1)[0])

            try:
                sig = json.loads(r["signals_json"]) if r.get("signals_json") else {}
            except (json.JSONDecodeError, TypeError):
                sig = {}
            if isinstance(sig, dict):
                rs = sig.get("routed_structure")
                if isinstance(rs, str) and rs:
                    bump(routed, rs)
                    rows_with_provenance += 1
                if sig.get("premium_flip") == 1:
                    flips += 1
                rv = sig.get("regime_vetoes")
                if isinstance(rv, str) and rv:
                    for v in rv.split(","):
                        bump(regime_vetoes, v.split(":", 1)[0])

            g = r.get("gex_pct_rank")
            if isinstance(g, (int, float)):
                gex_vals.append(float(g))
                if abs(g - 0.5) < 1e-12:
                    gex_warmup += 1

        gex: dict = {"n": len(gex_vals)}
        if gex_vals:
            gex.update({
                "mean": round(sum(gex_vals) / len(gex_vals), 4),
                "frac_at_warmup_neutral": round(gex_warmup / len(gex_vals), 4),
                "frac_below_gate_floor": round(
                    sum(1 for v in gex_vals if v < gate_gex_floor) / len(gex_vals), 4),
                "gate_floor": gate_gex_floor,
                "note": ("exactly 0.5 is the GexRankWindow warm-up sentinel "
                         "(can rarely also be a genuine median print)"),
            })

        return {
            "n": len(rows),
            "sessions": len({r["session_date"] for r in rows}),
            "routed_structures": routed,
            "structure_mix": structure_mix,
            "class_mix": class_mix,
            "no_trade_reasons": no_trade_reasons,
            "gate_failures": gate_failures,
            "selector_vetoes": selector_vetoes,
            "regime_vetoes": regime_vetoes,
            "premium_flips": {
                "n": flips,
                "rows_with_provenance": rows_with_provenance,
                "note": "credit cell forced to a debit cousin by a dealer veto",
            },
            "gex_rank": gex,
        }

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

    def gex_variant_comparison(self, session_date: Optional[str] = None) -> dict:
        """
        PR 9 / §16.4 — compare journaled GEX variant signals vs realized P&L
        and vs each other. Observation-only readout; does not change policy.
        """
        rows = self.fetch(session_date=session_date, settled_only=True)
        rows = [r for r in rows if r.get("realized_pnl") is not None]
        if len(rows) < 3:
            return {"n": len(rows), "note": "need >=3 settled rows"}

        def _sig(r):
            try:
                s = json.loads(r["signals_json"]) if r.get("signals_json") else {}
            except (json.JSONDecodeError, TypeError):
                s = {}
            return s if isinstance(s, dict) else {}

        sigs = [_sig(r) for r in rows]
        y = [r["realized_pnl"] for r in rows]

        def corr(xs, ys):
            n = len(xs)
            if n < 3:
                return None
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            vx = sum((a - mx) ** 2 for a in xs)
            vy = sum((b - my) ** 2 for b in ys)
            return round(cov / (vx * vy) ** 0.5, 3) if vx > 0 and vy > 0 else None

        variants = ("oi", "weekly", "volume", "hybrid")
        out: dict = {"n": len(rows), "variants": {}}
        for v in variants:
            key = f"gex_{v}_net_gex"
            pairs = [(s[key], yy) for s, yy in zip(sigs, y)
                     if isinstance(s.get(key), (int, float))]
            panel = {
                "n": len(pairs),
                "corr_vs_pnl": corr([p for p, _ in pairs],
                                    [p for _, p in pairs]) if pairs else None,
            }
            # disagreement rate when this variant's sign differs from OI
            if v != "oi":
                disagree = 0
                n_cmp = 0
                for s in sigs:
                    a, b = s.get("gex_oi_net_gex"), s.get(key)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        n_cmp += 1
                        if (a > 0) != (b > 0):
                            disagree += 1
                panel["sign_disagree_vs_oi"] = (
                    round(disagree / n_cmp, 3) if n_cmp else None)
            out["variants"][v] = panel

        # Aggregate disagreement feature vs PnL
        dkey = "gex_disagree_sign"
        dpairs = [(s[dkey], yy) for s, yy in zip(sigs, y)
                  if isinstance(s.get(dkey), (int, float))]
        out["disagree_sign_corr_vs_pnl"] = (
            corr([p for p, _ in dpairs], [p for _, p in dpairs])
            if dpairs else None)
        out["disagree_sign_rate"] = (
            round(sum(p for p, _ in dpairs) / len(dpairs), 3)
            if dpairs else None)
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

        The per-tick numbers OVERSTATE the sample: two predictions a minute
        apart share almost the entire future path to settlement, so a few
        dozen sessions can look like thousands of observations. The
        "sessions" panel is the honest view — per-session hit rates, the
        independent session count, and a session-bootstrap 95% CI.
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

        def session_stats(rs):
            by_session: dict[str, list] = {}
            for r in rs:
                by_session.setdefault(r["session_date"], []).append(r)
            per_session = {d: stats(srs)["hit_rate"]
                           for d, srs in by_session.items()}
            out = {"n_sessions": len(per_session),
                   "mean_session_hit_rate": None,
                   "hit_rate_ci95": None}
            if per_session:
                from validation.bootstrap import session_bootstrap
                boot = session_bootstrap(per_session)
                out["mean_session_hit_rate"] = boot["stat"]
                if boot["ci_low"] is not None:
                    out["hit_rate_ci95"] = [boot["ci_low"], boot["ci_high"]]
            return out

        overall = stats(rows)
        return {
            "overall": overall,
            "sessions": session_stats(rows),
            "by_direction": {
                d: stats([r for r in rows if r["regime_direction"] == d])
                for d in ("call", "put")
            },
            "traded_only": stats([r for r in rows if r["was_traded"] == 1]),
            "note": ("hit = settlement moved in the bias direction; "
                     "avg_fwd_move_pct is the bias-signed mean move (edge in %, "
                     "positive means the bias points the right way on average); "
                     "'sessions' resamples complete sessions — the per-tick n "
                     "overstates the independent sample size"),
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
