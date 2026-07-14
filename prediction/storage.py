"""
prediction/storage.py
=====================
SQLite persistence + Parquet export for the canonical V2 dataset
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §8.4, §20.2).

Tables (spec §20.2, plus observation_labels for the §9 label rows):

  feature_snapshots   one row per observation (snapshot_id PRIMARY KEY)
  observation_labels  one row per labeled observation
  prediction_outputs  PredictionBundle rows (written from PR 4 onward)
  candidate_snapshots one row per generated candidate per observation
  candidate_outcomes  one row per settled candidate

All flexible payloads are canonical JSON (sorted keys) so that rebuilding
from identical recordings yields byte-identical rows — `dataset_hash()` is
the acceptance check for deterministic rebuilds. Writes are idempotent
(INSERT OR REPLACE on the primary key).

Parquet export (§8.4) lazily requires pyarrow; the SQLite store works
without it.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

_CREATE = """
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_date TEXT NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    minutes_since_open REAL,
    minutes_to_close REAL,
    spot REAL,
    features_json TEXT NOT NULL,
    standardized_json TEXT,
    missingness_json TEXT,
    source_ages_json TEXT,
    quality_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_fsnap_session ON feature_snapshots(session_date);

CREATE TABLE IF NOT EXISTS observation_labels (
    snapshot_id TEXT PRIMARY KEY,
    label_version TEXT NOT NULL,
    labels_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_group_version TEXT NOT NULL,
    predictions_json TEXT NOT NULL,
    uncertainty REAL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pout_snapshot ON prediction_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS candidate_snapshots (
    candidate_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    family TEXT NOT NULL,
    legs_json TEXT NOT NULL,
    quote_json TEXT,
    geometry_json TEXT,
    legacy_metrics_json TEXT,
    execution_estimate_json TEXT,
    prediction_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_csnap_snapshot ON candidate_snapshots(snapshot_id);

CREATE TABLE IF NOT EXISTS candidate_outcomes (
    candidate_id TEXT PRIMARY KEY,
    settled INTEGER NOT NULL DEFAULT 0,
    settlement_price REAL,
    pnl_mid REAL,
    pnl_expected_fill REAL,
    pnl_conservative REAL,
    pnl_policy REAL,
    mfe REAL,
    mae REAL,
    target_hit INTEGER,
    stop_hit INTEGER,
    first_event TEXT,
    outcome_json TEXT
);

CREATE TABLE IF NOT EXISTS sigma_cone_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    ts TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    spot REAL NOT NULL,
    model_version TEXT NOT NULL,
    sigma REAL NOT NULL,
    horizon_min REAL NOT NULL,
    lo REAL NOT NULL,
    hi REAL NOT NULL,
    mid REAL,
    settle_by TEXT NOT NULL,
    settled INTEGER NOT NULL DEFAULT 0,
    realized_spot REAL,
    realized_ts TEXT,
    inside INTEGER,
    error_mid REAL,
    coverage_note TEXT,
    UNIQUE(snapshot_id, timeframe, sigma)
);
CREATE INDEX IF NOT EXISTS ix_cone_session ON sigma_cone_journal(session_date);
CREATE INDEX IF NOT EXISTS ix_cone_settle ON sigma_cone_journal(settled, settle_by);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_type TEXT NOT NULL,
    target TEXT NOT NULL,
    horizon TEXT,
    feature_version TEXT NOT NULL,
    label_version TEXT,
    fold_definition_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    slice_metrics_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uncertainty_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_group_version TEXT NOT NULL,
    composite REAL NOT NULL,
    components_json TEXT NOT NULL,
    reasons_json TEXT,
    diagnostics_json TEXT,
    generated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_uncertainty_snapshot
ON uncertainty_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS structural_states (
    snapshot_id TEXT PRIMARY KEY,
    structural_version TEXT NOT NULL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regime_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    probabilities_json TEXT NOT NULL,
    uncertainty REAL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_regime_snapshot
ON regime_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS competing_risk_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    horizon TEXT NOT NULL,
    model_version TEXT NOT NULL,
    forecast_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_competing_snapshot
ON competing_risk_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS path_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon TEXT NOT NULL,
    event_probabilities_json TEXT NOT NULL,
    distribution_json TEXT,
    diagnostics_json TEXT,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_path_snapshot
ON path_forecasts(snapshot_id);

CREATE TABLE IF NOT EXISTS ensemble_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    horizon TEXT NOT NULL,
    model_version TEXT NOT NULL,
    forecast_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ensemble_snapshot
ON ensemble_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS candidate_rank_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_candidate_rank_snapshot
ON candidate_rank_outputs(snapshot_id);

CREATE TABLE IF NOT EXISTS fill_records (
    fill_record_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fill_session
ON fill_records(session_date);
CREATE INDEX IF NOT EXISTS ix_fill_candidate
ON fill_records(candidate_id);

CREATE TABLE IF NOT EXISTS meta_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT,
    model_version TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_meta_snapshot
ON meta_decisions(snapshot_id);

CREATE TABLE IF NOT EXISTS ensemble_weight_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_session TEXT NOT NULL,
    target TEXT NOT NULL,
    horizon TEXT,
    weights_json TEXT NOT NULL,
    losses_json TEXT NOT NULL,
    penalties_json TEXT,
    configuration_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_weight_session
ON ensemble_weight_history(as_of_session);

CREATE TABLE IF NOT EXISTS drift_events (
    event_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    as_of_session TEXT NOT NULL,
    severity TEXT NOT NULL,
    status_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_drift_model_session
ON drift_events(model_id, as_of_session);

CREATE TABLE IF NOT EXISTS promotion_reviews (
    review_id TEXT PRIMARY KEY,
    model_group_id TEXT NOT NULL,
    current_status TEXT NOT NULL,
    proposed_status TEXT NOT NULL,
    review_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS deployment_history (
    deployment_id TEXT PRIMARY KEY,
    deployed_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    active_models_json TEXT NOT NULL,
    prior_models_json TEXT,
    configuration_hash TEXT NOT NULL,
    rollback_target_json TEXT,
    deployed_by TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS canonical_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    session_date TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    snapshot_schema_version TEXT,
    raw_features_json TEXT,
    standardized_features_json TEXT,
    missingness_json TEXT,
    source_timestamps_json TEXT,
    source_ages_json TEXT,
    quality_json TEXT,
    snapshot_hash TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    deployment_id TEXT,
    model_group_id TEXT,
    forecast_json TEXT NOT NULL,
    uncertainty REAL,
    ood_score REAL,
    data_quality REAL,
    generated_at TEXT NOT NULL,
    mode TEXT
);

CREATE TABLE IF NOT EXISTS candidate_universes (
    snapshot_id TEXT PRIMARY KEY,
    generator_version TEXT,
    configuration_hash TEXT,
    candidate_count INTEGER,
    excluded_count INTEGER,
    generated_at TEXT NOT NULL,
    diagnostics_json TEXT
);

CREATE TABLE IF NOT EXISTS unified_decisions (
    snapshot_id TEXT PRIMARY KEY,
    deployment_id TEXT,
    deployment_mode TEXT,
    authority_source TEXT,
    legacy_action TEXT,
    legacy_candidate_id TEXT,
    v3_statistical_action TEXT,
    v3_final_action TEXT,
    v3_candidate_id TEXT,
    final_action TEXT,
    selected_candidate_id TEXT,
    hard_vetoes_json TEXT,
    reasons_json TEXT,
    fallback_used INTEGER,
    fallback_reason TEXT,
    configuration_hash TEXT,
    decision_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    final_rank INTEGER,
    absolute_utility REAL,
    expected_net_pnl REAL,
    p_positive_pnl REAL,
    expected_shortfall REAL,
    fill_probability REAL,
    expected_order_value REAL,
    vetoes_json TEXT,
    pnl_quantiles_json TEXT,
    model_versions_json TEXT,
    evaluation_json TEXT NOT NULL,
    UNIQUE(snapshot_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS candidate_execution_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    mid_credit REAL,
    natural_credit REAL,
    expected_credit REAL,
    p_fill REAL,
    estimate_json TEXT NOT NULL,
    UNIQUE(snapshot_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS candidate_ranks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    final_rank INTEGER NOT NULL,
    ranking_uncertainty REAL,
    UNIQUE(snapshot_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS fill_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    candidate_id TEXT,
    attempt_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_decision_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    features_json TEXT,
    action TEXT,
    row_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_OUTCOME_COLS = ("settled", "settlement_price", "pnl_mid", "pnl_expected_fill",
                 "pnl_conservative", "pnl_policy", "mfe", "mae",
                 "target_hit", "stop_hit", "first_event")


def _canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace variance."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def make_candidate_id(snapshot_id: str, family: str, legs: list) -> str:
    """Stable candidate identity: observation + family + exact leg geometry."""
    geom = _canonical_json([[lg["strike"], lg["kind"], lg["qty"]]
                            for lg in legs])
    payload = f"{snapshot_id}|{family}|{geom}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PredictionStore:
    db_path: str = "prediction_store.sqlite"
    schema_ok: bool = True
    schema_error: Optional[str] = None

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.executescript(_CREATE)
            self.conn.commit()
            self.schema_ok = True
            self.schema_error = None
        except sqlite3.Error as exc:
            self.schema_ok = False
            self.schema_error = str(exc)

    def require_schema(self) -> None:
        """Raise if migrations failed — V3 shadow path must stop."""
        if not self.schema_ok:
            raise RuntimeError(
                "prediction store schema migration failed: "
                f"{self.schema_error or 'unknown error'}")

    # ---- feature snapshots ---------------------------------------------------
    def log_feature_snapshot(self, row) -> None:
        """`row` is a prediction.dataset.ObservationRow (duck-typed)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_snapshots "
            "(snapshot_id, session_date, ts, symbol, feature_version, "
            "minutes_since_open, minutes_to_close, spot, features_json, "
            "standardized_json, missingness_json, source_ages_json, "
            "quality_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row.snapshot_id, row.session_date, row.ts, row.symbol,
             row.feature_version, row.minutes_since_open, row.minutes_to_close,
             row.spot, _canonical_json(row.features),
             _canonical_json(row.standardized),
             _canonical_json(row.missingness),
             _canonical_json(row.source_ages),
             _canonical_json(row.quality)),
        )
        self.conn.commit()

    def fetch_feature_snapshots(self, session_date: Optional[str] = None
                                ) -> list[dict]:
        sql = "SELECT * FROM feature_snapshots"
        args: list = []
        if session_date:
            sql += " WHERE session_date=?"
            args.append(session_date)
        sql += " ORDER BY session_date, ts, snapshot_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (("features_json", "features"),
                              ("standardized_json", "standardized"),
                              ("missingness_json", "missingness"),
                              ("source_ages_json", "source_ages"),
                              ("quality_json", "quality")):
                try:
                    row[dest] = json.loads(row.pop(src) or "null") or {}
                except (json.JSONDecodeError, TypeError):
                    row[dest] = {}
            out.append(row)
        return out

    # ---- labels ---------------------------------------------------------------
    def log_labels(self, snapshot_id: str, labels: dict,
                   label_version: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO observation_labels "
            "(snapshot_id, label_version, labels_json) VALUES (?,?,?)",
            (snapshot_id, label_version, _canonical_json(labels)),
        )
        self.conn.commit()

    def fetch_labels(self, snapshot_id: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM observation_labels"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY snapshot_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["labels"] = json.loads(row.pop("labels_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["labels"] = {}
            out.append(row)
        return out

    # ---- prediction outputs (written from PR 4 onward) -------------------------
    def log_prediction(self, snapshot_id: str, model_group_version: str,
                       predictions: dict, uncertainty: Optional[float],
                       generated_at: str, mode: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO prediction_outputs (snapshot_id, model_group_version, "
            "predictions_json, uncertainty, generated_at, mode) "
            "VALUES (?,?,?,?,?,?)",
            (snapshot_id, model_group_version, _canonical_json(predictions),
             uncertainty, generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_predictions(self, snapshot_id: Optional[str] = None,
                          mode: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM prediction_outputs"
        conds, args = [], []
        if snapshot_id:
            conds.append("snapshot_id=?")
            args.append(snapshot_id)
        if mode:
            conds.append("mode=?")
            args.append(mode)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["predictions"] = json.loads(row.pop("predictions_json")
                                                or "{}")
            except (json.JSONDecodeError, TypeError):
                row["predictions"] = {}
            out.append(row)
        return out

    # ---- model evaluations + uncertainty (V3 Part 1 §9) ------------------------
    def log_model_evaluation(
        self,
        evaluation_id: str,
        *,
        model_id: str,
        model_type: str,
        target: str,
        feature_version: str,
        fold_definition: dict,
        metrics: dict,
        horizon: Optional[str] = None,
        label_version: Optional[str] = None,
        slice_metrics: Optional[dict] = None,
        created_at: Optional[str] = None,
    ) -> None:
        self.require_schema()
        import datetime as _dt
        created_at = created_at or _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO model_evaluations "
            "(evaluation_id, model_id, model_type, target, horizon, "
            "feature_version, label_version, fold_definition_json, "
            "metrics_json, slice_metrics_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (evaluation_id, model_id, model_type, target, horizon,
             feature_version, label_version,
             _canonical_json(fold_definition), _canonical_json(metrics),
             _canonical_json(slice_metrics) if slice_metrics is not None else None,
             created_at),
        )
        self.conn.commit()

    def fetch_model_evaluations(self, model_id: Optional[str] = None) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM model_evaluations"
        args: list = []
        if model_id:
            sql += " WHERE model_id=?"
            args.append(model_id)
        sql += " ORDER BY created_at, evaluation_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (("fold_definition_json", "fold_definition"),
                              ("metrics_json", "metrics"),
                              ("slice_metrics_json", "slice_metrics")):
                raw = row.pop(src, None)
                try:
                    row[dest] = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    row[dest] = None
            out.append(row)
        return out

    def log_uncertainty_output(
        self,
        snapshot_id: str,
        model_group_version: str,
        composite: float,
        components: dict,
        *,
        reasons: Optional[list] = None,
        diagnostics: Optional[dict] = None,
        generated_at: Optional[str] = None,
    ) -> int:
        self.require_schema()
        import datetime as _dt
        generated_at = generated_at or _dt.datetime.now(
            _dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO uncertainty_outputs "
            "(snapshot_id, model_group_version, composite, components_json, "
            "reasons_json, diagnostics_json, generated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (snapshot_id, model_group_version, float(composite),
             _canonical_json(components),
             _canonical_json(reasons or []),
             _canonical_json(diagnostics or {}),
             generated_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_uncertainty_outputs(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM uncertainty_outputs"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (("components_json", "components"),
                              ("reasons_json", "reasons"),
                              ("diagnostics_json", "diagnostics")):
                raw = row.pop(src, None)
                try:
                    row[dest] = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    row[dest] = None
            out.append(row)
        return out

    # ---- structural states (V3 Part 2 §35 / PR 7) ------------------------------
    def log_structural_state(
        self,
        snapshot_id: str,
        state: dict,
        *,
        structural_version: Optional[str] = None,
    ) -> None:
        """Persist a StructuralState.to_dict() payload (idempotent)."""
        self.require_schema()
        version = structural_version or str(
            state.get("version") or "v3.0.0")
        self.conn.execute(
            "INSERT OR REPLACE INTO structural_states "
            "(snapshot_id, structural_version, state_json) VALUES (?,?,?)",
            (snapshot_id, version, _canonical_json(state)),
        )
        self.conn.commit()

    def fetch_structural_state(
        self, snapshot_id: str,
    ) -> Optional[dict]:
        self.require_schema()
        row = self.conn.execute(
            "SELECT * FROM structural_states WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        try:
            out["state"] = json.loads(out.pop("state_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            out["state"] = {}
        return out

    def fetch_structural_states(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM structural_states"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY snapshot_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["state"] = json.loads(row.pop("state_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["state"] = {}
            out.append(row)
        return out

    # ---- regime outputs (V3 Part 2 §35 / PR 9) ---------------------------------
    def log_regime_output(
        self,
        snapshot_id: str,
        model_version: str,
        probabilities: dict,
        *,
        uncertainty: Optional[float] = None,
        generated_at: Optional[str] = None,
        mode: str = "shadow",
    ) -> int:
        self.require_schema()
        import datetime as _dt
        generated_at = generated_at or _dt.datetime.now(
            _dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO regime_outputs "
            "(snapshot_id, model_version, probabilities_json, uncertainty, "
            "generated_at, mode) VALUES (?,?,?,?,?,?)",
            (snapshot_id, model_version, _canonical_json(probabilities),
             uncertainty, generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_regime_outputs(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM regime_outputs"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["probabilities"] = json.loads(
                    row.pop("probabilities_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["probabilities"] = {}
            out.append(row)
        return out

    # ---- competing risk / path / ensemble (V3 Part 2 §35 / PR 16) --------------
    def log_competing_risk_output(
        self, snapshot_id: str, target_name: str, horizon: str,
        model_version: str, forecast: dict, *,
        generated_at: Optional[str] = None, mode: str = "shadow",
    ) -> int:
        self.require_schema()
        import datetime as _dt
        generated_at = generated_at or _dt.datetime.now(
            _dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO competing_risk_outputs "
            "(snapshot_id, target_name, horizon, model_version, forecast_json, "
            "generated_at, mode) VALUES (?,?,?,?,?,?,?)",
            (snapshot_id, target_name, horizon, model_version,
             _canonical_json(forecast), generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_competing_risk_outputs(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM competing_risk_outputs"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["forecast"] = json.loads(row.pop("forecast_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["forecast"] = {}
            out.append(row)
        return out

    def log_path_forecast(
        self, snapshot_id: str, model_version: str, horizon: str,
        event_probabilities: dict, *,
        distribution: Optional[dict] = None,
        diagnostics: Optional[dict] = None,
        generated_at: Optional[str] = None, mode: str = "shadow",
    ) -> int:
        self.require_schema()
        import datetime as _dt
        generated_at = generated_at or _dt.datetime.now(
            _dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO path_forecasts "
            "(snapshot_id, model_version, horizon, event_probabilities_json, "
            "distribution_json, diagnostics_json, generated_at, mode) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (snapshot_id, model_version, horizon,
             _canonical_json(event_probabilities),
             _canonical_json(distribution or {}),
             _canonical_json(diagnostics or {}),
             generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_path_forecasts(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM path_forecasts"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (
                ("event_probabilities_json", "event_probabilities"),
                ("distribution_json", "distribution"),
                ("diagnostics_json", "diagnostics"),
            ):
                raw = row.pop(src, None)
                try:
                    row[dest] = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    row[dest] = {}
            out.append(row)
        return out

    def log_ensemble_output(
        self, snapshot_id: str, target_name: str, horizon: str,
        model_version: str, forecast: dict, *,
        generated_at: Optional[str] = None, mode: str = "shadow",
    ) -> int:
        self.require_schema()
        import datetime as _dt
        generated_at = generated_at or _dt.datetime.now(
            _dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO ensemble_outputs "
            "(snapshot_id, target_name, horizon, model_version, forecast_json, "
            "generated_at, mode) VALUES (?,?,?,?,?,?,?)",
            (snapshot_id, target_name, horizon, model_version,
             _canonical_json(forecast), generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_ensemble_outputs(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM ensemble_outputs"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["forecast"] = json.loads(row.pop("forecast_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["forecast"] = {}
            out.append(row)
        return out

    # ---- candidate ranking (Part 3) -------------------------------------------
    def log_candidate_ranking(
        self,
        snapshot_id: str,
        model_version: str,
        ranking: dict,
        generated_at: str,
        mode: str,
    ) -> int:
        self.require_schema()
        cur = self.conn.execute(
            "INSERT INTO candidate_rank_outputs "
            "(snapshot_id, model_version, ranking_json, generated_at, mode) "
            "VALUES (?,?,?,?,?)",
            (snapshot_id, model_version, _canonical_json(ranking),
             generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_candidate_rankings(
        self, snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM candidate_rank_outputs"
        args: list = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["ranking"] = json.loads(row.pop("ranking_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["ranking"] = {}
            out.append(row)
        return out

    # ---- fill records (Part 3) ------------------------------------------------
    def log_fill_record(self, record) -> None:
        """Idempotent upsert by fill_record_id. Validates via execution module."""
        self.require_schema()
        from execution.fill_records import FillRecord, validate_fill_record
        if hasattr(record, "to_dict"):
            rec = record
            payload = record.to_dict()
        else:
            rec = FillRecord.from_dict(record)
            payload = rec.to_dict()
        validate_fill_record(rec)
        self.conn.execute(
            "INSERT OR REPLACE INTO fill_records "
            "(fill_record_id, snapshot_id, candidate_id, session_date, "
            "record_json) VALUES (?,?,?,?,?)",
            (rec.fill_record_id, rec.snapshot_id, rec.candidate_id,
             rec.session_date, _canonical_json(payload)),
        )
        self.conn.commit()

    def fetch_fill_records(
        self,
        *,
        session_date: Optional[str] = None,
        candidate_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> list[dict]:
        self.require_schema()
        sql = "SELECT * FROM fill_records"
        conds, args = [], []
        if session_date:
            conds.append("session_date=?")
            args.append(session_date)
        if candidate_id:
            conds.append("candidate_id=?")
            args.append(candidate_id)
        if snapshot_id:
            conds.append("snapshot_id=?")
            args.append(snapshot_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY session_date, fill_record_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["record"] = json.loads(row.pop("record_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["record"] = {}
            out.append(row)
        return out

    # ---- meta decisions (Part 3) ----------------------------------------------
    def log_meta_decision(
        self,
        snapshot_id: str,
        model_version: str,
        decision: dict,
        generated_at: str,
        mode: str,
        candidate_id=None,
    ) -> int:
        self.require_schema()
        cur = self.conn.execute(
            "INSERT INTO meta_decisions "
            "(snapshot_id, candidate_id, model_version, decision_json, "
            "generated_at, mode) VALUES (?,?,?,?,?,?)",
            (snapshot_id, candidate_id, model_version,
             _canonical_json(decision), generated_at, mode),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_meta_decisions(self, snapshot_id=None) -> list:
        self.require_schema()
        sql = "SELECT * FROM meta_decisions"
        args = []
        if snapshot_id:
            sql += " WHERE snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["decision"] = json.loads(row.pop("decision_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["decision"] = {}
            out.append(row)
            out.append(row)
        return out

    # ---- ensemble weights / drift (Part 3) ------------------------------------
    def log_ensemble_weights(
        self, *, as_of_session, target, weights, losses, configuration_hash,
        created_at, horizon=None, penalties=None,
    ) -> int:
        self.require_schema()
        cur = self.conn.execute(
            "INSERT INTO ensemble_weight_history "
            "(as_of_session, target, horizon, weights_json, losses_json, "
            "penalties_json, configuration_hash, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (as_of_session, target, horizon, _canonical_json(weights),
             _canonical_json(losses),
             _canonical_json(penalties) if penalties is not None else None,
             configuration_hash, created_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_drift_event(self, event_id, model_id, as_of_session, severity,
                        status, created_at) -> None:
        self.require_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO drift_events "
            "(event_id, model_id, as_of_session, severity, status_json, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (event_id, model_id, as_of_session, severity,
             _canonical_json(status), created_at),
        )
        self.conn.commit()

    def fetch_drift_events(self, model_id=None) -> list:
        self.require_schema()
        sql = "SELECT * FROM drift_events"
        args = []
        if model_id:
            sql += " WHERE model_id=?"
            args.append(model_id)
        sql += " ORDER BY created_at, event_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            try:
                row["status"] = json.loads(row.pop("status_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                row["status"] = {}
            out.append(row)
        return out

    def log_promotion_review(self, review_id, model_group_id, current_status,
                             proposed_status, review, created_at,
                             resolved_at=None) -> None:
        self.require_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO promotion_reviews "
            "(review_id, model_group_id, current_status, proposed_status, "
            "review_json, created_at, resolved_at) VALUES (?,?,?,?,?,?,?)",
            (review_id, model_group_id, current_status, proposed_status,
             _canonical_json(review), created_at, resolved_at),
        )
        self.conn.commit()

    def log_deployment_history(self, deployment_id, deployed_at, mode,
                               active_models, configuration_hash,
                               prior_models=None, rollback_target=None,
                               deployed_by=None, note=None) -> None:
        self.require_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO deployment_history "
            "(deployment_id, deployed_at, mode, active_models_json, "
            "prior_models_json, configuration_hash, rollback_target_json, "
            "deployed_by, note) VALUES (?,?,?,?,?,?,?,?,?)",
            (deployment_id, deployed_at, mode,
             _canonical_json(active_models),
             _canonical_json(prior_models) if prior_models is not None else None,
             configuration_hash,
             _canonical_json(rollback_target) if rollback_target is not None else None,
             deployed_by, note),
        )
        self.conn.commit()

    def fetch_deployment_history(self) -> list:
        self.require_schema()
        out = []
        for r in self.conn.execute(
            "SELECT * FROM deployment_history ORDER BY deployed_at"
        ).fetchall():
            row = dict(r)
            for src, dest in (
                ("active_models_json", "active_models"),
                ("prior_models_json", "prior_models"),
                ("rollback_target_json", "rollback_target"),
            ):
                try:
                    row[dest] = json.loads(row.pop(src) or "null")
                except (json.JSONDecodeError, TypeError, KeyError):
                    row[dest] = None
            out.append(row)
        return out

    # ---- candidates ------------------------------------------------------------
    def log_candidate_snapshot(self, candidate_id: str, snapshot_id: str,
                               family: str, legs: list,
                               quote: Optional[dict] = None,
                               geometry: Optional[dict] = None,
                               legacy_metrics: Optional[dict] = None,
                               execution_estimate: Optional[dict] = None,
                               prediction: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO candidate_snapshots "
            "(candidate_id, snapshot_id, family, legs_json, quote_json, "
            "geometry_json, legacy_metrics_json, execution_estimate_json, "
            "prediction_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (candidate_id, snapshot_id, family, _canonical_json(legs),
             _canonical_json(quote) if quote is not None else None,
             _canonical_json(geometry) if geometry is not None else None,
             _canonical_json(legacy_metrics) if legacy_metrics is not None else None,
             _canonical_json(execution_estimate) if execution_estimate is not None else None,
             _canonical_json(prediction) if prediction is not None else None),
        )
        self.conn.commit()

    def log_candidate_outcome(self, candidate_id: str, outcome: dict) -> None:
        """`outcome` is a prediction.labels.candidate_outcome_labels dict;
        fields beyond the explicit columns land in outcome_json."""
        extras = {k: v for k, v in outcome.items() if k not in _OUTCOME_COLS}
        self.conn.execute(
            "INSERT OR REPLACE INTO candidate_outcomes "
            "(candidate_id, settled, settlement_price, pnl_mid, "
            "pnl_expected_fill, pnl_conservative, pnl_policy, mfe, mae, "
            "target_hit, stop_hit, first_event, outcome_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (candidate_id,
             *[outcome.get(c) for c in _OUTCOME_COLS],
             _canonical_json(extras)),
        )
        self.conn.commit()

    def fetch_candidates(self, snapshot_id: Optional[str] = None) -> list[dict]:
        sql = ("SELECT s.*, o.settled, o.settlement_price, o.pnl_mid, "
               "o.pnl_expected_fill, o.pnl_conservative, o.pnl_policy, "
               "o.mfe, o.mae, o.target_hit, o.stop_hit, o.first_event, "
               "o.outcome_json "
               "FROM candidate_snapshots s "
               "LEFT JOIN candidate_outcomes o USING (candidate_id)")
        args: list = []
        if snapshot_id:
            sql += " WHERE s.snapshot_id=?"
            args.append(snapshot_id)
        sql += " ORDER BY s.candidate_id"
        out = []
        for r in self.conn.execute(sql, args).fetchall():
            row = dict(r)
            for src, dest in (("legs_json", "legs"), ("quote_json", "quote"),
                              ("geometry_json", "geometry"),
                              ("legacy_metrics_json", "legacy_metrics"),
                              ("execution_estimate_json", "execution_estimate"),
                              ("prediction_json", "prediction"),
                              ("outcome_json", "outcome_extras")):
                try:
                    row[dest] = json.loads(row.pop(src) or "null")
                except (json.JSONDecodeError, TypeError):
                    row[dest] = None
            out.append(row)
        return out

    # ---- sigma cone journal (MTF outward-looking predictions) ----------------
    def log_sigma_cones(self, cones) -> int:
        """Persist ConeForecast panes (one row per timeframe × σ). Idempotent."""
        n = 0
        for cone in cones or []:
            for band in getattr(cone, "bands", ()) or ():
                self.conn.execute(
                    "INSERT OR REPLACE INTO sigma_cone_journal "
                    "(snapshot_id, session_date, ts, timeframe, spot, "
                    "model_version, sigma, horizon_min, lo, hi, mid, settle_by, "
                    "settled, realized_spot, realized_ts, inside, error_mid, "
                    "coverage_note) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,NULL,NULL,NULL,NULL,NULL)",
                    (cone.snapshot_id, cone.session_date, cone.ts,
                     cone.timeframe, float(cone.spot), cone.model_version,
                     float(band.sigma), float(band.horizon_min),
                     float(band.lo), float(band.hi), float(band.mid),
                     band.settle_by),
                )
                n += 1
        self.conn.commit()
        return n


    # ---- UNIFIED decision-graph persistence (handoff §16) --------------------
    def log_canonical_snapshot(self, row) -> None:
        """Persist CanonicalSnapshot dict. Idempotent on snapshot_id."""
        self.require_schema()
        d = row if isinstance(row, dict) else (
            row.to_dict() if hasattr(row, "to_dict") else dict(row))
        import datetime as _dt
        self.conn.execute(
            "INSERT OR REPLACE INTO canonical_snapshots "
            "(snapshot_id, symbol, ts, session_date, feature_version, "
            "snapshot_schema_version, raw_features_json, "
            "standardized_features_json, missingness_json, "
            "source_timestamps_json, source_ages_json, quality_json, "
            "snapshot_hash, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                d.get("snapshot_id"), d.get("symbol"), d.get("ts"),
                d.get("session_date"), d.get("feature_version"),
                d.get("snapshot_schema_version"),
                _canonical_json(d.get("raw_features") or {}),
                _canonical_json(d.get("standardized_features") or {}),
                _canonical_json(d.get("missingness") or {}),
                _canonical_json(d.get("source_timestamps") or {}),
                _canonical_json(d.get("source_ages_seconds") or {}),
                _canonical_json(d.get("quality") or {}),
                d.get("snapshot_hash"),
                _dt.datetime.now(_dt.timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def log_forecast_bundle(self, payload) -> None:
        self.require_schema()
        d = payload if isinstance(payload, dict) else (
            payload.to_dict() if hasattr(payload, "to_dict") else dict(payload))
        import datetime as _dt
        self.conn.execute(
            "INSERT INTO forecast_bundles "
            "(snapshot_id, deployment_id, model_group_id, forecast_json, "
            "uncertainty, ood_score, data_quality, generated_at, mode) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                d.get("snapshot_id"),
                d.get("deployment_id"),
                d.get("model_group_id") or (
                    (d.get("model_versions") or {}).get("group")),
                _canonical_json(d),
                d.get("uncertainty"),
                d.get("ood_score"),
                d.get("data_quality"),
                _dt.datetime.now(_dt.timezone.utc).isoformat(),
                d.get("mode"),
            ),
        )
        self.conn.commit()

    def log_candidate_universe(self, row) -> None:
        self.require_schema()
        d = row if isinstance(row, dict) else (
            row.to_dict() if hasattr(row, "to_dict") else dict(row))
        self.conn.execute(
            "INSERT OR REPLACE INTO candidate_universes "
            "(snapshot_id, generator_version, configuration_hash, "
            "candidate_count, excluded_count, generated_at, diagnostics_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                d.get("snapshot_id"),
                d.get("generator_version"),
                d.get("generator_configuration_hash") or d.get("configuration_hash"),
                int(d.get("candidate_count") or 0),
                int(d.get("excluded_count") or 0),
                d.get("generated_at") or "",
                _canonical_json(d.get("diagnostics") or {}),
            ),
        )
        self.conn.commit()

    def log_unified_decision(self, row) -> None:
        self.require_schema()
        d = row if isinstance(row, dict) else (
            row.to_dict() if hasattr(row, "to_dict") else dict(row))
        import datetime as _dt
        self.conn.execute(
            "INSERT OR REPLACE INTO unified_decisions "
            "(snapshot_id, deployment_id, deployment_mode, authority_source, "
            "legacy_action, legacy_candidate_id, v3_statistical_action, "
            "v3_final_action, v3_candidate_id, final_action, "
            "selected_candidate_id, hard_vetoes_json, reasons_json, "
            "fallback_used, fallback_reason, configuration_hash, "
            "decision_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                d.get("snapshot_id"), d.get("deployment_id"),
                d.get("deployment_mode"), d.get("authority_source"),
                d.get("legacy_action"), d.get("legacy_candidate_id"),
                d.get("v3_statistical_action"), d.get("v3_final_action"),
                d.get("v3_candidate_id"), d.get("final_action"),
                d.get("selected_candidate_id"),
                _canonical_json(d.get("hard_vetoes") or []),
                _canonical_json(d.get("reasons") or []),
                1 if d.get("fallback_used") else 0,
                d.get("fallback_reason"),
                d.get("configuration_hash"),
                _canonical_json(d),
                _dt.datetime.now(_dt.timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def persist_decision_graph(
        self,
        *,
        snapshot=None,
        forecast=None,
        universe=None,
        evaluations=None,
        decision=None,
        fill_attempts=None,
        meta_row=None,
    ) -> None:
        """
        Atomically persist the learning graph for one tick.

        One transaction covers snapshot, forecast, universe, per-candidate
        evaluations / ranks / execution estimates, fill attempts, meta row,
        and the unified decision. A crash mid-write rolls back entirely.
        """
        self.require_schema()
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()

        def _as_dict(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return dict(obj)

        snap_d = _as_dict(snapshot)
        fc_d = _as_dict(forecast)
        uni_d = _as_dict(universe)
        dec_d = _as_dict(decision)
        snapshot_id = None
        for src in (dec_d, snap_d, uni_d, fc_d):
            if src and src.get("snapshot_id"):
                snapshot_id = src["snapshot_id"]
                break

        try:
            if snap_d is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO canonical_snapshots "
                    "(snapshot_id, symbol, ts, session_date, feature_version, "
                    "snapshot_schema_version, raw_features_json, "
                    "standardized_features_json, missingness_json, "
                    "source_timestamps_json, source_ages_json, quality_json, "
                    "snapshot_hash, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snap_d.get("snapshot_id"), snap_d.get("symbol"),
                        snap_d.get("ts"), snap_d.get("session_date"),
                        snap_d.get("feature_version"),
                        snap_d.get("snapshot_schema_version"),
                        _canonical_json(snap_d.get("raw_features") or {}),
                        _canonical_json(
                            snap_d.get("standardized_features") or {}),
                        _canonical_json(snap_d.get("missingness") or {}),
                        _canonical_json(snap_d.get("source_timestamps") or {}),
                        _canonical_json(
                            snap_d.get("source_ages_seconds") or {}),
                        _canonical_json(snap_d.get("quality") or {}),
                        snap_d.get("snapshot_hash"),
                        now,
                    ),
                )
            if fc_d is not None:
                self.conn.execute(
                    "INSERT INTO forecast_bundles "
                    "(snapshot_id, deployment_id, model_group_id, "
                    "forecast_json, uncertainty, ood_score, data_quality, "
                    "generated_at, mode) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        fc_d.get("snapshot_id") or snapshot_id,
                        fc_d.get("deployment_id"),
                        fc_d.get("model_group_id") or (
                            (fc_d.get("model_versions") or {}).get("group")),
                        _canonical_json(fc_d),
                        fc_d.get("uncertainty"),
                        fc_d.get("ood_score"),
                        fc_d.get("data_quality"),
                        now,
                        fc_d.get("mode"),
                    ),
                )
            if uni_d is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO candidate_universes "
                    "(snapshot_id, generator_version, configuration_hash, "
                    "candidate_count, excluded_count, generated_at, "
                    "diagnostics_json) VALUES (?,?,?,?,?,?,?)",
                    (
                        uni_d.get("snapshot_id") or snapshot_id,
                        uni_d.get("generator_version"),
                        uni_d.get("generator_configuration_hash")
                        or uni_d.get("configuration_hash"),
                        int(uni_d.get("candidate_count") or 0),
                        int(uni_d.get("excluded_count") or 0),
                        uni_d.get("generated_at") or now,
                        _canonical_json(uni_d.get("diagnostics") or {}),
                    ),
                )
            for ev in (evaluations or ()):
                ed = _as_dict(ev) or {}
                cid = ed.get("candidate_id")
                sid = ed.get("snapshot_id") or snapshot_id
                if not cid or not sid:
                    continue
                self.conn.execute(
                    "INSERT OR REPLACE INTO candidate_evaluations "
                    "(snapshot_id, candidate_id, final_rank, absolute_utility, "
                    "expected_net_pnl, p_positive_pnl, expected_shortfall, "
                    "fill_probability, expected_order_value, vetoes_json, "
                    "pnl_quantiles_json, model_versions_json, evaluation_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sid, cid, ed.get("final_rank"),
                        ed.get("absolute_utility"),
                        ed.get("expected_net_pnl"),
                        ed.get("p_positive_pnl"),
                        ed.get("expected_shortfall"),
                        ed.get("fill_probability"),
                        ed.get("expected_order_value"),
                        _canonical_json(ed.get("vetoes") or []),
                        _canonical_json(ed.get("pnl_quantiles") or {}),
                        _canonical_json(ed.get("model_versions") or {}),
                        _canonical_json(ed),
                    ),
                )
                if ed.get("final_rank") is not None:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO candidate_ranks "
                        "(snapshot_id, candidate_id, final_rank, "
                        "ranking_uncertainty) VALUES (?,?,?,?)",
                        (
                            sid, cid, int(ed["final_rank"]),
                            ed.get("ranking_uncertainty"),
                        ),
                    )
                # Execution estimate fields from CandidateEvaluation contract
                # + diagnostics (mid/natural stamped by Part 3 path).
                diag = ed.get("diagnostics") or {}
                mid_c = (
                    ed.get("mid_credit")
                    if ed.get("mid_credit") is not None
                    else diag.get("mid_credit"))
                nat_c = (
                    ed.get("natural_credit")
                    if ed.get("natural_credit") is not None
                    else diag.get("natural_credit"))
                exp_c = (
                    ed.get("expected_fill_price")
                    if ed.get("expected_fill_price") is not None
                    else ed.get("expected_credit")
                    if ed.get("expected_credit") is not None
                    else diag.get("expected_credit"))
                p_fill = ed.get("fill_probability")
                if any(v is not None for v in (mid_c, nat_c, exp_c, p_fill)):
                    self.conn.execute(
                        "INSERT OR REPLACE INTO candidate_execution_estimates "
                        "(snapshot_id, candidate_id, mid_credit, "
                        "natural_credit, expected_credit, p_fill, "
                        "estimate_json) VALUES (?,?,?,?,?,?,?)",
                        (
                            sid, cid,
                            mid_c,
                            nat_c,
                            exp_c,
                            p_fill,
                            _canonical_json({
                                "diagnostics": diag,
                                "conservative_fill_price": ed.get(
                                    "conservative_fill_price"),
                                "expected_concession": ed.get(
                                    "expected_concession"),
                                "fees": ed.get("fees"),
                                "expected_order_value": ed.get(
                                    "expected_order_value"),
                            }),
                        ),
                    )
            for attempt in (fill_attempts or ()):
                ad = _as_dict(attempt) or {}
                self.conn.execute(
                    "INSERT INTO fill_attempts "
                    "(snapshot_id, candidate_id, attempt_json, created_at) "
                    "VALUES (?,?,?,?)",
                    (
                        ad.get("snapshot_id") or snapshot_id,
                        ad.get("candidate_id"),
                        _canonical_json(ad),
                        now,
                    ),
                )
            if meta_row is not None:
                md = _as_dict(meta_row) or {}
                self.conn.execute(
                    "INSERT INTO meta_decision_rows "
                    "(snapshot_id, features_json, action, row_json, "
                    "created_at) VALUES (?,?,?,?,?)",
                    (
                        md.get("snapshot_id") or snapshot_id,
                        _canonical_json(md.get("features") or {}),
                        md.get("action"),
                        _canonical_json(md),
                        now,
                    ),
                )
            if dec_d is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO unified_decisions "
                    "(snapshot_id, deployment_id, deployment_mode, "
                    "authority_source, legacy_action, legacy_candidate_id, "
                    "v3_statistical_action, v3_final_action, v3_candidate_id, "
                    "final_action, selected_candidate_id, hard_vetoes_json, "
                    "reasons_json, fallback_used, fallback_reason, "
                    "configuration_hash, decision_json, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        dec_d.get("snapshot_id") or snapshot_id,
                        dec_d.get("deployment_id"),
                        dec_d.get("deployment_mode"),
                        dec_d.get("authority_source"),
                        dec_d.get("legacy_action"),
                        dec_d.get("legacy_candidate_id"),
                        dec_d.get("v3_statistical_action"),
                        dec_d.get("v3_final_action"),
                        dec_d.get("v3_candidate_id"),
                        dec_d.get("final_action"),
                        dec_d.get("selected_candidate_id"),
                        _canonical_json(dec_d.get("hard_vetoes") or []),
                        _canonical_json(dec_d.get("reasons") or []),
                        1 if dec_d.get("fallback_used") else 0,
                        dec_d.get("fallback_reason"),
                        dec_d.get("configuration_hash"),
                        _canonical_json(dec_d),
                        now,
                    ),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def settle_sigma_cones(self, now_iso: str, realized_spot: float,
                           *, realized_ts: Optional[str] = None) -> int:
        """
        Match due cone bands against the true spot.

        Any unsettled row with settle_by <= now_iso is scored: inside [lo,hi],
        error vs mid, coverage note. Returns count newly settled.
        """
        from prediction.sigma_cone import ConeBand, settle_band

        px = float(realized_spot)
        rts = realized_ts or now_iso
        rows = self.conn.execute(
            "SELECT id, sigma, lo, hi, mid, horizon_min, settle_by "
            "FROM sigma_cone_journal "
            "WHERE settled=0 AND settle_by<=?",
            (now_iso,),
        ).fetchall()
        n = 0
        for r in rows:
            band = ConeBand(
                sigma=float(r["sigma"]), lo=float(r["lo"]), hi=float(r["hi"]),
                horizon_min=float(r["horizon_min"]), mid=float(r["mid"]),
                settle_by=r["settle_by"],
            )
            s = settle_band(band, realized_spot=px, realized_ts=rts)
            self.conn.execute(
                "UPDATE sigma_cone_journal SET settled=1, realized_spot=?, "
                "realized_ts=?, inside=?, error_mid=?, coverage_note=? "
                "WHERE id=?",
                (s.realized_spot, s.realized_ts, 1 if s.inside else 0,
                 s.error_mid, s.coverage_note, r["id"]),
            )
            n += 1
        if n:
            self.conn.commit()
        return n

    def fetch_sigma_cones(self, *, session_date: Optional[str] = None,
                          settled: Optional[bool] = None,
                          timeframe: Optional[str] = None,
                          limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM sigma_cone_journal"
        conds, args = [], []
        if session_date:
            conds.append("session_date=?")
            args.append(session_date)
        if settled is not None:
            conds.append("settled=?")
            args.append(1 if settled else 0)
        if timeframe:
            conds.append("timeframe=?")
            args.append(timeframe)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY ts DESC, timeframe, sigma LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def sigma_cone_coverage(self, *, session_date: Optional[str] = None) -> dict:
        """Hit-rate of settled cones by σ level (and overall)."""
        sql = ("SELECT sigma, COUNT(*) AS n, "
               "SUM(CASE WHEN inside=1 THEN 1 ELSE 0 END) AS n_inside, "
               "AVG(error_mid) AS mean_error_mid, "
               "AVG(ABS(error_mid)) AS mae_mid "
               "FROM sigma_cone_journal WHERE settled=1")
        args: list = []
        if session_date:
            sql += " AND session_date=?"
            args.append(session_date)
        sql += " GROUP BY sigma ORDER BY sigma"
        by_sigma = {}
        total_n = total_in = 0
        for r in self.conn.execute(sql, args).fetchall():
            n = int(r["n"] or 0)
            nin = int(r["n_inside"] or 0)
            total_n += n
            total_in += nin
            by_sigma[str(r["sigma"])] = {
                "n": n,
                "n_inside": nin,
                "hit_rate": (nin / n) if n else None,
                "mean_error_mid": r["mean_error_mid"],
                "mae_mid": r["mae_mid"],
            }
        return {
            "n_settled": total_n,
            "n_inside": total_in,
            "hit_rate": (total_in / total_n) if total_n else None,
            "by_sigma": by_sigma,
        }

    # ---- deterministic rebuild check --------------------------------------------
    def dataset_hash(self) -> str:
        """
        SHA256 over every feature_snapshot + observation_label row in canonical
        order. Two stores built from identical recordings MUST agree
        (acceptance criterion for PR 3).
        """
        h = hashlib.sha256()
        for r in self.conn.execute(
                "SELECT snapshot_id, session_date, ts, symbol, feature_version, "
                "minutes_since_open, minutes_to_close, spot, features_json, "
                "standardized_json, missingness_json, source_ages_json, "
                "quality_json FROM feature_snapshots "
                "ORDER BY session_date, ts, snapshot_id"):
            h.update(_canonical_json(list(r)).encode("utf-8"))
        for r in self.conn.execute(
                "SELECT snapshot_id, label_version, labels_json "
                "FROM observation_labels ORDER BY snapshot_id"):
            h.update(_canonical_json(list(r)).encode("utf-8"))
        return h.hexdigest()

    # ---- Parquet export (§8.4) ---------------------------------------------------
    def export_features_parquet(self, out_dir: str) -> list[str]:
        """
        Materialize feature_snapshots as Parquet, partitioned
        data/derived-style: <out_dir>/version=<fv>/session_date=<d>/part.parquet.
        Requires pyarrow (lazy); raises a clear RuntimeError when unavailable.
        """
        try:
            import pandas as pd
            import pyarrow  # noqa: F401 — pandas' to_parquet engine
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow "
                "(pip install pyarrow)") from exc

        rows = self.fetch_feature_snapshots()
        written: list[str] = []
        by_part: dict = {}
        for r in rows:
            by_part.setdefault((r["feature_version"], r["session_date"]),
                               []).append(r)
        for (fv, session), part_rows in sorted(by_part.items()):
            recs = []
            for r in part_rows:
                rec = {k: r[k] for k in
                       ("snapshot_id", "symbol", "session_date", "ts",
                        "minutes_since_open", "minutes_to_close", "spot",
                        "feature_version")}
                for name, v in r["features"].items():
                    rec[f"feat_{name}"] = v
                for name, v in r["missingness"].items():
                    rec[f"miss_{name}"] = v
                recs.append(rec)
            d = os.path.join(out_dir, f"version={fv}", f"session_date={session}")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "part-0.parquet")
            pd.DataFrame(recs).to_parquet(path, index=False)
            written.append(path)
        return written

    def close(self):
        self.conn.close()
