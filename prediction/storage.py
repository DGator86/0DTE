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

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_CREATE)
        self.conn.commit()

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
        sql = ("SELECT s.*, o.settled, o.settlement_price, o.pnl_mid, o.mfe, "
               "o.mae, o.target_hit, o.stop_hit, o.first_event, o.outcome_json "
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
