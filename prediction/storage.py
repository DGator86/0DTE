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
