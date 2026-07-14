"""
tests/test_storage_v3_part1.py
==============================
V3 Part 1 §9 — model_evaluations / uncertainty_outputs migrations.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from prediction.storage import PredictionStore


def test_idempotent_migration(tmp_path):
    db = str(tmp_path / "p.sqlite")
    s1 = PredictionStore(db_path=db)
    assert s1.schema_ok
    s1.log_uncertainty_output(
        "snap1", "v3", 0.4, {"ensemble": 0.3}, reasons=["x"])
    s1.log_model_evaluation(
        "ev1", model_id="m1", model_type="direction", target="up_30m",
        feature_version="v2.0.0", fold_definition={"fold": 0},
        metrics={"brier": 0.2})
    # Re-open: CREATE IF NOT EXISTS must not fail or drop data
    s2 = PredictionStore(db_path=db)
    assert s2.schema_ok
    assert len(s2.fetch_uncertainty_outputs("snap1")) == 1
    assert len(s2.fetch_model_evaluations("m1")) == 1


def test_existing_db_gains_new_tables(tmp_path):
    db = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE feature_snapshots ("
        "snapshot_id TEXT PRIMARY KEY, session_date TEXT, ts TEXT, "
        "symbol TEXT, feature_version TEXT, features_json TEXT)")
    conn.commit()
    conn.close()
    store = PredictionStore(db_path=db)
    assert store.schema_ok
    # Old table still present; new tables exist
    cur = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in cur.fetchall()}
    assert "feature_snapshots" in names
    assert "model_evaluations" in names
    assert "uncertainty_outputs" in names


def test_no_tables_dropped(tmp_path):
    db = str(tmp_path / "p.sqlite")
    store = PredictionStore(db_path=db)
    before = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    store2 = PredictionStore(db_path=db)
    after = {r[0] for r in store2.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert before <= after


def test_uncertainty_roundtrip(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "u.sqlite"))
    store.log_uncertainty_output(
        "s", "grp", 0.7,
        {"ensemble": 0.5, "out_of_distribution": 0.8, "conformal": None},
        reasons=["missing_conformal_component"],
        diagnostics={"note": "test"},
        generated_at="2026-07-14T00:00:00Z",
    )
    rows = store.fetch_uncertainty_outputs("s")
    assert len(rows) == 1
    assert rows[0]["composite"] == pytest.approx(0.7)
    assert rows[0]["components"]["ensemble"] == 0.5
    assert "missing_conformal_component" in rows[0]["reasons"]
