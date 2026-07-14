"""
tests/test_storage_part2.py
===========================
V3 Part 2 PR7 — structural_states persistence / idempotent migration.
"""
from __future__ import annotations

import sqlite3

import pytest

from prediction.storage import PredictionStore
from prediction.structural_state import StructuralStateBuilder


def test_structural_table_created(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "p.sqlite"))
    assert store.schema_ok
    names = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "structural_states" in names
    # Part 1 tables still present
    assert "uncertainty_outputs" in names
    assert "model_evaluations" in names


def test_structural_roundtrip(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "p.sqlite"))
    state = StructuralStateBuilder().build(
        ts="2026-07-14T15:00:00Z",
        symbol="SPY",
        spot=600.0,
        expected_remaining_move=2.0,
        current_sources={
            "oi": {
                "net_gex": 1e9, "gamma_flip": 598.0,
                "call_wall": 610.0, "put_wall": 590.0,
                "abs_gamma_by_strike": {600.0: 5.0, 605.0: 3.0},
            },
            "volume": {
                "net_gex": 0.8e9, "gamma_flip": 597.0,
                "call_wall": 609.0, "put_wall": 591.0,
            },
        },
        historical_states=[],
    )
    store.log_structural_state("snap-1", state.to_dict())
    row = store.fetch_structural_state("snap-1")
    assert row is not None
    assert row["structural_version"] == "v3.0.0"
    assert row["state"]["net_gex_oi"] == pytest.approx(1e9)
    assert row["state"]["spot"] == pytest.approx(600.0)
    # Idempotent replace
    store.log_structural_state("snap-1", state.to_dict())
    assert len(store.fetch_structural_states("snap-1")) == 1


def test_existing_db_gains_structural_table(tmp_path):
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
    names = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "feature_snapshots" in names
    assert "structural_states" in names


def test_no_tables_dropped(tmp_path):
    db = str(tmp_path / "p.sqlite")
    store = PredictionStore(db_path=db)
    before = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    store2 = PredictionStore(db_path=db)
    after = {r[0] for r in store2.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert before <= after


def test_config_json_loads():
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1]
            / "configs" / "prediction_v3_part2.json")
    cfg = json.loads(path.read_text())
    assert cfg["structural_state"]["fallback_order"] == [
        "hybrid", "oi", "volume"]
    assert cfg["regime_model"]["minimum_sessions"] == 40
