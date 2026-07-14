"""
tests/test_part3_storage.py
===========================
V3 Part 3 — storage migrations for Part 3 tables (§28–§29).
"""
from __future__ import annotations

from prediction.storage import PredictionStore


def test_part3_tables_created(tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    tables = {
        r[0] for r in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for name in (
        "fill_records", "candidate_rank_outputs", "meta_decisions",
        "ensemble_weight_history", "drift_events", "promotion_reviews",
        "deployment_history",
    ):
        assert name in tables
    store.close()


def test_idempotent_open(tmp_path):
    path = str(tmp_path / "p.sqlite")
    a = PredictionStore(path)
    a.log_deployment_history(
        "d1", "2026-07-01T00:00:00Z", "shadow",
        {"meta": "m1"}, "hash1",
    )
    a.close()
    b = PredictionStore(path)
    assert len(b.fetch_deployment_history()) == 1
    b.close()
