"""Offline tests for journal → MarketPacket experience backfill."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.backfill_experience_from_journal import backfill, row_to_packets


def _seed_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE evaluations (
            id INTEGER PRIMARY KEY,
            session_date TEXT,
            ts TEXT,
            spot REAL,
            selected_family TEXT,
            regime_direction TEXT,
            max_loss REAL,
            credit REAL,
            ev REAL,
            ev_per_risk REAL,
            decision TEXT,
            settled INTEGER,
            realized_pnl REAL,
            was_traded INTEGER,
            candidate_present INTEGER,
            snapshot_id TEXT,
            gate_failed TEXT,
            veto_reasons TEXT,
            signals_json TEXT
        )
        """
    )
    rows = [
        (
            1,
            "2026-07-22",
            "2026-07-22T10:00:00-04:00",
            600.0,
            "call_credit",
            "call",
            1.0,
            0.25,
            0.1,
            0.1,
            "TRADE",
            1,
            12.5,
            1,
            1,
            "snap-a",
            '["LATE"]',
            "[]",
            json.dumps({"legacy_top_candidate_id": "cand-a", "v2_fc_uncertainty": 0.2}),
        ),
        (
            2,
            "2026-07-23",
            "2026-07-23T11:00:00-04:00",
            601.0,
            "put_credit",
            "put",
            1.0,
            0.3,
            0.05,
            0.05,
            "NO_TRADE",
            1,
            -2.0,
            0,
            1,
            "snap-b",
            "[]",
            '["EV<=0"]',
            json.dumps({"legacy_top_candidate_id": "cand-b"}),
        ),
        (
            3,
            "2026-07-24",
            "2026-07-24T12:00:00-04:00",
            602.0,
            "iron_fly",
            "both",
            2.0,
            0.4,
            0.2,
            0.1,
            "TRADE",
            1,
            5.0,
            1,
            1,
            "snap-c",
            "[]",
            "[]",
            json.dumps({"legacy_top_candidate_id": "cand-c"}),
        ),
    ]
    conn.executemany(
        "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_row_to_packets_builds_market_and_outcome() -> None:
    row = {
        "id": 9,
        "session_date": "2026-07-24",
        "ts": "2026-07-24T15:00:00-04:00",
        "spot": 738.0,
        "selected_family": "call_credit",
        "regime_direction": "call",
        "max_loss": 0.98,
        "credit": 0.02,
        "ev": 0.01,
        "ev_per_risk": 0.015,
        "decision": "NO_TRADE",
        "settled": 1,
        "realized_pnl": 0.02,
        "was_traded": 0,
        "candidate_present": 1,
        "snapshot_id": "abc",
        "gate_failed": '["LATE"]',
        "veto_reasons": "[]",
        "signals_json": json.dumps(
            {"legacy_top_candidate_id": "c1", "v2_fc_uncertainty": 0.35}
        ),
    }
    market, outcome = row_to_packets(row)
    assert market["snapshot_id"] == "abc"
    assert market["schema_version"] == "zerodte.spyder.market.v1"
    assert market["candidates"][0]["candidate_id"] == "c1"
    assert "LATE" in market["hard_vetoes"][0]
    assert outcome is not None
    assert outcome["realized_pnl"] == "0.02"


def test_backfill_writes_sessions(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    root = tmp_path / "experience"
    _seed_db(db)
    result = backfill(db_path=db, root=root, stride=1, recent_days=0, dry_run=False)
    assert result["n_sessions"] == 3
    assert result["markets_written"] == 3
    assert result["outcomes_written"] == 3
    assert (root / "snapshots" / "snap-a.json").is_file()
    assert (root / "outcomes" / "snap-a.json").is_file()
    sessions = json.loads((root / "sessions.json").read_text(encoding="utf-8"))
    assert sessions == ["2026-07-22", "2026-07-23", "2026-07-24"]
