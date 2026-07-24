"""RecordedFeed.recent_days windowing (the dojo 'review the past N days' knob)."""
from __future__ import annotations

import gzip
import json
import os

from chain_store import RecordedFeed


def _write_session(directory: str, date: str, n_ticks: int = 5) -> None:
    """Minimal recording file: n tick rows + one settle, enough for load()
    and the windowing logic (snapshot() internals are not exercised here)."""
    path = os.path.join(directory, f"ticks_{date}.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for m in range(n_ticks):
            ts = f"{date}T{10 + m:02d}:00:00-04:00"
            f.write(json.dumps({"t": "tick", "ts": ts, "seq": m,
                                "market": {}, "bars": [], "chain": None}) + "\n")
        f.write(json.dumps({"t": "settle", "date": date, "price": 600.0}) + "\n")


def test_recent_days_keeps_only_last_n_sessions(tmp_path):
    dates = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]
    for d in dates:
        _write_session(str(tmp_path), d)

    full = RecordedFeed(str(tmp_path))
    assert len({t.date().isoformat() for t in full.timestamps()}) == 5
    assert len(full.timestamps()) == 25

    windowed = RecordedFeed(str(tmp_path), recent_days=2)
    kept = sorted({t.date().isoformat() for t in windowed.timestamps()})
    assert kept == ["2026-07-23", "2026-07-24"]
    assert len(windowed.timestamps()) == 10
    # settlements are windowed too
    assert windowed.settlement_price("2026-07-24") == 600.0
    assert windowed.settlement_price("2026-07-20") is None


def test_recent_days_zero_is_noop(tmp_path):
    for d in ["2026-07-23", "2026-07-24"]:
        _write_session(str(tmp_path), d)
    a = RecordedFeed(str(tmp_path))
    b = RecordedFeed(str(tmp_path), recent_days=0)
    assert len(a.timestamps()) == len(b.timestamps()) == 10


def test_recent_days_larger_than_history_keeps_all(tmp_path):
    for d in ["2026-07-23", "2026-07-24"]:
        _write_session(str(tmp_path), d)
    w = RecordedFeed(str(tmp_path), recent_days=30)
    assert len({t.date().isoformat() for t in w.timestamps()}) == 2
