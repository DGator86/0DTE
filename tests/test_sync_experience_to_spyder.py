"""Offline smoke for deploy/ops/sync-experience-to-spyder.sh."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "ops" / "sync-experience-to-spyder.sh"


def _write_packet(path: Path, *, session_date: str, snapshot_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "zerodte.spyder.market.v1",
                "snapshot_id": snapshot_id,
                "session_date": session_date,
                "symbol": "SPY",
                "underlying_price": 500.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_sync_copies_packets_and_merges_sessions(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "snapshots").mkdir(parents=True)
    (src / "outcomes").mkdir(parents=True)
    (dst / "snapshots").mkdir(parents=True)
    (dst / "outcomes").mkdir(parents=True)

    _write_packet(src / "snapshots" / "a.json", session_date="2026-07-21", snapshot_id="a")
    _write_packet(src / "snapshots" / "b.json", session_date="2026-07-22", snapshot_id="b")
    (src / "sessions.json").write_text(
        json.dumps(["2026-07-21"]) + "\n", encoding="utf-8"
    )
    (dst / "sessions.json").write_text(
        json.dumps(["2026-07-20"]) + "\n", encoding="utf-8"
    )
    _write_packet(dst / "snapshots" / "old.json", session_date="2026-07-20", snapshot_id="old")

    env = os.environ.copy()
    env["ZERODTE_SPYDER_EXPERIENCE_ROOT"] = str(src)
    env["SPY_DER_EXPERIENCE_INBOX"] = str(dst)
    env["SYNC_ALLOW_NONROOT"] = "1"

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (dst / "snapshots" / "a.json").is_file()
    assert (dst / "snapshots" / "b.json").is_file()
    assert (dst / "snapshots" / "old.json").is_file()

    sessions = json.loads((dst / "sessions.json").read_text(encoding="utf-8"))
    assert sessions == ["2026-07-20", "2026-07-21", "2026-07-22"]


def test_sync_empty_source_exits_clean(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "snapshots").mkdir(parents=True)
    (src / "outcomes").mkdir(parents=True)
    dst.mkdir()

    env = os.environ.copy()
    env["ZERODTE_SPYDER_EXPERIENCE_ROOT"] = str(src)
    env["SPY_DER_EXPERIENCE_INBOX"] = str(dst)
    env["SYNC_ALLOW_NONROOT"] = "1"

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not (dst / "sessions.json").exists()
