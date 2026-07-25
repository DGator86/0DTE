"""The dashboard must say *why* SPY-DER state is unavailable.

Every read failure used to collapse to `None`, and the caller reported "not
found" for all of them. A Dojo report that existed but could not be read was
therefore displayed as one that had never been written, under a message telling
the operator to enable timers that were already running.

Permission denied is the failure that actually happens: SPY-DER writes this
state as `spy-der` and the dashboard reads it as `zerodte`.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from integrations.spy_der.dashboard_reader import (
    read_dashboard_bundle,
    read_dojo_latest,
    read_live_state,
)

DOJO_REPORT = {
    "report_date": "2026-07-25",
    "summary": "recorded tape: ok · universe sparring: 6 universes, 1 weak archetype(s)",
    "flags": [{"severity": "warn", "flag": "weak_archetype:chop", "detail": "-0.0120"}],
    "metrics": {"phases": {"recorded": {"status": "ok"}, "universe": {"status": "ok"}}},
    "generated_at": "2026-07-25T06:30:00-04:00",
}


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(DOJO_REPORT), encoding="utf-8")
    return path


def _deny(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Make exactly `target` raise PermissionError on open.

    Patched rather than chmod'ed because the suite may run as root, and root
    bypasses the mode bits that produce this failure in production.
    """
    real_open = builtins.open

    def guarded(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(file) == target:
            raise PermissionError(13, "Permission denied", str(target))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)


# --------------------------------------------------------------------------- #
# The happy path still works                                                  #
# --------------------------------------------------------------------------- #
def test_a_readable_report_comes_back_whole(report: Path) -> None:
    got = read_dojo_latest(report)
    assert got["summary"] == DOJO_REPORT["summary"]
    assert got["flags"][0]["flag"] == "weak_archetype:chop"
    assert "note" not in got


# --------------------------------------------------------------------------- #
# Missing and unreadable are different, and must read differently             #
# --------------------------------------------------------------------------- #
def test_missing_report_says_not_found(tmp_path: Path) -> None:
    note = read_dojo_latest(tmp_path / "absent.json")["note"]
    assert "not found" in note
    assert "permission" not in note.lower()


def test_unreadable_report_says_permission_denied(
    report: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case that made a working Dojo look like one that never ran."""
    _deny(monkeypatch, report)
    note = read_dojo_latest(report)["note"]
    assert "permission denied" in note.lower()
    assert "not found" not in note
    assert str(report) in note


def test_permission_note_names_the_fix(report: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator reading this in the UI should know what to change."""
    _deny(monkeypatch, report)
    note = read_dojo_latest(report)["note"].lower()
    assert "0644" in note or "readable" in note


def test_malformed_json_is_reported_as_unreadable_not_missing(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{ truncated", encoding="utf-8")
    note = read_dojo_latest(path)["note"]
    assert "not found" not in note
    assert "unreadable" in note


def test_a_json_array_is_rejected_with_its_type(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    note = read_dojo_latest(path)["note"]
    assert "JSON object" in note
    assert "list" in note


def test_a_directory_in_place_of_the_report_is_reported(tmp_path: Path) -> None:
    directory = tmp_path / "latest.json"
    directory.mkdir()
    note = read_dojo_latest(directory)["note"]
    assert "directory" in note


# --------------------------------------------------------------------------- #
# live_state gets the same treatment                                          #
# --------------------------------------------------------------------------- #
def test_live_state_missing_says_not_found(tmp_path: Path) -> None:
    assert "not found" in read_live_state(tmp_path / "absent.json")["note"]


def test_live_state_unreadable_says_permission_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "live_state.json"
    path.write_text(json.dumps({"schema_version": "spyder.dashboard.v1"}), encoding="utf-8")
    _deny(monkeypatch, path)
    assert "permission denied" in read_live_state(path)["note"].lower()


# --------------------------------------------------------------------------- #
# The bundle the dashboard actually serves                                    #
# --------------------------------------------------------------------------- #
def test_bundle_surfaces_the_permission_reason_as_the_dojo_summary(
    tmp_path: Path, report: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/api/learning` shows dojo_status.summary — it must carry the reason."""
    _deny(monkeypatch, report)
    bundle = read_dashboard_bundle(
        live_path=tmp_path / "absent_live.json", dojo_path=report
    )
    assert "permission denied" in bundle["dojo_status"]["summary"].lower()


def test_bundle_keeps_the_report_path_for_the_operator(
    tmp_path: Path, report: Path
) -> None:
    bundle = read_dashboard_bundle(
        live_path=tmp_path / "absent_live.json", dojo_path=report
    )
    assert bundle["dojo_status"]["latest_report_path"] == str(report)
