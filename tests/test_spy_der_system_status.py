"""`/api/system` — the panel that removes the need to SSH.

Reads SPY-DER's published files directly rather than calling its `/v1/system`.
That is deliberate: status matters most when something is down, and that
includes SPY-DER's own API. Files keep working when the service does not.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from integrations.spy_der.dashboard_reader import read_system_status

UTC = timezone.utc


def _hb(root: Path, service: str, *, interval: float, age_seconds: float, detail: str = "") -> None:
    directory = root / "health"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
    (directory / f"{service}.json").write_text(
        json.dumps(
            {
                "service": service,
                "updated_at": stamp.isoformat(),
                "interval_seconds": interval,
                "detail": detail,
            }
        ),
        encoding="utf-8",
    )


def _recording(root: Path, session: str = "2026-07-22", ticks: int = 3) -> None:
    market = root / "market"
    market.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)
    lines = [
        json.dumps(
            {
                "snapshot": {
                    "timestamp": (now - timedelta(seconds=30 * (ticks - i))).isoformat(),
                    "selected_providers": [{"component": "spot", "provider": "tradier"}],
                }
            }
        )
        for i in range(ticks)
    ]
    (market / f"{session}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _healthy(root: Path) -> None:
    _hb(root, "market", interval=60, age_seconds=5, detail="390 tick(s)")
    _hb(root, "engine", interval=30, age_seconds=5)
    _hb(root, "settlement", interval=300, age_seconds=5)
    _recording(root)


def _by_name(status: dict) -> dict:
    return {s["service"]: s for s in status["services"]}


# --------------------------------------------------------------------------- #
# Services                                                                    #
# --------------------------------------------------------------------------- #
def test_healthy_deployment_is_ok(tmp_path: Path) -> None:
    _healthy(tmp_path)
    assert read_system_status(tmp_path)["overall"] == "ok"


def test_a_service_that_never_ran_is_reported(tmp_path: Path) -> None:
    """Omitting it would hide exactly the failure the operator is looking for."""
    _recording(tmp_path)
    services = _by_name(read_system_status(tmp_path))
    assert services["market"]["state"] == "never_seen"
    assert services["engine"]["state"] == "never_seen"


def test_stale_service_degrades_the_banner(tmp_path: Path) -> None:
    _healthy(tmp_path)
    _hb(tmp_path, "settlement", interval=300, age_seconds=7200)
    status = read_system_status(tmp_path)
    assert _by_name(status)["settlement"]["state"] == "stale"
    assert status["overall"] == "degraded"


def test_late_service_warns_but_is_not_degraded(tmp_path: Path) -> None:
    _healthy(tmp_path)
    _hb(tmp_path, "engine", interval=30, age_seconds=75)  # within 3x
    status = read_system_status(tmp_path)
    assert _by_name(status)["engine"]["state"] == "late"
    assert status["overall"] == "warn"


def test_staleness_is_relative_to_each_services_own_interval(tmp_path: Path) -> None:
    """60s is healthy for settlement (300s) and stale for engine (10s)."""
    _healthy(tmp_path)
    _hb(tmp_path, "settlement", interval=300, age_seconds=60)
    _hb(tmp_path, "engine", interval=10, age_seconds=60)
    services = _by_name(read_system_status(tmp_path))
    assert services["settlement"]["state"] == "ok"
    assert services["engine"]["state"] == "stale"


def test_service_detail_is_surfaced(tmp_path: Path) -> None:
    _healthy(tmp_path)
    assert "390" in _by_name(read_system_status(tmp_path))["market"]["detail"]


def test_an_unexpected_service_is_still_listed(tmp_path: Path) -> None:
    _healthy(tmp_path)
    _hb(tmp_path, "experimental", interval=60, age_seconds=5)
    assert "experimental" in _by_name(read_system_status(tmp_path))


def test_unreadable_heartbeat_does_not_crash_the_panel(tmp_path: Path) -> None:
    _healthy(tmp_path)
    (tmp_path / "health" / "market.json").write_text("{ truncated", encoding="utf-8")
    status = read_system_status(tmp_path)
    assert status["services"]  # still renders


# --------------------------------------------------------------------------- #
# Feed — read from the tape, not a self-report                                #
# --------------------------------------------------------------------------- #
def test_feed_reports_ticks_and_provider(tmp_path: Path) -> None:
    _healthy(tmp_path)
    feed = read_system_status(tmp_path)["feed"]
    assert feed["state"] == "recording"
    assert feed["ticks"] == 3
    assert feed["provider"] == "tradier"
    assert feed["last_tick_age_seconds"] is not None


def test_a_live_service_recording_nothing_still_warns(tmp_path: Path) -> None:
    """The disagreement worth surfacing: loop turning, no data landing."""
    _hb(tmp_path, "market", interval=60, age_seconds=5)
    _hb(tmp_path, "engine", interval=30, age_seconds=5)
    _hb(tmp_path, "settlement", interval=300, age_seconds=5)
    status = read_system_status(tmp_path)
    assert _by_name(status)["market"]["state"] == "ok"
    assert status["feed"]["state"] == "no_recordings"
    assert status["overall"] == "warn"


def test_feed_uses_the_latest_session(tmp_path: Path) -> None:
    _healthy(tmp_path)
    _recording(tmp_path, session="2026-07-23", ticks=7)
    feed = read_system_status(tmp_path)["feed"]
    assert feed["session"] == "2026-07-23"
    assert feed["ticks"] == 7
    assert feed["sessions_recorded"] == 2


# --------------------------------------------------------------------------- #
# Deploy                                                                      #
# --------------------------------------------------------------------------- #
def test_deploy_reports_the_live_commit(tmp_path: Path) -> None:
    _healthy(tmp_path)
    (tmp_path / "deploy.json").write_text(
        json.dumps(
            {
                "commit_short": "b0ba921",
                "subject": "Port the Tradier provider",
                "deployed_at": (datetime.now(tz=UTC) - timedelta(minutes=8)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    deploy = read_system_status(tmp_path)["deploy"]
    assert deploy["state"] == "ok"
    assert deploy["commit_short"] == "b0ba921"
    assert 460 < deploy["deployed_age_seconds"] < 500


def test_missing_deploy_file_is_unknown_not_fatal(tmp_path: Path) -> None:
    _healthy(tmp_path)
    assert read_system_status(tmp_path)["deploy"]["state"] == "unknown"


# --------------------------------------------------------------------------- #
# It has to work on a box where nothing has run                               #
# --------------------------------------------------------------------------- #
def test_bare_state_root_still_answers(tmp_path: Path) -> None:
    """That is exactly when an operator opens the page."""
    status = read_system_status(tmp_path)
    assert status["overall"] == "degraded"
    assert len(status["services"]) == 3
    assert status["feed"]["state"] == "no_recordings"


def test_absent_state_root_still_answers(tmp_path: Path) -> None:
    status = read_system_status(tmp_path / "nope")
    assert status["overall"] == "degraded"


# --------------------------------------------------------------------------- #
# Served by the API the dashboard calls                                       #
# --------------------------------------------------------------------------- #
def test_api_system_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    _healthy(tmp_path)
    monkeypatch.setenv("SPY_DER_STATE_ROOT", str(tmp_path))
    from dashboard import server

    data = asyncio.run(server.api_system())
    assert data["overall"] == "ok"
    assert data["source"] == "spy-der"
    assert {s["service"] for s in data["services"]} >= {"market", "engine", "settlement"}
