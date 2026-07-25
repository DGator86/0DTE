"""Read SPY-DER dashboard outputs — the only AI surface the 0DTE UI may see."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from integrations.spy_der.contracts import (
    DASHBOARD_SCHEMA,
    DEFAULT_DOJO_LATEST,
    DEFAULT_LIVE_STATE,
    DEFAULT_STATE_ROOT,
    validate_dashboard_packet,
)

log = logging.getLogger("integrations.spy_der.dashboard_reader")


def state_root() -> Path:
    env = os.environ.get("SPY_DER_STATE_ROOT", "").strip()
    return Path(env or DEFAULT_STATE_ROOT)


def live_state_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SPY_DER_LIVE_STATE", "").strip()
    return Path(env or DEFAULT_LIVE_STATE)


def dojo_latest_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SPY_DER_DOJO_LATEST", "").strip()
    return Path(env or DEFAULT_DOJO_LATEST)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed reading %s: %s", path, exc)
        return None


def read_live_state(
    path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return spyder.dashboard.v1 or a soft note when missing/invalid."""
    p = live_state_path(path)
    data = _read_json(p)
    if data is None:
        return {"note": f"spy-der live_state not found: {p}", "path": str(p)}
    try:
        validate_dashboard_packet(data)
    except ValueError as exc:
        # Still return payload for display if schema drifts slightly, but flag it.
        log.warning("live_state schema issue at %s: %s", p, exc)
        return {
            "note": f"spy-der live_state schema issue: {exc}",
            "path": str(p),
            "raw": data,
            "schema_version": data.get("schema_version"),
        }
    return data


def read_dojo_latest(
    path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return dojo latest report JSON or a soft note when missing."""
    p = dojo_latest_path(path)
    data = _read_json(p)
    if data is None:
        return {"note": f"spy-der dojo latest not found: {p}", "path": str(p)}
    return data


def read_dashboard_bundle(
    *,
    live_path: Optional[str | Path] = None,
    dojo_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Combined view for dashboard APIs."""
    live = read_live_state(live_path)
    dojo = read_dojo_latest(dojo_path)
    # Prefer embedded dojo status from live_state when present.
    embedded = live.get("dojo") if isinstance(live.get("dojo"), dict) else None
    return {
        "schema_version": live.get("schema_version") or DASHBOARD_SCHEMA,
        "live": live,
        "dojo": dojo,
        "dojo_status": embedded or {
            "latest_status": dojo.get("status") or dojo.get("latest_status") or "UNKNOWN",
            "summary": dojo.get("summary") or dojo.get("note") or "",
            "latest_report_path": str(dojo_latest_path(dojo_path)),
        },
    }
