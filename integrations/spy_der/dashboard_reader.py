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


def _read_json(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return ``(data, note)``. ``note`` says why ``data`` is ``None``.

    The reason matters more than it looks. This used to collapse every failure
    to ``None``, and the caller reported "not found" for all of them — so a
    report that existed but could not be *read* was displayed as a report that
    had never been written, and the dashboard told the operator to enable timers
    that were already running. Permission denied is the failure that actually
    happens here: SPY-DER writes this state as ``spy-der`` and the dashboard
    reads it as ``zerodte``.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, f"not found: {path}"
    except PermissionError:
        log.warning("permission denied reading %s", path)
        return None, (
            f"permission denied: {path} — the dashboard reads this as a different "
            "user than the one that wrote it; the file must be readable by "
            "others (0644) and every parent directory traversable"
        )
    except IsADirectoryError:
        return None, f"expected a file, found a directory: {path}"
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed reading %s: %s", path, exc)
        return None, f"unreadable ({type(exc).__name__}): {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"expected a JSON object, got {type(data).__name__}: {path}"
    return data, None


def read_live_state(
    path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return spyder.dashboard.v1 or a soft note when missing/invalid."""
    p = live_state_path(path)
    data, note = _read_json(p)
    if data is None:
        return {"note": f"spy-der live_state {note}", "path": str(p)}
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
    """Return dojo latest report JSON, or a soft note explaining why not."""
    p = dojo_latest_path(path)
    data, note = _read_json(p)
    if data is None:
        return {"note": f"spy-der dojo latest {note}", "path": str(p)}
    return data


def _age_seconds(value: Any, now: Any) -> Optional[float]:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = _dt.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_tz.utc)
    return max(0.0, (now - parsed).total_seconds())


#: Services SPY-DER publishes a heartbeat for, and what each is for. A service
#: absent from `health/` is reported "never_seen" rather than omitted — silence
#: about a service that should exist is the failure being hidden.
EXPECTED_SERVICES = {
    "market": "provider ingestion",
    "engine": "deterministic stages",
    "settlement": "session settlement",
}

#: Beyond this multiple of a service's own interval, its heartbeat is stale.
#: Matches SPY-DER's own classification so both sides agree.
STALE_INTERVAL_MULTIPLE = 3.0


def read_system_status(state_root_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Service health, feed progress and deployed commit, read from files.

    Deliberately reads the published files rather than calling SPY-DER's own
    `/v1/system`: status is most needed when something is down, and that
    includes SPY-DER's API. Files keep working when the service does not.

    No SPY-DER import — same ownership boundary as the rest of this module.
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    root = Path(state_root_path) if state_root_path is not None else state_root()
    now = _dt.now(tz=_tz.utc)

    services: list[dict[str, Any]] = []
    health = root / "health"
    seen: dict[str, dict[str, Any]] = {}
    if health.is_dir():
        for path in sorted(health.glob("*.json")):
            body = _read_json(path)[0] or {}
            seen[path.stem] = body if isinstance(body, dict) else {}
    for name, purpose in EXPECTED_SERVICES.items():
        body = seen.pop(name, None)
        if body is None:
            services.append(
                {"service": name, "purpose": purpose, "state": "never_seen",
                 "detail": "no heartbeat has ever been published"}
            )
            continue
        interval = float(body.get("interval_seconds") or 0.0)
        age = _age_seconds(body.get("updated_at"), now)
        if age is None:
            state = "unknown"
        elif interval <= 0 or age <= interval:
            state = "ok"
        elif age <= interval * STALE_INTERVAL_MULTIPLE:
            state = "late"
        else:
            state = "stale"
        services.append(
            {"service": name, "purpose": purpose, "state": state,
             "age_seconds": age, "detail": str(body.get("detail") or "")}
        )
    for name, body in seen.items():
        services.append({"service": name, "purpose": "", "state": "ok",
                         "detail": str(body.get("detail") or "")})

    feed = _feed_status(root, now)
    deploy = _read_json(root / "deploy.json")[0] or {}
    if deploy:
        deploy["deployed_age_seconds"] = _age_seconds(deploy.get("deployed_at"), now)
        deploy["state"] = "ok"
    else:
        deploy = {"state": "unknown"}

    states = {s["state"] for s in services}
    if states & {"stale", "never_seen"} or feed.get("state") == "unreadable":
        overall = "degraded"
    elif states & {"late", "unknown"} or feed.get("state") in {"no_recordings", "empty"}:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "generated_at": now.isoformat(),
        "overall": overall,
        "services": services,
        "feed": feed,
        "deploy": deploy,
        "source": "spy-der",
    }


def _feed_status(root: Path, now: Any) -> dict[str, Any]:
    """Read from the tape, not a self-report.

    A heartbeat says the loop is turning; this says data actually landed. They
    disagree exactly when something is wrong.
    """
    market = root / "market"
    if not market.is_dir():
        return {"state": "no_recordings", "note": f"{market} does not exist"}
    sessions = sorted(p.stem for p in market.glob("*.jsonl"))
    if not sessions:
        return {"state": "no_recordings", "note": "no session recordings yet"}
    latest = sessions[-1]
    ticks = 0
    last_line = ""
    try:
        with open(market / f"{latest}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ticks += 1
                    last_line = line
    except PermissionError:
        return {"state": "unreadable", "session": latest, "note": "permission denied"}
    except OSError as exc:
        return {"state": "unreadable", "session": latest, "note": str(exc)}

    observed_at = None
    provider = None
    if last_line:
        try:
            snapshot = (json.loads(last_line) or {}).get("snapshot") or {}
            observed_at = snapshot.get("timestamp")
            selected = snapshot.get("selected_providers") or []
            if selected and isinstance(selected[0], dict):
                provider = selected[0].get("provider")
        except ValueError:
            observed_at = None
    return {
        "state": "recording" if ticks else "empty",
        "session": latest,
        "sessions_recorded": len(sessions),
        "ticks": ticks,
        "last_tick_at": observed_at,
        "last_tick_age_seconds": _age_seconds(observed_at, now),
        "provider": provider,
    }


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
