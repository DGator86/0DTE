"""Publish MarketPacket JSON for SPY-DER FileMarketExperienceProvider."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional  # noqa: F401 — Any used in annotations

from integrations.spy_der.contracts import (
    DEFAULT_EXPERIENCE_ROOT,
    build_market_packet,
    shadow_candidate_to_view,
    validate_market_packet,
)

log = logging.getLogger("integrations.spy_der.market_publisher")


def experience_root(root: Optional[str | Path] = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("ZERODTE_SPYDER_EXPERIENCE_ROOT", "").strip()
    return Path(env or DEFAULT_EXPERIENCE_ROOT)


def build_market_packet_from_tick(
    *,
    snapshot_id: str,
    session_date: Any,
    symbol: str,
    underlying_price: float,
    shadow_candidates: list[Any] | None = None,
    hard_vetoes: tuple[str, ...] | list[str] = (),
    forecast: Optional[dict[str, Any]] = None,
    forecast_uncertainty: float = 0.0,
    track_record: Optional[dict[str, Any]] = None,
    data_quality: float = 1.0,
    risk_max_size_scalar: float = 1.0,
    generated_at: Any = None,
) -> dict[str, Any]:
    views = [
        shadow_candidate_to_view(c, index=i, session_date=session_date)
        for i, c in enumerate(shadow_candidates or [])
    ]
    packet = build_market_packet(
        snapshot_id=snapshot_id,
        session_date=session_date,
        symbol=symbol,
        underlying_price=underlying_price,
        data_quality=data_quality,
        forecast_uncertainty=forecast_uncertainty,
        hard_vetoes=hard_vetoes,
        forecast=forecast,
        candidates=views,
        track_record=track_record,
        generated_at=generated_at,
        risk_max_size_scalar=risk_max_size_scalar,
    )
    return validate_market_packet(packet)


def publish_market_packet(
    packet: dict[str, Any],
    *,
    root: Optional[str | Path] = None,
) -> Path:
    """Atomically write snapshots/{snapshot_id}.json and update sessions.json."""
    packet = validate_market_packet(packet)
    base = experience_root(root)
    snap_dir = base / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_id = str(packet["snapshot_id"])
    # Avoid path traversal from untrusted snapshot ids.
    safe_id = snap_id.replace("/", "_").replace("\\", "_")
    path = snap_dir / f"{safe_id}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(packet, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    _append_session(base, str(packet["session_date"]))
    log.debug("published MarketPacket %s -> %s", snap_id, path)
    return path


def _append_session(base: Path, session_date: str) -> None:
    sessions_path = base / "sessions.json"
    sessions: list[str] = []
    if sessions_path.is_file():
        try:
            with open(sessions_path, encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, list):
                sessions = [str(x) for x in raw]
        except (OSError, ValueError, TypeError):
            sessions = []
    if session_date not in sessions:
        sessions.append(session_date)
        sessions = sorted(set(sessions))
        tmp = sessions_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(sessions, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, sessions_path)
