"""Publish OutcomePacket JSON for SPY-DER FileMarketExperienceProvider."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from integrations.spy_der.contracts import (
    build_outcome_packet,
    validate_outcome_packet,
)
from integrations.spy_der.market_publisher import experience_root

log = logging.getLogger("integrations.spy_der.outcome_publisher")


def publish_outcome_packet(
    packet: dict[str, Any],
    *,
    root: Optional[str | Path] = None,
) -> Path:
    packet = validate_outcome_packet(packet)
    base = experience_root(root)
    out_dir = base / "outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_id = str(packet["snapshot_id"]).replace("/", "_").replace("\\", "_")
    path = out_dir / f"{snap_id}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(packet, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    log.debug("published OutcomePacket %s -> %s", packet["snapshot_id"], path)
    return path


def publish_settlement_outcomes(
    *,
    session_date: str,
    symbol: str,
    rows: list[dict[str, Any]],
    settled_at: Any = None,
    root: Optional[str | Path] = None,
) -> list[Path]:
    """Publish one OutcomePacket per settled journal row that has a snapshot_id."""
    written: list[Path] = []
    for row in rows or []:
        snap_id = row.get("snapshot_id")
        if not snap_id:
            continue
        action = str(row.get("decision") or row.get("action") or "UNKNOWN")
        packet = build_outcome_packet(
            snapshot_id=str(snap_id),
            session_date=session_date,
            symbol=symbol,
            candidate_id=row.get("candidate_id") or row.get("selected_candidate_id"),
            action=action,
            realized_pnl=row.get("realized_pnl"),
            settled=True,
            labels={
                "track": row.get("track") or row.get("paper_track"),
                "family": row.get("selected_family") or row.get("family"),
                "direction": row.get("regime_direction") or row.get("direction"),
                "session_date": session_date,
            },
            settled_at=settled_at,
        )
        written.append(publish_outcome_packet(packet, root=root))
    return written
