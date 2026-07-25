"""MarketExperienceProvider over published packets / recorded experience.

Does not move journal or tape data; exposes the filesystem layout SPY-DER
``FileMarketExperienceProvider`` already consumes.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Optional

from integrations.spy_der.contracts import (
    validate_market_packet,
    validate_outcome_packet,
)
from integrations.spy_der.market_publisher import experience_root


class MarketExperienceProvider:
    """Read versioned MarketPacket / OutcomePacket documents from disk.

    Directory layout::

        root/
          sessions.json
          snapshots/*.json
          outcomes/*.json
    """

    def __init__(self, root: Optional[str | Path] = None) -> None:
        self.root = experience_root(root)
        self._snapshots_dir = self.root / "snapshots"
        self._outcomes_dir = self.root / "outcomes"

    def sessions(self) -> Sequence[date]:
        sessions_file = self.root / "sessions.json"
        if sessions_file.is_file():
            try:
                with open(sessions_file, encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, list):
                    return sorted({date.fromisoformat(str(item)) for item in raw})
            except (OSError, ValueError, TypeError):
                pass
        found: set[date] = set()
        for packet in self._iter_markets():
            try:
                found.add(date.fromisoformat(str(packet["session_date"])))
            except (KeyError, ValueError, TypeError):
                continue
        return sorted(found)

    def snapshots(self, session: date) -> Iterable[dict[str, Any]]:
        for packet in self._iter_markets():
            try:
                if date.fromisoformat(str(packet["session_date"])) == session:
                    yield packet
            except (KeyError, ValueError, TypeError):
                continue

    def outcome(self, snapshot_id: str) -> Optional[dict[str, Any]]:
        safe = str(snapshot_id).replace("/", "_").replace("\\", "_")
        path = self._outcomes_dir / f"{safe}.json"
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return None
            return validate_outcome_packet(data)
        except (OSError, ValueError, TypeError):
            return None

    def _iter_markets(self) -> Iterable[dict[str, Any]]:
        if not self._snapshots_dir.is_dir():
            return
        for path in sorted(self._snapshots_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    yield validate_market_packet(data)
            except (OSError, ValueError, TypeError):
                continue
