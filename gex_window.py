"""
gex_window.py
=============
Shared, disk-persisted rolling window for the |net GEX| percentile rank.

Why this exists: each feed kept its own in-memory deque of the last ~100
SIGNED net-GEX prints. Two problems the dashboard made visible:

  1. Restart cold-start. The window died with the process, so every deploy or
     crash reset `gex_pct_rank` — and the gate's GEX_WEAK check (rank >= 0.60)
     then suppressed premium entries for no market reason.
  2. Wrong quantity and wrong horizon. The gate's documented semantics are
     "|GEX| must be in the top 40% of its trailing range": a MAGNITUDE rank.
     Ranking signed values over a ~100-minute intraday window instead measures
     "is GEX rising this hour", which pins at 0 on any slow decline.

This window ranks abs(net_gex) against a multi-day history persisted to JSON,
and reports a neutral 0.5 until it has a minimum sample — insufficient data
should read as "no opinion", not as an extreme that trips a gate.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GexRankWindow:
    """Rolling |net GEX| history -> percentile rank, persisted across restarts.

    path=None keeps it memory-only (tests, ad-hoc runs). Corrupt or missing
    state files are ignored — the window just re-warms.
    """
    path: Optional[str] = None
    max_age_days: float = 10.0        # trailing horizon for the rank
    max_entries: int = 5000           # hard cap on stored samples
    min_samples: int = 30             # below this, rank reads neutral 0.5
    _entries: list = field(default_factory=list)   # [(epoch_seconds, abs_gex)]

    def __post_init__(self):
        self._load()

    # -- public ---------------------------------------------------------------
    def rank(self, net_gex: float, now_epoch: Optional[float] = None) -> float:
        """Record this print and return the percentile rank of |net_gex|."""
        # Re-read before appending: failover means several feed instances share
        # one file, and each tick is served by only one of them.
        self._load()
        now_epoch = time.time() if now_epoch is None else float(now_epoch)
        mag = abs(float(net_gex))
        self._entries.append((now_epoch, mag))
        self._prune(now_epoch)
        self._save()
        if len(self._entries) < self.min_samples:
            return 0.5
        below = sum(1 for _, m in self._entries if m < mag)
        return below / len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals --------------------------------------------------------------
    def _prune(self, now_epoch: float) -> None:
        cutoff = now_epoch - self.max_age_days * 86400.0
        self._entries = [(t, m) for t, m in self._entries if t >= cutoff]
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def _load(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [(float(t), float(m)) for t, m in data.get("entries", [])]
        except Exception:
            self._entries = []          # corrupt state: re-warm, don't crash the feed

    def _save(self) -> None:
        if not self.path:
            return
        try:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".gex_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries}, f)
            os.replace(tmp, self.path)
        except Exception:
            pass                         # persistence is best-effort; never break a tick
