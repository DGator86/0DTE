"""
chain_store.py
==============
Record live TickSnapshots to disk and replay them as a DataFeed.

Why this exists: every feed adapter serves LIVE snapshots only, so the
walk-forward/backtest harness had nothing real to chew on — the only
real-data instrument was the shadow journal accumulating at one day per day.
Recording costs ~1 MB/session gzipped; a few weeks of recordings turn
walk_forward.run_walk_forward into an actual out-of-sample test on actual
markets. Start recording early; you cannot backfill what you never saved.

Format — one gzipped JSONL file per session (ticks_YYYY-MM-DD.jsonl.gz),
appended live (gzip members concatenate transparently on read):

  {"t":"tick","ts":...,"market":{...},"chain":{...},"bars":[[iso,o,h,l,c,v],...]}
  {"t":"settle","date":"YYYY-MM-DD","price":...}

Bars are stored incrementally (only bars newer than the previous record), so a
session file stays small; RecordedFeed reassembles the rolling window.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import glob
import gzip
import json
import os
from typing import Optional

import numpy as np

from gate_scorer import MarketSnapshot
from resample import RawBars
from rnd_extractor import ChainQuote, ChainSnapshot
from unified_loop import TickSnapshot

ET_FMT = "%Y-%m-%d"


# --------------------------------------------------------------------------- #
# Serialization helpers                                                        #
# --------------------------------------------------------------------------- #
def _market_to_dict(m: MarketSnapshot) -> dict:
    d = dataclasses.asdict(m)
    d["now"] = m.now.isoformat()
    return d


def _market_from_dict(d: dict) -> MarketSnapshot:
    d = dict(d)
    d["now"] = dt.datetime.fromisoformat(d["now"])
    return MarketSnapshot(**d)


def _chain_to_dict(c: ChainSnapshot) -> dict:
    return {
        "spot": c.spot, "t_years": c.t_years, "r": c.r,
        "quotes": [[q.strike, q.call_bid, q.call_ask, q.put_bid, q.put_ask]
                   for q in c.quotes],
    }


def _chain_from_dict(d: dict) -> ChainSnapshot:
    return ChainSnapshot(
        quotes=[ChainQuote(*q) for q in d["quotes"]],
        spot=d["spot"], t_years=d["t_years"], r=d["r"],
    )


def _bars_rows(bars: RawBars, after: Optional[np.datetime64]) -> list:
    ts = np.asarray(bars.ts, dtype="datetime64[ns]")
    mask = np.ones(len(ts), dtype=bool) if after is None else ts > after
    idx = np.nonzero(mask)[0]
    return [[str(ts[i]), float(bars.open[i]), float(bars.high[i]),
             float(bars.low[i]), float(bars.close[i]), float(bars.volume[i])]
            for i in idx]


# --------------------------------------------------------------------------- #
# Recorder                                                                     #
# --------------------------------------------------------------------------- #
class ChainRecorder:
    """Append-only tick/settlement recorder. Best-effort: a failed write must
    never break a live tick, so record() swallows exceptions."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._last_bar_ts: Optional[np.datetime64] = None
        self._last_session: Optional[str] = None
        self._seq = 0                               # per-session source sequence
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_date: str) -> str:
        return os.path.join(self.directory, f"ticks_{session_date}.jsonl.gz")

    def _append(self, session_date: str, obj: dict) -> None:
        with gzip.open(self._path(session_date), "at", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def record(self, now: dt.datetime, snap: TickSnapshot) -> None:
        try:
            session = now.date().isoformat()
            if session != self._last_session:
                self._last_bar_ts = None          # new session file: full bar window once
                self._last_session = session
                self._seq = 0
            rec = {
                "t": "tick",
                "ts": now.isoformat(),
                "seq": self._seq,                 # stable per-session source sequence
                "market": _market_to_dict(snap.market),
                "chain": _chain_to_dict(snap.chain) if snap.chain is not None else None,
                "bars": _bars_rows(snap.bars, self._last_bar_ts) if snap.bars is not None else [],
            }
            self._seq += 1
            if snap.bars is not None and len(snap.bars.ts):
                self._last_bar_ts = np.asarray(snap.bars.ts, dtype="datetime64[ns]")[-1]
            self._append(session, rec)
        except Exception:
            pass

    def record_settlement(self, session_date: str, price: float) -> None:
        try:
            self._append(session_date, {"t": "settle", "date": session_date,
                                        "price": float(price)})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Replay feed                                                                  #
# --------------------------------------------------------------------------- #
class RecordedFeed:
    """
    Replay recorded sessions as a unified_loop.DataFeed.

    snapshot(now) serves the next unserved recorded tick with ts <= now (so it
    composes with run_replay/walk-forward driven by .timestamps()). Bars are
    reassembled from the incremental rows into a rolling lookback window.
    """

    def __init__(self, directory: str, lookback_minutes: int = 7800) -> None:
        self.directory = directory
        self.lookback = lookback_minutes
        self._ticks: list[dict] = []
        self._settles: dict[str, float] = {}
        self._bars_acc: list[list] = []            # accumulated [iso,o,h,l,c,v]
        self._idx = 0
        self._load()

    # -- loading ---------------------------------------------------------------
    def _load(self) -> None:
        for path in sorted(glob.glob(os.path.join(self.directory, "ticks_*.jsonl.gz"))):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue                    # truncated tail from a crash
                    if obj.get("t") == "tick":
                        self._ticks.append(obj)
                    elif obj.get("t") == "settle":
                        self._settles[obj["date"]] = float(obj["price"])
        self._ticks.sort(key=lambda o: o["ts"])

    def timestamps(self) -> list[dt.datetime]:
        """All recorded tick timestamps — feed these to run_replay/walk-forward."""
        return [dt.datetime.fromisoformat(o["ts"]) for o in self._ticks]

    def replay_ticks(self):
        """
        Yield (seq, ts, TickSnapshot) for every recorded tick WITHOUT touching
        the serving state (`_idx`/`_bars_acc`) — the dataset builder's entry
        point (prediction/dataset.py). Bars are reassembled the same way
        snapshot() does, then defensively as-of filtered so a malformed
        recording can never leak a future bar into an earlier observation.

        seq is the recorded per-session source sequence; recordings made
        before seq existed fall back to the tick's position within the run.
        """
        from prediction.asof import bars_asof
        bars_acc: list[list] = []
        for i, rec in enumerate(self._ticks):
            ts = dt.datetime.fromisoformat(rec["ts"])
            bars_acc.extend(rec.get("bars") or [])
            if len(bars_acc) > self.lookback:
                bars_acc = bars_acc[-self.lookback:]
            bars = None
            if bars_acc:
                bars = RawBars(
                    ts=np.array([r[0] for r in bars_acc], dtype="datetime64[ns]"),
                    open=np.array([r[1] for r in bars_acc], dtype=float),
                    high=np.array([r[2] for r in bars_acc], dtype=float),
                    low=np.array([r[3] for r in bars_acc], dtype=float),
                    close=np.array([r[4] for r in bars_acc], dtype=float),
                    volume=np.array([r[5] for r in bars_acc], dtype=float),
                )
                bars = bars_asof(bars, ts)
            snap = TickSnapshot(
                market=_market_from_dict(rec["market"]),
                bars=bars,
                chain=_chain_from_dict(rec["chain"]) if rec.get("chain") else None,
            )
            yield rec.get("seq", i), ts, snap

    def __len__(self) -> int:
        return len(self._ticks)

    # -- DataFeed protocol -------------------------------------------------------
    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        if self._idx >= len(self._ticks):
            return None
        rec = self._ticks[self._idx]
        if dt.datetime.fromisoformat(rec["ts"]) > now:
            return None                             # not yet reached this recording
        self._idx += 1

        self._bars_acc.extend(rec.get("bars") or [])
        if len(self._bars_acc) > self.lookback:
            self._bars_acc = self._bars_acc[-self.lookback:]

        bars = None
        if self._bars_acc:
            arr = self._bars_acc
            bars = RawBars(
                ts=np.array([r[0] for r in arr], dtype="datetime64[ns]"),
                open=np.array([r[1] for r in arr], dtype=float),
                high=np.array([r[2] for r in arr], dtype=float),
                low=np.array([r[3] for r in arr], dtype=float),
                close=np.array([r[4] for r in arr], dtype=float),
                volume=np.array([r[5] for r in arr], dtype=float),
            )

        return TickSnapshot(
            market=_market_from_dict(rec["market"]),
            bars=bars,
            chain=_chain_from_dict(rec["chain"]) if rec.get("chain") else None,
        )

    def settlement_price(self, session_date: str) -> Optional[float]:
        return self._settles.get(session_date)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "ticks"
    feed = RecordedFeed(d)
    ts = feed.timestamps()
    print(f"{len(feed)} recorded ticks in {d!r}"
          + (f" spanning {ts[0]} → {ts[-1]}" if ts else ""))
    print(f"settlements: {feed._settles}")
