"""
prediction/asof.py
==================
As-of (point-in-time) source rules for the canonical V2 dataset
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §8.3).

The one rule: a feature may be included in an observation only when its
source timestamp is <= the observation timestamp. A one-minute bar ending
at 10:01 may be used for a 10:01 observation; a bar ending at 10:02 may
not; an option quote stamped 10:01:07 may not enter a 10:01:00 observation.

Bar-timestamp convention: resample.py labels bars with `label="right",
closed="right"`, and every live feed derives RawBars.ts from epoch seconds
(naive UTC). So RawBars timestamps ARE bar END times on a naive-UTC clock,
and `ts <= observation_ts` is exactly the leak-free filter.

Missing values are recorded as missing, never replaced with neutral values
(§8.7) — the statistical model must know whether a neutral-looking value
was observed or imputed.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from resample import RawBars

UTC = dt.timezone.utc


class AsOfViolation(ValueError):
    """A source timestamped AFTER the observation tried to enter it."""


def _to_naive_utc(ts: dt.datetime) -> dt.datetime:
    """Aware -> naive UTC (the feeds' bar clock); naive passes through."""
    if ts.tzinfo is not None:
        return ts.astimezone(UTC).replace(tzinfo=None)
    return ts


def ensure_asof(name: str, source_ts: dt.datetime,
                observation_ts: dt.datetime) -> float:
    """
    Assert `source_ts <= observation_ts`; return the source age in seconds.
    Raises AsOfViolation when the source is from the future. Both timestamps
    are normalized to naive UTC before comparison so aware/naive mixes
    cannot silently compare wrong.
    """
    src = _to_naive_utc(source_ts)
    obs = _to_naive_utc(observation_ts)
    age = (obs - src).total_seconds()
    if age < 0:
        raise AsOfViolation(
            f"source {name!r} is {-age:.3f}s in the FUTURE of the "
            f"observation ({source_ts.isoformat()} > "
            f"{observation_ts.isoformat()})")
    return age


def bars_asof(bars: RawBars, observation_ts: dt.datetime) -> RawBars:
    """
    Return only the bars whose END timestamp is <= the observation timestamp.
    This is the defensive filter for replay/backfill paths: no future bar can
    influence an earlier feature snapshot even if the recording contains one.
    """
    cutoff = np.datetime64(_to_naive_utc(observation_ts))
    ts = np.asarray(bars.ts, dtype="datetime64[ns]")
    n = int(np.searchsorted(ts, cutoff, side="right"))
    return RawBars(
        ts=ts[:n], open=bars.open[:n], high=bars.high[:n],
        low=bars.low[:n], close=bars.close[:n], volume=bars.volume[:n],
        signed_volume=(bars.signed_volume[:n]
                       if bars.signed_volume is not None else None),
        tick=bars.tick[:n] if bars.tick is not None else None,
    )


@dataclass
class AsOfFeatureBuilder:
    """
    Collect one observation's raw features under the as-of rule, producing
    the three parallel dicts the canonical feature table stores (§8.4/8.7):

      features     name -> value (None when missing)
      missingness  name -> 0/1 (1 = the source did not provide a value)
      source_ages  name -> age in seconds at observation time (None when
                   missing or when the source carries no timestamp)

    `add()` REJECTS future-stamped sources (AsOfViolation) rather than
    silently dropping them: a future source reaching this point is a
    pipeline bug that must surface, not degrade.
    """
    observation_ts: dt.datetime
    features: dict = field(default_factory=dict)
    missingness: dict = field(default_factory=dict)
    source_ages: dict = field(default_factory=dict)

    def add(self, name: str, value,
            source_ts: Optional[dt.datetime] = None) -> None:
        if value is None or (isinstance(value, float) and value != value):
            self.add_missing(name)
            return
        age = None
        if source_ts is not None:
            age = ensure_asof(name, source_ts, self.observation_ts)
        self.features[name] = value
        self.missingness[name] = 0
        self.source_ages[name] = age

    def add_missing(self, name: str) -> None:
        self.features[name] = None
        self.missingness[name] = 1
        self.source_ages[name] = None

    def coverage(self) -> float:
        """Fraction of registered features that were actually observed."""
        if not self.missingness:
            return 0.0
        present = sum(1 for m in self.missingness.values() if m == 0)
        return present / len(self.missingness)

    def build(self) -> dict:
        return {
            "features": dict(self.features),
            "missingness": dict(self.missingness),
            "source_ages": dict(self.source_ages),
            "coverage": round(self.coverage(), 6),
        }
