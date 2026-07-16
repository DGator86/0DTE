"""Canonical market-data contracts.

These objects deliberately contain no provider-specific SDK types. They are the
boundary between ingestion and all downstream processing.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


MARKET_SNAPSHOT_SCHEMA = "market.snapshot.v1"


class FeedStatus(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class FeedObservation:
    name: str
    status: FeedStatus
    provider: str = ""
    observed_at: dt.datetime | None = None
    age_seconds: float | None = None
    freshness_limit_seconds: float | None = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {FeedStatus.LIVE, FeedStatus.FALLBACK}


@dataclass(frozen=True)
class DataQuality:
    score: float
    coverage: float
    hard_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("data-quality score must be within [0, 1]")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("data-quality coverage must be within [0, 1]")

    @property
    def valid(self) -> bool:
        return not self.hard_failures


@dataclass(frozen=True)
class CanonicalMarketSnapshot:
    snapshot_id: str
    timestamp: dt.datetime
    symbol: str
    spot: float
    schema_version: str = MARKET_SNAPSHOT_SCHEMA
    feeds: Mapping[str, FeedObservation] = field(default_factory=dict)
    market: Mapping[str, Any] = field(default_factory=dict)
    bars: Any = None
    option_chain: Any = None
    option_rows: tuple[Any, ...] = ()
    weekly_option_rows: tuple[Any, ...] = ()
    data_quality: DataQuality = field(
        default_factory=lambda: DataQuality(score=0.0, coverage=0.0)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "feeds", MappingProxyType(dict(self.feeds)))
        object.__setattr__(self, "market", MappingProxyType(dict(self.market)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "option_rows", tuple(self.option_rows))
        object.__setattr__(self, "weekly_option_rows", tuple(self.weekly_option_rows))

    @property
    def has_chain(self) -> bool:
        return self.option_chain is not None

    @property
    def required_feeds_live(self) -> bool:
        required = ("spot", "bars", "option_chain", "settlement")
        return all(name in self.feeds and self.feeds[name].usable for name in required)
