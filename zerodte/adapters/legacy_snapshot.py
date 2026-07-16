"""Bridge the existing ``unified_loop.TickSnapshot`` into the canonical contract.

The adapter uses structural typing so importing it does not import live feeds or
the legacy orchestrator. This keeps the new package dependency-light.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Any

from zerodte.contracts.market import (
    CanonicalMarketSnapshot,
    DataQuality,
    FeedObservation,
    FeedStatus,
)


def canonical_snapshot_from_tick(
    tick: Any,
    *,
    snapshot_id: str,
    symbol: str = "SPY",
    feed_observations: Mapping[str, FeedObservation] | None = None,
) -> CanonicalMarketSnapshot:
    market = getattr(tick, "market", None)
    if market is None:
        raise ValueError("legacy tick has no market snapshot")
    timestamp = getattr(market, "now", None)
    if not isinstance(timestamp, dt.datetime) or timestamp.tzinfo is None:
        raise ValueError("legacy market timestamp must be timezone-aware")
    spot = float(getattr(market, "spot", 0.0) or 0.0)
    bars = getattr(tick, "bars", None)
    chain = getattr(tick, "chain", None)

    feeds = dict(feed_observations or {})
    feeds.setdefault(
        "spot",
        FeedObservation(
            name="spot",
            status=FeedStatus.LIVE if spot > 0 else FeedStatus.INVALID,
            provider=str(getattr(tick, "gex_feed_source", "") or "legacy"),
            observed_at=timestamp,
        ),
    )
    feeds.setdefault(
        "bars",
        FeedObservation(
            name="bars",
            status=FeedStatus.LIVE if bars is not None else FeedStatus.MISSING,
            provider="legacy",
            observed_at=timestamp if bars is not None else None,
        ),
    )
    feeds.setdefault(
        "option_chain",
        FeedObservation(
            name="option_chain",
            status=FeedStatus.LIVE if chain is not None else FeedStatus.MISSING,
            provider=str(getattr(tick, "gex_feed_source", "") or "legacy"),
            observed_at=timestamp if chain is not None else None,
        ),
    )
    feeds.setdefault(
        "settlement",
        FeedObservation(
            name="settlement",
            status=FeedStatus.MISSING,
            provider="legacy",
            detail="settlement is resolved by the feed after the session",
        ),
    )

    hard_failures: list[str] = []
    if spot <= 0:
        hard_failures.append("invalid_spot")
    if bars is None:
        hard_failures.append("missing_bars")
    warnings: list[str] = []
    if chain is None:
        warnings.append("missing_option_chain")
    present = sum((spot > 0, bars is not None, chain is not None))

    return CanonicalMarketSnapshot(
        snapshot_id=snapshot_id,
        timestamp=timestamp,
        symbol=symbol,
        spot=spot,
        feeds=feeds,
        market=_object_mapping(market),
        bars=bars,
        option_chain=chain,
        option_rows=tuple(getattr(tick, "option_rows", None) or ()),
        weekly_option_rows=tuple(
            getattr(tick, "weekly_option_rows", None) or ()
        ),
        data_quality=DataQuality(
            score=present / 3.0,
            coverage=present / 3.0,
            hard_failures=tuple(hard_failures),
            warnings=tuple(warnings),
        ),
        metadata={
            "adapter": "legacy_tick.v1",
            "gex_feed_source": str(
                getattr(tick, "gex_feed_source", "") or ""
            ),
        },
    )


def _object_mapping(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"cannot convert {type(value).__name__} to mapping")
