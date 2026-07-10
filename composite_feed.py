"""
composite_feed.py  —  failover wrapper that chains multiple DataFeeds.

WHY THIS EXISTS
    A single feed returning None (API outage, rate limit, a one-sided book) makes
    the tick loop go dark for that minute. CompositeFeed wraps an ordered list of
    feeds and, per tick, returns the first non-None TickSnapshot — so the chain
    provider can fail over from Tradier -> Tastytrade -> Massive without the
    orchestrator knowing. settlement_price() fails over independently, with the
    free Yahoo backstop as the final guarantee that a session can always settle.

    build_default_feed() auto-detects which providers are usable from the
    environment (which credentials are present) and assembles the composite in
    priority order, so the VPS service "just works" with whatever you've funded.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Optional, Sequence

from unified_loop import TickSnapshot, DataFeed
from yahoo_feed import YahooBackstop

log = logging.getLogger("composite_feed")


class CompositeFeed:
    """Try each member feed in order; first to produce a snapshot wins."""

    def __init__(
        self,
        feeds: Sequence[DataFeed],
        settlement_backstop: Optional[YahooBackstop] = None,
    ) -> None:
        if not feeds:
            raise ValueError("CompositeFeed needs at least one feed")
        self.feeds = list(feeds)
        self._settlement_backstop = settlement_backstop
        self._last_source: Optional[str] = None

    @property
    def last_source(self) -> Optional[str]:
        """Class name of the feed that served the most recent snapshot."""
        return self._last_source

    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        for feed in self.feeds:
            name = type(feed).__name__
            try:
                snap = feed.snapshot(now)
            except Exception as exc:
                log.warning("%s.snapshot raised: %s", name, exc)
                continue
            if snap is not None:
                if self._last_source != name:
                    log.info("snapshot served by %s", name)
                    self._last_source = name
                # Tag provenance for GEX variant journaling (PR 9); does not
                # alter MarketSnapshot policy fields.
                try:
                    if not getattr(snap, "gex_feed_source", ""):
                        snap.gex_feed_source = name
                except Exception:
                    pass
                return snap
        return None

    def settlement_price(self, session_date: str) -> Optional[float]:
        for feed in self.feeds:
            try:
                px = feed.settlement_price(session_date)
            except Exception:
                px = None
            if px is not None:
                return px
        if self._settlement_backstop is not None:
            return self._settlement_backstop.settlement(session_date)
        return None


# --------------------------------------------------------------------------- #
# Auto-assembly from the environment                                            #
# --------------------------------------------------------------------------- #
def build_default_feed(symbol: str = "SPY", **feed_kwargs) -> DataFeed:
    """
    Assemble a CompositeFeed from whatever providers are credentialed in the
    environment, in priority order:

        1. Tradier    (TRADIER_ACCESS_TOKEN)        real NBBO + execution venue
        2. Tastytrade (TASTYTRADE_CLIENT_SECRET +   real NBBO fallback
                       TASTYTRADE_REFRESH_TOKEN)
        3. Massive    (MASSIVE_API_KEY)             greeks/OI (no real-time NBBO)

    Yahoo is always attached as the settlement backstop. Raises if no provider
    is credentialed (Yahoo alone serves no option chain, so it can't drive the
    pipeline on its own).
    """
    feeds: list[DataFeed] = []

    if os.environ.get("TRADIER_ACCESS_TOKEN"):
        from tradier_feed import TradierDataFeed
        feeds.append(TradierDataFeed(underlying=symbol, **feed_kwargs))
        log.info("Tradier feed enabled (primary).")

    if (os.environ.get("TASTYTRADE_CLIENT_SECRET")
            and os.environ.get("TASTYTRADE_REFRESH_TOKEN")):
        from tastytrade_feed import TastytradeDataFeed
        feeds.append(TastytradeDataFeed(underlying=symbol, **feed_kwargs))
        log.info("Tastytrade feed enabled (fallback).")

    if os.environ.get("MASSIVE_API_KEY"):
        from massive_feed import MassiveDataFeed
        feeds.append(MassiveDataFeed(underlying=symbol, **feed_kwargs))
        log.info("Massive feed enabled (fallback; no real-time NBBO).")

    if not feeds:
        raise RuntimeError(
            "No market-data provider credentialed. Set at least one of "
            "TRADIER_ACCESS_TOKEN, TASTYTRADE_CLIENT_SECRET+TASTYTRADE_REFRESH_TOKEN, "
            "or MASSIVE_API_KEY."
        )

    return CompositeFeed(feeds, settlement_backstop=YahooBackstop(symbol))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Providers credentialed in this environment:")
    print("  Tradier:    ", bool(os.environ.get("TRADIER_ACCESS_TOKEN")))
    print("  Tastytrade: ", bool(os.environ.get("TASTYTRADE_CLIENT_SECRET")
                                  and os.environ.get("TASTYTRADE_REFRESH_TOKEN")))
    print("  Massive:    ", bool(os.environ.get("MASSIVE_API_KEY")))
    try:
        feed = build_default_feed()
        names = [type(f).__name__ for f in feed.feeds]  # type: ignore[attr-defined]
        print("Composite order:", " -> ".join(names))
    except RuntimeError as e:
        print("build_default_feed:", e)
