"""
tests/test_feeds.py
===================
Unit tests for the market-data feed layer added for provider redundancy:

  - CompositeFeed failover (snapshot + settlement), provider-agnostic
  - TastytradeDataFeed pure logic (side mapping, row assembly) against the real
    SDK enums/events — these guard the one adapter that can't be exercised live
    in CI (no broker creds), so a future SDK bump or refactor can't silently
    break chain parsing.

No network and no credentials are used: feeds are stubbed and the Tastytrade
event objects are duck-typed namespaces matching the SDK's field names.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------------- #
# CompositeFeed                                                                 #
# --------------------------------------------------------------------------- #
class _Dead:
    def snapshot(self, now): return None
    def settlement_price(self, d): return None


class _Raises:
    def snapshot(self, now): raise RuntimeError("boom")
    def settlement_price(self, d): raise RuntimeError("boom")


class _Live:
    def __init__(self, tag): self.tag = tag
    def snapshot(self, now): return f"snap:{self.tag}"
    def settlement_price(self, d): return 123.45


def test_composite_skips_dead_and_raising_feeds():
    from composite_feed import CompositeFeed
    c = CompositeFeed([_Raises(), _Dead(), _Live("A")])
    assert c.snapshot(dt.datetime(2026, 6, 29, 12, 0)) == "snap:A"
    assert c.last_source == "_Live"


def test_composite_first_live_wins_and_short_circuits():
    from composite_feed import CompositeFeed
    c = CompositeFeed([_Live("first"), _Live("second")])
    assert c.snapshot(dt.datetime(2026, 6, 29, 12, 0)) == "snap:first"


def test_composite_settlement_failover():
    from composite_feed import CompositeFeed
    c = CompositeFeed([_Dead(), _Live("A")])
    assert c.settlement_price("2026-06-29") == 123.45


def test_composite_settlement_uses_backstop_when_all_dead():
    from composite_feed import CompositeFeed

    class _Backstop:
        def settlement(self, d): return 999.0

    c = CompositeFeed([_Dead(), _Dead()], settlement_backstop=_Backstop())
    assert c.snapshot(dt.datetime(2026, 6, 29, 12, 0)) is None
    assert c.settlement_price("2026-06-29") == 999.0


def test_composite_requires_at_least_one_feed():
    from composite_feed import CompositeFeed
    with pytest.raises(ValueError):
        CompositeFeed([])


def test_build_default_feed_raises_without_credentials(monkeypatch):
    import composite_feed
    for var in ("TRADIER_ACCESS_TOKEN", "TASTYTRADE_CLIENT_SECRET",
                "TASTYTRADE_REFRESH_TOKEN", "MASSIVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        composite_feed.build_default_feed()


# --------------------------------------------------------------------------- #
# TastytradeDataFeed pure logic (validated against the installed SDK)           #
# --------------------------------------------------------------------------- #
def test_side_mapping_matches_real_optiontype_enum():
    tf = pytest.importorskip("tastytrade_feed")
    OptionType = pytest.importorskip("tastytrade.instruments").OptionType
    for ot in OptionType:                      # CALL='C', PUT='P'
        opt = SimpleNamespace(option_type=ot)
        expected = "call" if ot.name == "CALL" else "put"
        assert tf._side_of(opt) == expected


def test_row_from_builds_valid_row():
    tf = pytest.importorskip("tastytrade_feed")
    meta = {"side": "call", "strike": 500.0}
    q = SimpleNamespace(bid_price=1.20, ask_price=1.30)
    g = SimpleNamespace(gamma=0.05, delta=0.42)
    row = tf._row_from(meta, q, g, oi=1234)
    assert row is not None
    assert row.side == "call" and row.strike == 500.0
    assert row.oi == 1234 and row.gamma == 0.05
    assert row.bid == 1.20 and row.ask == 1.30
    assert row.delta == 0.42                   # abs() of a positive call delta
    assert row.quote_valid is True
    assert row.quote_source == "tastytrade_live"


def test_row_from_takes_abs_of_put_delta():
    tf = pytest.importorskip("tastytrade_feed")
    meta = {"side": "put", "strike": 480.0}
    q = SimpleNamespace(bid_price=0.90, ask_price=1.00)
    g = SimpleNamespace(gamma=0.04, delta=-0.38)   # puts have negative delta
    row = tf._row_from(meta, q, g, oi=10)
    assert row.delta == pytest.approx(0.38)


def test_row_from_missing_oi_defaults_zero():
    tf = pytest.importorskip("tastytrade_feed")
    meta = {"side": "call", "strike": 500.0}
    q = SimpleNamespace(bid_price=1.0, ask_price=1.1)
    g = SimpleNamespace(gamma=0.05, delta=0.4)
    assert tf._row_from(meta, q, g, oi=None).oi == 0


@pytest.mark.parametrize("q,g", [
    (None, SimpleNamespace(gamma=0.05, delta=0.4)),                 # no quote
    (SimpleNamespace(bid_price=1.0, ask_price=1.1), None),          # no greeks
    (SimpleNamespace(bid_price=0.0, ask_price=1.1),
     SimpleNamespace(gamma=0.05, delta=0.4)),                       # one-sided (bid<=0)
    (SimpleNamespace(bid_price=1.0, ask_price=0.0),
     SimpleNamespace(gamma=0.05, delta=0.4)),                       # one-sided (ask<=0)
])
def test_row_from_rejects_unusable_quotes(q, g):
    tf = pytest.importorskip("tastytrade_feed")
    meta = {"side": "call", "strike": 500.0}
    assert tf._row_from(meta, q, g, oi=5) is None
