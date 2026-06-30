"""Tests for NYSE market calendar."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from market_calendar import (
    is_market_open,
    market_status,
    next_market_close,
    next_market_open,
)

ET = ZoneInfo("America/New_York")


def _et(y, m, d, h=0, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=ET)


def test_weekend_closed():
    sat = _et(2026, 6, 27, 12, 0)
    assert not is_market_open(sat)
    nxt = next_market_open(sat)
    assert nxt.weekday() == 0  # Monday
    assert nxt.hour == 9 and nxt.minute == 30


def test_nyse_holiday_closed():
    new_years = _et(2026, 1, 1, 11, 0)
    assert not is_market_open(new_years)


def test_mid_session_open():
    tue = _et(2026, 6, 30, 10, 0)
    assert is_market_open(tue)
    close = next_market_close(tue)
    assert close is not None
    assert close.hour == 16 and close.minute == 0


def test_early_close_day():
    # Day after Thanksgiving 2025 — NYSE early close at 1:00 PM ET
    early = _et(2025, 11, 28, 12, 0)
    assert is_market_open(early)
    close = next_market_close(early)
    assert close is not None
    assert close.hour == 13 and close.minute == 0


def test_market_status_countdown_fields():
    closed = _et(2026, 6, 28, 12, 0)
    st = market_status(closed)
    assert st["is_open"] is False
    assert st["label_closed"] == "Market is Closed"
    assert st["seconds_until_open"] is not None
    assert st["seconds_until_open"] > 0
    assert st["next_close"] is None

    open_ = _et(2026, 6, 30, 10, 0)
    st2 = market_status(open_)
    assert st2["is_open"] is True
    assert st2["seconds_until_close"] is not None
    assert st2["seconds_until_close"] > 0
