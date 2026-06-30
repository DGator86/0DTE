"""
market_calendar.py
==================
NYSE (XNYS) regular-session calendar for SPY. Single source of truth for
market-open checks used by shadow_runner and the observability dashboard.

Uses exchange_calendars for holidays and early-close sessions.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

ET = ZoneInfo("America/New_York")
EXCHANGE = "XNYS"
SYMBOL = "SPY"
_REGULAR_CLOSE_HOUR_ET = 16


@lru_cache(maxsize=1)
def _calendar():
    return xcals.get_calendar(EXCHANGE)


def _to_et(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ET)
    return ts.astimezone(ET)


def _pd_et(ts: dt.datetime) -> pd.Timestamp:
    return pd.Timestamp(_to_et(ts))


def _session_date(ts: dt.datetime) -> str:
    return _to_et(ts).date().isoformat()


def _session_for_date(date_str: str) -> Optional[str]:
    cal = _calendar()
    if not cal.is_session(date_str):
        return None
    return date_str


def _session_open_close(session: str) -> tuple[dt.datetime, dt.datetime]:
    cal = _calendar()
    open_ts = cal.session_open(session).tz_convert(ET).to_pydatetime()
    close_ts = cal.session_close(session).tz_convert(ET).to_pydatetime()
    return open_ts, close_ts


def _session_type(close_et: dt.datetime) -> str:
    if close_et.hour < _REGULAR_CLOSE_HOUR_ET:
        return "early_close"
    return "regular"


def is_market_open(now: Optional[dt.datetime] = None) -> bool:
    """True when SPY/NYSE regular session is open."""
    now = _to_et(now or dt.datetime.now(ET))
    cal = _calendar()
    ts = _pd_et(now)
    if not cal.is_session(_session_date(now)):
        return False
    return bool(cal.is_open_on_minute(ts))


def next_market_open(now: Optional[dt.datetime] = None) -> dt.datetime:
    """Next session open (9:30 ET or early-close day's open)."""
    now = _to_et(now or dt.datetime.now(ET))
    cal = _calendar()
    ts = _pd_et(now)
    date_str = _session_date(now)

    if cal.is_session(date_str):
        open_ts, close_ts = _session_open_close(date_str)
        if now < open_ts:
            return open_ts

    session = cal.date_to_session(date_str, direction="next")
    session_str = session.strftime("%Y-%m-%d")
    open_ts, _ = _session_open_close(session_str)
    return open_ts


def next_market_close(now: Optional[dt.datetime] = None) -> Optional[dt.datetime]:
    """Current session close if market is open, else None."""
    now = _to_et(now or dt.datetime.now(ET))
    if not is_market_open(now):
        return None
    session = _session_date(now)
    _, close_ts = _session_open_close(session)
    return close_ts


def _seconds_until(target: Optional[dt.datetime], now: dt.datetime) -> Optional[int]:
    if target is None:
        return None
    return max(0, int((target - now).total_seconds()))


def market_status(now: Optional[dt.datetime] = None) -> dict:
    """Full market status dict for API/UI countdown anchors."""
    now = _to_et(now or dt.datetime.now(ET))
    open_now = is_market_open(now)
    nxt_open = next_market_open(now)
    nxt_close = next_market_close(now) if open_now else None

    session_type = "regular"
    if open_now and nxt_close is not None:
        session_type = _session_type(nxt_close)
    elif not open_now:
        cal = _calendar()
        session = cal.date_to_session(_session_date(now), direction="next")
        session_str = session.strftime("%Y-%m-%d")
        _, close_et = _session_open_close(session_str)
        session_type = _session_type(close_et)

    return {
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
        "timezone": str(ET),
        "is_open": open_now,
        "session_type": session_type,
        "next_open": nxt_open.isoformat(),
        "next_close": nxt_close.isoformat() if nxt_close else None,
        "seconds_until_open": _seconds_until(nxt_open, now) if not open_now else 0,
        "seconds_until_close": _seconds_until(nxt_close, now) if open_now else None,
        "label_closed": "Market is Closed",
        "label_open": "Market Open",
        "as_of": now.isoformat(),
    }
