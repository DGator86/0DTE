"""
yahoo_feed.py  —  free, key-less backstop for the EASY half of the data contract.

WHY THIS EXISTS
    Three of the four data roles the pipeline needs are commodities that should
    never take the system offline:
        - underlying spot + 1-min OHLCV bars   (Track B technicals)
        - VIX term structure (9D / 30 / 3M / VVIX)
        - EOD settlement close                 (journal settlement)
    The hard role — real-time 0DTE option NBBO + greeks — stays with a broker
    (Tradier / Tastytrade). This module backstops ONLY the easy three, from
    Yahoo Finance's public chart endpoint: no API key, no account, free.

    It is NOT a standalone live feed (it serves no option chain), so it cannot
    drive Track A on its own. Its job is to make bars/VIX/settlement resilient
    inside CompositeFeed and the broker feeds, and to provide a free
    settlement_price() fallback.

    Bonus over the broker feeds: Yahoo serves REAL CBOE index quotes
    (^VIX9D / ^VIX / ^VIX3M / ^VVIX), upgrading the vol surface from the
    chain-IV proxy to actual values when the broker isn't entitled to VIX.

SECURITY: no credentials. Public endpoint only.
NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from resample import RawBars

ET = ZoneInfo("America/New_York")
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
# Yahoo serves the CBOE vol indices under caret tickers.
_VIX_TICKERS = {"vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX"}


# --------------------------------------------------------------------------- #
# HTTP                                                                          #
# --------------------------------------------------------------------------- #
def _chart(symbol: str, params: dict) -> dict:
    """GET the Yahoo chart endpoint for a symbol. A browser User-Agent is
    required or Yahoo returns 429/403."""
    url = f"{_CHART}{urllib.parse.quote(symbol)}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; zerodte-shadow/1.0)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    })
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    data = json.loads(raw)
    err = (data.get("chart") or {}).get("error")
    if err:
        raise RuntimeError(f"Yahoo error for {symbol}: {err}")
    results = (data.get("chart") or {}).get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo returned no result for {symbol}")
    return results[0]


# --------------------------------------------------------------------------- #
# Market data                                                                   #
# --------------------------------------------------------------------------- #
def get_spot(symbol: str = "SPY") -> float:
    """Last/regular-market price for the underlying (delayed ~15m on Yahoo)."""
    meta = _chart(symbol, {"interval": "1d", "range": "1d"}).get("meta") or {}
    px = meta.get("regularMarketPrice") or meta.get("previousClose") or 0.0
    return float(px)


def get_bars_raw(symbol: str = "SPY", lookback_minutes: int = 7800) -> RawBars:
    """1-min OHLCV bars. Yahoo caps 1m history at ~8 days; we request the widest
    window it allows and trim NaN rows (Yahoo pads gaps with nulls)."""
    # 1m data is only retained ~8 days regardless of the requested lookback.
    res = _chart(symbol, {"interval": "1m", "range": "8d", "includePrePost": "false"})
    ts_epoch = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts_epoch or not quote:
        raise RuntimeError(f"No 1m bars for {symbol}")

    o = quote.get("open", []); h = quote.get("high", [])
    lo = quote.get("low", []); c = quote.get("close", []); v = quote.get("volume", [])

    rows = []
    for i, t in enumerate(ts_epoch):
        # Yahoo emits null for empty minutes; drop any bar missing a field.
        vals = (o[i], h[i], lo[i], c[i])
        if any(x is None for x in vals):
            continue
        rows.append((t, o[i], h[i], lo[i], c[i], v[i] if v[i] is not None else 0.0))
    if not rows:
        raise RuntimeError(f"All {symbol} 1m bars were null")

    arr = np.array(rows, dtype=float)
    ts = (arr[:, 0].astype("int64") * 1_000_000_000).astype("datetime64[ns]")
    return RawBars(
        ts=ts,
        open=arr[:, 1], high=arr[:, 2], low=arr[:, 3],
        close=arr[:, 4], volume=arr[:, 5],
    )


def get_vix_term_structure() -> Optional[dict]:
    """Real CBOE vol indices from Yahoo: {vix9d, vix, vix3m, vvix}.
    Missing legs fall back to the 30-day VIX so the ordering stays sane.
    Returns None only if VIX itself can't be fetched."""
    out: dict[str, float] = {}
    for key, ticker in _VIX_TICKERS.items():
        try:
            meta = _chart(ticker, {"interval": "1d", "range": "1d"}).get("meta") or {}
            px = meta.get("regularMarketPrice") or meta.get("previousClose")
            if px:
                out[key] = float(px)
        except Exception:
            continue
    vix = out.get("vix")
    if not vix:
        return None
    return {
        "vix9d": out.get("vix9d", vix),
        "vix": vix,
        "vix3m": out.get("vix3m", vix),
        "vvix": out.get("vvix", 95.0),
    }


def get_settlement(symbol: str, session_date: str) -> Optional[float]:
    """EOD close for a session date (YYYY-MM-DD). Pulls a small daily window and
    selects the matching day, so it works whether or not it's the latest bar."""
    try:
        res = _chart(symbol, {"interval": "1d", "range": "1mo"})
        ts_epoch = res.get("timestamp") or []
        closes = (((res.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
        for t, close in zip(ts_epoch, closes):
            if close is None:
                continue
            d = datetime.fromtimestamp(t, ET).strftime("%Y-%m-%d")
            if d == session_date:
                return float(close)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Backstop object (consumed by the broker feeds + CompositeFeed)               #
# --------------------------------------------------------------------------- #
class YahooBackstop:
    """Bundles the easy-three fetches behind one object. Every method swallows
    failures and returns None so a caller can fall through to its own source or
    a cached value — the backstop must never raise into the tick loop."""

    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol

    def spot(self) -> Optional[float]:
        try:
            px = get_spot(self.symbol)
            return px if px > 0 else None
        except Exception:
            return None

    def bars(self, lookback_minutes: int = 7800) -> Optional[RawBars]:
        try:
            return get_bars_raw(self.symbol, lookback_minutes)
        except Exception:
            return None

    def vix_term_structure(self) -> Optional[dict]:
        return get_vix_term_structure()

    def settlement(self, session_date: str) -> Optional[float]:
        return get_settlement(self.symbol, session_date)


# --------------------------------------------------------------------------- #
# Diagnostic                                                                    #
# --------------------------------------------------------------------------- #
def diagnose(symbol: str = "SPY") -> None:
    """Confirm Yahoo reachability and the three backstop roles."""
    print(f"Yahoo backstop diagnostic for {symbol}")
    try:
        print(f"  spot           = {get_spot(symbol)}")
    except Exception as e:
        print(f"  spot           FAILED: {e}")
    try:
        raw = get_bars_raw(symbol)
        print(f"  1m bars        = {len(raw.close)} rows, last close {raw.close[-1]:.2f}")
    except Exception as e:
        print(f"  1m bars        FAILED: {e}")
    ts = get_vix_term_structure()
    print(f"  VIX term       = {ts if ts else 'unavailable'}")
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"  settlement {today} = {get_settlement(symbol, today)}")


if __name__ == "__main__":
    import sys
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "SPY")
