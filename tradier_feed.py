"""
tradier_feed.py  —  live DataFeed adapter backed by the Tradier Brokerage API.

WHY THIS EXISTS
    The Massive/Polygon snapshot plan in use returns no real-time option NBBO
    (every contract falls back to day.close, quote_valid=False), so Track A
    prices on stale marks. A Tradier brokerage account includes real-time
    option chains WITH greeks + OI (greeks courtesy of ORATS) and real-time
    equity quotes — at no extra market-data fee — and doubles as the execution
    venue. This adapter produces the same TickSnapshot as MassiveDataFeed, so it
    is a drop-in for UnifiedOrchestrator:

        feed = TradierDataFeed("SPY")
        orch = UnifiedOrchestrator(feed=feed, journal=Journal("live.sqlite"))

SECURITY: credentials are read from environment variables ONLY. Nothing is
hardcoded. Never paste a token into this file or any chat.
    export TRADIER_ACCESS_TOKEN=...                         # your token
    export TRADIER_BASE_URL=https://api.tradier.com/v1      # or sandbox host

Endpoints used (all documented, JSON):
    GET /v1/markets/quotes?symbols=SPY                      real-time spot
    GET /v1/markets/options/chains?symbol=SPY&expiration=YYYY-MM-DD&greeks=true
    GET /v1/markets/options/expirations?symbol=SPY          (0DTE check)
    GET /v1/markets/timesales?symbol=SPY&interval=1min&...  1-min bars
    GET /v1/markets/history?symbol=SPY&interval=daily&...   settlement close

NOTE: bid/ask are real-time; greeks/OI come from ORATS and can lag a few
minutes. That is far better than day-close marks — quote_valid is True here.

NOT financial advice.
"""
from __future__ import annotations

import os
import json
import gzip
import urllib.parse
import urllib.request
import urllib.error
import datetime as dt
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np

from spy0dte import OptionRow, build_gamma_map
from resample import RawBars
from unified_loop import TickSnapshot
from gate_scorer import MarketSnapshot
# Reuse the chain/technical helpers already proven against the Massive feed.
from massive_feed import (
    _option_rows_to_chain_snapshot, _bar_technicals,
    _session_vwap_and_reversions, _atm_straddle_price,
)

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# HTTP                                                                          #
# --------------------------------------------------------------------------- #
def _get(path: str, params: dict[str, Any]) -> dict:
    """GET a Tradier endpoint. path is relative to TRADIER_BASE_URL."""
    token = os.environ.get("TRADIER_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("TRADIER_ACCESS_TOKEN not set in environment")
    base = os.environ.get("TRADIER_BASE_URL", "https://api.tradier.com/v1").strip().rstrip("/")
    if any(c.isspace() or c == "#" for c in base):
        raise RuntimeError(
            f"TRADIER_BASE_URL is malformed: {base!r}. systemd's EnvironmentFile "
            "does not strip inline '# ...' comments — put comments on their own "
            "line in /etc/zerodte/zerodte.env, not after the value."
        )
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"Tradier HTTP {e.code}: {body}") from None


def _as_list(node: Any) -> list:
    """Tradier returns a bare object for single-element results and a list for
    many. Normalize to a list; treat None / 'null' as empty."""
    if node is None or node == "null":
        return []
    return node if isinstance(node, list) else [node]


# --------------------------------------------------------------------------- #
# Market data                                                                   #
# --------------------------------------------------------------------------- #
def get_spot(symbol: str) -> float:
    """Real-time last price for the underlying."""
    data = _get("/markets/quotes", {"symbols": symbol})
    quotes = _as_list((data.get("quotes") or {}).get("quote"))
    for q in quotes:
        if q.get("symbol") == symbol:
            return float(q.get("last") or q.get("close") or 0.0)
    return float(quotes[0].get("last")) if quotes else 0.0


def todays_expiry(symbol: str) -> Optional[str]:
    """Return today's expiration (YYYY-MM-DD) if the symbol expires today
    (0DTE session), else None."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    data = _get("/markets/options/expirations",
                {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"})
    dates = _as_list((data.get("expirations") or {}).get("date"))
    return today if today in dates else None


def _row_from_option(o: dict) -> Optional[OptionRow]:
    """Map one Tradier option to an OptionRow. Real-time bid/ask -> quote_valid.
    Returns None if greeks/quotes are incomplete."""
    side = o.get("option_type")             # "call" | "put"
    strike = o.get("strike")
    oi = o.get("open_interest")
    greeks = o.get("greeks") or {}
    gamma = greeks.get("gamma")
    delta = greeks.get("delta")
    bid = o.get("bid")
    ask = o.get("ask")
    if None in (side, strike, oi, gamma, delta, bid, ask):
        return None
    if bid <= 0 or ask <= 0:
        return None                         # no two-sided market -> unusable
    return OptionRow(side=side, strike=float(strike), oi=int(oi),
                     gamma=float(gamma), bid=float(bid), ask=float(ask),
                     delta=abs(float(delta)),
                     quote_source="tradier_live", quote_valid=True)


def get_chain(symbol: str, expiration: str) -> list[OptionRow]:
    """Real-time option chain with greeks for a single expiration."""
    data = _get("/markets/options/chains",
                {"symbol": symbol, "expiration": expiration, "greeks": "true"})
    opts = _as_list((data.get("options") or {}).get("option"))
    rows = [_row_from_option(o) for o in opts]
    return [r for r in rows if r is not None]


def get_bars_raw(symbol: str, lookback_minutes: int = 7800) -> RawBars:
    """1-min OHLCV bars from /markets/timesales."""
    now_et = datetime.now(ET)
    start = (now_et - dt.timedelta(minutes=lookback_minutes + 60)).strftime("%Y-%m-%d %H:%M")
    end = now_et.strftime("%Y-%m-%d %H:%M")
    data = _get("/markets/timesales",
                {"symbol": symbol, "interval": "1min", "start": start, "end": end})
    pts = _as_list((data.get("series") or {}).get("data"))
    if not pts:
        raise RuntimeError(f"No bars returned for {symbol}")
    ts = np.array([int(p["timestamp"]) for p in pts], dtype="datetime64[s]").astype("datetime64[ns]")
    return RawBars(
        ts=ts,
        open=np.array([p["open"] for p in pts], dtype=float),
        high=np.array([p["high"] for p in pts], dtype=float),
        low=np.array([p["low"] for p in pts], dtype=float),
        close=np.array([p["close"] for p in pts], dtype=float),
        volume=np.array([p.get("volume", 0) for p in pts], dtype=float),
    )


def get_settlement(symbol: str, session_date: str) -> Optional[float]:
    """EOD close for a session date (YYYY-MM-DD)."""
    try:
        data = _get("/markets/history",
                    {"symbol": symbol, "interval": "daily",
                     "start": session_date, "end": session_date})
        days = _as_list((data.get("history") or {}).get("day"))
        if days:
            return float(days[0]["close"])
    except Exception:
        pass
    return None


def get_vix_term_structure() -> Optional[dict]:
    """VIX term structure straight from CBOE index quotes Tradier exposes.
    VIX9D / VIX3M may not be entitled on every account; missing legs fall back
    to the 30-day VIX so the ordering stays sane. None if VIX itself is absent."""
    try:
        data = _get("/markets/quotes", {"symbols": "VIX,VIX9D,VIX3M"})
    except Exception:
        return None
    out: dict[str, float] = {}
    for q in _as_list((data.get("quotes") or {}).get("quote")):
        last = q.get("last")
        if last:
            out[q.get("symbol")] = float(last)
    vix = out.get("VIX")
    if not vix:
        return None
    return {"vix9d": out.get("VIX9D", vix), "vix": vix, "vix3m": out.get("VIX3M", vix)}


# --------------------------------------------------------------------------- #
# Feed                                                                          #
# --------------------------------------------------------------------------- #
class TradierDataFeed:
    """
    Drop-in live DataFeed for UnifiedOrchestrator backed by Tradier (real-time
    option NBBO + greeks). Mirrors MassiveDataFeed's interface and reuses its
    chain/technical helpers; the difference that matters is quote_valid=True.

    SECURITY: credentials from environment ONLY (TRADIER_ACCESS_TOKEN).
    """

    def __init__(
        self,
        underlying: str = "SPY",
        lookback_minutes: int = 7800,
        r: float = 0.05,
        # Vol surface (overridden each tick when use_live_vix and VIX is entitled)
        vix9d: float = 14.0,
        vix: float = 15.0,
        vix3m: float = 17.0,
        vvix: float = 92.0,
        vvix_baseline: float = 95.0,
        use_live_vix: bool = True,
        vix_refresh_seconds: int = 600,
        gex_history_len: int = 100,
        has_catalyst: bool = False,
        catalyst_label: Optional[str] = None,
    ) -> None:
        from collections import deque
        self.underlying = underlying
        self.lookback_minutes = lookback_minutes
        self.r = r
        self._vix9d, self._vix, self._vix3m = vix9d, vix, vix3m
        self._vvix, self._vvix_baseline = vvix, vvix_baseline
        self._use_live_vix = use_live_vix
        self._vix_refresh_seconds = vix_refresh_seconds
        self._vix_ts: Optional[datetime] = None
        self._gex_history: "deque[float]" = deque(maxlen=gex_history_len)
        self.has_catalyst = has_catalyst
        self.catalyst_label = catalyst_label

    def _gex_pct_rank(self, net_gex: float) -> float:
        self._gex_history.append(net_gex)
        h = list(self._gex_history)
        return float(sum(1 for x in h if x < net_gex) / len(h)) if len(h) > 1 else 0.5

    def _t_years(self, now: dt.datetime) -> float:
        today = now.astimezone(ET)
        expiry = dt.datetime(today.year, today.month, today.day, 16, 0, 0, tzinfo=ET)
        return max((expiry - today).total_seconds(), 60.0) / (365.25 * 24.0 * 3600.0)

    def _maybe_refresh_vix(self, now: dt.datetime) -> None:
        if not self._use_live_vix:
            return
        if (self._vix_ts is not None
                and (now - self._vix_ts).total_seconds() < self._vix_refresh_seconds):
            return
        ts = get_vix_term_structure()
        if ts and ts["vix"] > 0:
            self._vix9d, self._vix, self._vix3m = ts["vix9d"], ts["vix"], ts["vix3m"]
            self._vix_ts = now

    # -- DataFeed protocol --
    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        expiry = todays_expiry(self.underlying)
        if expiry is None:
            return None                     # not a 0DTE session for this symbol

        try:
            spot = get_spot(self.underlying)
            rows = get_chain(self.underlying, expiry)
            raw = get_bars_raw(self.underlying, self.lookback_minutes)
        except Exception:
            return None
        if not rows or spot <= 0.0:
            return None

        self._maybe_refresh_vix(now)

        gm = build_gamma_map(rows, spot)
        gex_rank = self._gex_pct_rank(gm.net_gex)
        chain = _option_rows_to_chain_snapshot(spot, rows, self._t_years(now), self.r)
        tech = _bar_technicals(raw)
        vwap, vwap_rev = _session_vwap_and_reversions(raw, now)
        straddle_be = _atm_straddle_price(rows, spot)

        market = MarketSnapshot(
            spot=spot, net_gex=gm.net_gex, gamma_flip=gm.gamma_flip,
            call_wall=gm.call_wall, put_wall=gm.put_wall, gex_pct_rank=gex_rank,
            vix9d=self._vix9d, vix=self._vix, vix3m=self._vix3m,
            vvix=self._vvix, vvix_baseline=self._vvix_baseline,
            straddle_breakeven=straddle_be, expected_range=straddle_be / 1.25,
            adx=tech["adx"], rsi=tech["rsi"],
            bb_width=tech["bb_width"], bb_width_baseline=tech["bb_width_baseline"],
            vwap=vwap, vwap_reversion_count=vwap_rev,
            tick_abs_mean=480.0,            # $TICK not sourced here; calm default
            cvd_slope=tech["cvd_slope"],
            now=now, has_catalyst=self.has_catalyst, catalyst_label=self.catalyst_label,
        )
        return TickSnapshot(market=market, bars=raw, chain=chain)

    def settlement_price(self, session_date: str) -> Optional[float]:
        return get_settlement(self.underlying, session_date)


# --------------------------------------------------------------------------- #
# Diagnostic                                                                    #
# --------------------------------------------------------------------------- #
def diagnose(symbol: str = "SPY") -> None:
    """Confirm auth + real-time NBBO without exposing the token."""
    if not os.environ.get("TRADIER_ACCESS_TOKEN"):
        print("TRADIER_ACCESS_TOKEN not set — export it and rerun.")
        return
    try:
        spot = get_spot(symbol)
        print(f"AUTH OK. {symbol} spot = {spot}")
    except RuntimeError as e:
        print("CALL FAILED:", e)
        print("If 401: token wrong/expired. If using sandbox, set "
              "TRADIER_BASE_URL=https://sandbox.tradier.com/v1")
        return
    exp = todays_expiry(symbol)
    print(f"0DTE today? {'YES — ' + exp if exp else 'no expiration today'}")
    if exp:
        rows = get_chain(symbol, exp)
        live = sum(1 for r in rows if r.quote_valid)
        print(f"chain: {len(rows)} contracts, {live} with real-time NBBO")
        for r in rows[:5]:
            print(f"  {r.side:4s} {r.strike:7.1f} OI={r.oi:<6d} "
                  f"Γ={r.gamma:.4f} Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")
    ts = get_vix_term_structure()
    print(f"VIX term structure: {ts if ts else 'VIX not entitled on this account'}")


if __name__ == "__main__":
    import sys
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "SPY")
