"""
massive_feed.py  —  live adapter: Massive options chain snapshot -> OptionRow.

SECURITY: credentials are read from environment variables ONLY. Nothing is
hardcoded. Never paste a key into this file or any chat.
    export MASSIVE_API_KEY=...           # your key
    export MASSIVE_BASE_URL=https://api.massive.com   # confirm host in your dashboard

Endpoint used (Polygon-compatible path, which Massive mirrors):
    GET /v3/snapshot/options/{underlyingAsset}

Confirmed field mappings (from live feed test):
  - details.contract_type, details.strike_price, details.expiration_date  OK
  - greeks.gamma, greeks.delta                                             OK
  - open_interest                                                          OK
  - last_quote.bid/ask                    ABSENT — falls back to day.close
  - underlying_asset.price                ABSENT — estimated from deep-ITM call
"""
from __future__ import annotations
import math
import os
import json
import gzip
import urllib.request
import urllib.error
import datetime as dt
from collections import deque
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from spy0dte import OptionRow, build_gamma_map
from resample import RawBars, compute_tf_features, resample_ohlcv
from rnd_extractor import ChainSnapshot, ChainQuote
from gate_scorer import MarketSnapshot
from unified_loop import TickSnapshot

ET = ZoneInfo("America/New_York")


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _get(url: str, key: str) -> dict:
    # Massive may authenticate via Bearer header or ?apiKey= query param.
    # Default Bearer; set MASSIVE_AUTH=query to switch if you get a 401.
    auth_mode = os.environ.get("MASSIVE_AUTH", "bearer").lower()
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if auth_mode == "query":
        url = url + ("&" if "?" in url else "?") + f"apiKey={key}"
    else:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"Massive HTTP {e.code} ({auth_mode} auth): {body}") from None


def _row_from_contract(c: dict) -> OptionRow | None:
    """Map one snapshot contract to an OptionRow. Returns None if data is incomplete.
    If live bid/ask is absent, fall back to the day close BUT tag it so live
    selection rejects it — a fallback quote has spread 0 and would defeat the
    liquidity filter."""
    details = c.get("details", {})              # CONFIRM: contract_type, strike_price
    greeks = c.get("greeks", {})                # CONFIRM: gamma, delta
    side = details.get("contract_type")         # "call" | "put"
    quote = c.get("last_quote", {}) or {}       # CONFIRM: bid, ask
    day = c.get("day", {}) or {}                # daily agg, has close
    side = details.get("contract_type")
    """Map one snapshot contract to an OptionRow. Returns None if data is incomplete."""
    details = c.get("details", {})
    greeks = c.get("greeks", {})
    side = details.get("contract_type")     # "call" | "put"
    strike = details.get("strike_price")
    gamma = greeks.get("gamma")
    delta = greeks.get("delta")
    oi = c.get("open_interest")

    if None in (side, strike, gamma, delta, oi):
        return None

    # Prefer real-time last_quote; fall back to day.close
    # Prefer real-time last_quote; fall back to day.close (confirmed absent in live API)
    quote = c.get("last_quote", {})
    bid = quote.get("bid")
    ask = quote.get("ask")
    if bid is None or ask is None:
        close = c.get("day", {}).get("close")
        if close is None:
            return None
        bid = ask = close   # spread_pct=0; mid=close

    if bid <= 0 or ask <= 0:
        return None
    if None in (side, strike, gamma, delta, oi):
        return None

    bid, ask = quote.get("bid"), quote.get("ask")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        source, valid = "live_quote", True
    else:
        close = day.get("close")
        if not close or close <= 0:
            return None                          # no quote AND no close -> unusable
        bid = ask = float(close)                 # fallback: diagnostics only
        source, valid = "day_close_fallback", False

    return OptionRow(side=side, strike=float(strike), oi=int(oi),
                     gamma=float(gamma), bid=float(bid), ask=float(ask),
                     delta=abs(float(delta)), quote_source=source, quote_valid=valid)


def _estimate_spot(rows: list[OptionRow]) -> float:
    """If underlying_asset.price is missing, estimate spot from a deep-ITM call:
    spot ~= strike + call_mid (intrinsic-dominated). Crude but bounded."""
    deep = [r for r in rows if r.side == "call" and r.delta >= 0.9]
    if not deep:
        return 0.0
    r = max(deep, key=lambda x: x.delta)
    return round(r.strike + r.mid, 2)


def get_chain(underlying: str, zero_dte_only: bool = True) -> tuple[float, list[OptionRow]]:
    """Fetch the chain snapshot, filter to today's expiry, map to OptionRows.
    Returns (spot, rows). Raises if the key/host are wrong."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY not set in environment")
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")

    url = f"{base}/v3/snapshot/options/{underlying}?limit=250"
    today = _today_et()
    rows: list[OptionRow] = []
    spot = 0.0
    best_delta = 0.0  # fallback: track deepest-ITM call for spot estimation
    best_delta = 0.0    # tracks deepest-ITM call for spot estimation fallback
    pages = 0

    while url and pages < 25:            # safety cap on pagination
        data = _get(url, key)
        for c in data.get("results", []):
            # Prefer API-provided spot; fall back to deep-ITM call intrinsic value
            ua = c.get("underlying_asset", {})          # CONFIRM: underlying_asset.price
            # underlying price lives on each contract's snapshot
            ua = c.get("underlying_asset", {})          # CONFIRM: underlying_asset.price
            if ua.get("price"):
                spot = float(ua["price"])
            if zero_dte_only and c.get("details", {}).get("expiration_date") != today:
    while url and pages < 25:           # safety cap on pagination
        data = _get(url, key)
        for c in data.get("results", []):
            # Prefer API-provided spot; fall back to deep-ITM call intrinsic value
            # (underlying_asset.price confirmed absent in live Massive snapshot)
            ua = c.get("underlying_asset", {})
            if ua.get("price"):
                spot = float(ua["price"])
            elif not spot:
                d = abs((c.get("greeks") or {}).get("delta") or 0)
                close = (c.get("day") or {}).get("close") or 0
                strike_k = (c.get("details") or {}).get("strike_price") or 0
                if ((c.get("details") or {}).get("contract_type") == "call"
                        and d > best_delta and d > 0.95 and close > 0):
                    best_delta = d
                    spot = float(strike_k) + float(close)

            if zero_dte_only and (c.get("details") or {}).get("expiration_date") != today:
                continue
            row = _row_from_contract(c)
            if row:
                rows.append(row)
        nxt = data.get("next_url")
        url = nxt if nxt else None
        url = data.get("next_url") or None
        pages += 1

    if spot <= 0:
        spot = _estimate_spot(rows)              # underlying price absent -> estimate

    live = sum(1 for r in rows if r.quote_valid)
    if rows and live == 0:
        print("WARNING: 0 live quotes — entire chain is day-close fallback. "
              "This is NOT tradeable (no NBBO). Check your Massive plan's real-time "
              "entitlement and that you're calling during market hours.")
    elif rows and live < len(rows) * 0.5:
        print(f"WARNING: only {live}/{len(rows)} contracts have live quotes — "
              "live selection will reject the rest.")

    return spot, rows


def diagnose(underlying: str) -> None:
    """Run ONE call and print the response STRUCTURE so we can confirm field
    mappings. Prints key paths and a sample mapped row — never your key."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        print("MASSIVE_API_KEY not set"); return
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")
    url = f"{base}/v3/snapshot/options/{underlying}?limit=5"
    try:
        data = _get(url, key)
    except RuntimeError as e:
        print("CALL FAILED:", e)
        print("If 401/403: try  export MASSIVE_AUTH=query  and rerun.")
        print("If 404: the base host or path is wrong — check the API tab in your dashboard.")
        return

    print("AUTH OK. Top-level keys:", list(data.keys()))
    results = data.get("results", [])
    print("results count on page:", len(results))
    if not results:
        print("No contracts returned — check the ticker or your plan's entitlement.")
        return

    def paths(d, prefix=""):
        out = []
        if isinstance(d, dict):
            for k, v in d.items():
                out += paths(v, f"{prefix}{k}.")
        else:
            out.append(f"{prefix[:-1]} = {type(d).__name__}")
        return out

    print("\n--- structure of results[0] (paths, no values for safety) ---")
    for p in paths(results[0]):
        print("  ", p)

    print("\n--- mapping check (what the adapter extracts) ---")
    row = _row_from_contract(results[0])
    if row:
        src = "LIVE" if row.quote_valid else "FALLBACK(day close)"
        print(f"  OK -> {row.side} {row.strike} OI={row.oi} gamma={row.gamma} "
              f"delta={row.delta} {row.bid}/{row.ask} [{src}]")
    else:
        print("  MAPPING FAILED -> one of contract_type/strike_price/gamma/delta/"
              "open_interest/day.close is named differently. Paste the structure above.")


# --------------------------------------------------------------------------- #
# Task #3: MassiveDataFeed — live DataFeed for UnifiedOrchestrator            #
# --------------------------------------------------------------------------- #

def get_bars_raw(symbol: str, lookback_minutes: int = 7800) -> RawBars:
    """
    Fetch 1-min OHLCV bars from the Polygon-compatible /v2/aggs endpoint.

    lookback_minutes=7800 ≈ 20 trading days; enough for all MTF indicators.
    Credentials from MASSIVE_API_KEY / MASSIVE_BASE_URL env vars.
    """
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY not set")
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")

    now_et = datetime.now(ET)
    from_dt = now_et - dt.timedelta(minutes=lookback_minutes + 60)
    from_str = from_dt.strftime("%Y-%m-%d")
    to_str = now_et.strftime("%Y-%m-%d")

    url = (f"{base}/v2/aggs/ticker/{symbol}/range/1/minute/{from_str}/{to_str}"
           f"?adjusted=true&sort=asc&limit=50000")

    bars: list[dict] = []
    while url:
        data = _get(url, key)
        bars.extend(data.get("results", []))
        url = data.get("next_url") or None

    if not bars:
        raise RuntimeError(f"No bars returned for {symbol}")

    # Polygon bar format: {t: epoch_ms, o, h, l, c, v}
    ts = np.array([b["t"] for b in bars], dtype="datetime64[ms]").astype("datetime64[ns]")
    return RawBars(
        ts=ts,
        open=np.array([b["o"] for b in bars], dtype=float),
        high=np.array([b["h"] for b in bars], dtype=float),
        low=np.array([b["l"] for b in bars], dtype=float),
        close=np.array([b["c"] for b in bars], dtype=float),
        volume=np.array([b["v"] for b in bars], dtype=float),
    )


def _option_rows_to_chain_snapshot(spot: float, rows: list[OptionRow],
                                    t_years: float, r: float = 0.05) -> ChainSnapshot | None:
    """
    Convert a flat list of OptionRows (one per contract) into a ChainSnapshot.

    Groups by strike, requiring both a call and a put at each strike for
    put-call parity and RND extraction. Drops any strike missing either side.
    """
    calls: dict[float, OptionRow] = {}
    puts: dict[float, OptionRow] = {}
    for row in rows:
        bucket = calls if row.side == "call" else puts
        # prefer tighter quote if the same strike appears twice
        if row.strike not in bucket or row.spread_pct < bucket[row.strike].spread_pct:
            bucket[row.strike] = row

    quotes: list[ChainQuote] = []
    for strike in sorted(set(calls) & set(puts)):
        c, p = calls[strike], puts[strike]
        quotes.append(ChainQuote(
            strike=strike,
            call_bid=c.bid, call_ask=c.ask,
            put_bid=p.bid, put_ask=p.ask,
        ))

    if len(quotes) < 5:
        return None
    return ChainSnapshot(quotes=quotes, spot=spot, t_years=t_years, r=r)


def get_settlement(symbol: str, session_date: str) -> float | None:
    """EOD close price for a session date (YYYY-MM-DD) used in P&L settlement."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return None
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")
    url = (f"{base}/v2/aggs/ticker/{symbol}/range/1/day/{session_date}/{session_date}"
           f"?adjusted=true&sort=asc&limit=1")
    try:
        data = _get(url, key)
        results = data.get("results", [])
        if results:
            return float(results[0]["c"])
    except Exception:
        pass
    return None


def _atm_straddle_price(rows: list[OptionRow], spot: float) -> float:
    """Dollar value of the ATM straddle (call_mid + put_mid at the nearest strike)."""
    if not rows:
        return 0.0
    strikes = sorted(set(r.strike for r in rows))
    atm = min(strikes, key=lambda k: abs(k - spot))
    call_mid = put_mid = 0.0
    for r in rows:
        if r.strike == atm:
            if r.side == "call":
                call_mid = r.mid
            else:
                put_mid = r.mid
    return call_mid + put_mid


def _bb_width_from_bars(close: np.ndarray, p: int = 20) -> tuple[float, float]:
    """
    Returns (current_bb_width_pct, baseline_bb_width_pct) as % of price.
    baseline = trailing median of the width series.
    """
    if len(close) < p * 2:
        return 1.0, 2.0
    s = pd.Series(close.astype(float))
    mid = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0)
    width_pct = (4.0 * sd / mid * 100.0).dropna()
    if len(width_pct) < 1 or not np.isfinite(width_pct.iloc[-1]):
        return 1.0, 2.0
    return float(width_pct.iloc[-1]), float(width_pct.median())


def _session_vwap_and_reversions(raw: RawBars, now: dt.datetime) -> tuple[float, int]:
    """
    VWAP from the session open (09:30 ET) and count of price-to-VWAP crossings.
    If bar timestamps are naive they are treated as UTC.
    """
    today = now.astimezone(ET)
    session_open = today.replace(hour=9, minute=30, second=0, microsecond=0)

    ts_pd = pd.DatetimeIndex(raw.ts.astype("datetime64[ms]"))
    if ts_pd.tzinfo is None:
        ts_pd = ts_pd.tz_localize("UTC").tz_convert(ET)
    else:
        ts_pd = ts_pd.tz_convert(ET)

    mask = np.array(ts_pd >= session_open, dtype=bool)
    if not mask.any():
        return float(raw.close[-1]) if len(raw.close) else 0.0, 0

    c = raw.close[mask].astype(float)
    v = raw.volume[mask].astype(float)
    cum_v = np.cumsum(v)
    cum_v = np.where(cum_v > 0, cum_v, 1.0)
    vwap_series = np.cumsum(c * v) / cum_v
    vwap = float(vwap_series[-1])

    if len(c) > 1:
        diff = c - vwap_series
        crossings = int(np.sum(np.diff(np.sign(diff)) != 0))
    else:
        crossings = 0

    return vwap, crossings


def _bar_technicals(raw: RawBars) -> dict:
    """
    Compute bar-derived MarketSnapshot fields by resampling to 5m.
    Returns: adx, rsi, cvd_slope, bb_width, bb_width_baseline.
    """
    df = raw.to_frame()
    rs5 = resample_ohlcv(df, "5min")
    feats = compute_tf_features(rs5)
    bb_w, bb_base = _bb_width_from_bars(raw.close)
    return {
        "adx":                feats.get("adx_strength") or 20.0,
        "rsi":                feats.get("rsi") or 50.0,
        "cvd_slope":          feats.get("cvd_persistence") or 0.0,
        "bb_width":           bb_w,
        "bb_width_baseline":  bb_base,
    }


class MassiveDataFeed:
    """
    Live DataFeed adapter for UnifiedOrchestrator backed by the Massive API.

    SECURITY: credentials from environment ONLY.
        export MASSIVE_API_KEY=...
        export MASSIVE_BASE_URL=https://api.massive.com

    VIX / VVIX are not available from the chain endpoint. Supply them from a
    separate feed (e.g. CBOE data) or keep the configurable defaults until
    that pipe is built. The system degrades gracefully: regime classifier uses
    whatever values are present.

    GEX percentile rank is maintained as a rolling window across ticks; it
    starts at 0.5 until enough history accumulates.
    """

    def __init__(
        self,
        underlying: str = "SPY",
        lookback_minutes: int = 7800,           # ~20 trading days
        r: float = 0.05,
        # Vol surface defaults until a VIX feed is wired
        vix9d: float = 14.0,
        vix: float = 15.0,
        vix3m: float = 17.0,
        vvix: float = 92.0,
        vvix_baseline: float = 95.0,
        # State: rolling GEX history for percentile rank
        gex_history_len: int = 100,
        has_catalyst: bool = False,
        catalyst_label: str | None = None,
    ) -> None:
        self.underlying = underlying
        self.lookback_minutes = lookback_minutes
        self.r = r
        self._vix9d = vix9d
        self._vix = vix
        self._vix3m = vix3m
        self._vvix = vvix
        self._vvix_baseline = vvix_baseline
        self._gex_history: deque[float] = deque(maxlen=gex_history_len)
        self.has_catalyst = has_catalyst
        self.catalyst_label = catalyst_label

    # -- vol overrides (wire live VIX feed here) --
    def set_vix(self, vix9d: float, vix: float, vix3m: float) -> None:
        self._vix9d, self._vix, self._vix3m = vix9d, vix, vix3m

    def set_vvix(self, vvix: float, vvix_baseline: float) -> None:
        self._vvix, self._vvix_baseline = vvix, vvix_baseline

    def _gex_pct_rank(self, net_gex: float) -> float:
        self._gex_history.append(net_gex)
        h = list(self._gex_history)
        return float(sum(1 for x in h if x < net_gex) / len(h)) if len(h) > 1 else 0.5

    def _t_years(self, now: dt.datetime) -> float:
        """Minutes remaining to 4 pm ET expiry, expressed as a fraction of a year."""
        today = now.astimezone(ET)
        expiry = dt.datetime(today.year, today.month, today.day,
                             16, 0, 0, tzinfo=ET)
        secs = max((expiry - today).total_seconds(), 60.0)
        return secs / (365.25 * 24.0 * 3600.0)

    # -- DataFeed protocol --

    def snapshot(self, now: dt.datetime) -> TickSnapshot | None:
        try:
            raw = get_bars_raw(self.underlying, self.lookback_minutes)
        except Exception:
            return None

        try:
            spot, rows = get_chain(self.underlying, zero_dte_only=True)
        except Exception:
            return None
        if not rows or spot <= 0.0:
            return None

        # GEX structure
        gm = build_gamma_map(rows, spot)
        gex_rank = self._gex_pct_rank(gm.net_gex)

        # Options chain for Track A (RND + spread selector)
        chain = _option_rows_to_chain_snapshot(spot, rows, self._t_years(now), self.r)

        # Bar-derived technicals
        tech = _bar_technicals(raw)
        vwap, vwap_rev = _session_vwap_and_reversions(raw, now)

        # Straddle implied move
        straddle_be = _atm_straddle_price(rows, spot)
        expected_range = straddle_be / 1.25   # rough log-normal 1-sigma scaling

        market = MarketSnapshot(
            spot=spot,
            net_gex=gm.net_gex,
            gamma_flip=gm.gamma_flip,
            call_wall=gm.call_wall,
            put_wall=gm.put_wall,
            gex_pct_rank=gex_rank,
            vix9d=self._vix9d,
            vix=self._vix,
            vix3m=self._vix3m,
            vvix=self._vvix,
            vvix_baseline=self._vvix_baseline,
            straddle_breakeven=straddle_be,
            expected_range=expected_range,
            adx=tech["adx"],
            rsi=tech["rsi"],
            bb_width=tech["bb_width"],
            bb_width_baseline=tech["bb_width_baseline"],
            vwap=vwap,
            vwap_reversion_count=vwap_rev,
            tick_abs_mean=480.0,    # $TICK not available from Massive chain; use calm default
            cvd_slope=tech["cvd_slope"],
            now=now,
            has_catalyst=self.has_catalyst,
            catalyst_label=self.catalyst_label,
        )
        return TickSnapshot(market=market, bars=raw, chain=chain)

    def settlement_price(self, session_date: str) -> float | None:
        return get_settlement(self.underlying, session_date)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose(sys.argv[2] if len(sys.argv) > 2 else "SPY")
        sys.exit(0)
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    spot, rows = get_chain(sym)
    live = sum(1 for r in rows if r.quote_valid)
    print(f"{sym}: spot={spot} | {len(rows)} 0DTE rows ({live} live quotes, "
          f"{len(rows)-live} day-close fallback)")
    print(f"{sym}: spot={spot:.2f} | {len(rows)} 0DTE rows with complete greeks/quotes")
    for r in rows[:6]:
        src = "L" if r.quote_valid else "F"
        print(f"  [{src}] {r.side:4s} {r.strike:7.1f}  OI={r.oi:<6d} Γ={r.gamma:.4f} "
              f"Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")
    if rows:
        import spy0dte as eng
        gm = eng.build_gamma_map(rows, spot)
        print(f"  -> netGEX ratio {gm.net_ratio} | flip {gm.gamma_flip} | "
              f"walls {gm.put_wall}/{gm.call_wall} | regime {gm.regime.upper()}")
