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
from datetime import datetime

from gex_window import GexRankWindow
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
    Falls back to day.close if live bid/ask absent; tags fallback rows as invalid
    for live selection."""
    details = c.get("details", {}) or {}
    greeks = c.get("greeks", {}) or {}
    side = details.get("contract_type")     # "call" | "put"
    strike = details.get("strike_price")
    gamma = greeks.get("gamma")
    delta = greeks.get("delta")
    oi = c.get("open_interest")

    if None in (side, strike, gamma, delta, oi):
        return None

    # Prefer real-time last_quote; fall back to day.close (confirmed absent in live API)
    quote = c.get("last_quote", {}) or {}
    day = c.get("day", {}) or {}
    bid = quote.get("bid")
    ask = quote.get("ask")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        source, valid = "live_quote", True
    else:
        close = day.get("close")
        if not close or close <= 0:
            return None
        bid = ask = float(close)
        source, valid = "day_close_fallback", False

    return OptionRow(side=side, strike=float(strike), oi=int(oi),
                     gamma=float(gamma), bid=float(bid), ask=float(ask),
                     delta=abs(float(delta)), quote_source=source, quote_valid=valid,
                     volume=int(day.get("volume") or 0))


def _estimate_spot(rows: list[OptionRow]) -> float:
    """Estimate spot when underlying_asset.price is absent.

    Primary method — put-call parity at the money:
        C - P = S - K * e^(-rT)   ->   S ≈ (C_mid - P_mid) + K
    For 0DTE the discount factor is ~1 (K*r*T is a few cents), so the strike
    where |C_mid - P_mid| is smallest is the ATM crossing, and the parity
    relation there recovers spot to within the bid/ask. We take the median of
    the parity estimate over the few strikes nearest that crossing for
    robustness against a single bad mark.

    Fallbacks: the 0.5-delta call strike; else 0.0 (caller treats as no data).
    This replaces the old deep-ITM-intrinsic estimate, which was biased high
    (~1% on the live SPY chain: estimated 735 vs ATM 727-728)."""
    calls: dict[float, OptionRow] = {}
    puts: dict[float, OptionRow] = {}
    for r in rows:
        bucket = calls if r.side == "call" else puts
        # prefer the tighter/real quote if a strike appears more than once
        if r.strike not in bucket or r.spread_pct < bucket[r.strike].spread_pct:
            bucket[r.strike] = r

    paired = sorted(set(calls) & set(puts))
    if paired:
        atm = min(paired, key=lambda k: abs(calls[k].mid - puts[k].mid))
        window = sorted(paired, key=lambda k: abs(k - atm))[:5]
        est = sorted((calls[k].mid - puts[k].mid) + k for k in window)
        return round(est[len(est) // 2], 2)   # median parity spot near ATM

    # Fallback: strike of the call whose delta is closest to 0.5 (≈ ATM forward).
    call_rows = [r for r in rows if r.side == "call" and r.delta > 0]
    if call_rows:
        return round(min(call_rows, key=lambda r: abs(r.delta - 0.5)).strike, 2)
    return 0.0


def get_chain(underlying: str, zero_dte_only: bool = True) -> tuple[float, list[OptionRow]]:
    """Fetch the chain snapshot, filter to today's expiry, map to OptionRows.
    Returns (spot, rows). Raises if the key/host are wrong."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY not set in environment")
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")

    today = _today_et()
    # Filter to today's expiry server-side when 0DTE: the snapshot is otherwise
    # the entire multi-expiry chain (thousands of contracts, many pages) of which
    # we discard all but one expiry. The exact-match expiration_date param cuts
    # that to a single expiry's strikes — far fewer pages per tick. The
    # client-side check below stays as a backstop in case the param is ignored.
    url = f"{base}/v3/snapshot/options/{underlying}?limit=250"
    if zero_dte_only:
        url += f"&expiration_date={today}"
    rows: list[OptionRow] = []
    spot = 0.0
    pages = 0

    while url and pages < 25:           # safety cap on pagination
        data = _get(url, key)
        for c in data.get("results", []):
            # underlying_asset.price is the authoritative spot when present.
            # It is confirmed ABSENT on the live Massive snapshot, so in practice
            # spot is recovered post-loop via ATM put-call parity (_estimate_spot).
            ua = c.get("underlying_asset", {}) or {}
            if ua.get("price"):
                spot = float(ua["price"])

            if zero_dte_only and (c.get("details") or {}).get("expiration_date") != today:
                continue
            row = _row_from_contract(c)
            if row:
                rows.append(row)
        url = data.get("next_url") or None
        pages += 1

    if spot <= 0:
        spot = _estimate_spot(rows)

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


def flow_lite(rows: list[OptionRow]) -> dict:
    """Options-flow lite from the chain rows every feed already fetches.

    Observation-only signals (see journal.py's admission rule): put/call
    volume ratio and volume/OI participation shock. NaN when the provider
    supplies no volume — absent must never read as zero flow.
    """
    import math as _math
    call_vol = sum(r.volume for r in rows if r.side == "call")
    put_vol = sum(r.volume for r in rows if r.side == "put")
    total_oi = sum(r.oi for r in rows)
    total_vol = call_vol + put_vol
    return {
        "pcr_volume": (put_vol / call_vol) if call_vol > 0 else float("nan"),
        "volume_oi_ratio": (total_vol / total_oi)
                           if (total_oi > 0 and total_vol > 0) else float("nan"),
    }


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


def get_iv_term_structure(underlying: str, spot: float,
                          tenors_days: tuple[int, int, int] = (9, 30, 90)
                          ) -> dict | None:
    """ATM implied-vol term structure derived from the live option chain.

    Replaces the hardcoded VIX / VIX9D / VIX3M defaults with a market-derived
    proxy. For each target tenor we find the expiry whose days-to-expiry is
    closest, take the ATM strike (nearest spot), and average the call & put
    implied vol there. Returned in VIX-style points (IV * 100). None on
    insufficient data (caller keeps its existing values).

    NOTE: a SPY ATM-IV proxy is not the CBOE VIX (a variance-swap strip over
    all strikes). But the gate only consumes the *ordering / slope* of the
    three tenors — vix9d/vix ratio and vix vs vix3m (contango vs backwardation)
    — which the ATM proxy tracks closely. The chain already carries
    `implied_volatility` per contract, so this costs no extra entitlement.
    """
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return None
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")
    today = datetime.now(ET).date()

    # Filter server-side to a near-ATM strike band and a bounded expiry window so
    # we fetch only the contracts the term structure needs (a few hundred, not
    # the whole chain). Polygon/Massive support these query params natively.
    lo, hi = round(spot * 0.97, 2), round(spot * 1.03, 2)
    exp_hi = (today + dt.timedelta(days=max(tenors_days) + 20)).isoformat()
    url = (f"{base}/v3/snapshot/options/{underlying}"
           f"?strike_price.gte={lo}&strike_price.lte={hi}"
           f"&expiration_date.gte={today.isoformat()}&expiration_date.lte={exp_hi}"
           f"&limit=250")

    # expiration_date -> { strike -> {"call": iv, "put": iv} }
    by_exp: dict[str, dict[float, dict[str, float]]] = {}
    pages = 0
    while url and pages < 10:
        data = _get(url, key)
        for c in data.get("results", []):
            det = c.get("details", {}) or {}
            iv = c.get("implied_volatility")
            exp = det.get("expiration_date")
            strike = det.get("strike_price")
            side = det.get("contract_type")
            if not iv or iv <= 0 or not exp or strike is None or side not in ("call", "put"):
                continue
            by_exp.setdefault(exp, {}).setdefault(float(strike), {})[side] = float(iv)
        url = data.get("next_url") or None
        pages += 1

    if not by_exp:
        return None

    atm: list[tuple[int, float]] = []   # (days_to_expiry, atm_iv)
    for exp, strikes in by_exp.items():
        try:
            y, m, d = (int(x) for x in exp.split("-"))
            dte = (dt.date(y, m, d) - today).days
        except Exception:
            continue
        if dte < 0:
            continue
        atm_k = min(strikes, key=lambda k: abs(k - spot))
        legs = strikes[atm_k]
        ivs = [v for v in (legs.get("call"), legs.get("put")) if v]
        if not ivs:
            continue
        atm.append((dte, sum(ivs) / len(ivs)))

    if not atm:
        return None
    atm.sort()

    def nearest(target: int) -> float:
        _, iv = min(atm, key=lambda t: abs(t[0] - target))
        return round(iv * 100.0, 2)

    t9, t30, t90 = tenors_days
    return {"vix9d": nearest(t9), "vix": nearest(t30), "vix3m": nearest(t90),
            "n_expiries": len(atm)}


class MassiveDataFeed:
    """
    Live DataFeed adapter for UnifiedOrchestrator backed by the Massive API.

    SECURITY: credentials from environment ONLY.
        export MASSIVE_API_KEY=...
        export MASSIVE_BASE_URL=https://api.massive.com

    VIX / VIX9D / VIX3M are derived from the chain's own ATM implied-vol term
    structure when use_chain_vix=True (the default), refreshed every
    vix_refresh_seconds. Pass use_chain_vix=False to pin the constructor
    defaults, or call set_vix() to override from a dedicated feed. VVIX is not
    derivable from the equity chain (it needs VIX options) and keeps its
    configurable default; the system degrades gracefully.

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
        # Derive VIX term structure from the chain's ATM IV (no extra entitlement)
        use_chain_vix: bool = True,
        vix_refresh_seconds: int = 600,
        # State: rolling GEX history for percentile rank
        gex_history_len: int = 100,        # retained for API compat; window is time-based now
        has_catalyst: bool = False,
        catalyst_label: str | None = None,
        gex_history_path: str | None = None,      # persist |GEX| rank window across restarts
    ) -> None:
        self.underlying = underlying
        self.lookback_minutes = lookback_minutes
        self.r = r
        self._vix9d = vix9d
        self._vix = vix
        self._vix3m = vix3m
        self._vvix = vvix
        self._vvix_baseline = vvix_baseline
        self._use_chain_vix = use_chain_vix
        self._vix_refresh_seconds = vix_refresh_seconds
        self._vix_ts: datetime | None = None    # last term-structure refresh
        self._gex_window = GexRankWindow(path=gex_history_path)
        self.has_catalyst = has_catalyst
        self.catalyst_label = catalyst_label

    def _maybe_refresh_vix(self, spot: float, now: dt.datetime) -> None:
        """Refresh the ATM-IV term structure from the chain, at most once every
        vix_refresh_seconds. Silently keeps prior values on any failure."""
        if not self._use_chain_vix or spot <= 0:
            return
        if (self._vix_ts is not None
                and (now - self._vix_ts).total_seconds() < self._vix_refresh_seconds):
            return
        try:
            ts = get_iv_term_structure(self.underlying, spot)
        except Exception:
            ts = None
        if ts and ts["vix"] > 0:
            self._vix9d, self._vix, self._vix3m = ts["vix9d"], ts["vix"], ts["vix3m"]
            self._vix_ts = now

    # -- vol overrides (wire live VIX feed here) --
    def set_vix(self, vix9d: float, vix: float, vix3m: float) -> None:
        self._vix9d, self._vix, self._vix3m = vix9d, vix, vix3m

    def set_vvix(self, vvix: float, vvix_baseline: float) -> None:
        self._vvix, self._vvix_baseline = vvix, vvix_baseline

    def _gex_pct_rank(self, net_gex: float) -> float:
        return self._gex_window.rank(net_gex)

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

        # VIX term structure from the chain's own ATM IV (cached refresh)
        self._maybe_refresh_vix(spot, now)

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

        flow = flow_lite(rows)
        market = MarketSnapshot(
            spot=spot,
            net_gex=gm.net_gex,
            gamma_flip=gm.gamma_flip,
            call_wall=gm.call_wall,
            put_wall=gm.put_wall,
            gex_pct_rank=gex_rank,
            gex_rank_warm=self._gex_window.is_warm,
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
            pcr_volume=flow["pcr_volume"], volume_oi_ratio=flow["volume_oi_ratio"],
        )
        return TickSnapshot(
            market=market, bars=raw, chain=chain,
            option_rows=rows,
            gex_feed_source="MassiveDataFeed",
        )

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
