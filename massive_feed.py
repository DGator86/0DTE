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
import os
import json
import gzip
import urllib.request
import urllib.error
from datetime import datetime
from zoneinfo import ZoneInfo

from spy0dte import OptionRow   # reuse the engine's dataclass — single source of truth

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
