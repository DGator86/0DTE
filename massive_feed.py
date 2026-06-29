"""
massive_feed.py  —  live adapter: Massive options chain snapshot -> OptionRow.

SECURITY: credentials are read from environment variables ONLY. Nothing is
hardcoded. Never paste a key into this file or any chat.
    export MASSIVE_API_KEY=...           # API key or dashboard-generated OAuth token
    export MASSIVE_OAUTH_CLIENT_ID=...   # Personal OAuth2 app (alternative to API key)
    export MASSIVE_OAUTH_CLIENT_SECRET=...
    export MASSIVE_BASE_URL=https://api.massive.com   # confirm host in your dashboard

Endpoint used (Polygon-compatible path, which Massive mirrors):
    GET /v3/snapshot/options/{underlyingAsset}
Returns every contract for the underlying with greeks, OI, quotes, and the
underlying price — exactly the fields the gamma map consumes.

The field mapping below follows the Polygon/Massive snapshot convention. Run the
self-check against ONE real response (python massive_feed.py SPY) and confirm the
three field paths flagged with #CONFIRM before trusting it live.
"""
from __future__ import annotations
import os
import json
import gzip
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from zoneinfo import ZoneInfo

from spy0dte import OptionRow   # reuse the engine's dataclass — single source of truth

ET = ZoneInfo("America/New_York")
_OAUTH_TOKEN_URL = "https://auth.massive.com/oauth2/token"
_oauth_cache: dict[str, float | str] = {"token": "", "expires_at": 0.0}


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _oauth_client_credentials(client_id: str, client_secret: str) -> str:
    """Exchange Personal OAuth2 app credentials for a short-lived bearer token."""
    now = time.time()
    cached = _oauth_cache.get("token")
    if cached and now < float(_oauth_cache.get("expires_at", 0)):
        return str(cached)

    token_url = os.environ.get("MASSIVE_OAUTH_TOKEN_URL", _OAUTH_TOKEN_URL)
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        token_url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"Massive OAuth token HTTP {e.code}: {detail}") from None

    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Massive OAuth token response missing access_token: {payload}")
    expires_in = int(payload.get("expires_in", 3600))
    _oauth_cache["token"] = token
    _oauth_cache["expires_at"] = now + max(expires_in - 60, 30)
    return token


def _resolve_auth_token() -> str:
    """Return a bearer token from API key, OAuth access token, or OAuth app creds."""
    for env in ("MASSIVE_API_KEY", "MASSIVE_OAUTH_ACCESS_TOKEN"):
        token = os.environ.get(env, "").strip()
        if token:
            return token

    client_id = os.environ.get("MASSIVE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("MASSIVE_OAUTH_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return _oauth_client_credentials(client_id, client_secret)

    raise RuntimeError(
        "Set MASSIVE_API_KEY, MASSIVE_OAUTH_ACCESS_TOKEN, or "
        "MASSIVE_OAUTH_CLIENT_ID + MASSIVE_OAUTH_CLIENT_SECRET"
    )


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
    """Map one snapshot contract to an OptionRow. Returns None if data is incomplete."""
    details = c.get("details", {})              # CONFIRM: contract_type, strike_price
    greeks = c.get("greeks", {})                # CONFIRM: gamma, delta
    side = details.get("contract_type")         # "call" | "put"
    strike = details.get("strike_price")
    gamma = greeks.get("gamma")
    delta = greeks.get("delta")
    oi = c.get("open_interest")
    if None in (side, strike, gamma, delta, oi):
        return None

    # Prefer real-time last_quote; fall back to day.close
    quote = c.get("last_quote", {})             # CONFIRM: bid, ask
    bid = quote.get("bid")
    ask = quote.get("ask")
    if bid is None or ask is None:
        close = c.get("day", {}).get("close")
        if close is None:
            return None
        bid = ask = close  # spread_pct=0; mid=close

    if bid <= 0 or ask <= 0:
        return None
    return OptionRow(side=side, strike=float(strike), oi=int(oi),
                     gamma=float(gamma), bid=float(bid), ask=float(ask),
                     delta=abs(float(delta)))


def get_chain(underlying: str, zero_dte_only: bool = True) -> tuple[float, list[OptionRow]]:
    """Fetch the chain snapshot, filter to today's expiry, map to OptionRows.
    Returns (spot, rows). Raises if the key/host are wrong."""
    key = _resolve_auth_token()
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")

    url = f"{base}/v3/snapshot/options/{underlying}?limit=250"
    today = _today_et()
    rows: list[OptionRow] = []
    spot = 0.0
    pages = 0

    best_delta = 0.0  # track deepest-ITM call for spot estimation

    while url and pages < 25:            # safety cap on pagination
        data = _get(url, key)
        for c in data.get("results", []):
            # Prefer API-provided spot; fall back to deep-ITM call intrinsic value
            ua = c.get("underlying_asset", {})
            if ua.get("price"):
                spot = float(ua["price"])
            details = c.get("details", {})
            greeks = c.get("greeks", {}) or {}
            d = abs(greeks.get("delta") or 0)
            close = c.get("day", {}).get("close") or 0
            if (details.get("contract_type") == "call"
                    and d > best_delta and d > 0.95 and close > 0):
                best_delta = d
                spot = float(details.get("strike_price", 0)) + float(close)

            if zero_dte_only and c.get("details", {}).get("expiration_date") != today:
                continue
            row = _row_from_contract(c)
            if row:
                rows.append(row)
        url = data.get("next_url")
        pages += 1

    return spot, rows


def diagnose(underlying: str) -> None:
    """Run ONE call and print the response STRUCTURE so we can confirm field
    mappings. Prints key paths and a sample mapped row — never your key."""
    try:
        key = _resolve_auth_token()
    except RuntimeError as e:
        print(e); return
    base = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")
    url = f"{base}/v3/snapshot/options/{underlying}?limit=5"
    try:
        data = _get(url, key)
    except RuntimeError as e:
        print("CALL FAILED:", e)
        if "OAuth token" in str(e):
            print("OAuth hint: Personal OAuth2 apps may require generating an access token")
            print("from the Massive dashboard, then set MASSIVE_OAUTH_ACCESS_TOKEN or MASSIVE_API_KEY.")
            print("Or use a dashboard API key from https://massive.com/dashboard/api-keys")
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
    row = None
    for c in results:
        row = _row_from_contract(c)
        if row:
            break
    if row:
        print(f"  OK -> {row.side} {row.strike} OI={row.oi} gamma={row.gamma} delta={row.delta} {row.bid}/{row.ask}")
    else:
        print("  MAPPING FAILED -> no contract on this page had complete greeks/OI/quotes.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose(sys.argv[2] if len(sys.argv) > 2 else "SPY")
        sys.exit(0)
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    spot, rows = get_chain(sym)
    print(f"{sym}: spot={spot} | {len(rows)} 0DTE rows with complete greeks/quotes")
    for r in rows[:6]:
        print(f"  {r.side:4s} {r.strike:7.1f}  OI={r.oi:<6d} Γ={r.gamma:.4f} "
              f"Δ={r.delta:.2f}  {r.bid:.2f}/{r.ask:.2f}")
    # end-to-end: feed straight into the engine
    if rows:
        import spy0dte as eng
        gm = eng.build_gamma_map(rows, spot)
        print(f"  -> netGEX ratio {gm.net_ratio} | flip {gm.gamma_flip} | "
              f"walls {gm.put_wall}/{gm.call_wall} | regime {gm.regime.upper()}")
        d = eng.decide(gm, price_accepting=0)
        print(f"DECISION: {d.action} — {d.reason}")
        if d.action in ("CALL", "PUT"):
            risk, _ = eng.scale_risk(n_trades=0, win_rate=0.0, avg_win=0, avg_loss=0)
            order = eng.select_order(rows, d, equity=1000.0, risk_frac=risk)
            if order:
                print("ORDER:", order.thesis, f"| risk ${order.dollar_risk:.0f}")
        elif d.action == "SELL_CONDOR":
            risk, _ = eng.scale_risk(n_trades=0, win_rate=0.0, avg_win=0, avg_loss=0)
            condor = eng.select_condor(rows, gm, equity=1000.0, risk_frac=risk)
            if condor:
                print("ORDER:", condor.thesis, f"| risk ${condor.dollar_risk:.0f}")
