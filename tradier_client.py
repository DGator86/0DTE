"""
tradier_client.py — Tradier Brokerage API helper (stdlib only).

Credentials from environment only. Never hardcode tokens.
    export TRADIER_API_TOKEN=...          # from web.tradier.com/user/api
    export TRADIER_ACCOUNT_ID=6YB70758    # optional; auto-detected from profile
    export TRADIER_SANDBOX=0              # set 1 for sandbox.tradier.com

Docs: https://docs.tradier.com/docs/getting-started
MCP:  https://docs.tradier.com/docs/tradier-mcp
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

PROD_BASE = "https://api.tradier.com"
SANDBOX_BASE = "https://sandbox.tradier.com"


def _api_base() -> str:
    if os.environ.get("TRADIER_SANDBOX", "").lower() in ("1", "true", "yes"):
        return os.environ.get("TRADIER_API_BASE", SANDBOX_BASE).rstrip("/")
    return os.environ.get("TRADIER_API_BASE", PROD_BASE).rstrip("/")


def _token() -> str:
    token = os.environ.get("TRADIER_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TRADIER_API_TOKEN not set")
    return token


def api_request(method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
    """Authenticated JSON request to Tradier REST API."""
    url = f"{_api_base()}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(f"Tradier API HTTP {e.code} {path}: {detail}") from None


def get_profile() -> dict:
    return api_request("GET", "/v1/user/profile")


def account_id() -> str:
    explicit = os.environ.get("TRADIER_ACCOUNT_ID", "").strip()
    if explicit:
        return explicit
    profile = get_profile()
    accounts = profile.get("profile", {}).get("account", [])
    if isinstance(accounts, dict):
        accounts = [accounts]
    if not accounts:
        raise RuntimeError("No Tradier accounts found on profile")
    return str(accounts[0]["account_number"])


def get_quotes(symbols: str, greeks: bool = False) -> dict:
    return api_request("GET", "/v1/markets/quotes", {
        "symbols": symbols,
        "greeks": "true" if greeks else "false",
    })


def get_options_expirations(symbol: str) -> list[str]:
    data = api_request("GET", "/v1/markets/options/expirations", {
        "symbol": symbol,
        "includeAllRoots": "true",
    })
    dates = data.get("expirations", {}).get("date", [])
    if isinstance(dates, str):
        return [dates]
    return list(dates or [])


def get_options_chain(symbol: str, expiration: str, greeks: bool = True) -> list[dict]:
    data = api_request("GET", "/v1/markets/options/chains", {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": "true" if greeks else "false",
    })
    options = data.get("options", {}).get("option", [])
    if isinstance(options, dict):
        return [options]
    return list(options or [])


def diagnose() -> None:
    print("Tradier API diagnose")
    print(f"  api_base: {_api_base()}")
    try:
        profile = get_profile()
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    p = profile.get("profile", {})
    print(f"  user: {p.get('name', '?')} ({p.get('id', '?')})")
    accounts = p.get("account", [])
    if isinstance(accounts, dict):
        accounts = [accounts]
    print(f"  accounts: {len(accounts)}")
    for acct in accounts:
        num = acct.get("account_number", "?")
        opt_lvl = acct.get("option_level", "?")
        acct_type = acct.get("type", "?")
        print(f"    - {num} ({acct_type}, option level {opt_lvl})")

    try:
        q = get_quotes("SPY")
        quote = q.get("quotes", {}).get("quote", {})
        if isinstance(quote, list):
            quote = quote[0] if quote else {}
        print(f"  SPY quote: last={quote.get('last')} bid={quote.get('bid')} ask={quote.get('ask')}")
    except RuntimeError as e:
        print(f"  quote check FAILED: {e}")

    try:
        exps = get_options_expirations("SPY")
        print(f"  SPY expirations: {len(exps)} available, nearest={exps[0] if exps else 'none'}")
    except RuntimeError as e:
        print(f"  options expirations FAILED: {e}")


if __name__ == "__main__":
    diagnose()
