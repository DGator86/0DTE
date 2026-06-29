"""
tastytrade_auth.py — Tastytrade OAuth2 session helper (stdlib only).

Credentials come from environment variables only. Never hardcode secrets.

One-time setup (after creating your OAuth app):
  1. developer.tastytrade.com → OAuth Applications → Manage → Create Grant
  2. Save the refresh token into .env as TASTYTRADE_REFRESH_TOKEN
  3. Set TASTYTRADE_CLIENT_ID, TASTYTRADE_CLIENT_SECRET, and scopes you granted

Ongoing use: call refresh_access_token() before API requests. Access tokens last
~15 minutes; refresh tokens do not expire.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

PROD_API_BASE = "https://api.tastyworks.com"
SANDBOX_API_BASE = "https://api.cert.tastyworks.com"
AUTH_PAGE = "https://my.tastytrade.com/auth.html"
_DEFAULT_SCOPE = "read trade openid"
_token_cache: dict[str, float | str] = {"token": "", "expires_at": 0.0}


def _api_base() -> str:
    if os.environ.get("TASTYTRADE_SANDBOX", "").lower() in ("1", "true", "yes"):
        return os.environ.get("TASTYTRADE_API_BASE", SANDBOX_API_BASE).rstrip("/")
    return os.environ.get("TASTYTRADE_API_BASE", PROD_API_BASE).rstrip("/")


def _client_id() -> str:
    client_id = os.environ.get("TASTYTRADE_CLIENT_ID", "").strip()
    if not client_id:
        raise RuntimeError("TASTYTRADE_CLIENT_ID not set")
    return client_id


def _client_secret() -> str:
    secret = os.environ.get("TASTYTRADE_CLIENT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("TASTYTRADE_CLIENT_SECRET not set")
    return secret


def _refresh_token() -> str:
    token = os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TASTYTRADE_REFRESH_TOKEN not set — create a grant at "
            "developer.tastytrade.com → OAuth Applications → Manage → Create Grant"
        )
    return token


def _scope() -> str:
    return os.environ.get("TASTYTRADE_SCOPE", _DEFAULT_SCOPE).strip() or _DEFAULT_SCOPE


def authorization_url(redirect_uri: str | None = None, scope: str | None = None) -> str:
    """Build the browser URL for the one-time authorization-code flow."""
    redirect = redirect_uri or os.environ.get("TASTYTRADE_REDIRECT_URI", "http://localhost:8000")
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": redirect,
        "scope": scope or _scope(),
    })
    return f"{AUTH_PAGE}?{params}"


def _token_request(body: dict[str, str]) -> dict:
    url = f"{_api_base()}/oauth/token"
    data = urllib.parse.urlencode(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(f"Tastytrade OAuth HTTP {e.code}: {detail}") from None


def exchange_code(code: str, redirect_uri: str | None = None) -> dict:
    """Exchange an authorization code for access + refresh tokens."""
    redirect = redirect_uri or os.environ.get("TASTYTRADE_REDIRECT_URI", "http://localhost:8000")
    return _token_request({
        "grant_type": "authorization_code",
        "code": code,
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "redirect_uri": redirect,
    })


def refresh_access_token(force: bool = False) -> str:
    """Return a valid session token, refreshing from the long-lived refresh token."""
    now = time.time()
    cached = _token_cache.get("token")
    if not force and cached and now < float(_token_cache.get("expires_at", 0)):
        return str(cached)

    payload = _token_request({
        "grant_type": "refresh_token",
        "client_secret": _client_secret(),
        "refresh_token": _refresh_token(),
        "scope": _scope(),
    })
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Tastytrade token response missing access_token: {payload}")

    expires_in = int(payload.get("expires_in", 900))
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + max(expires_in - 60, 30)
    return token


def api_request(method: str, path: str, body: dict | None = None) -> dict:
    """Authenticated JSON request to the Tastytrade REST API."""
    token = refresh_access_token()
    url = f"{_api_base()}{path}"
    data = None
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(f"Tastytrade API HTTP {e.code} {path}: {detail}") from None


def diagnose() -> None:
    """Verify OAuth app credentials and, if configured, list linked accounts."""
    print("Tastytrade OAuth diagnose")
    print(f"  api_base: {_api_base()}")
    try:
        _client_id()
        _client_secret()
        print("  client_id / client_secret: OK (set)")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    if not os.environ.get("TASTYTRADE_REFRESH_TOKEN", "").strip():
        print("  refresh_token: MISSING")
        print("  Next step: developer.tastytrade.com → OAuth Applications → Manage → Create Grant")
        print("  Then set TASTYTRADE_REFRESH_TOKEN in .env")
        print("  Or open this URL once, approve, and exchange the callback code:")
        print(f"  {authorization_url()}")
        return

    try:
        refresh_access_token(force=True)
        print("  refresh_token: OK (access token acquired)")
    except RuntimeError as e:
        print(f"  token refresh FAILED: {e}")
        return

    try:
        data = api_request("GET", "/customers/me/accounts")
        accounts = data.get("data", {}).get("items", [])
        print(f"  accounts: {len(accounts)} linked")
        for acct in accounts[:5]:
            num = acct.get("account", {}).get("account-number", "?")
            acct_type = acct.get("account", {}).get("account-type-name", "?")
            print(f"    - {num} ({acct_type})")
    except RuntimeError as e:
        print(f"  account lookup FAILED: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "auth-url":
        print(authorization_url())
    elif len(sys.argv) > 2 and sys.argv[1] == "exchange":
        result = exchange_code(sys.argv[2])
        print("access_token: acquired")
        if result.get("refresh_token"):
            print("refresh_token:", result["refresh_token"])
            print("Save it as TASTYTRADE_REFRESH_TOKEN in .env")
    else:
        diagnose()
