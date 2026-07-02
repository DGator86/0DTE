"""
dashboard/auth.py
=================
Bearer-token auth for the read-only dashboard.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def get_dashboard_token() -> Optional[str]:
    return os.environ.get("DASHBOARD_TOKEN") or None


def check_token(request: Request) -> None:
    expected = get_dashboard_token()
    if not expected:
        raise HTTPException(status_code=503, detail="DASHBOARD_TOKEN not configured")

    # Header only — a ?token= query param would end up in access logs and
    # browser history. (The UI's ?token= bookmark never reaches /api/*: app.js
    # moves it into sessionStorage and sends the Authorization header.)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], expected):
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


class ReadOnlyMiddleware(BaseHTTPMiddleware):
    """Reject mutating HTTP methods."""

    async def dispatch(self, request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            return JSONResponse(
                status_code=405,
                content={"detail": "Method not allowed — read-only dashboard"},
            )
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require token on /api/* routes."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            try:
                check_token(request)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)
