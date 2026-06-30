#!/usr/bin/env bash
# Run SPY 0DTE decision engine with live Massive feed (if configured).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8

PY="$ROOT/venv/bin/python3"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${MASSIVE_API_KEY:-}" && -z "${MASSIVE_OAUTH_ACCESS_TOKEN:-}" \
      && ( -z "${MASSIVE_OAUTH_CLIENT_ID:-}" || -z "${MASSIVE_OAUTH_CLIENT_SECRET:-}" ) ]]; then
  echo "[0DTE] No Massive credentials set — running synthetic demo"
  "$PY" spy0dte.py
else
  "$PY" massive_feed.py SPY
fi
