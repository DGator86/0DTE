#!/usr/bin/env bash
# Copy dashboard static assets into public/ for Vercel hosting.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$ROOT/public"
mkdir -p "$ROOT/public/static"
cp "$ROOT/dashboard/static/index.html" "$ROOT/public/"
cp "$ROOT/dashboard/static/app.js" "$ROOT/dashboard/static/style.css" "$ROOT/public/static/"
