#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/DGator86/0DTE.git"
BRANCH="claude/new-session-3o0dzc"
TARGET="/root/0DTE"
BACKUP="/tmp/0dte-migrate-$$"

echo "=== backup secrets and logs ==="
mkdir -p "$BACKUP"
cp -a "$TARGET/.env" "$BACKUP/.env"
cp -a "$TARGET/logs" "$BACKUP/logs" 2>/dev/null || mkdir -p "$BACKUP/logs"

echo "=== replace with git clone ==="
rm -rf "$TARGET"
git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$TARGET"

echo "=== restore local state ==="
cp -a "$BACKUP/.env" "$TARGET/.env"
chmod 600 "$TARGET/.env"
mkdir -p "$TARGET/logs"
cp -a "$BACKUP/logs/." "$TARGET/logs/" 2>/dev/null || true
chmod +x "$TARGET/run_0dte.sh"

echo "=== python venv ==="
cd "$TARGET"
python3 -m venv venv
venv/bin/pip install -q -r requirements.txt
venv/bin/python -m py_compile spy0dte.py mc.py journal.py massive_feed.py tastytrade_auth.py tradier_client.py tradier_feed.py

echo "=== verify ==="
git -C "$TARGET" remote -v
git -C "$TARGET" log -1 --oneline
/root/0DTE/run_0dte.sh 2>&1 | tail -3
crontab -l | grep 0DTE

rm -rf "$BACKUP"
echo "=== git pull ready at $TARGET ==="
