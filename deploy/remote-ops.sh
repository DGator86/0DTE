#!/usr/bin/env bash
# remote-ops.sh — operational commands run ON the VPS over SSH by the "VPS Ops"
# GitHub Action (.github/workflows/ops.yml). It is the read/observe + light
# control surface for the zerodte-shadow service, so the box can be inspected and
# nudged through auditable, logged Action runs without anyone holding the SSH key
# locally.
#
# Invoked as:  ssh root@VPS "COMMAND='status' ARG='' bash -s" < deploy/remote-ops.sh
#
# SAFETY: prints no secrets. Diagnostics source the env file (as root, which can
# read the 0600 file) only to authenticate the feed check — the feed diagnostics
# print AUTH status + contract counts, never the token. No command places an
# order; the service is shadow-mode only.
set -euo pipefail

SVC=zerodte-shadow
APP=/opt/zerodte
DB=/var/lib/zerodte/shadow.db
ENVF=/etc/zerodte/zerodte.env
PY="$APP/venv/bin/python"
RUN_USER=zerodte

CMD="${COMMAND:-status}"
ARG="${ARG:-}"

# Run a command as the unprivileged service user (owns the journal DB).
as_svc() { sudo -u "$RUN_USER" "$@"; }

# Print the service status, tolerating ONLY the inactive-unit case. `systemctl
# status` exits 0 when active and 3 when inactive/dead (both informational here);
# any other code (e.g. 4 = no such unit) is a real error and must fail the job.
show_status() {
    local lines="${1:-60}" out rc=0
    out="$(systemctl status "$SVC" --no-pager -l 2>&1)" || rc=$?
    printf '%s\n' "$out" | head -n "$lines"
    case "$rc" in
        0|3) return 0 ;;
        *) echo "systemctl status: unexpected exit $rc" >&2; return "$rc" ;;
    esac
}

echo "== zerodte ops: ${CMD} ${ARG:+(arg: $ARG)} =="

case "$CMD" in
  status)
    show_status 60
    ;;

  logs)
    n="$ARG"; case "$n" in ''|*[!0-9]*) n=200 ;; esac
    journalctl -u "$SVC" -n "$n" --no-pager
    ;;

  report)
    # Calibration summary from the journal (gate effectiveness + correlations).
    as_svc "$PY" "$APP/shadow_runner.py" --report --db "$DB"
    ;;

  diagnose-tradier)
    # Confirms real-time NBBO entitlement. Runs as root to read the 0600 env file;
    # the diagnostic prints AUTH OK + a live-quote count, never the token.
    bash -c "set -a; . '$ENVF'; set +a; '$PY' '$APP/tradier_feed.py' '${ARG:-SPY}'"
    ;;

  diagnose-tastytrade)
    bash -c "set -a; . '$ENVF'; set +a; '$PY' '$APP/tastytrade_feed.py' '${ARG:-SPY}'"
    ;;

  restart)
    systemctl restart "$SVC"
    sleep 2
    show_status 20
    ;;

  settle)
    [ -n "$ARG" ] || { echo "settle requires a date arg (YYYY-MM-DD)"; exit 2; }
    as_svc "$PY" "$APP/shadow_runner.py" --settle "$ARG" --db "$DB"
    ;;

  test-notify)
    # Send a test push through the SAME ntfy path real trade signals use, reading
    # the topic from the 0600 env file (as root). The topic is never printed —
    # only the HTTP result — so it stays private.
    set -a; . "$ENVF"; set +a
    [ -n "${NOTIFY_NTFY_TOPIC:-}" ] || { echo "NOTIFY_NTFY_TOPIC not set in $ENVF"; exit 2; }
    NOTIFY_NTFY_TOPIC="$NOTIFY_NTFY_TOPIC" NOTIFY_NTFY_TOKEN="${NOTIFY_NTFY_TOKEN:-}" "$PY" - <<'PYEOF'
import os, urllib.request
topic = os.environ["NOTIFY_NTFY_TOPIC"]
token = os.environ.get("NOTIFY_NTFY_TOKEN", "")
req = urllib.request.Request(
    f"https://ntfy.sh/{topic}",
    data="If you can read this on your phone, your zerodte trade alerts are wired up correctly.".encode(),
    headers={"Title": "zerodte test alert", "Priority": "high", "Tags": "white_check_mark"},
)
if token:
    req.add_header("Authorization", f"Bearer {token}")
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"ntfy POST HTTP {r.status} — check your phone (topic hidden)")
        raise SystemExit(0 if r.status == 200 else 1)
except urllib.error.HTTPError as e:
    print(f"ntfy POST failed: HTTP {e.code}"); raise SystemExit(1)
PYEOF
    ;;

  *)
    echo "Unknown command: $CMD" >&2
    echo "Valid: status | logs | report | diagnose-tradier | diagnose-tastytrade | restart | settle | test-notify" >&2
    exit 2
    ;;
esac
