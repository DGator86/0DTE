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
PAPER_DB=/var/lib/zerodte/paper.sqlite
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

  paper-report)
    # Paper-trading P&L: equity, win rate, total P&L, exits breakdown.
    as_svc "$PY" "$APP/shadow_runner.py" --paper-report --paper-db "$PAPER_DB"
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

  validate)
    # Run the validation pipeline on demand (ARG: daily | weekly; default daily).
    # Same run the scheduled timers perform; the report lands in
    # validation_reports and shows up in the dashboard's Validation tab.
    mode="${ARG:-daily}"
    case "$mode" in daily|weekly) ;; *) echo "validate arg must be 'daily' or 'weekly'"; exit 2 ;; esac
    as_svc "$PY" "$APP/validation_pipeline.py" --mode "$mode" \
        --db "$DB" --record-dir /var/lib/zerodte/ticks
    ;;

  learn)
    # DEPRECATED (Phase 5): adaptive learning is owned by SPY-DER.
    echo "learn: removed from 0DTE — run SPY-DER learning / Dojo instead" >&2
    echo "See docs/migrations/PHASE5_AI_OWNERSHIP_REMOVAL.md" >&2
    echo "Enable spy-der-dojo-* timers and spy-der-agent.service on the VPS." >&2
    exit 2
    ;;

  experience-status)
    # Read-only: confirm 0DTE outbox → SPY-DER Dojo inbox sync state.
    SRC="${ZERODTE_SPYDER_EXPERIENCE_ROOT:-/var/lib/zerodte/spyder_experience}"
    DST="${SPY_DER_EXPERIENCE_INBOX:-/var/lib/spy-der/inbox/experience}"
    echo "--- source outbox: $SRC ---"
    if [ -d "$SRC/snapshots" ]; then
      echo "snapshots: $(find "$SRC/snapshots" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
      echo "outcomes:  $(find "$SRC/outcomes" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    else
      echo "snapshots dir missing"
    fi
    if [ -f "$SRC/sessions.json" ]; then
      echo "sessions.json:"
      cat "$SRC/sessions.json"
    else
      echo "sessions.json: (missing)"
    fi
    echo
    echo "--- dojo inbox: $DST ---"
    if [ -d "$DST/snapshots" ]; then
      echo "snapshots: $(find "$DST/snapshots" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
      echo "outcomes:  $(find "$DST/outcomes" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    else
      echo "snapshots dir missing"
    fi
    if [ -f "$DST/sessions.json" ]; then
      echo "sessions.json:"
      cat "$DST/sessions.json"
    else
      echo "sessions.json: (missing)"
    fi
    echo
    echo "--- spy-der-dojo-recent.service ---"
    systemctl status spy-der-dojo-recent.service --no-pager -l 2>&1 | head -n 40 || true
    echo
    echo "--- zerodte-sync-experience.timer ---"
    systemctl status zerodte-sync-experience.timer --no-pager -l 2>&1 | head -n 20 || true
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
    echo "Valid: status | logs | report | paper-report | diagnose-tradier | diagnose-tastytrade | restart | settle | validate | learn | experience-status | test-notify" >&2
    exit 2
    ;;
esac
