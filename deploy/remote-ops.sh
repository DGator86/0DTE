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

echo "== zerodte ops: ${CMD} ${ARG:+(arg: $ARG)} =="

case "$CMD" in
  status)
    # `systemctl status` exits 3 for an inactive unit — that's informational
    # here, not a failure, so don't let it fail the job.
    systemctl status "$SVC" --no-pager -l | head -60 || true
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
    systemctl status "$SVC" --no-pager | head -20 || true
    ;;

  settle)
    [ -n "$ARG" ] || { echo "settle requires a date arg (YYYY-MM-DD)"; exit 2; }
    as_svc "$PY" "$APP/shadow_runner.py" --settle "$ARG" --db "$DB"
    ;;

  *)
    echo "Unknown command: $CMD" >&2
    echo "Valid: status | logs | report | diagnose-tradier | diagnose-tastytrade | restart | settle" >&2
    exit 2
    ;;
esac
