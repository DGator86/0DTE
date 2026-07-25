#!/usr/bin/env bash
# sync-experience-to-spyder.sh — copy 0DTE MarketPacket / OutcomePacket outbox
# into the SPY-DER Dojo experience inbox.
#
# Why: 0DTE publishes to /var/lib/zerodte/spyder_experience (writable by the
# zerodte user). SPY-DER Dojo timers read /var/lib/spy-der/inbox/experience.
# Without this sync, Dojo sees "0 sessions" even when shadow has been publishing.
#
# Safe to re-run. Merges sessions.json. Does not delete anything on either side.
# Run as root on the VPS (needs to chown the inbox to spy-der).
set -euo pipefail

SRC="${ZERODTE_SPYDER_EXPERIENCE_ROOT:-/var/lib/zerodte/spyder_experience}"
DST="${SPY_DER_EXPERIENCE_INBOX:-/var/lib/spy-der/inbox/experience}"
DST_USER="${SPY_DER_USER:-spy-der}"
DST_GROUP="${SPY_DER_GROUP:-spy-der}"

log() { printf '==> %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ] && [ "${SYNC_ALLOW_NONROOT:-0}" != "1" ]; then
    echo "run as root (need to chown $DST to $DST_USER)" >&2
    exit 1
fi

if [ ! -d "$SRC" ]; then
    echo "source experience root missing: $SRC" >&2
    echo "Nothing to sync — has zerodte-shadow been publishing MarketPackets?" >&2
    exit 2
fi

mkdir -p "$DST/snapshots" "$DST/outcomes"

snap_src=$(find "$SRC/snapshots" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
out_src=$(find "$SRC/outcomes" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
log "source $SRC — snapshots=$snap_src outcomes=$out_src"

if [ "$snap_src" = "0" ] && [ "$out_src" = "0" ]; then
    echo "source has no packets yet — inbox left unchanged" >&2
    exit 0
fi

# Copy packet files (no --delete: inbox may also receive native SPY-DER writes).
_copy_tree() {
    local from="$1" to="$2"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$from/" "$to/"
    else
        mkdir -p "$to"
        # Portable fallback when rsync is absent (dev/CI images).
        cp -a "$from"/. "$to"/
    fi
}
if [ -d "$SRC/snapshots" ]; then
    _copy_tree "$SRC/snapshots" "$DST/snapshots"
fi
if [ -d "$SRC/outcomes" ]; then
    _copy_tree "$SRC/outcomes" "$DST/outcomes"
fi

# Merge sessions.json (union of both sides).
python3 - "$SRC" "$DST" <<'PY'
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])

def load(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]

sessions = sorted(set(load(src / "sessions.json")) | set(load(dst / "sessions.json")))
# Also harvest session_date from snapshot packets if sessions.json was never written.
snap_dir = dst / "snapshots"
if snap_dir.is_dir():
    for path in snap_dir.glob("*.json"):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and body.get("session_date"):
            sessions.append(str(body["session_date"]))
sessions = sorted(set(sessions))
tmp = dst / "sessions.json.tmp"
tmp.write_text(json.dumps(sessions, indent=2) + "\n", encoding="utf-8")
tmp.replace(dst / "sessions.json")
print(f"sessions={len(sessions)} -> {dst / 'sessions.json'}")
if sessions:
    print("  " + ", ".join(sessions[:12]) + ("…" if len(sessions) > 12 else ""))
PY

# Inbox must be readable/owned by spy-der (Dojo runs as that user).
# World-readable packets so dashboard / other readers can inspect.
find "$DST" -type d -exec chmod 0755 {} +
find "$DST" -type f -exec chmod 0644 {} +
if id -u "$DST_USER" >/dev/null 2>&1; then
    chown -R "$DST_USER:$DST_GROUP" /var/lib/spy-der/inbox
else
    log "WARN: user $DST_USER missing — left inbox root-owned (modes 0644/0755)"
fi

snap_dst=$(find "$DST/snapshots" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
out_dst=$(find "$DST/outcomes" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
log "destination $DST — snapshots=$snap_dst outcomes=$out_dst"
log "sync complete — re-run spy-der dojo (sessions > 0 if source had tape)"
