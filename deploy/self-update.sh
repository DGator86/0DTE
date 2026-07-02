#!/usr/bin/env bash
# self-update.sh — pull-based deploy: poll origin for a new commit on the
# deploy branch and, when one lands, run the repo's own remote-deploy.sh.
#
# Runs ON the VPS as root from zerodte-update.timer (every 2 minutes). This is
# the deploy path that needs NO inbound SSH: pushes to main land on the box
# because the box pulls, so a Hostinger firewall change, an IP rotation, or a
# fail2ban ban of GitHub's runners can never strand a release again. The SSH
# push-to-deploy workflow still works when reachable; both converge on the
# same idempotent remote-deploy.sh.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/zerodte}
BRANCH=${DEPLOY_BRANCH:-main}

if [ ! -d "$APP_DIR/.git" ]; then
    echo "self-update: no checkout at $APP_DIR — run remote-deploy.sh once first."
    exit 0
fi

git -C "$APP_DIR" fetch --quiet --prune origin "$BRANCH"
local_sha=$(git -C "$APP_DIR" rev-parse HEAD)
remote_sha=$(git -C "$APP_DIR" rev-parse "origin/$BRANCH")

if [ "$local_sha" = "$remote_sha" ]; then
    exit 0                          # up to date; stay silent in the journal
fi

echo "self-update: ${local_sha:0:8} -> ${remote_sha:0:8} on $BRANCH"
# Run the NEW deploy script (from the fetched ref, not the old checkout) so
# unit-file and dependency changes ship atomically with the code needing them.
git -C "$APP_DIR" show "origin/$BRANCH:deploy/remote-deploy.sh" \
    | DEPLOY_REF="$remote_sha" bash -s
