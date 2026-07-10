#!/usr/bin/env bash
# remote-deploy.sh — idempotent deploy of the 0DTE shadow pipeline on a VPS.
#
# Runs ON the Hostinger VPS, as root (Hostinger's default SSH user). The CI
# workflow in .github/workflows/deploy.yml pipes this script over SSH:
#
#     ssh root@VPS 'DEPLOY_REF=origin/main bash -s' < deploy/remote-deploy.sh
#
# First run provisions everything (service user, code, venv, systemd unit).
# Every later run just fast-forwards the checkout and restarts the service.
# It is safe to re-run any number of times.
#
# The ONE thing it never touches is the secrets file (/etc/zerodte/zerodte.env):
# that is created by hand once (see deploy/README.md) and the script refuses to
# start the service until it exists, so a key is never overwritten by a deploy.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/DGator86/0DTE.git}"
DEPLOY_REF="${DEPLOY_REF:-origin/main}"   # what the workflow asked us to deploy
APP_DIR=/opt/zerodte
ENV_FILE=/etc/zerodte/zerodte.env
DATA_DIR=/var/lib/zerodte
SVC=zerodte-shadow
SVC_USER=zerodte

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "remote-deploy.sh must run as root (got uid $(id -u))." >&2
    exit 1
fi

log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null

log "Service user + directories"
id -u "$SVC_USER" >/dev/null 2>&1 || \
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
mkdir -p "$APP_DIR" /etc/zerodte "$DATA_DIR"
chown "$SVC_USER:$SVC_USER" "$DATA_DIR"

log "Code -> $DEPLOY_REF"
if [ ! -d "$APP_DIR/.git" ]; then
    git clone "$REPO_URL" "$APP_DIR"
fi
git -C "$APP_DIR" remote set-url origin "$REPO_URL"
git -C "$APP_DIR" fetch --prune origin
# Resolve the ref: a branch name ("main") tracks the remote tip, while a raw
# commit SHA is used as-is. The checkout is read-only at runtime, so a clean
# hard-reset is the whole update — there are no local commits to preserve.
if git -C "$APP_DIR" rev-parse --verify -q "origin/$DEPLOY_REF^{commit}" >/dev/null; then
    TARGET="origin/$DEPLOY_REF"
else
    TARGET="$DEPLOY_REF"
fi
git -C "$APP_DIR" reset --hard "$TARGET"
echo "Deployed commit: $(git -C "$APP_DIR" rev-parse --short HEAD)"

log "Virtualenv + dependencies"
if [ ! -x "$APP_DIR/venv/bin/python" ]; then
    python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

log "systemd unit"
install -m 644 "$APP_DIR/deploy/$SVC.service" "/etc/systemd/system/$SVC.service"
systemctl daemon-reload

log "Self-update timer (pull-based deploys; inbound SSH not required)"
install -m 644 "$APP_DIR/deploy/zerodte-update.service" /etc/systemd/system/zerodte-update.service
install -m 644 "$APP_DIR/deploy/zerodte-update.timer" /etc/systemd/system/zerodte-update.timer
systemctl daemon-reload
systemctl enable --now zerodte-update.timer >/dev/null 2>&1 || true

log "Validation timers (daily post-close + weekly deep review)"
for unit in zerodte-validate-daily zerodte-validate-weekly; do
    install -m 644 "$APP_DIR/deploy/$unit.service" "/etc/systemd/system/$unit.service"
    install -m 644 "$APP_DIR/deploy/$unit.timer" "/etc/systemd/system/$unit.timer"
done
systemctl daemon-reload
systemctl enable --now zerodte-validate-daily.timer >/dev/null 2>&1 || true
systemctl enable --now zerodte-validate-weekly.timer >/dev/null 2>&1 || true

if [ ! -f "$ENV_FILE" ]; then
    # printf renders the color; a plain heredoc can't interpret \033 escapes
    # and would print them literally — and this is the first-run message.
    printf '\n\033[1;33m%s\033[0m\n' "Secrets file $ENV_FILE not found — service NOT started." >&2
    cat >&2 <<EOF
Code is deployed, but the pipeline needs your API key first. One-time setup:

    sudo install -D -m 600 -o root -g $SVC_USER \\
         $APP_DIR/deploy/zerodte.env.example $ENV_FILE
    sudo nano $ENV_FILE          # set MASSIVE_API_KEY + NOTIFY_NTFY_TOPIC

Then re-run the deploy (push again or trigger the workflow) and it will start.
EOF
    exit 0
fi

log "Enable + restart service"
systemctl enable "$SVC" >/dev/null 2>&1 || true
systemctl restart "$SVC"
sleep 2
systemctl --no-pager --lines=0 status "$SVC" || true

DASHBOARD_SVC=zerodte-dashboard
if grep -q '^DASHBOARD_TOKEN=.' "$ENV_FILE" 2>/dev/null; then
    log "Dashboard service (DASHBOARD_TOKEN set)"
    install -m 644 "$APP_DIR/deploy/$DASHBOARD_SVC.service" "/etc/systemd/system/$DASHBOARD_SVC.service"
    systemctl daemon-reload
    systemctl enable "$DASHBOARD_SVC" >/dev/null 2>&1 || true
    systemctl restart "$DASHBOARD_SVC"
    sleep 1
    systemctl --no-pager --lines=0 status "$DASHBOARD_SVC" || true
else
    log "Skipping dashboard (set DASHBOARD_TOKEN in $ENV_FILE to enable)"
fi

log "Deploy complete."
