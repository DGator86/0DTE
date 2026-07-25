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
# SPY-DER checkout (separate venv + spy-der-agent.service own AI).
# 0DTE talks to SPY-DER only via HTTP :8787 and /var/lib/spy-der/* files.
SPY_DER_REPO_URL="${SPY_DER_REPO_URL:-https://github.com/DGator86/SPY-DER.git}"
SPY_DER_DIR="${SPY_DER_DIR:-/opt/spy-der}"
SPY_DER_REF="${SPY_DER_REF:-origin/main}"
SPY_DER_ENABLED="${SPY_DER_ENABLED:-1}"

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

# SPY-DER now ships itself: it has its own poller (spy-der-update.timer), its
# own remote-deploy.sh, its own venv and its own units. When that is installed,
# this deploy must keep its hands off /opt/spy-der — two deploys resetting the
# same checkout on independent 2-minute timers is a race with no upside.
if [ -f /etc/systemd/system/spy-der-update.timer ]; then
    log "SPY-DER self-deploys (spy-der-update.timer present) — not touching $SPY_DER_DIR"
elif [ "$SPY_DER_ENABLED" = "1" ]; then
    log "SPY-DER checkout -> $SPY_DER_REF (bootstrap only; no pip install here)"
    if [ ! -d "$SPY_DER_DIR/.git" ]; then
        git clone "$SPY_DER_REPO_URL" "$SPY_DER_DIR"
    fi
    git -C "$SPY_DER_DIR" remote set-url origin "$SPY_DER_REPO_URL"
    git -C "$SPY_DER_DIR" fetch --prune origin
    # Same ref resolution as 0DTE: branch name -> origin/<branch>, else SHA.
    if git -C "$SPY_DER_DIR" rev-parse --verify -q "origin/${SPY_DER_REF#origin/}^{commit}" >/dev/null; then
        SPY_TARGET="origin/${SPY_DER_REF#origin/}"
    else
        SPY_TARGET="$SPY_DER_REF"
    fi
    git -C "$SPY_DER_DIR" reset --hard "$SPY_TARGET"
    echo "SPY-DER commit: $(git -C "$SPY_DER_DIR" rev-parse --short HEAD)"
    echo "Run $SPY_DER_DIR/deploy/remote-deploy.sh once to hand SPY-DER its own deploy."
else
    log "SPY-DER checkout disabled (SPY_DER_ENABLED=$SPY_DER_ENABLED)"
fi

# Experience hand-off lives under this project's data dir and stays ours.
mkdir -p "$DATA_DIR/spyder_experience/snapshots" "$DATA_DIR/spyder_experience/outcomes"
chown -R "$SVC_USER:$SVC_USER" "$DATA_DIR/spyder_experience" 2>/dev/null || true

# /var/lib/spy-der is deliberately NOT chowned here. Those units declare
# StateDirectory=spy-der, so systemd resets the tree to the spy-der user every
# time one starts — a chown to this project's user does not win, it just flaps,
# and the two fight on independent schedules. Read access comes from the modes
# SPY-DER publishes (0644 files, 0755 directories), not from ownership. If the
# dashboard cannot read a report, fix the mode, never the owner.
mkdir -p /var/lib/spy-der/reports/dojo

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

log "Legacy configs/ticks dirs (champion now preferred from /var/lib/spy-der)"
# Durable configs under the data dir survive /opt resets. Learning/Dojo timers
# are DEPRECATED — install unit files for cutover reference but do not enable.
# Operators should enable spy-der-dojo-* + SPY-DER learning from SPY-DER deploy.
mkdir -p "$DATA_DIR/configs/candidates" "$DATA_DIR/configs/promoted" \
         "$DATA_DIR/configs/archive" "$DATA_DIR/reports/promotion" \
         "$DATA_DIR/ticks"
if [ ! -f "$DATA_DIR/configs/champion.json" ] \
   && [ -f "$APP_DIR/configs/champion.json" ]; then
    cp -a "$APP_DIR/configs/champion.json" "$DATA_DIR/configs/champion.json"
fi
chown -R "$SVC_USER:$SVC_USER" "$DATA_DIR/configs" "$DATA_DIR/reports" "$DATA_DIR/ticks"
for unit in zerodte-learn-evening zerodte-learn-weekly \
            zerodte-dojo-daily zerodte-dojo-recent zerodte-dojo-weekly; do
    if [ -f "$APP_DIR/deploy/$unit.service" ]; then
        install -m 644 "$APP_DIR/deploy/$unit.service" "/etc/systemd/system/$unit.service"
    fi
    if [ -f "$APP_DIR/deploy/$unit.timer" ]; then
        install -m 644 "$APP_DIR/deploy/$unit.timer" "/etc/systemd/system/$unit.timer"
    fi
    # Do not enable deprecated AI timers; stop them if previously enabled.
    systemctl disable --now "$unit.timer" >/dev/null 2>&1 || true
done
systemctl daemon-reload
log "Deprecated zerodte-learn-* / zerodte-dojo-* timers left disabled (use spy-der-*)"

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
