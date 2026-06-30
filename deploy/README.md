# Deploying the 0DTE shadow pipeline on a Hostinger VPS

This runs `shadow_runner.py` as an always-on **systemd** service: it ticks the
full pipeline every 60s during market hours, journals every evaluation to
SQLite, auto-settles at 4:15 PM ET, and pushes a notification to your phone when
a signal fires. **It never places orders** — you execute the tickets manually.

> Not financial advice. Run in shadow mode (this is shadow mode) for 2–4 weeks
> and read `--report` before risking real money.

Target: a Hostinger VPS running **Ubuntu 22.04/24.04**. All commands are run over
SSH as a sudo-capable user. Paths used:

| Path | Purpose |
|---|---|
| `/opt/zerodte` | the code (git checkout) + virtualenv — read-only at runtime |
| `/etc/zerodte/zerodte.env` | secrets, `chmod 600` — never in the repo |
| `/var/lib/zerodte/shadow.db` | the journal — persists across restarts/updates |

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

## 2. Dedicated service user + directories

A locked-down system user (no login, no home) owns the process — least privilege
for a box that holds an API key.

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin zerodte
sudo mkdir -p /opt/zerodte /etc/zerodte /var/lib/zerodte
sudo chown zerodte:zerodte /var/lib/zerodte
```

## 3. Get the code + virtualenv

```bash
sudo git clone https://github.com/DGator86/0DTE.git /opt/zerodte
cd /opt/zerodte
sudo python3 -m venv venv
sudo ./venv/bin/pip install --upgrade pip
sudo ./venv/bin/pip install -r requirements.txt
```

## 4. Secrets

```bash
sudo install -D -m 600 -o root -g zerodte \
     deploy/zerodte.env.example /etc/zerodte/zerodte.env
sudo nano /etc/zerodte/zerodte.env     # set MASSIVE_API_KEY + NOTIFY_NTFY_TOPIC
```

Quick check that the key works (read-only diagnostic, prints no secrets):

```bash
sudo -u zerodte bash -c 'set -a; . /etc/zerodte/zerodte.env; set +a; \
     /opt/zerodte/venv/bin/python /opt/zerodte/massive_feed.py diagnose SPY'
```

You should see `AUTH OK`. (`0 live quotes` is expected on the Massive plan and
off-hours — see "Real-time quotes" below.)

## 5. Install and start the service

```bash
sudo cp deploy/zerodte-shadow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zerodte-shadow
```

Verify:

```bash
systemctl status zerodte-shadow
journalctl -u zerodte-shadow -f      # live logs; one line per tick
```

During market hours you'll see ticks like
`10:42:00  regime=range_compression  IC  gate=PASS  decision=TRADE  x1.00`.
Off-hours it logs the next open and sleeps.

## 6. Phone notifications

Install the **ntfy** app (iOS/Android), and subscribe to the exact string you
set as `NOTIFY_NTFY_TOPIC`. Every `TRADE` signal arrives as a push with the
structure, strikes, credit, and size — that's your cue to place the trade.

## 7. Observability dashboard (read-only)

A mobile-friendly web UI shows what the pipeline is doing, what market data it
is evaluating, and why each decision was made. **It cannot place trades or
change configuration.**

1. Add a long random `DASHBOARD_TOKEN` to `/etc/zerodte/zerodte.env`
2. Redeploy (or `sudo systemctl enable --now zerodte-dashboard`)
3. Expose locally-bound port 8765 via **Cloudflare Tunnel** or Tailscale:

```bash
# Example: Cloudflare Tunnel (install cloudflared first)
cloudflared tunnel --url http://127.0.0.1:8765
```

Open the tunnel URL on your phone or PC. On first visit, paste your
`DASHBOARD_TOKEN` (or bookmark `https://<tunnel>/?token=<token>` once).

The banner shows **Market Open** / **Market is Closed** with a live countdown
to the next open or close, using the NYSE calendar (holidays and early closes).

```bash
sudo systemctl status zerodte-dashboard
sudo journalctl -u zerodte-dashboard -f
```

---

## Operating it

```bash
# Calibration report (gate effectiveness + score correlations)
sudo -u zerodte /opt/zerodte/venv/bin/python /opt/zerodte/shadow_runner.py \
     --report --db /var/lib/zerodte/shadow.db

# Backfill a session that missed auto-settlement
sudo -u zerodte /opt/zerodte/venv/bin/python /opt/zerodte/shadow_runner.py \
     --settle 2026-06-29 --db /var/lib/zerodte/shadow.db

# Restart / stop
sudo systemctl restart zerodte-shadow
sudo systemctl stop zerodte-shadow
```

## Updating the code

Manually, on the box:

```bash
cd /opt/zerodte && sudo git pull
sudo ./venv/bin/pip install -r requirements.txt   # if deps changed
sudo systemctl restart zerodte-shadow
```

…or let CI do it for you — see **Push-to-deploy** below.

The journal in `/var/lib/zerodte` is untouched by updates. Avoid needless
mid-session restarts: the GEX-percentile window and VIX cache are in-memory and
re-warm after a restart (the journal itself persists).

---

## Push-to-deploy (GitHub Actions)

`.github/workflows/deploy.yml` SSHes into the VPS on every push to `main` and
runs [`deploy/remote-deploy.sh`](remote-deploy.sh), which fast-forwards the
checkout under `/opt/zerodte` and restarts the service. The deploy script is
**idempotent** — the first run also provisions the service user, venv, and
systemd unit, so it covers steps 1–3 and 5 above automatically. The only manual
step that remains is the **secrets file** (step 4): the script never writes it
and refuses to start the service until it exists, so a deploy can never clobber
your API key.

### One-time setup

**1. Generate a dedicated deploy key** (on your laptop — no passphrase, so CI can
use it unattended):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/zerodte_deploy -C "github-actions-deploy" -N ""
```

**2. Authorize the public key on the VPS:**

```bash
ssh-copy-id -i ~/.ssh/zerodte_deploy.pub root@YOUR_VPS_IP
# or paste ~/.ssh/zerodte_deploy.pub into /root/.ssh/authorized_keys by hand
```

**3. Pin the host key** (so CI verifies the box instead of trust-on-first-use):

```bash
ssh-keyscan YOUR_VPS_IP
```

**4. Add repository secrets** — GitHub → repo **Settings → Secrets and variables
→ Actions → New repository secret:**

| Secret | Value | Required |
|---|---|---|
| `VPS_HOST` | the VPS IP / hostname | ✅ |
| `VPS_SSH_KEY` | contents of `~/.ssh/zerodte_deploy` (the **private** key) | ✅ |
| `VPS_USER` | SSH user, if not `root` | optional |
| `VPS_SSH_PORT` | SSH port, if not `22` | optional |
| `VPS_KNOWN_HOSTS` | the `ssh-keyscan` output from step 3 | recommended |

**5. Seed the secrets file once** (the only thing CI won't do — see step 4 of the
manual guide above), then trigger the first deploy by pushing to `main` or via
**Actions → Deploy to VPS → Run workflow**. The first run provisions the box;
every push after that is a one-click redeploy.

> Least privilege: this is a deploy-only key for one box. To lock it down further
> you can prefix the `authorized_keys` entry with
> `command="...",no-port-forwarding`, or run the service as a non-root sudoer and
> have the workflow `sudo` — but root-over-SSH matches Hostinger's default and
> the rest of this guide.

## Real-time quotes (going to "maximal status")

The Massive plan returns no real-time option NBBO, so Track A prices on
day-close marks — fine for shadow journaling, not for executable credits. When
your **Tradier** token is ready:

1. Add `TRADIER_ACCESS_TOKEN` (+ `TRADIER_BASE_URL`) to `/etc/zerodte/zerodte.env`.
2. In `shadow_runner.py`, swap the feed import/instantiation
   (`MassiveDataFeed` → `TradierDataFeed`).
3. `sudo systemctl restart zerodte-shadow`.

Confirm first with:
`sudo -u zerodte bash -c 'set -a; . /etc/zerodte/zerodte.env; set +a; /opt/zerodte/venv/bin/python /opt/zerodte/tradier_feed.py'`

## Notes

- **Outbound-only.** The process opens no listening ports; your VPS firewall can
  deny all inbound except SSH.
- **One instance per DB.** Don't run a second copy against the same `shadow.db`.
- **Timezone-independent.** All trading logic uses ET internally regardless of
  the VPS clock; only log timestamps follow system local time.
