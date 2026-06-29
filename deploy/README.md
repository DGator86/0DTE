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

```bash
cd /opt/zerodte && sudo git pull
sudo ./venv/bin/pip install -r requirements.txt   # if deps changed
sudo systemctl restart zerodte-shadow
```

The journal in `/var/lib/zerodte` is untouched by updates. Avoid needless
mid-session restarts: the GEX-percentile window and VIX cache are in-memory and
re-warm after a restart (the journal itself persists).

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
