# Vercel dashboard (frontend) + VPS API (data)

The observability UI can be hosted on **Vercel** while data stays on your **VPS**
(`shadow.db`, `live_state.json`, `paper.sqlite`). Vercel serverless functions
proxy read-only `/api/*` requests to the VPS dashboard API. Your `DASHBOARD_TOKEN`
lives only in Vercel env vars — visitors never see it.

```
Phone/PC  →  your-app.vercel.app  →  Vercel /api proxy  →  VPS tunnel  →  dashboard :8765
```

## 1. VPS: expose the read-only API

On the VPS, the dashboard service must already be running (see
[`deploy/README.md`](README.md) §7).

Bind stays on `127.0.0.1:8765`. Expose it with a **stable HTTPS URL** using
Cloudflare Tunnel (recommended):

```bash
# One-time: install cloudflared, then run a named tunnel or quick tunnel:
cloudflared tunnel --url http://127.0.0.1:8765
```

Copy the HTTPS URL (e.g. `https://zerodte-api.example.com` or a `trycloudflare.com`
URL for testing). This becomes `VPS_API_URL`.

Ensure `/etc/zerodte/zerodte.env` has:

```bash
DASHBOARD_TOKEN=your-long-random-string   # same value used on Vercel
```

```bash
sudo systemctl enable --now zerodte-dashboard
curl -s -H "Authorization: Bearer $DASHBOARD_TOKEN" http://127.0.0.1:8765/api/health
```

## 2. Vercel: deploy the UI + proxy

### Option A — Vercel CLI

```bash
npm i -g vercel
vercel login
vercel link
vercel env add VPS_API_URL      # https://your-tunnel-host (no trailing slash)
vercel env add DASHBOARD_TOKEN  # same token as VPS
vercel --prod
```

### Option B — GitHub integration

1. Import the repo in [vercel.com/new](https://vercel.com/new)
2. Framework preset: **Other** (uses `vercel.json` build script)
3. Add environment variables (Production + Preview):

| Variable | Value |
|----------|--------|
| `VPS_API_URL` | `https://your-tunnel-host` (HTTPS URL to VPS dashboard) |
| `DASHBOARD_TOKEN` | Same secret as `/etc/zerodte/zerodte.env` |

4. Deploy

Open `https://your-project.vercel.app` — no token prompt when the proxy is
configured correctly.

## 3. Verify

- **Vercel UI** loads with market banner and tabs
- **Vercel** `/api/health` returns `{"ok":true,...}`
- During market hours, **Now** tab updates from VPS `live_state.json`

If you see `502 VPS dashboard unreachable`, check the tunnel and `VPS_API_URL`.

## 4. Dojo tab — SPY-DER reports on the dashboard

The Dojo tab renders SPY-DER's report. Nothing is copied or duplicated: the VPS
dashboard reads SPY-DER's published state file directly.

```
Vercel /api/dojo  →  VPS dashboard :8765 /api/dojo
                     →  /var/lib/spy-der/reports/dojo/latest.json
```

The `spy-der-dojo-*` timers write that file; `integrations/spy_der/dashboard_reader`
reads it. The dashboard never imports SPY-DER's Dojo, learning or agent modules —
only this file and `live_state.json`, per `docs/OWNERSHIP_BOUNDARY.md`.

### The permission requirement

This is the step that bites. SPY-DER writes the report as the **`spy-der`** user;
the dashboard reads it as **`zerodte`**. The report must therefore be readable by
others, and every parent directory must be traversable:

```bash
sudo chmod 0644 /var/lib/spy-der/reports/dojo/*.json
sudo chmod 0755 /var/lib/spy-der /var/lib/spy-der/reports /var/lib/spy-der/reports/dojo
```

SPY-DER now publishes these at `0644` (minus the operator umask), so new runs are
readable automatically. Files written before that change keep `0600` until the
next run overwrites them — hence the one-time `chmod` above.

### Verify, VPS first

```bash
# 1. the report exists and is world-readable
ls -l /var/lib/spy-der/reports/dojo/latest.json

# 2. the dashboard user can actually open it
sudo -u zerodte cat /var/lib/spy-der/reports/dojo/latest.json | head -c 200

# 3. the API serves it
curl -s -H "Authorization: Bearer $DASHBOARD_TOKEN" \
  http://127.0.0.1:8765/api/dojo | jq '.reports[0].summary, .note'

# 4. and through Vercel
curl -s https://your-project.vercel.app/api/dojo | jq '.reports | length'
```

Step 3 returning `{"reports": [], "note": ...}` means the note tells you the
cause — it distinguishes *not found* (no Dojo run yet) from *permission denied*
(the case above) from *unreadable*. The Dojo tab shows that same note rather
than assuming the timers are off.

### If the tab is empty

| Note | Cause | Fix |
|---|---|---|
| `not found` | no Dojo run has completed | `systemctl list-timers 'spy-der-dojo-*'`; run it manually |
| `permission denied` | mode/ownership | the `chmod` above |
| `unreadable (JSONDecodeError)` | truncated write | re-run the Dojo; the writer is atomic, so this implies disk trouble |
| tab shows `Unauthorized` | token mismatch | `DASHBOARD_TOKEN` must match on VPS and Vercel |

`zerodte-dashboard.service` passes `--spy-der-dojo-latest` and
`--spy-der-live-state` explicitly and declares `ReadOnlyPaths=/var/lib/spy-der`,
so the dependency survives a sandbox tightening. After editing the unit:

```bash
sudo systemctl daemon-reload && sudo systemctl restart zerodte-dashboard
```

## Notes

- **Read-only** — same GET-only API as the VPS dashboard; no trades or config changes
- **Polling** — UI refreshes every 15s; SSE is not used through Vercel
- **Direct VPS access** — still works via Cloudflare Tunnel + token in browser
- **Security** — use a named Cloudflare Tunnel for production, not ephemeral
  `trycloudflare.com` URLs
