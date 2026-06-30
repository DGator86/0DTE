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

## Notes

- **Read-only** — same GET-only API as the VPS dashboard; no trades or config changes
- **Polling** — UI refreshes every 15s; SSE is not used through Vercel
- **Direct VPS access** — still works via Cloudflare Tunnel + token in browser
- **Security** — use a named Cloudflare Tunnel for production, not ephemeral
  `trycloudflare.com` URLs
