# SPY-DER parallel track on the VPS

SPY-DER appears beside Legacy / V2 / V3 in the dashboard **Parallel decisions**
panel and as a fourth independent paper ledger (`spy_der`).

## Auto-deploy (preferred)

The VPS `zerodte-update.timer` already pulls `0DTE` `main` every ~2 minutes
and runs `remote-deploy.sh`. That script now also:

1. Fast-forwards `/opt/spy-der` to `SPY-DER` `main`
2. `pip install -e /opt/spy-der` into `/opt/zerodte/venv`
3. Restarts shadow + dashboard

So once this branch is **merged to 0DTE `main`**, the scanner picks up the
parallel panel with no manual VPS SSH.

Optional Grok key in `/etc/zerodte/zerodte.env`:

```bash
XAI_API_KEY=...
```

Without it, SPY-DER uses the deterministic agent so the panel still updates.

Disable with `SPY_DER_ENABLED=0` in the deploy environment if needed.

## What you should see

1. **Parallel decisions** panel — four cards: Legacy, V2, V3, SPY-DER.
2. **Paper** metrics — `SPY-DER P&L` beside the other tracks.
3. Open positions tagged `fill_track=spy_der`.

Live broker routing remains disabled. This is paper/shadow comparison only.

## Rollback

```bash
# Roll 0DTE main back before the parallel-track commit (self-update will follow),
# or temporarily:
sudo SPY_DER_ENABLED=0 bash /opt/zerodte/deploy/remote-deploy.sh
```
