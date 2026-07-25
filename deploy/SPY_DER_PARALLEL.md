# SPY-DER parallel track on the VPS

SPY-DER appears beside Legacy / V2 / V3 in the dashboard **Parallel decisions**
panel and as a fourth independent paper ledger (`spy_der`).

Phase 5 ownership: **SPY-DER owns all AI**. 0DTE publishes `MarketPacket` /
`OutcomePacket` and calls `POST http://127.0.0.1:8787/v1/decision`. See
[`docs/migrations/PHASE5_AI_OWNERSHIP_REMOVAL.md`](../docs/migrations/PHASE5_AI_OWNERSHIP_REMOVAL.md).

## Runtime wiring

1. 0DTE shadow loop builds a `MarketPacket` and writes it under
   `/var/lib/zerodte/spyder_experience/snapshots/` (zerodte-owned outbox).
2. `zerodte-sync-experience.timer` copies that outbox into
   `/var/lib/spy-der/inbox/experience` (spy-der-owned Dojo inbox) every 15
   minutes; deploy also runs a one-shot sync.
3. 0DTE `DecisionClient` POSTs that packet to `SPY_DER_DECISION_URL`
   (default `http://127.0.0.1:8787/v1/decision`).
4. SPY-DER `spy-der-agent.service` answers with `spyder.dashboard.v1`.
5. Dashboard reads `/var/lib/spy-der/live_state.json` and
   `/var/lib/spy-der/reports/dojo/latest.json` only.

Manual one-shot (root):

```bash
# If the outbox is empty but shadow.db has history, backfill first:
sudo -u zerodte /opt/zerodte/venv/bin/python \
  /opt/zerodte/scripts/backfill_experience_from_journal.py \
  --db /var/lib/zerodte/shadow.db \
  --root /var/lib/zerodte/spyder_experience
sudo bash /opt/zerodte/deploy/ops/sync-experience-to-spyder.sh
sudo systemctl start spy-der-dojo-recent.service
```

AI keys (`XAI_API_KEY`, model routing) belong in the **SPY-DER** environment,
not `/etc/zerodte/zerodte.env`.

## Deploy notes

`remote-deploy.sh` still fast-forwards `/opt/spy-der` when
`SPY_DER_ENABLED=1`, but **does not** `pip install` SPY-DER into the 0DTE
venv (in-process coupling removed). SPY-DER must run with its own venv and
`spy-der-agent.service`.

Enable SPY-DER Dojo timers from the SPY-DER repo:

- `spy-der-dojo-daily.timer`
- `spy-der-dojo-recent.timer`
- `spy-der-dojo-weekly.timer`

0DTE `zerodte-dojo-*` / `zerodte-learn-*` unit files remain in-tree as
deprecated cutover references and are **not** enabled by deploy.

Disable the SPY-DER checkout step with `SPY_DER_ENABLED=0` if needed.

## What you should see

1. **Parallel decisions** panel — four cards: Legacy, V2, V3, SPY-DER.
2. **Paper** metrics — `SPY-DER P&L` beside the other tracks.
3. Open positions tagged `fill_track=spy_der`.
4. Dojo / Learning tabs populated from `/var/lib/spy-der/*` (empty/`note` until
   SPY-DER is running).

Live broker routing remains disabled. This is paper/shadow comparison only.

## Rollback

```bash
# Temporarily skip SPY-DER checkout during 0DTE deploy:
sudo SPY_DER_ENABLED=0 bash /opt/zerodte/deploy/remote-deploy.sh
# Shadow loop will fail closed to UNAVAILABLE for the spy_der track.
```
