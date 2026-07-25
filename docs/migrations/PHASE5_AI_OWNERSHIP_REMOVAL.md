# Phase 5 — 0DTE AI Ownership Removal & SPY-DER Integration

This PR finishes the 0DTE side of the ownership boundary established by
SPY-DER PR #41 / #43. **SPY-DER is not modified here.**

End state:

- **0DTE** = market engine, packet publisher, thin decision client, dashboard
- **SPY-DER** = AI brain, Dojo, learning, promotion, model routing

## Removed from 0DTE (manifest)

| Removed component | Former path | SPY-DER replacement |
|---|---|---|
| Dojo runner | `dojo.py` | `spy_der.dojo.runner` + `spy-der-dojo-*` units |
| Sequential Dojo | `sequential_dojo.py` | `spy_der.dojo.sequential` |
| Adaptive Learning Engine | `adaptive_learning/*` | `spy_der.learning/*` under `/var/lib/spy-der/configs/` |
| In-process decision bridge | `spy_der_bridge.py` | `POST http://127.0.0.1:8787/v1/decision` |
| In-process prediction bridge | `spy_der_predict.py` | SPY-DER decision / dashboard packet |
| Dojo docs | `docs/dojo.md`, `docs/sequential_dojo.md` | SPY-DER `docs/ops/dojo.md` |
| AI-only tests | `tests/test_dojo.py`, `test_sequential_dojo.py`, `test_adaptive_learning.py`, `test_spy_der_predict.py` | SPY-DER unit tests |

Deploy unit **files kept** (deprecated, not auto-deleted):

- `deploy/zerodte-dojo-{daily,recent,weekly}.{service,timer}`
- `deploy/zerodte-learn-{evening,weekly}.{service,timer}`

## Added on 0DTE

| Module | Role |
|---|---|
| `integrations/spy_der/contracts.py` | Schema constants + packet builders (`zerodte.spyder.*` / `spyder.*`) |
| `integrations/spy_der/market_publisher.py` | Publish `MarketPacket` → experience `snapshots/` |
| `integrations/spy_der/outcome_publisher.py` | Publish `OutcomePacket` → experience `outcomes/` |
| `integrations/spy_der/decision_client.py` | HTTP client with retry + deterministic fallback |
| `integrations/spy_der/dashboard_reader.py` | Read `/var/lib/spy-der/live_state.json` + dojo `latest.json` |
| `integrations/spy_der/experience.py` | `MarketExperienceProvider` over published packets |
| `integrations/spy_der/synthetic.py` | `SyntheticUniverseProvider` wrapping matrix/coupled feeds |
| `integrations/spy_der/champion_reader.py` | Mechanical champion apply (no promote/learn) |

## Contracts (authoritative — do not redesign)

- `zerodte.spyder.market.v1` — MarketPacket
- `zerodte.spyder.outcome.v1` — OutcomePacket
- `spyder.decision.request.v1` / `spyder.decision.response.v1` — HTTP wrapper
- `spyder.dashboard.v1` — dashboard display packet

Experience filesystem layout (SPY-DER `FileMarketExperienceProvider`):

```
$ZERODTE_SPYDER_EXPERIENCE_ROOT/   # default /var/lib/zerodte/spyder_experience
  sessions.json
  snapshots/{snapshot_id}.json
  outcomes/{snapshot_id}.json
```

Decision URL: `SPY_DER_DECISION_URL` (default `http://127.0.0.1:8787/v1/decision`).

Champion search order: `/var/lib/spy-der/configs/champion.json` →
`/var/lib/zerodte/configs/champion.json` → `configs/champion.json`.

## VPS cutover checklist (ops-only — not performed by this PR)

1. Ensure SPY-DER is deployed with its **own** venv under `/opt/spy-der`.
2. Start `spy-der-agent.service` listening on `127.0.0.1:8787`.
3. Enable SPY-DER Dojo timers (from SPY-DER `deploy/`):
   - `spy-der-dojo-daily.timer`
   - `spy-der-dojo-recent.timer`
   - `spy-der-dojo-weekly.timer`
4. Disable (do not delete yet) superseded 0DTE timers:
   - `zerodte-dojo-daily.timer`
   - `zerodte-dojo-recent.timer`
   - `zerodte-dojo-weekly.timer`
   - `zerodte-learn-evening.timer`
   - `zerodte-learn-weekly.timer`
5. Move AI secrets (`XAI_API_KEY`, model routing) into the SPY-DER env — not
   `/etc/zerodte/zerodte.env`.
6. Confirm dashboard Dojo/Learning tabs read `/var/lib/spy-der/*`.
7. Confirm shadow loop publishes experience packets and receives HTTP decisions.

## Backward compatibility until cutover

- Decision client fails closed (`UNAVAILABLE` / ABSTAIN) when `:8787` is down.
- Champion loader still accepts legacy `/var/lib/zerodte/configs/champion.json`.
- Deprecated systemd unit files remain in-tree for operators.
- Dashboard degrades to empty/`note` when SPY-DER state files are missing.
