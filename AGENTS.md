# AGENTS.md

See `HANDOFF.md` for the full system architecture (the two tracks, module
reference, conventions, and known gaps). This file only adds environment /
operating notes.

## Cursor Cloud specific instructions

### What this repo is
A Python 0DTE options decision-support system plus a **read-only FastAPI
observability dashboard** (`dashboard/`). The Vercel bits (`api/[...path].js`,
`public/`, `vercel.json`) are only a thin proxy to a remote VPS — not a local
app. Nothing here places live orders.

### This repository is being retired

**Stop new development here.** 0DTE is the legacy implementation being absorbed
into `DGator86/SPY-DER` and archived. It remains a temporary upstream market
provider during full-stack migration; the final target is an independently
operating SPY-DER system with no runtime dependency on 0DTE.

Read SPY-DER `docs/TARGET_ARCHITECTURE.md` before starting anything, and
`docs/CUTOVER_PLAN.md` for the retirement sequence. Every path in this
repository already has a disposition — move, reimplement, replace, archive or
delete — recorded in SPY-DER `migrations/inventory/zerodte_disposition.json`.

An earlier revision of this file told agents to migrate the repository
incrementally *into* the `zerodte/` package. That is no longer the plan:
`zerodte/**` and `integrations/**` are both marked **delete**, so work invested
there is thrown away at cutover.

For new work:

- **Default to adding it in SPY-DER**, not here. New capability belongs to the
  system that survives.
- Do not add capability to `zerodte/**` or `integrations/**`.
- Do not treat `MarketPacket` as a permanent cross-repository dependency; after
  cutover it is an internal SPY-DER boundary or an external API schema.
- Changes here should be limited to keeping the existing runtime alive and
  supporting parity validation until cutover.
- AI ownership already lives in **SPY-DER**: integrate only through
  `integrations/spy_der/` (HTTP `DecisionClient` to `:8787`, dashboard reader for
  `/var/lib/spy-der/*`). Do not import `spy_der.*` internals or add
  Grok/Dojo/learning modules here.
- `integrations/spy_der/synthetic.py` is **dead**. SPY-DER owns synthetic
  universe production outright (`spy_der.synthetic`) and the Dojo calls it
  natively. Do not extend it.
- If you must touch shared mathematics, change it in SPY-DER and hold 0DTE to
  the parity tolerances in SPY-DER `docs/CUTOVER_PLAN.md`.
- `zerodte.agent.AgentProvider` remains a fail-closed protocol scaffold only.
- Keep hard vetoes, candidate construction, payoff validation, risk, sizing,
  execution, deployment promotion, and rollback deterministic.
- Do not move the live entrypoint or merge large file relocations together with
  behavior changes. Each migration PR must preserve current authority unless
  its scope explicitly says otherwise.

See `docs/ARCHITECTURE_V4.md` for the target layout and migration sequence.

### Dependencies / environment
- Python 3.11+ (repo/CI target 3.11; the VM's 3.12 works fine). Deps are in
  `requirements.txt`; `pytest` is installed separately (as CI does).
- The VM's system Python is PEP-668 "externally managed", so pip needs
  `--break-system-packages`. The startup update script already installs deps.
- Installed console scripts (`pytest`, `uvicorn`, …) land in `~/.local/bin`,
  which is not on `PATH` — invoke them as modules (`python3 -m pytest`,
  `python3 -m dashboard.server`).

### Test / lint / smoke (all offline — synthetic data, no creds, no network)
- Tests: `python3 -m pytest tests/ -q` (offline; ~50s).
- No linter is configured. CI (`.github/workflows/ci.yml`) is just pytest plus
  engine smoke demos: `python3 spy0dte.py`, `python3 mc.py`, `python3 journal.py`
  (each exits 0; `journal.py` is silent by design), and a feed-import check.
- Every core module has a `__main__` demo (see `HANDOFF.md` §7). The full
  unified pipeline: `python3 unified_loop.py`.

### Live feeds require credentials (and are not needed for dev)
`composite_feed.build_default_feed()` and `shadow_runner.py` need real feed
creds (`TRADIER_ACCESS_TOKEN`, `TASTYTRADE_*`, `MASSIVE_API_KEY`). With no
creds `build_default_feed()` **raises by design** — `shadow_runner.py` has no
synthetic mode and cannot run offline. Use the synthetic path below instead.

### Running the dashboard locally (the one runnable "app")
`python3 -m dashboard.server --db <shadow.db> --paper-db <paper.sqlite> --live-state <live_state.json> --host 127.0.0.1 --port 8765`
- All `/api/*` routes require auth: set env `DASHBOARD_TOKEN` and send
  `Authorization: Bearer <token>`. In the browser, load once with
  `http://127.0.0.1:8765/?token=<token>` (the SPA stores it in sessionStorage).
- Missing DB/state files degrade gracefully to `{"note": "... not found"}` —
  they are not errors; the UI just shows empty/waiting panels.

### Getting authentic dashboard data WITHOUT live feeds (non-obvious)
There is no offline flag to populate the journal. Drive
`unified_loop.UnifiedOrchestrator` with `synthetic_world.CoupledSyntheticFeed`
(the *coupled* world — GEX drives price so settlement is measurable), journal to
a `shadow.db`, `orch.settle(session_date)` for each session, and write
`live_state.json` via `dashboard.state.serialize_tick_result(...)`. Then point
the dashboard `--db/--live-state` at those files.

### Vercel frontend (proxy only)
`bash scripts/vercel-build.sh` copies `dashboard/static/*` into `public/`. The
proxy needs `VPS_API_URL` + `DASHBOARD_TOKEN` to reach a live VPS; not required
for local work.
