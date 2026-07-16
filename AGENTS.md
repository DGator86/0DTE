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

### Canonical architecture migration

The repository is being migrated incrementally into the `zerodte/` package.
The existing top-level modules and `shadow_runner.py` remain the production
baseline until a later, separately reviewed promotion PR.

For new work:

- Put cross-stage data contracts in `zerodte/contracts/`.
- Put orchestration interfaces in `zerodte/runtime/`; orchestration must not
  contain pricing, forecasting, risk, or execution mathematics.
- Use `zerodte/adapters/` to bridge legacy objects. Do not add new downstream
  imports of `unified_loop.TickResult` or provider SDK response types.
- AI providers belong behind `zerodte.agent.AgentProvider`. They may select only
  canonical candidate IDs and may reduce, but never raise, deterministic size.
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
- Tests: `python3 -m pytest tests/ -q` (234 tests, ~50s).
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
