# Post-PR119 Baseline: Runtime Contract Inventory

This document locks the known-good baseline restored by PR #119 and inventories
the runtime contracts that later integration PRs (feed status, canonical
snapshot, versioned `/api/live`, dashboard migration, prediction runtime,
unified decision stack) will change deliberately. `tests/test_baseline_contracts.py`
turns each inventory below into a regression test so contract drift is visible
in CI rather than discovered in production.

Nothing in this baseline PR changes runtime behavior.

## Baseline identity

| Fact | Value |
| --- | --- |
| Baseline tree | `4e2fe756788231a41fc5130dcd173c87521c14be` (PR #118 merge) |
| Rollback merge on main | `dc80273d22bcbde800d9bf5b12e381f3302160cb` (PR #119) |
| Backed out | PR #115, PR #116, PR #117 (unified-stack integration sequence) |
| Preserved | PR #118 (V3 Part 1 uncertainty/model changes) |

## Authoritative decision path (V1)

The production path is `shadow_runner.ShadowRunner` driving
`unified_loop.UnifiedOrchestrator.tick()`:

```
CompositeFeed.snapshot()            composite_feed.py — first non-None feed wins,
        |                           last_source records which provider answered
        v
UnifiedOrchestrator.tick()          unified_loop.py
  regime_classifier -> RegimeState
  decision_matrix   -> TradeIntent (structure/direction/size_mult)
  spread_selector.decide() -> TradeDecision (gate, candidate, vetoes)
  V2/V3 shadow paths (signals only, no authority):
    _route_policy, _resolve_prediction_bundle, _run_v2_shadow_ranking,
    part3 shadow decision (result.part3, labeled SHADOW)
  risk gate: RiskManager.check(candidate) — flips TRADE -> NO_TRADE
             with no_trade_reason="risk:..."; record_trade on approval
  journal.log(row)
  _build_paper_intents -> TickResult.paper_intents
        |
        v
TickResult (final_size_mult, vetoes, signals, part3, paper_intents)
        |
        +-> PaperBroker.on_tick()               paper execution (3 tracks)
        +-> serialize_tick_result()             dashboard/state.py
              -> write_live_state(live_state.json)
                    -> GET /api/live serves the file verbatim
```

Authority notes locked by tests:

- V1 is the only authority. V2/V3 outputs live in `TickResult.signals`
  (prefixes `policy_`, `v2_`, `phys_`, `gex_`, `pin_`, `cf_`, `cone_`) and in
  the shadow-labeled `part3` payload.
- The risk gate runs inside `tick()` **before** journaling and paper intents;
  `UnifiedOrchestrator.risk_manager` is a single optional `RiskManager`.
- `final_size_mult` is zeroed unless the (single) decision is `TRADE`.

## API serializer

`dashboard/state.py::serialize_tick_result()` builds a **live.v1** payload
(`schema_version: "live.v1"`) with explicit sections: `snapshot`, `feeds`,
`market`, `legacy`, `forecast`, `v3`, `accounts`, `risk`, `paper`, `system`.
Per-source feed status lives under `feeds` (overall LIVE only when every
required source is fresh). Flat aliases are **removed**
(`system.compat_flat_keys=false`); the dashboard reads sections only (PR D).

`write_live_state()` sanitizes non-finite floats and writes atomically;
`GET /api/live` returns the file content unmodified.

Top-level live.v1 keys (locked by fixture
`tests/fixtures/live_state_baseline.json`):

```
schema_version, generated_at, snapshot, feeds, market, legacy, forecast,
v3, accounts, risk, paper, system
```

`heartbeat_state()` is the no-tick variant; its `system.status` values are
`market_closed | feed_not_ready | feed_error` (plus `live` on real ticks).
Overall feed status is never LIVE on heartbeat.

## Dashboard API endpoints (dashboard/server.py)

```
GET /                      GET /api/health           GET /api/market-status
GET /api/live              GET /api/ticks            GET /api/ticks/{row_id}
GET /api/paper             GET /api/trades           GET /api/ras
GET /api/report            GET /api/gex-variants     GET /api/predictions
GET /api/sigma-cones       GET /api/validation       GET /api/validation/{report_id}
GET /api/learning          GET /api/candidates       GET /api/promotions
GET /api/feature-scores    GET /api/drift            GET /api/readiness
GET /api/stream (SSE)      /static (mount)
```

There is no cross-endpoint consistency contract: `/api/live`,
`/api/market-status`, `/api/ticks`, `/api/paper`, etc. are fetched
independently by the frontend and merged client-side.

## Dashboard refresh functions (dashboard/static/app.js)

One `setInterval(refresh, REFRESH_MS)` polling loop plus per-tab refreshers:

- `refresh()` — main loop; `Promise.all` over 9 endpoints, then
  `requireLiveV1(live)` once, then ~35 `render*` calls (see the inventory
  test for the exact list). Invalid schema → `showLiveUnavailable`.
- `refreshJournal()`, `refreshValidation()`, `refreshLearning()`,
  `refreshPrediction()` — tab-scoped.

PR D closed the baseline weaknesses: renderers read live.v1 sections
(`legacy`, `forecast`, `v3`, `feeds`, `market`, `system`), do not fall back
across versions (no `v2_policy_* || policy_*`), and feed health comes from
`feeds` (not a truthy `feed_source` / `chain_available`).

## Feed status logic

Per-source feed status is serialized under `feeds` (PR C / live.v1):

- Required sources: `spot`, `bars`, `option_chain`, `settlement`
- Each carries `status` ∈ {LIVE, DELAYED, STALE, MISSING, INVALID, FALLBACK},
  provider, ages, and freshness limits (`dashboard/live_schema.py`,
  `prediction/feed_status.py`).
- `feeds.overall_status` is LIVE only when every required source is LIVE.
  A truthy `feed_source` alone is **not** sufficient (age-unknown ⇒ DELAYED).
- Callers may pass explicit `feed_statuses` / `feed_ages_seconds` into
  `serialize_tick_result`; otherwise statuses are synthesized honestly from
  `feed_source` + `chain_available`.

Provider / chain availability for the tick live under `snapshot.feed_source`
and `snapshot.chain_available` (not top-level flat aliases).

## Paper accounts

One `PaperBroker` instance (one `paper.sqlite`) with three virtual cash
ledgers, keyed by `entry_ctx.fill_track`:

```
PAPER_TRACKS = ("legacy", "v2", "v3")   # paper_broker.py
```

- Per-track: cash ledger, open-position cap (`max_open_positions`), daily
  loss/entry counters (`_day_realized`, `_day_entries`).
- Shared: the sqlite db, the broker instance, fill/slippage model, and the
  single `RiskManager` (see below). Unknown tracks fall back to `legacy`.

## Risk-manager instances

Exactly one `RiskManager` for everything:

- Constructed in `ShadowRunner.__init__` (`self._risk`) and passed to
  `UnifiedOrchestrator(risk_manager=...)`.
- Scope: max positions, daily loss, gamma/exposure — all global, not
  per-track. Paper tracks share this risk state.
- `PositionMonitor` (RAS-based per-position exits) is separate and also
  singular.

## Model registry loading path

`prediction/registry.py::ModelRegistry` — file-based (`models/*.joblib` +
`*.json` metadata):

- `load()` fails closed (raises `RegistryError`) on: missing metadata,
  unreadable metadata, unsupported schema version, missing artifact,
  artifact-hash mismatch, feature-version / target / horizon mismatch,
  model feature version newer than live, missing required input fields
  (schema v2), and missing v2 audit fields (calibration/fold/OOF).
- `prediction/deployment.py` + `configs/champion.json` +
  `configs/deployment.json` hold the current deployment pointers; there is no
  unified `DeploymentBundle` yet.

## What the baseline tests lock

`tests/test_baseline_contracts.py` asserts, against committed snapshots:

1. `serialize_tick_result` / `heartbeat_state` key structure (recursive shape
   vs `tests/fixtures/live_state_baseline.json`).
2. `/api/live` serves the serialized file verbatim (no envelope added).
3. The dashboard server route inventory.
4. The `app.js` render/refresh function inventory and the set of `/api/*`
   endpoints it calls; exactly one polling loop.
5. `PAPER_TRACKS` and the single-broker / single-risk-manager wiring.
6. `ModelRegistry.load` fail-closed behavior (missing metadata, tampered
   artifact).

When a later PR intentionally changes one of these contracts, it must update
the corresponding snapshot **in the same PR** — that is the point: contract
changes become explicit, reviewed diffs.
