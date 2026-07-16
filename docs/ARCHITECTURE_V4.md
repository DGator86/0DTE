# 0DTE Architecture V4 — Canonical Runtime Migration

Status: incremental migration scaffold  
Production authority: unchanged (`shadow_runner.py` → `unified_loop.py`)  
Live execution: not authorized  
AI authority in this PR: none

## Objective

Reorganize the repository around versioned contracts and independently testable
processing stages without repeating the failed all-at-once V1/V2/V3 integration.
The current production path remains intact while legacy modules are wrapped and
migrated one boundary at a time.

The target decision flow is:

```text
provider feeds
    -> canonical market snapshot
    -> features + structural state
    -> calibrated forecasts
    -> deterministic candidate universe
    -> executable candidate economics
    -> baseline policy comparisons
    -> canonical AgentDecisionPacket
    -> constrained AI decision agent
    -> deterministic risk/sizing validation
    -> paper or advisory execution
    -> append-only audit events
```

The AI agent is the final statistical choice layer, not the source of market
features, option geometry, payoff mathematics, risk limits, or order authority.

## Non-negotiable safety rules

1. Every permitted structure is options-only and requires no stock ownership.
2. Maximum loss must be deterministically finite before a candidate is exposed
   to any policy or AI agent.
3. The agent may select only an existing canonical `candidate_id`.
4. The agent may reduce deterministic size but may never increase it.
5. Operational hard stops and portfolio risk remain deterministic.
6. Missing data, malformed model output, provider failure, schema mismatch, or
   an unknown candidate fails closed to `ABSTAIN`.
7. Research, shadow, advisory, candidate, champion, promotion, and rollback
   remain governed by the deployment bundle and human review.
8. This migration does not change the production authority path implicitly.

## New package boundary

```text
zerodte/
├── contracts/
│   ├── market.py       canonical snapshot, feed provenance, data quality
│   ├── candidates.py   option legs, bounded-loss economics, candidate IDs
│   ├── decisions.py    policy views, agent packet, structured agent output
│   └── risk.py         operational, portfolio, and sizing envelope
├── runtime/
│   ├── services.py     narrow stage protocols
│   └── pipeline.py     orchestration only
├── agent/
│   ├── contracts.py    provider-neutral model interface
│   └── runtime.py      validation and fail-closed behavior
└── adapters/
    └── legacy_snapshot.py
```

The package is deliberately dependency-light. Provider SDK objects and live
runner classes must not leak into the contracts.

## Ownership by stage

### Ingestion

Responsible for provider failover, timestamps, freshness, and conversion into
`CanonicalMarketSnapshot`. No prediction or trading decision belongs here.

### Features and structural state

Responsible for RND moments, GEX variants, MTF features, volatility channels,
pin state, market internals, and robust normalization. It emits evidence; it
does not choose a trade.

### Forecasting

Responsible for calibrated direction, return quantiles, realized-move, range
survival, barrier-touch, physical-density, uncertainty, and drift outputs.
Forecasts must be generated independently of the selected structure.

### Candidate service

Responsible for all legal option geometry and payoff calculations. Candidate
families may include long calls/puts, debit spreads, bull put credit spreads,
bear call credit spreads, iron condors, iron butterflies, and broken-wing
butterflies when their maximum loss is deterministically bounded.

The service assigns stable IDs and calculates the same economics for Legacy,
V2, V3, and the future agent. No policy receives a private candidate universe.

### Policies

Legacy, V2, and V3 become policy implementations over common inputs. Their
outputs are compact `PolicyDecisionView` records included in the agent packet so
model disagreement is explicit rather than hidden in unrelated signal keys.

### AI agent

The agent consumes processed outputs and a bounded candidate whitelist. Its
valid entry response is equivalent to:

```json
{
  "action": "SELECT_CANDIDATE",
  "candidate_id": "cand_...",
  "size_scalar": 0.55,
  "confidence": 0.71,
  "uncertainty": 0.29,
  "supporting_evidence_ids": ["forecast:p_range_30m"],
  "contradictory_evidence_ids": ["policy:v3:left_tail"],
  "rationale": "..."
}
```

It cannot send arbitrary legs or strikes. Provider integrations such as xAI,
OpenAI, or local models implement the same `AgentProvider` protocol.

### Risk and execution

Risk revalidates the selected candidate against current data immediately before
paper/advisory execution. It owns maximum dollars at risk, position count,
daily loss, exposure, entry lockouts, and emergency exits. Agent confidence is
never a risk measurement.

## Compatibility strategy

The top-level repository is not moved wholesale. During migration:

- `unified_loop.TickSnapshot` is converted through
  `zerodte.adapters.canonical_snapshot_from_tick`.
- Legacy modules remain callable and continue to supply calculations.
- New code consumes canonical contracts rather than adding more imports of
  `TickResult`, dashboard JSON, or provider responses.
- Each migrated service gets parity tests against the current implementation.
- Only after parity and replay validation does a later PR switch the live
  entrypoint.

## Pull-request sequence

### PR 1 — Contracts and orchestration scaffold (this change)

- Add canonical package boundary.
- Add immutable snapshot, candidate, decision, and risk contracts.
- Add service protocols and staged pipeline.
- Add provider-neutral, fail-closed agent runtime.
- Add legacy snapshot adapter and safety regression tests.
- Do not change live runtime behavior.

### PR 2 — Canonical snapshot assembler

- Wrap `CompositeFeed` and produce canonical feed provenance directly.
- Remove downstream provider guessing.
- Add snapshot parity fixtures and freshness tests.

### PR 3 — Feature and structural-state service

- Extract RND, GEX, MTF, pin, volatility-channel, and dynamics work from
  `UnifiedOrchestrator.tick()`.
- Preserve exact current outputs through compatibility adapters.

### PR 4 — PredictionRuntime

- Load the validated `DeploymentBundle` and registered model group.
- Fail closed on artifact or schema mismatch.
- Produce one canonical forecast bundle per snapshot.

### PR 5 — Shared candidate universe

- Move candidate generation and V3 execution economics behind
  `CandidateService`.
- Assign stable candidate IDs.
- Eliminate placeholder candidate metrics and policy-specific fill assumptions.

### PR 6 — Policy adapters

- Convert Legacy, V2, and V3 outputs into `PolicyDecisionView`.
- Compare policies over identical candidates and risk assumptions.

### PR 7 — Agent shadow runtime

- Refactor useful audit/cost/cadence concepts from draft PR #125.
- Replace arbitrary leg construction with candidate-ID selection.
- Run as an isolated shadow policy with no submission authority.

### PR 8 — Risk/execution boundary and event journal

- Revalidate decisions immediately before paper execution.
- Emit versioned events for snapshot, forecast, candidate, policy, agent, risk,
  fill, mark, close, and settlement.
- Derive dashboard state from one consistent event snapshot.

### PR 9 — Controlled runtime cutover

- Run old and new orchestrators in replay and shadow parity.
- Require an explicit review packet and rollback target.
- Switch the live entrypoint only in a dedicated PR.

## Definition of done

The reorganization is complete when:

- `UnifiedOrchestrator.tick()` is orchestration-only or retired.
- Every policy and the agent sees identical snapshot, forecast, candidate,
  execution, and risk assumptions.
- No candidate can be created outside deterministic option geometry.
- No agent output can bypass hard vetoes or raise size.
- Every decision is reproducible from a versioned packet and deployment hash.
- Dashboard data is generated from a consistent event state.
- Session-grouped replay and shadow comparisons show no unexplained behavior
  drift before production cutover.
