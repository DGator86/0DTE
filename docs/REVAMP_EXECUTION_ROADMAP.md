# 0DTE Revamp Execution Roadmap

Status: active execution plan  
Architecture: V4 canonical runtime  
Current production authority: `shadow_runner.py` → `UnifiedOrchestrator`  
Target production authority: versioned canonical runtime with constrained AI meta-policy  
Live broker execution: out of scope until separately authorized

## Mission

Transform the repository from a large, tightly coupled decision loop into a
versioned quantitative decision platform in which:

- market data becomes one immutable, provenance-aware snapshot;
- feature engineering, forecasting, candidate generation, policy, risk, and
  execution are independent services;
- Legacy, V2, V3, and the AI agent evaluate the same candidates using the same
  economics and risk assumptions;
- the AI agent is the final constrained decision layer over processed outputs;
- deterministic code remains authoritative for option geometry, maximum loss,
  sizing, operational vetoes, execution validation, deployment, and rollback.

## Current state

Architecture V4 PR #126 established:

- `zerodte/contracts/`
- `zerodte/runtime/`
- `zerodte/agent/`
- `zerodte/adapters/`
- canonical candidate IDs and decision packets
- fail-closed agent validation
- initial safety tests

The live path is still the existing orchestrator. Draft PR #125 contains useful
AI audit and operational ideas, but its arbitrary trade-construction model must
not become the final architecture.

## Execution principles

1. One behavioral concern per pull request.
2. Every extraction receives parity tests against the current implementation.
3. New code enters through canonical contracts; no new downstream dependency on
   `TickResult`, dashboard JSON, or provider-specific SDK objects.
4. Shadow first, then paper, then advisory. No implicit promotion.
5. The existing live path remains rollback-capable until the final cutover.
6. All trade candidates are options-only, require no stock ownership, and have
   deterministically bounded maximum loss.
7. AI failure, bad schema, stale inputs, missing artifacts, or unknown candidate
   always resolves to `ABSTAIN`.

# Program structure

## Workstream A — Canonical data plane

### PR 2 — Canonical snapshot assembler

**Goal:** make feed provenance and freshness first-class before any processing.

Changes:

- add `zerodte/ingestion/assembler.py`;
- wrap individual feeds and `CompositeFeed` without changing their current
  provider priority;
- produce `CanonicalMarketSnapshot` directly;
- record provider attempts, selected source, observation timestamps, source age,
  fallback status, missing components, and chain coverage;
- create a deterministic snapshot ID at ingestion;
- preserve the legacy `TickSnapshot` through an adapter during migration.

Tests:

- primary feed success;
- fallback feed success;
- provider exception isolation;
- missing chain;
- stale bars;
- invalid spot;
- timezone enforcement;
- deterministic snapshot IDs;
- parity with the current synthetic feed path.

Exit gate:

- canonical and legacy snapshots are produced from the same feed call;
- no downstream module has to infer the provider from class names.

### PR 3 — Snapshot journal and replay envelope

**Goal:** make every decision reproducible from the exact input state.

Changes:

- serialize a compact canonical snapshot envelope;
- add schema version and content hash;
- persist source freshness and quality status;
- add recorded-tick replay loader;
- prohibit mutable provider objects in persisted contracts.

Exit gate:

- a captured snapshot can be replayed offline and retains the same ID and hash.

## Workstream B — Processing plane

### PR 4 — Feature and structural-state service

**Goal:** remove analytical computation from orchestration.

Extract from `UnifiedOrchestrator.tick()`:

- risk-neutral density moments and edge diagnostics;
- realized-volatility physical baseline;
- GEX variants;
- volatility channels;
- pin assessment;
- market dynamics;
- MTF input and standardized matrix;
- regime classification;
- feature quality and missingness.

New modules:

```text
zerodte/features/rnd.py
zerodte/features/gex.py
zerodte/features/volatility.py
zerodte/features/pin.py
zerodte/features/dynamics.py
zerodte/features/mtf.py
zerodte/features/service.py
```

The result should be a versioned `FeatureBundle` and `StructuralState`, not a
large untyped `signals` dictionary.

Exit gate:

- replay parity for all decision-relevant current signals;
- `UnifiedOrchestrator.tick()` delegates feature computation through one service.

### PR 5 — PredictionRuntime

**Goal:** complete the missing validated serving path for trained forecasts.

Changes:

- load `DeploymentBundle` once at startup;
- validate artifact IDs, hashes, feature version, label version, schema version,
  model status, and mode permissions;
- produce one forecast bundle per canonical snapshot;
- support explicit research/shadow/advisory/candidate/champion modes;
- forbid silent heuristic substitution in candidate/champion modes;
- label heuristic baselines explicitly in research/shadow;
- expose uncertainty, data quality, calibration version, and drift state.

Exit gate:

- identical snapshot + bundle + artifacts yields identical forecast output;
- any artifact mismatch fails closed and records the exact reason.

## Workstream C — Shared trade economics

### PR 6 — Canonical candidate factory

**Goal:** create one legal candidate universe for every decision layer.

Changes:

- wrap the current leg-based spread selector behind `CandidateService`;
- support only approved bounded-loss, no-stock structures;
- normalize expiration, leg quantities, option type, strikes, family, and
  direction;
- assign stable canonical candidate IDs;
- calculate payoff, maximum profit, maximum loss, breakevens, and probability
  of profit from the same payoff engine;
- reject naked, cash-secured, covered, or stock-dependent candidates regardless
  of legacy selector configuration.

Exit gate:

- Legacy, V2, V3, and AI receive the exact same candidate IDs and geometry.

### PR 7 — Executable economics service

**Goal:** stop comparing strategies with inconsistent or midpoint-only economics.

Changes:

- centralize mid, natural, expected-fill, conservative-fill, fees, and slippage;
- produce fill probability and concession estimates;
- compute executable EV, utility, CVaR, max loss, liquidity, touch probability,
  wall safety, gamma exposure, and data-quality penalties;
- remove placeholder candidate economics from AI integrations;
- make one economics version part of the deployment hash.

Exit gate:

- all policy rankings use the same execution estimate and risk denominator;
- synthetic and recorded tests prove payoff and fill monotonicity.

## Workstream D — Decision plane

### PR 8 — Legacy, V2, and V3 policy adapters

**Goal:** make existing systems comparable without rewriting their logic at once.

Changes:

- convert each system into a `PolicyService` adapter;
- emit `PolicyDecisionView` with action, candidate ID, confidence, uncertainty,
  size cap, rationale, hard vetoes, and version;
- include disagreement diagnostics;
- remove private candidate selection paths from policy adapters;
- preserve current authority through deployment mode.

Exit gate:

- each policy is evaluated over the same packet and candidate universe;
- no policy can mutate a candidate or risk limit.

### PR 9 — Canonical DecisionPacket builder

**Goal:** provide the AI layer with a complete but bounded decision context.

Packet contents:

- canonical snapshot identity and quality;
- structural state;
- calibrated forecasts and uncertainty;
- drift and model-health state;
- ranked candidate summaries with executable economics;
- Legacy/V2/V3 decisions and disagreement;
- portfolio state;
- operational status;
- deterministic risk envelope;
- deployment and configuration hashes.

The packet excludes:

- credentials;
- broker mutation methods;
- unrestricted repository or shell access;
- arbitrary option-chain trade construction tools;
- current-session unsettled labels.

Exit gate:

- packet serialization is deterministic and schema validated;
- prompt-injection strings in market data cannot alter the tool contract.

## Workstream E — AI decision agent

### PR 10 — Provider-neutral agent shadow runtime

**Goal:** introduce the final AI decision layer safely.

Refactor useful components from draft PR #125:

- audit trail;
- cadence and cost controls;
- provider timeout and retry rules;
- restart recovery;
- structured decision parsing;
- decision evidence logging.

Discard or replace:

- arbitrary model-generated legs and strikes;
- global mutation of paper tracks;
- monkey-patching broker methods;
- model self-reported confidence as candidate economics;
- decision blinding that removes processed outputs.

Valid agent actions:

- `SELECT_CANDIDATE`
- `ABSTAIN`
- `HOLD`
- `REDUCE`
- `CLOSE`

Entry constraints:

- candidate ID must exist in the packet;
- candidate must have no hard veto;
- size scalar may only reduce the deterministic cap;
- exit policy must be selected from an approved registry;
- malformed, late, unavailable, or contradictory output fails to `ABSTAIN`.

Exit gate:

- agent runs in shadow and cannot submit paper or live orders;
- every agent response is reproducible from packet hash, model ID, prompt
  version, and provider response ID.

### PR 11 — Agent evaluation and promotion harness

**Goal:** prove whether the agent adds value rather than merely sounding smart.

Measurements:

- trade/no-trade precision;
- incremental executable EV;
- utility and CVaR improvement;
- drawdown and loss concentration;
- calibration of confidence and abstention;
- decision latency and stale-decision rate;
- agreement/disagreement performance versus Legacy/V2/V3;
- cost per valid decision;
- performance by regime, time of day, data quality, and candidate family.

Required comparisons:

- Legacy alone;
- V2 alone;
- V3 alone;
- deterministic ensemble;
- AI meta-policy;
- AI meta-policy with each evidence class ablated.

Exit gate:

- agent clears pre-registered shadow thresholds over multiple settled sessions;
- no evaluation uses current-session unsettled outcomes.

## Workstream F — Risk, execution, and state

### PR 12 — Deterministic pre-trade firewall

**Goal:** revalidate everything immediately before submission.

Checks:

- market/session status;
- freshness and quote age;
- candidate ID and leg parity;
- bounded maximum loss;
- expected-fill economics;
- account equity and daily loss budget;
- position count and concentration;
- portfolio delta/gamma limits;
- duplicate order and race protection;
- entry cutoff and emergency lockout;
- deployment mode authorization.

Exit gate:

- no policy or agent object can call the broker directly;
- execution receives only a validated `OrderIntent`.

### PR 13 — Append-only event journal

**Goal:** replace fragmented state writes with one auditable event stream.

Events:

- `snapshot_created`
- `features_computed`
- `forecasts_generated`
- `candidates_generated`
- `policy_evaluated`
- `agent_decided`
- `risk_evaluated`
- `order_simulated`
- `position_opened`
- `position_marked`
- `position_reduced`
- `position_closed`
- `outcome_settled`
- `deployment_changed`

Exit gate:

- dashboard state can be reconstructed from events;
- cross-endpoint views share one consistent state version.

### PR 14 — Paper execution authority

**Goal:** allow the promoted agent to control only its own isolated paper account.

Requirements:

- deployment mode `candidate`;
- explicit candidate account ID;
- deterministic firewall approval;
- mandatory exits and restart recovery;
- independent comparison accounts for Legacy/V2/V3;
- one-click rollback to shadow.

Exit gate:

- the agent can create no option geometry and cannot access other tracks.

## Workstream G — Operational hardening

### PR 15 — Package, dependency, and CI modernization

Changes:

- `pyproject.toml` and `src`-compatible package configuration;
- pinned lock file;
- formatter and lint rules;
- type checking for canonical modules;
- dependency and secret scanning;
- schema/contract tests;
- replay determinism tests;
- property tests for payoffs and bounded loss;
- load and latency tests;
- prompt-injection and malformed-output tests.

Exit gate:

- canonical package cannot merge with lint, type, contract, or safety failures.

### PR 16 — Dashboard V4

Changes:

- one canonical state endpoint keyed by state version;
- feed freshness and component health;
- forecast calibration and drift;
- candidate economics comparison;
- Legacy/V2/V3/AI decision comparison;
- risk veto and sizing explanation;
- event timeline and replay link;
- deployment bundle and rollback status.

Exit gate:

- dashboard never merges unrelated endpoint timestamps client-side.

## Workstream H — Cutover

### PR 17 — Dual-runtime shadow parity

Run old and new runtimes from the same recorded and live snapshots.

Promotion requirements:

- unexplained decision drift below threshold;
- snapshot and feature parity;
- candidate geometry parity;
- settlement and P&L parity;
- stable latency and memory;
- tested rollback package.

### PR 18 — Canonical runtime production cutover

Changes:

- switch service entrypoint to canonical runtime;
- retain old runtime as rollback target for a defined observation window;
- keep AI in shadow or paper mode according to its independent promotion state;
- publish deployment review packet and rollback command.

### PR 19 — Advisory authority

The agent may produce actionable recommendations, but not direct live orders.
This stage requires established paper evidence and human approval.

### Future PR — Live execution

Live broker authority is a separate product and risk decision. It is not implied
by completion of the architecture revamp and requires explicit authorization,
broker controls, regulatory review as applicable, and a separate release plan.

# Dependency graph

```text
PR2 Snapshot assembler
  -> PR3 Replay envelope
  -> PR4 Feature service
  -> PR5 PredictionRuntime
  -> PR6 Candidate factory
  -> PR7 Executable economics
  -> PR8 Policy adapters
  -> PR9 DecisionPacket
  -> PR10 Agent shadow
  -> PR11 Agent evaluation
  -> PR12 Risk firewall
  -> PR13 Event journal
  -> PR14 Paper authority

PR15 CI hardening runs in parallel after PR2.
PR16 Dashboard depends on PR13.
PR17 Dual-runtime parity depends on PR4–PR13.
PR18 Cutover depends on PR17.
```

# Recommended execution order

## Immediate critical path

1. Canonical snapshot assembler.
2. Replay envelope.
3. Feature service extraction.
4. PredictionRuntime.
5. Shared candidate factory and executable economics.
6. Policy adapters and DecisionPacket.
7. AI shadow runtime.

## Parallel track

1. CI modernization.
2. Event schema design.
3. Dashboard V4 wireframes and endpoint contract.
4. Recorded-session fixture library.
5. Agent evaluation specification.

# Program-level acceptance criteria

The revamp is complete only when:

- the live orchestrator contains sequencing rather than analytical logic;
- each processing stage has a versioned input/output contract;
- every decision can be replayed from immutable inputs and deployment artifacts;
- every policy and the AI agent sees identical candidates and economics;
- no AI output can create option legs, override vetoes, or increase risk;
- risk and execution are independent of forecast and policy implementation;
- dashboard state derives from a consistent event version;
- the old runtime remains a tested rollback target through cutover;
- AI promotion is evidence-based and independent from runtime migration.

# Decisions already made

- Architecture V4 is the target architecture.
- PR #126 is the foundation and has been merged.
- Draft PR #125 will not be merged as the final AI architecture.
- Useful PR #125 operational components will be selectively ported.
- AI is a constrained meta-policy using processed outputs, not an autonomous
  unrestricted trade constructor.
- Undefined-risk and stock-dependent structures are excluded.
- The implementation proceeds through incremental PRs with parity gates rather
  than a repository-wide rewrite.
