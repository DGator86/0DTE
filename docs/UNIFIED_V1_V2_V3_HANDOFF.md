# 0DTE Unified Prediction and Decision Engine

## V1 + V2 + V3 Integration Specification and Coding-Agent Handoff

| Field | Value |
|---|---|
| Repository | DGator86/0DTE |
| Document status | Implementation specification |
| Audience | Coding agent, quantitative developer, model-validation reviewer |
| Primary objective | Integrate the existing V1, V2, and V3 systems into one coherent runtime and learning architecture without losing deterministic safeguards, counterfactual journaling, session-safe validation, human-controlled promotion, or rollback capability. |

---

## 1. Executive Directive

Do not merge V1, V2, and V3 by deleting older components, copying code into one large module, or allowing the newest model to replace the entire system.

The versions must become separate layers of one decision stack:

* **V1** is the deterministic spine and safety baseline.
* **V2** is the base learned forecasting layer.
* **V3** is the advanced forecasting, candidate-ranking, execution, uncertainty, drift, abstention, and deployment layer.

The completed system must operate from:

1. One canonical market snapshot.
2. One as-of-safe feature set.
3. One candidate universe.
4. One unified decision record.
5. One deployment bundle.
6. One end-to-end learning and promotion workflow.

V1 must remain available as:

* A deterministic baseline.
* A shadow comparator.
* A fail-safe fallback.
* A rollback target.
* A source of hard operational vetoes.
* A source of centralized option candidate generation and payoff mathematics.

V2 and V3 must not become authoritative merely because their modules exist. They become authoritative only when:

* Valid trained artifacts are loaded.
* The artifacts pass registry and schema validation.
* Their complete stack is evaluated out of sample.
* A human approves a promotion packet.
* An atomic deployment pointer is updated.

**No automatic promotion is permitted.**

---

## 2. Current Repository State

The repository already contains most of the required components, but they are not yet assembled into one complete operating stack.

### 2.1 V1 operating path

The current live shadow path is centered on:

* `shadow_runner.py`
* `unified_loop.py`
* `decision_engine.py`
* `spread_selector.py`
* `gate_scorer.py`
* `regime_classifier.py`
* `decision_matrix.py`
* `journal.py`
* `paper_broker.py`
* `risk_manager.py`

`UnifiedOrchestrator` combines the original premium-selection and regime-routing tracks into one tick loop. It builds the market state, extracts the risk-neutral density, routes a structure, selects concrete option legs, applies gates, sizes the result, and journals trade and no-trade evaluations.

V1 is currently the practical runtime authority.

### 2.2 V2 operating path

V2 introduced:

* Canonical prediction datasets.
* Session-grouped training.
* Direction probability models.
* Return-quantile models.
* Volatility and range models.
* Independent physical distributions.
* Candidate-value models.
* Prediction-policy routing.
* A model registry.
* Shadow prediction persistence.

The V2 training pipeline is session-grouped and time ordered, compares learned models against legacy and naive baselines, and produces final shadow models.

However, the current `ShadowRunner` does not load a trained prediction model group into the bundle provider. It initializes the provider without a trained group and creates a `HeuristicCandidateValueModel`.

The V2 provider explicitly falls back to a heuristic bundle when no trained model group is supplied.

Therefore, current V2 parallel output must be treated as:

* A heuristic baseline.
* A wiring and dashboard test.
* Not a fully independent learned forecast.
* Not eligible for candidate or champion authority.

### 2.3 V3 operating path

V3 Part 1 added:

* Nested session cross-fitting.
* Independent calibration.
* Observation-specific uncertainty.
* Out-of-distribution detection.
* Session-level bootstrap intervals.
* Registry hardening.
* Fail-closed artifact rules.

V3 Part 2 added:

* Expanded structural state.
* Parallel GEX measurements.
* Probabilistic regime classification.
* Mixture-of-experts forecasting.
* Competing-risk models.
* Conformal return distributions.
* Empirical state-conditioned path simulation.
* Forecast ensembles.

V3 Part 3 added:

* Distributional candidate value.
* Pairwise candidate ranking.
* Fill probability.
* Fill concession.
* Executable order-value estimation.
* Trade/no-edge/abstain decisions.
* Dynamic model weighting.
* Drift monitoring and freeze behavior.
* Deployment modes.
* Promotion packets.
* Atomic rollback.

The V3 specifications correctly require research and shadow operation until explicit human promotion.

The current Part 2 and Part 3 shadow helpers are not yet integrated into the complete `ShadowRunner` tick path. `run_part2_shadow_tick` and `run_part3_shadow_decision` are primarily exercised through their own modules and tests.

The dashboard serializer accepts a Part 3 payload, but until live wiring landed, `ShadowRunner` often called it without one. Part 3 dashboard panels and triple-paper tracks exist; full unified authority routing does not.

---

## 3. Core Architectural Problem

The repository currently has two partially separate forms of learning and governance.

### 3.1 Legacy adaptive-learning loop

`adaptive_learning/learner.py` performs:

```
Journal
  → diagnostics
  → hypothesis generation
  → legacy parameter search
  → walk-forward evaluation
  → mandatory holdout
  → stability analysis
  → candidate rule configuration
  → pending human review
```

It optimizes `EngineConfig` and related deterministic rule parameters. It explicitly refuses searches with no holdout and never directly writes the champion configuration.

Its configuration store governs:

* `gate.*`
* `selector.*`
* `rnd.*`
* `classifier.*`
* Per-regime overrides.
* Per-regime size multipliers.

It writes a separate `configs/champion.json`.

### 3.2 Learned-model registry and deployment loop

V2 and V3 use:

* `prediction.registry.ModelRegistry`
* Versioned model artifacts.
* Artifact hashes.
* Feature and label versions.
* Fold hashes.
* Calibration metadata.
* OOS metrics.
* Model statuses.
* Deployment pointers.
* Promotion review packets.
* Rollback targets.

The registry fails closed when required metadata or artifacts are missing or inconsistent.

The deployment pointer separately identifies:

* Prediction model group.
* Candidate-value model.
* Candidate-rank model.
* Fill-probability model.
* Fill-concession model.
* Trade meta-model.

### 3.3 Required resolution

The repository must not maintain one champion system for V1 rule parameters and another unrelated champion system for learned models.

The complete decision stack must be deployed as one versioned deployment bundle.

A model group must not be evaluated under one V1 gate configuration and later deployed under another configuration without a new complete evaluation.

A candidate ranker must not be promoted independently of:

* The forecast group that generated its inputs.
* The candidate generator configuration.
* The execution model.
* The meta-decision thresholds.
* The V1 hard-veto configuration.
* The exact feature and label versions used in testing.

---

## 4. Mandatory Design Invariants

These invariants are non-negotiable.

### 4.1 No future leakage

No feature may use information timestamped after the prediction timestamp.

No session may appear on both sides of:

* Training.
* Calibration.
* Validation.
* Outer testing.
* Promotion holdout.

All folds must be grouped by complete market session.

Candidate rows from the same `snapshot_id` must never be split across training and test data.

### 4.2 Forecast-policy separation

Forecast models may not receive:

* Selected strategy family.
* Selected structure.
* Selected strikes.
* Gate result.
* Candidate score.
* Candidate rank.
* Policy direction.
* Human action.
* Future fills.
* Future P&L.
* Future stop or target outcome.

Policy may consume forecast outputs.

Policy may not alter or feed information back into the forecast.

This separation is already a formal V3 invariant and must remain intact.

### 4.3 Hard operational vetoes remain deterministic

The following remain outside learned models:

* Missing option chain.
* Invalid option surface.
* Broken arbitrage validation.
* Stale market data.
* Stale chain data.
* Prohibited catalyst window.
* Entry lockout time.
* Daily risk limit.
* Portfolio exposure limit.
* Broker unavailable.
* System unavailable.
* Insufficient executable liquidity.

A learned model may estimate market behavior during these conditions for research.

It may not override the operating restriction.

### 4.4 No silent heuristic substitution

In candidate or champion mode:

* Missing trained artifacts must produce `ABSTAIN` or an explicit configured fallback.
* Missing artifacts must never silently produce a heuristic forecast.
* Invalid schemas must fail closed.
* Invalid hashes must fail closed.
* Unsupported feature versions must fail closed.

Heuristic bundles may operate only as:

* research.
* shadow.
* Explicitly labeled baseline output.
* Cold-start dashboard assistance.

They are not learned model artifacts.

### 4.5 One candidate universe

Legacy and V3 must evaluate the same generated candidate set.

The system may not compare:

* A legacy candidate selected from one candidate universe.
* Against a V3 candidate selected from a broader or differently filtered universe.

Candidate generation must happen once.

Candidate scoring, ranking, policy, and execution estimates may then operate independently.

### 4.6 Counterfactuals remain first-class

The system must persist:

* Trades taken.
* No-trades.
* Gate-blocked candidates.
* Policy-rejected candidates.
* Meta-model `NO_EDGE` decisions.
* Meta-model `ABSTAIN` decisions.
* Hard-veto outcomes.
* Unfilled order attempts.
* Candidates not selected.
* Legacy selections.
* V3 selections.

The existing system’s strongest measurement property is that no-trades and blocked candidates are retained and later settled hypothetically.

This property must be extended to the entire V3 decision chain.

### 4.7 No intraday outcome learning

The current unsettled session may not update:

* Predictive model coefficients.
* Calibration models.
* Meta-model thresholds.
* Candidate-ranker coefficients.
* Fill-model coefficients.
* Champion promotion metrics.
* Ensemble loss weights that depend on outcomes.

As-of-safe normalization and market-state tracking may update intraday.

Outcome-dependent learning may update only after the session is completely settled.

### 4.8 Human-controlled promotion

No model, rule configuration, ensemble, threshold, or complete deployment may automatically become champion.

Promotion requires:

* A review packet.
* A reviewer identity.
* An approval note.
* Artifact and dataset hashes.
* Fold definitions.
* OOS metrics.
* Session-bootstrap intervals.
* Known weaknesses.
* Unsupported slices.
* A rollback target.

The existing promotion framework already requires this type of human approval.

### 4.9 Deterministic replay

Given identical:

* Recorded market data.
* Recorded chain data.
* Feature version.
* Label version.
* Configuration.
* Model artifacts.
* Random seeds.
* Deployment pointer.

The system must produce identical:

* Forecasts.
* Uncertainty.
* Candidate universe.
* Candidate rankings.
* Execution estimates.
* Meta-decisions.
* Hard-veto results.
* Authority decisions.
* Audit records.

### 4.10 Executable economics

Midpoint pricing is diagnostic only.

Decision-facing economics must use:

* Expected fill probability.
* Expected concession.
* Natural price.
* Fees.
* Slippage.
* Quote age.
* Exit costs.
* Stop slippage.
* Opportunity cost of an unfilled order.

---

## 5. Target System Architecture

```
Market bars + option chain + dealer/volatility data
                         |
                         v
              CanonicalSnapshotBuilder
                         |
              +----------+-----------+
              |                      |
              v                      v
       V1 Baseline Path       V2/V3 Forecast Runtime
       ----------------       -----------------------
       deterministic          direction probabilities
       regime state           return distributions
       matrix intent          realized-move forecast
       legacy policy          range survival
                              structural state
                              probabilistic regimes
                              competing risks
                              path simulations
                              uncertainty and OOD
              |                      |
              +----------+-----------+
                         |
                         v
            Independent Physical Distribution
                         |
                         v
              V1 Candidate Generator
          all feasible candidates generated once
                         |
              +----------+-----------+
              |                      |
              v                      v
       Legacy Candidate        V3 Candidate Evaluation
       Scoring                 -----------------------
                               candidate P&L distribution
                               absolute utility
                               pairwise ranking
                               ranking uncertainty
              |                      |
              +----------+-----------+
                         |
                         v
                V3 Execution Economics
                fill probability
                fill concession
                fees and slippage
                expected order value
                         |
                         v
                V3 Trade Meta-Decision
          TRADE | NO_EDGE | ABSTAIN
                         |
                         v
                V1 Hard Operational Vetoes
                         |
                         v
                   Authority Router
 legacy | shadow | advisory | candidate | champion
                         |
                         v
                UnifiedDecisionRecord
                         |
        +----------------+-----------------+
        |                |                 |
        v                v                 v
     Journal         Paper Broker       Dashboard
        |
        v
  Settlement and counterfactual outcome generation
        |
        v
  Unified learning, evaluation, promotion, rollback
```

---

## 6. Version Responsibilities

### 6.1 V1 responsibilities

V1 remains responsible for:

* Risk-neutral density extraction.
* Option-chain validation.
* Centralized spread enumeration.
* Payoff mathematics.
* Maximum loss.
* Greeks.
* Legacy EV.
* Legacy probability-of-profit calculations.
* Deterministic matrix policy.
* Existing regime classifier baseline.
* Hard operational gates.
* Risk manager.
* Journaling.
* Settlement.
* Paper broker.
* Notifications.
* Deterministic fallback behavior.

V1 must no longer be the only source of the physical forecast once a valid promoted learned forecast is available.

V1 remains an independent baseline.

### 6.2 V2 responsibilities

V2 becomes the base learned forecasting layer:

* Canonical feature dataset.
* Canonical observation labels.
* Session-grouped training.
* Direction probabilities.
* Return quantiles.
* Volatility forecasts.
* Range-survival probabilities.
* Independent physical distribution.
* Baseline candidate-value forecasting.
* Learned-versus-legacy baseline comparisons.
* Train/serve feature parity.

### 6.3 V3 Part 1 responsibilities

V3 Part 1 governs statistical validity:

* Nested cross-fitting.
* Independent probability calibration.
* OOF residuals.
* Observation-specific uncertainty.
* OOD scoring.
* Session-level confidence intervals.
* Fail-closed model loading.
* Deterministic seeds.
* Complete audit metadata.

### 6.4 V3 Part 2 responsibilities

V3 Part 2 governs advanced market forecasting:

* Expanded structural-state contract.
* Multiple GEX variants.
* Structural disagreement.
* Regime probabilities.
* Global and regime-specific experts.
* Shrinkage toward global models.
* Competing target/stop risks.
* Conformalized returns.
* Empirical path simulation.
* Forecast ensembles.
* Support-aware fallback.

A regime label is never sufficient by itself. Downstream models must consume the complete probability vector.

### 6.5 V3 Part 3 responsibilities

V3 Part 3 governs economic decisions:

* Candidate P&L distributions.
* Tail-aware utility.
* Pairwise ranking.
* Fill probability.
* Fill concession.
* Expected order value.
* `TRADE`.
* `NO_EDGE`.
* `ABSTAIN`.
* Dynamic OOS ensemble weights.
* Drift severity.
* Model freezing.
* Deployment modes.
* Human promotion.
* Atomic rollback.

---

## 7. New Core Contracts

Create these contracts before completing runtime integration.

### 7.1 CanonicalSnapshot

New module: `prediction/canonical_snapshot.py`

```python
@dataclass(frozen=True)
class CanonicalSnapshot:
    snapshot_id: str
    symbol: str
    ts: str
    session_date: str
    market: MarketSnapshot
    bars: RawBars
    chain: ChainSnapshot | None
    raw_features: dict[str, float | int | str | None]
    standardized_features: dict[str, float | None]
    missingness: dict[str, bool]
    source_timestamps: dict[str, str | None]
    source_ages_seconds: dict[str, float | None]
    quality: dict[str, float | int | bool | str | None]
    structural_sources: dict
    structural_state: StructuralState | None
    feature_version: str
    structural_state_version: str | None
    snapshot_schema_version: str
```

Requirements:

* Constructed exactly once per tick.
* Immutable after construction.
* One `snapshot_id` shared by all components.
* No routing or post-decision fields in model features.
* Missing values remain missing.
* Missing structural values must not be replaced with zero.
* Source timestamps and ages must be preserved.
* Snapshot hashing must be deterministic.

### 7.2 ForecastBundle

Use or extend `PredictionBundle` as the canonical forecast contract.

Required fields include:

```python
@dataclass(frozen=True)
class ForecastBundle:
    snapshot_id: str
    ts: str
    session_date: str
    symbol: str
    direction_probabilities: dict[str, float]
    return_quantiles: dict[str, dict[str, float]]
    expected_returns: dict[str, float]
    expected_realized_moves: dict[str, float]
    range_survival: dict[str, float]
    barrier_probabilities: dict[str, float]
    competing_risks: dict[str, dict[str, float]]
    regime_probabilities: dict[str, float]
    dominant_regime: str | None
    path_summary: dict
    physical_distribution_inputs: dict
    component_uncertainty: dict[str, float | None]
    composite_uncertainty: float
    ood_score: float
    data_quality: float
    feature_coverage: float
    model_versions: dict[str, str]
    feature_version: str
    label_version: str
    configuration_hash: str
    diagnostics: dict
```

Requirements:

* Produced before candidate selection.
* Does not contain selected trade information.
* Each probability family must satisfy normalization requirements.
* Missing components are explicit.
* Fallbacks must be recorded.
* Failed required components must increase uncertainty or cause unavailability.
* OOD observations must not claim ordinary conformal coverage.

### 7.3 CandidateUniverse

New module: `prediction/candidate_universe.py`

```python
@dataclass(frozen=True)
class CandidateUniverse:
    snapshot_id: str
    generated_at: str
    generator_version: str
    generator_configuration_hash: str
    candidates: tuple[SpreadCandidate, ...]
    excluded_at_generation: tuple[dict, ...]
    chain_quality: dict
    diagnostics: dict
```

Requirements:

* Generated once.
* Shared by legacy and V3.
* Exact candidate IDs must be deterministic.
* Candidate IDs must incorporate:
    * Snapshot ID.
    * Family.
    * Leg types.
    * Leg quantities.
    * Strikes.
    * Expiration.
* No learned model may alter the candidate universe before baseline comparison.
* Candidate generation constraints must be distinguished from downstream vetoes.

### 7.4 CandidateEvaluation

```python
@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    legacy_score: float | None
    legacy_ev: float | None
    legacy_prob_profit: float | None
    expected_net_pnl: float | None
    p_positive_pnl: float | None
    pnl_quantiles: dict[str, float]
    expected_shortfall: float | None
    absolute_utility: float | None
    pairwise_rank_score: float | None
    final_rank: int | None
    ranking_uncertainty: float | None
    fill_probability: float | None
    expected_fill_price: float | None
    conservative_fill_price: float | None
    expected_concession: float | None
    fees: float | None
    expected_exit_cost: float | None
    expected_order_value: float | None
    model_versions: dict[str, str]
    vetoes: tuple[str, ...]
    diagnostics: dict
```

### 7.5 UnifiedDecisionRecord

New module: `decision_stack/contracts.py`

```python
@dataclass(frozen=True)
class UnifiedDecisionRecord:
    snapshot_id: str
    ts: str
    session_date: str
    symbol: str
    deployment_id: str
    deployment_mode: str
    authority_source: str
    legacy_action: str
    legacy_candidate_id: str | None
    legacy_structure: str | None
    legacy_direction: str | None
    legacy_size_mult: float
    v3_statistical_action: str
    v3_final_action: str
    v3_candidate_id: str | None
    v3_structure: str | None
    v3_direction: str | None
    selected_candidate_id: str | None
    final_action: str
    final_structure: str | None
    final_direction: str | None
    final_size_mult: float
    hard_vetoes: tuple[str, ...]
    reasons: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None
    forecast_summary: dict
    selected_candidate_evaluation: dict | None
    legacy_v3_disagreement: dict
    model_versions: dict
    configuration_hash: str
    diagnostics: dict
```

Allowed final actions:

* `TRADE`
* `NO_EDGE`
* `ABSTAIN`
* `HARD_VETO`
* `NO_CANDIDATE`
* `UNAVAILABLE`

### 7.6 DeploymentBundle

Extend `prediction/deployment.py`.

```python
@dataclass(frozen=True)
class DeploymentBundle:
    deployment_id: str
    mode: str
    legacy_rule_config_id: str | None
    prediction_model_group_id: str | None
    candidate_value_model_id: str | None
    candidate_rank_model_id: str | None
    fill_probability_model_id: str | None
    fill_concession_model_id: str | None
    meta_model_id: str | None
    policy_version: str
    execution_version: str
    risk_version: str
    feature_version: str
    label_version: str
    structural_state_version: str
    authority_source: str
    fallback_policy: str
    previous_deployment_id: str | None
    rollback_deployment_id: str | None
    approved_review_id: str | None
    configuration_hash: str
    extras: dict
```

Authority sources: `legacy` | `v3` | `human`

Fallback policies: `abstain` | `legacy` | `no_trade`

`candidate` and `champion` modes must never default to heuristic fallback.

---

## 8. Deployment Modes

### 8.1 Research

* Offline use.
* Model training.
* Backtesting.
* Replay.
* No notifications.
* No paper-account authority.
* May use incomplete experimental components.
* Must label experimental outputs.

### 8.2 Shadow

* Legacy remains authoritative.
* V3 runs on identical snapshots.
* V3 evaluates the identical candidate universe.
* All disagreements are persisted.
* Paper broker follows legacy authority unless a separate comparison account is configured.
* No live order authority.

### 8.3 Advisory

* Legacy remains operational authority.
* V3 recommendations are displayed prominently.
* Human sees both.
* No automatic V3 ticket placement.
* Disagreement reasons must be available.

### 8.4 Candidate

* V3 controls a separate candidate paper account.
* Legacy controls the reference paper account.
* Both use:
    * Identical data.
    * Identical candidate generation.
    * Identical fill assumptions.
    * Identical fees.
    * Identical settlement logic.
* Candidate mode requires human-reviewed artifacts.

### 8.5 Champion

* V3 controls decision-ticket authority.
* V1 deterministic hard vetoes remain effective.
* The deployment bundle specifies whether a failed V3 component:
    * Abstains.
    * Falls back to legacy.
    * Produces no trade.
* V1 remains available for rollback.
* Champion status does not itself authorize real broker order placement.

The current deployment module also treats champion as decision authority rather than live-order authorization.

---

## 9. Unified Runtime Components

### 9.1 PredictionRuntime

New module: `prediction/runtime.py`

```python
class PredictionRuntime:
    @classmethod
    def from_deployment_bundle(
        cls,
        bundle: DeploymentBundle,
        registry: ModelRegistry,
    ) -> "PredictionRuntime":
        ...

    def forecast(
        self,
        snapshot: CanonicalSnapshot,
    ) -> ForecastBundle:
        ...

    def evaluate_candidates(
        self,
        snapshot: CanonicalSnapshot,
        forecast: ForecastBundle,
        universe: CandidateUniverse,
    ) -> tuple[CandidateEvaluation, ...]:
        ...
```

Responsibilities:

* Load every required artifact.
* Verify artifact hashes.
* Verify registry status permissions.
* Verify feature versions.
* Verify label versions.
* Verify required fields.
* Verify configuration compatibility.
* Build the complete V2/V3 forecast.
* Build candidate-value forecasts.
* Run pairwise ranking.
* Run fill models.
* Run the meta-model.
* Return structured errors.
* Never silently mutate deployment state.

**Artifact-loading behavior**

Research or shadow:

Optional components may fail if:

* Failure is recorded.
* Uncertainty increases.
* Decision-facing output is labeled unavailable.
* No false valid forecast is produced.

Candidate or champion:

Missing required components must:

* Fail closed.
* Produce `ABSTAIN`, `UNAVAILABLE`, or configured legacy fallback.
* Never silently use heuristic replacements.

### 9.2 UnifiedDecisionStack

New package: `decision_stack/`

Suggested files:

```
decision_stack/
    __init__.py
    contracts.py
    stack.py
    authority.py
    persistence.py
    diagnostics.py
```

Primary entry point:

```python
class UnifiedDecisionStack:
    def evaluate(
        self,
        snapshot: CanonicalSnapshot,
        *,
        position_contexts: list | None = None,
    ) -> UnifiedDecisionRecord:
        ...
```

Required evaluation order:

1. Validate canonical snapshot.
2. Run V1 deterministic baseline forecast and policy.
3. Run V2/V3 forecast runtime.
4. Build independent physical distribution.
5. Generate candidate universe once.
6. Score all candidates with legacy methods.
7. Evaluate all candidates with V3 methods.
8. Rank candidates.
9. Estimate execution.
10. Produce statistical action: `TRADE` | `NO_EDGE` | `ABSTAIN`
11. Apply hard operational vetoes.
12. Route authority according to deployment mode.
13. Persist the complete decision graph.
14. Return one unified record.

### 9.3 Authority router

New module: `decision_stack/authority.py`

```python
def resolve_authority(
    *,
    mode: str,
    legacy_decision,
    v3_decision,
    hard_vetoes: tuple[str, ...],
    fallback_policy: str,
) -> AuthorityResult:
    ...
```

Rules:

* **Hard veto** — always produces `HARD_VETO`, regardless of V1 or V3 statistical preference.
* **Shadow** — authoritative = legacy.
* **Advisory** — authoritative = legacy; advisory = v3.
* **Candidate** — authoritative reference account = legacy; authoritative candidate account = v3.
* **Champion** — authoritative = v3 unless the configured fail-closed fallback applies.

---

## 10. Unified Tick Flow

For every market tick:

1. **Acquire raw data** — market snapshot, bars, option chain, GEX/weekly rows, feed source, data/quote timestamps.
2. **Build canonical snapshot** — one `snapshot_id` shared by journal, features, structural state, prediction, candidates, rankings, fills, meta-decision, unified decision, dashboard.
3. **Run V1 baseline** — market dynamics, vol channels, pin, regime classifier, MTF matrix, legacy intent/policy/gates. Do not yet generate a separate candidate set.
4. **Run V2/V3 forecast** — complete forecast bundle from pre-decision information. On failure: structured record + deployment fallback; do not invent a neutral forecast.
5. **Build independent physical distribution** — from forecast outputs, not selected structure. Legacy directional tilt remains baseline/shadow/rollback only; must not contaminate V3 candidate economics.
6. **Generate candidate universe once** — centralized option candidate generator; all eligible families; generation restrictions versioned.
7. **Run legacy candidate scoring** — EV, PoP, safety, ranking, selected candidate, selector vetoes.
8. **Run V3 candidate evaluation** — P&L distribution, positive-P&L probability, tail risk, utility, pairwise rank, ranking uncertainty.
9. **Run execution models** — fill probability, concession, fees, fill price, expected order value; preserve unfilled attempts.
10. **Run trade meta-model** — `TRADE` | `NO_EDGE` | `ABSTAIN` with reason codes.
11. **Apply hard vetoes** — statistical `TRADE` + hard veto → `HARD_VETO`.
12. **Resolve authority** — deployment mode and fallback policy.
13. **Persist all paths** — legacy, V3, authoritative, universe, evaluations, ranks, fills, meta, vetoes, disagreement, fallback, deployment ID, configuration hash.
14. **Notify and paper trade** — read the authoritative decision contract, not a raw legacy `TradeDecision` alone.

---

## 11. Required Changes to Existing Modules

### 11.1 `shadow_runner.py`

Replace heuristic V2 initialization with:

```python
registry = ModelRegistry(models_dir)
deployment = load_deployment_bundle(deployment_path)
prediction_runtime = PredictionRuntime.from_deployment_bundle(
    deployment,
    registry,
)
decision_stack = UnifiedDecisionStack(
    deployment=deployment,
    prediction_runtime=prediction_runtime,
    legacy_config=loaded_rule_config,
    stores=...,
)
```

Add CLI arguments:

* `--deployment`
* `--models-dir`
* `--mode`
* `--reference-paper-db`
* `--candidate-paper-db`
* `--fallback-policy`
* `--strict-artifacts`

Required behavior:

* Load deployment atomically at startup.
* Reject invalid deployment.
* Log every loaded component ID.
* Log feature and label versions.
* Log fallback policy.
* Pass Part 3 payload to dashboard serializer.
* Use authoritative decision for notifications.
* Support separate reference and candidate paper accounts.
* Never use heuristic V2 in candidate or champion mode.

### 11.2 `unified_loop.py`

Refactor `UnifiedOrchestrator` to reduce separate injection points.

Deprecate:

* `physical_forecast`
* `physical_forecast_provider`
* `candidate_value_model`
* `candidate_ranker_cfg`
* `prediction_bundle`
* `prediction_bundle_provider`
* `policy_mode`
* `policy_router_cfg`

Replace with:

* `decision_stack: UnifiedDecisionStack`

During transition, compatibility adapters may map old inputs into the new stack.

Extend `TickResult` with:

* `legacy_decision`
* `v3_decision`
* `authoritative_decision`
* `authority_source`
* `deployment_id`
* `fallback_used`
* `candidate_universe_summary`

Retain the existing `decision` field temporarily as an alias to the authoritative result for backward compatibility.

### 11.3 `prediction/part2_shadow.py`

Refactor into a full forecast assembler:

```python
def build_v3_forecast(
    snapshot: CanonicalSnapshot,
    models: ForecastModelSet,
    store: PredictionStore | None,
    mode: str,
) -> ForecastBundle:
    ...
```

Required stages: structural state → regime probabilities → global models → regime experts → mixture blending → return distribution → conformal → competing risks → path model → ensemble → component/composite uncertainty → OOD → persistence.

Component failures must be explicit.

### 11.4 `prediction/part3_shadow.py`

Refactor into:

```python
def build_v3_decision(
    *,
    snapshot: CanonicalSnapshot,
    forecast: ForecastBundle,
    universe: CandidateUniverse,
    model_set: DecisionModelSet,
    hard_vetoes: tuple[str, ...],
    mode: str,
) -> V3DecisionResult:
    ...
```

Internally: candidate-value → utility → pairwise ranking → fill probability → fill concession → expected order value → meta-decision → hard-veto → persistence.

### 11.5 `prediction/deployment.py`

Extend the deployment pointer to include:

* `deployment_id`
* `legacy_rule_config_id`
* `feature_version` / `label_version` / `structural_state_version`
* `policy_version` / `execution_version` / `risk_version`
* `authority_source`
* `fallback_policy`

Configuration hashing must include every decision-relevant field (not only required model pointer keys).

### 11.6 `prediction/registry.py`

Add first-class model-group support:

```python
@dataclass(frozen=True)
class ModelGroupMetadata:
    group_id: str
    component_model_ids: dict[str, str]
    feature_version: str
    label_version: str
    structural_state_version: str
    configuration_hash: str
    training_sessions: list[str]
    calibration_sessions: list[str]
    outer_test_sessions: list[str]
    metrics: dict
    status: str
```

Methods: `save_group`, `load_group`, `set_group_status`, `validate_group`, `list_groups`.

A group must fail validation when components are missing, feature/label versions conflict, status is not allowed for the load mode, or a component hash is invalid.

### 11.7 `adaptive_learning/config_store.py`

Convert V1 rule configurations into registry-compatible artifacts.

Do not remove the existing file format immediately.

Add `RuleConfigArtifact` with `rule_config_id`, overrides, regime overrides, hash, parent, status, metrics.

The complete deployment pointer must reference a `legacy_rule_config_id`.

`configs/champion.json` becomes a compatibility export, not the primary source of deployment truth.

### 11.8 `adaptive_learning/learner.py`

Keep the legacy learner, but change its output responsibility:

* Diagnose V1 rule behavior.
* Search permitted V1 parameter spaces.
* Produce a versioned rule-config candidate.
* Never independently promote the live system.
* Forward the candidate into the unified deployment evaluator.

It must not write or stage a standalone champion disconnected from the V2/V3 model stack.

---

## 12. Unified Learning Architecture

Create:

```
learning/
    __init__.py
    orchestrator.py
    settlement.py
    labels.py
    model_training.py
    rule_training.py
    weight_updates.py
    drift_evaluation.py
    deployment_evaluation.py
    promotion_packet.py
```

Primary entry point:

```python
class LearningOrchestrator:
    def run_daily(...): ...
    def run_evening(...): ...
    def run_weekly(...): ...
    def run_manual(...): ...
```

---

## 13. Four Learning Speeds

### 13.1 Intraday state adaptation

**Allowed:** robust feature scaling, GEX percentile state, rolling windows, wall/flip velocity, data-quality tracking, feature staleness, volatility-state estimation.

**Not allowed:** outcome-dependent coefficient updates; current-session calibration / ranker / fill / meta-threshold changes.

### 13.2 Post-settlement adaptation

Run only after settlement is complete:

* Settle journal rows and hypothetical candidates.
* Create observation, candidate, fill, and meta-decision labels.
* Update model loss tables and ensemble weights.
* Update calibration diagnostics and drift state.
* Freeze degraded models when rules require it.

Dynamic weights must use settled sessions only.

### 13.3 Scheduled retraining

Retrain eligible families (direction, returns, volatility, range survival, regime, experts, competing risk, path conditioning, candidate value/rank, fill models, trade meta) each with its appropriate statistical loss.

Do not optimize the whole stack under one undifferentiated objective.

### 13.4 Human deployment learning

A new complete deployment bundle must progress through:

```
research → shadow → advisory → candidate → champion
```

No status transition may skip required evaluation. A shadow deployment may not directly become champion.

---

## 14. Training and Validation Requirements

### 14.1 Session grouping

Primary independent unit: **market session**.

Ticks may be model rows, but folds, bootstrap intervals, and promotion sample requirements are session-grouped. Candidate rows remain grouped by snapshot.

### 14.2 Nested evaluation

```
Outer session walk-forward
    |
    +-- Inner session folds for hyperparameter selection
    |
    +-- Cross-fitted predictions for calibration
    |
    +-- Untouched outer test sessions
```

Outer test sessions may not be used for hyperparameter selection, calibration, threshold selection, feature selection, model-family selection, or ensemble-weight selection.

### 14.3 Model-family metrics

| Family | Metrics |
|---|---|
| Direction | Log loss, Brier, calibration error, AUC (secondary), session-bootstrap intervals |
| Returns | Pinball, MAE, bias, interval coverage/width, downside miss, quantile crossing |
| Volatility / realized move | MAE, relative error, large-move underprediction, vol-regime calibration |
| Regime | Multiclass log loss, one-vs-rest Brier, calibration, confusion, support by session |
| Competing risk | Target-first / stop-first / neither calibration, time-dependent loss, same-bar ambiguity |
| Candidate value | Net P&L MAE/bias, positive-P&L Brier, quantile coverage, tail underprediction, ES calibration |
| Candidate rank | Ranking regret, top-1 realized utility, pairwise accuracy, Kendall/Spearman, by family/regime |
| Fill probability | Brier, calibration bins, log loss, fill-horizon calibration |
| Fill concession | MAE, quantile coverage, conservative-fill miss rate, family-level bias |
| Meta-model | TRADE precision, NO_EDGE effectiveness, ABSTAIN value, false-trade / missed-positive rates, EOV calibration |

---

## 15. Complete Deployment Evaluation

A deployment candidate is evaluated as a complete economic stack.

### 15.1 Forecast evaluation

Direction skill, return calibration, volatility accuracy, range-survival and barrier calibration, regime calibration, OOD behavior, uncertainty calibration.

### 15.2 Candidate evaluation

Realized executable P&L, ranking regret, tail loss, MAE, top-candidate performance, family and regime slices.

### 15.3 Execution evaluation

Fill-rate calibration, concession accuracy, unfilled opportunity cost, fees/slippage, expected-order-value calibration.

### 15.4 Decision evaluation

Trade / no-edge / abstain / hard-veto frequency, win rate, net P&L, drawdown, tail loss, risk-adjusted return, session-level consistency.

### 15.5 Legacy comparison

The promotion packet must compare V3 against V1 under identical data, candidate universe, fees, fill models, paper-account risk, settlement, and session ranges.

**Do not compare legacy midpoint P&L with V3 executable P&L.**

---

## 16. Persistence Requirements

Extend `PredictionStore` or add a coordinated decision store.

### 16.1 `canonical_snapshots`

`snapshot_id`, `symbol`, `ts`, `session_date`, `feature_version`, `snapshot_schema_version`, `raw_features_json`, `standardized_features_json`, `missingness_json`, `source_timestamps_json`, `source_ages_json`, `quality_json`, `snapshot_hash`

### 16.2 `forecast_bundles`

`snapshot_id`, `deployment_id`, `model_group_id`, `forecast_json`, `uncertainty`, `ood_score`, `data_quality`, `generated_at`, `mode`

### 16.3 `candidate_universes`

`snapshot_id`, `generator_version`, `configuration_hash`, `candidate_count`, `excluded_count`, `generated_at`, `diagnostics_json`

### 16.4 `candidate_evaluations`

One row per candidate: `snapshot_id`, `candidate_id`, `family`, `legs_json`, `legacy_metrics_json`, `v3_value_json`, `ranking_json`, `execution_json`, `vetoes_json`, `model_versions_json`

### 16.5 `unified_decisions`

`snapshot_id`, `deployment_id`, `deployment_mode`, `authority_source`, `legacy_action`, `legacy_candidate_id`, `v3_statistical_action`, `v3_final_action`, `v3_candidate_id`, `final_action`, `selected_candidate_id`, `hard_vetoes_json`, `reasons_json`, `fallback_used`, `fallback_reason`, `configuration_hash`, `decision_json`

### 16.6 `candidate_outcomes`

`snapshot_id`, `candidate_id`, `session_date`, `entry_assumption`, `fill_status`, `fill_price`, `exit_price`, `fees`, `net_pnl`, `max_adverse_excursion`, `max_favorable_excursion`, `target_first`, `stop_first`, `settled_at`, `label_version`

### 16.7 `deployment_evaluations`

`evaluation_id`, `deployment_id`, `comparison_deployment_id`, `session_start`, `session_end`, `sessions_count`, `metrics_json`, `slice_metrics_json`, `bootstrap_intervals_json`, `drift_json`, `created_at`

---

## 17. Label Construction

Labels must be versioned.

### 17.1 Market forecast labels

Examples: `up_15m`, `up_30m`, `up_60m`, `up_close`, `fwd_return_*`, `realized_move_*`, `range_survive_*`.

### 17.2 Barrier labels

Examples: `call_wall_touched_30m`, `put_wall_touched_30m`, `gamma_flip_crossed_30m`, `target_first`, `stop_first`, `neither`, `time_to_first_event`.

Levels must be frozen at prediction time. Do not use future-updated wall or flip levels as the historical reference.

### 17.3 Candidate labels

Examples: `fillable`, `fill_time_seconds`, `fill_concession`, `net_pnl`, `positive_pnl`, `pnl_quantiles_target`, `max_adverse_excursion`, `max_favorable_excursion`, `capital_required`, `realized_utility`.

### 17.4 Meta-decision labels

Evaluate whether `TRADE` generated positive executable value, whether `NO_EDGE` avoided negative value, whether `ABSTAIN` was justified by uncertainty/OOD, and whether a hard veto prevented loss or blocked profit.

These are evaluation labels, not automatic RL rewards.

---

## 18. Drift and Freeze Rules

Drift categories: `NORMAL` | `WATCH` | `DEGRADED` | `FREEZE`

Monitor: feature, prediction, residual, calibration, execution, economic-performance, missingness, and data-provider drift.

| Severity | Behavior |
|---|---|
| WATCH | Continue; reduce weight; increase observability; require review if persistent |
| DEGRADED | Materially reduce weight; increase uncertainty; restrict promotion; may fall back per policy |
| FREEZE | Exclude from decision-facing ensemble; do not delete; do not auto-promote replacement; record reason; require recovery evaluation |

---

## 19. Dashboard Requirements

The dashboard must show three clearly distinct views.

### 19.1 Legacy

Regime, matrix cell, structure, direction, gate result, candidate, legacy EV, reasons.

### 19.2 Forecast

Direction probabilities, return distribution, realized-move forecast, range survival, regime probabilities, competing risks, uncertainty, OOD, data quality, model versions.

### 19.3 V3 decision

Statistical action, final action, selected candidate, utility, rank, fill probability, expected fill, expected order value, hard vetoes, reasons, authority source, fallback, deployment ID, mode label.

### 19.4 Disagreement panel

Legacy vs V3 action / candidate / structure / direction; disagreement reason; which system is authoritative; whether fallback was used.

---

## 20. Logging and Error Handling

Every component failure must create a structured record with:

`snapshot_id`, `component`, `stage`, `exception_type`, `message`, `required_or_optional`, `fallback_action`, `deployment_mode`, `model_id`, `configuration_hash`, `timestamp`

Do not use broad exception handling that silently continues without a record.

The operating loop may continue when a noncritical research component fails, but the resulting forecast or decision must explicitly indicate degradation.

Candidate and champion modes must fail closed when a required component fails.

---

## 21. PR Implementation Sequence

Implement in this order. Do not combine runtime integration and champion promotion in one PR.

### PR 1 — Deployment bundle and runtime loader

**Scope:** Extend deployment pointer; deployment ID; legacy rule-config ID; feature/label/structural/policy/execution/risk versions; fallback policy; complete configuration hashing; model-group validation; `PredictionRuntime.from_deployment_bundle`.

**No behavior change** — legacy remains authoritative.

**Tests:** valid load; missing artifact / hash / feature-version / status permission failures; candidate/champion cannot use heuristic fallback; configuration hash changes when any decision-relevant field changes.

### PR 2 — Canonical snapshot

**Scope:** `CanonicalSnapshot`; build once per tick; preserve timestamps/missingness; one snapshot ID everywhere; persist.

**Tests:** deterministic hash; no post-routing fields; missing remains missing; identical input → identical snapshot; future-dated source rejected; snapshot ID shared across stores.

### PR 3 — Complete V3 forecast runtime

**Scope:** Full forecast assembly from trained V2/V3 models; structural state; regime probabilities; experts; competing risk; path; ensemble; uncertainty/OOD; persist.

**Tests:** forecast independent of selected candidate; regime probs sum to one; unsupported expert → global; component failure increases uncertainty; required failure → unavailable in strict mode; deterministic replay.

### PR 4 — Shared candidate universe

**Scope:** `CandidateUniverse`; generate once; deterministic IDs; identical feed to legacy and V3; persist every candidate.

**Tests:** identical candidate IDs; determinism; snapshot groups intact; no duplicate economic candidates; generator exclusions recorded.

### PR 5 — Complete V3 candidate and execution stack

**Scope:** Candidate-value distributions; utility; pairwise ranking; fill probability/concession; fees/slippage; expected order value; meta-decision; hard-veto application.

**Tests:** midpoint never treated as filled; unfilled remain evidence; hard veto overrides TRADE; required failure → ABSTAIN; deterministic rankings; identical universe; pairwise rows do not split snapshots.

### PR 6 — Authority router and unified decision

**Scope:** `UnifiedDecisionRecord`; authority router; all deployment modes; extend `TickResult`; update notifier, paper broker, dashboard; separate candidate and reference paper accounts.

**Tests:** shadow/advisory keep legacy authority; candidate dual paper; champion uses V3; fallback matches config; hard veto always wins; dashboard shows authority accurately.

### PR 7 — Unified settlement and labels

**Scope:** Settle all candidate counterfactuals and unfilled attempts; generate market/candidate/fill/meta labels; version all labels.

**Tests:** no-trades and nonselected settle; frozen structural levels; same-bar ambiguity adverse-first; current-session labels unavailable before settlement; idempotent settlement.

### PR 8 — Unified learning orchestrator

**Scope:** Coordinate V1 rule optimization and model-family retraining; dynamic weights; drift; evaluate complete deployment bundles; persist evaluation.

**Tests:** holdout mandatory; sessions never cross folds; outer test untouched; current-session outcomes excluded; rule/model candidates do not independently become champion.

### PR 9 — Joint promotion and rollback

**Scope:** Complete promotion packet (legacy rule config + all model artifacts); deployment status changes; atomic pointer swap and rollback; archive previous deployments.

**Tests:** promotion requires reviewer, approval note, rollback target, hashes and folds; shadow cannot skip to champion; rollback restores complete prior bundle; partial promotion impossible.

### PR 10 — True end-to-end replay

**Scope:** RecordedFeed → CanonicalSnapshot → V1 baseline → V3 forecast → CandidateUniverse → ranking → execution → meta → hard veto → authority → UnifiedDecisionRecord → journal → paper → settlement → LearningOrchestrator → DeploymentEvaluation → PromotionReviewPacket.

**Tests:** repeated replay identical; snapshot IDs align; no session/candidate leakage; no silent fallback; complete audit trail; promotion packet reconstructs every artifact and dataset version.

---

## 22. Required Test Matrix

At minimum, add:

* `tests/test_deployment_bundle.py`
* `tests/test_prediction_runtime_loader.py`
* `tests/test_canonical_snapshot.py`
* `tests/test_canonical_snapshot_asof.py`
* `tests/test_full_v3_forecast_runtime.py`
* `tests/test_forecast_policy_independence.py`
* `tests/test_shared_candidate_universe.py`
* `tests/test_candidate_id_determinism.py`
* `tests/test_v3_candidate_stack.py`
* `tests/test_execution_economics.py`
* `tests/test_authority_router.py`
* `tests/test_unified_decision_record.py`
* `tests/test_dual_paper_accounts.py`
* `tests/test_counterfactual_settlement.py`
* `tests/test_unified_labels.py`
* `tests/test_learning_orchestrator.py`
* `tests/test_joint_deployment_evaluation.py`
* `tests/test_joint_promotion.py`
* `tests/test_atomic_full_stack_rollback.py`
* `tests/test_true_end_to_end_replay.py`

---

## 23. Acceptance Criteria

The integration is complete only when every item below is true.

### Runtime

* One canonical snapshot is built per tick.
* One candidate universe is generated per tick.
* Legacy and V3 evaluate identical candidates.
* Trained artifacts are loaded from the registry.
* Heuristic outputs are explicitly labeled.
* V3 Part 2 runs in the `ShadowRunner` path.
* V3 Part 3 runs in the `ShadowRunner` path.
* Part 3 output appears in the dashboard.
* One unified decision record is returned.

### Safety

* Hard vetoes remain deterministic.
* Forecasts do not receive policy or candidate-selection information.
* Champion mode fails closed.
* Missing artifacts do not silently become heuristics.
* Rollback restores the complete stack.
* No automatic promotion exists.

### Learning

* Current-session outcomes do not affect current-session models.
* Labels are generated only after settlement.
* Dynamic weights update only from settled sessions.
* Model training is session grouped.
* Candidate rows remain snapshot grouped.
* Outer tests remain untouched.
* Promotion compares complete deployments.

### Measurement

* Trades and no-trades are retained.
* Nonselected candidates are retained.
* Unfilled attempts are retained.
* Legacy and V3 disagreements are retained.
* Gate effectiveness is measurable.
* Abstention value is measurable.
* Candidate-ranking regret is measurable.
* Fill-model calibration is measurable.
* Deployment-level economic performance is measurable.

### Governance

* One deployment pointer controls the complete stack.
* Rule configuration and model artifacts are promoted together.
* Promotion packet includes rollback target, known weaknesses, and unsupported slices.
* Promotion requires explicit human approval.

---

## 24. Explicit Non-Goals

This project does **not** authorize:

* Live broker order placement.
* Autonomous champion promotion.
* Reinforcement learning using live capital.
* Intraday retraining from unresolved outcomes.
* Deep neural networks solely for complexity.
* Deletion of the V1 baseline.
* Removal of hard operational controls.
* Assumption of midpoint fills.
* Treating synthetic results as proof of live edge.
* Treating tick count as independent sample size.
* Using future-revised market data.

---

## 25. Coding-Agent Operating Instructions

1. Refresh the repository and record the starting commit before editing.
2. Read the existing tests for every module before changing its contract.
3. Implement PRs in the specified order.
4. Do not combine runtime integration and champion promotion in one PR.
5. Preserve backward compatibility until the new authority router is tested.
6. Add migrations before writing new persistence fields.
7. Do not silently catch critical-path exceptions.
8. Seed every stochastic component.
9. Record all fallback behavior.
10. Do not replace missing values with zero unless the field contract explicitly defines zero.
11. Do not allow forecast models to consume decision outputs.
12. Do not allow current-session labels into current-session learning.
13. Do not split market sessions across folds.
14. Do not split candidates from one snapshot across folds.
15. Do not compare systems under different fill assumptions.
16. Do not call heuristic output a trained V2 or V3 model.
17. Do not allow incomplete deployment bundles into candidate or champion modes.
18. Run the full test suite after every PR.
19. Record the exact test command and result in every PR description.
20. Document every deviation from this specification.

---

## 26. Recommended Default Behavior

Until the entire integration passes acceptance:

| Setting | Value |
|---|---|
| Deployment mode | `shadow` |
| Authority source | `legacy` |
| Fallback policy | `abstain` |
| Heuristic forecast | allowed only as labeled baseline |
| V3 paper authority | disabled |
| Live order authority | disabled |
| Automatic promotion | disabled |
| Legacy directional tilt | baseline-only |

After sufficient complete-session evidence:

```
shadow → advisory → candidate dual-paper comparison → human-reviewed champion
```

Do not skip candidate dual-paper evaluation.

---

## 27. Definition of Done

The project is done when the repository has stopped behaving like three partially connected versions and instead behaves like one versioned decision platform:

* V1 supplies deterministic structure, safety, candidates, and fallback.
* V2 supplies learned base forecasts.
* V3 supplies statistically validated advanced forecasts, candidate economics, ranking, abstention, drift control, and deployment governance.
* Every path is evaluated from the same data and candidate universe.
* Every decision is reproducible.
* Every rejected trade remains measurable.
* Every promoted stack can be rolled back atomically.
* The learning loop improves components without allowing the system to rewrite itself intraday or silently promote unproven behavior.

**The next implementation priority is runtime integration, not additional model complexity.**

**First production-facing milestone:**

> A full V3 shadow decision generated inside `ShadowRunner` from trained registry artifacts, using the same canonical snapshot and candidate universe as V1, persisted as a unified decision record, displayed on the dashboard, and replayable deterministically end to end.

**Next useful step:** convert this handoff into individual GitHub issues with exact file-level tasks and PR acceptance checklists.
