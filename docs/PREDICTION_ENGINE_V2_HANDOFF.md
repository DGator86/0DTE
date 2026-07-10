# 0DTE Prediction Engine V2

**Technical Handoff and Implementation Specification**

- **Repository:** DGator86/0DTE
- **Baseline branch:** main
- **Baseline commit reviewed:** e9fc34a73eb8b2816738842d4948912ac46d293f
- **Document date:** July 10, 2026
- **Status:** Proposed architecture and implementation specification
- **Audience:** Repository owner, quantitative developer, data engineer, model-validation reviewer
- **Primary objective:** Convert the current deterministic rules engine into a properly validated probabilistic prediction and candidate-ranking system without discarding its existing risk controls, option-chain analytics, journaling, or champion/challenger infrastructure.

---

## 1. Executive Summary

The existing repository has unusually strong infrastructure for an early-stage 0DTE research system:

* Risk-neutral density extraction from the live options chain.
* Physical-density approximation using realized volatility.
* Multi-timeframe feature construction.
* Regime classification and structure routing.
* Concrete option-spread enumeration and payoff evaluation.
* Hard gate separation from candidate selection.
* Journaling of trades and hypothetical no-trades.
* Walk-forward testing.
* Parameter optimization.
* Champion/challenger configuration management.
* Human-reviewed promotion rather than automatic live mutation.

The live loop already combines the regime-routing layer with the option-chain selector and records every evaluation. The journal explicitly treats no-trades as first-class observations, allowing blocked candidates to be settled hypothetically.

The central weakness is not infrastructure. It is that the prediction layer is still primarily a collection of structured assumptions:

1. Features are standardized with fixed or slowly adaptive transforms.
2. Hand-set weights create regime scores.
3. A hand-authored 27-cell table selects the structure.
4. Direction is an equal-weighted composite of several indicators.
5. Fixed thresholds determine bull, bear, or neutral.
6. A fixed directional shift is inserted into the physical distribution after the rule engine selects a direction.
7. Option candidates are ranked with a hand-authored multiplicative formula.
8. Validation frequently treats correlated intraday ticks as though they were independent observations.

The repository itself accurately describes the regime classifier as deterministic and not machine-learned. Direction is currently constructed from an equal-weighted feature composite, then blended with fixed 40% fast and 60% slow weights and classified using fixed 58/42 thresholds.

Prediction Engine V2 will retain the deterministic system as a baseline and fallback while introducing:

* Session-safe data construction.
* Multi-horizon, strategy-relevant targets.
* Train-only, timeframe-specific feature normalization.
* Calibrated directional, volatility, range-survival, and barrier-touch probabilities.
* An independently generated physical distribution.
* Realistic execution-cost estimates.
* Candidate-level expected-utility ranking.
* Session-level nested walk-forward validation.
* Explicit model uncertainty.
* Shadow deployment and human-controlled promotion.

The first objective is not to produce a more complicated model. It is to make the evidence reliable enough that the system cannot mistake correlated observations, midpoint pricing, or circular assumptions for predictive edge.

---

## 2. Existing System: What Must Be Preserved

### 2.1 Risk-neutral density extraction

`rnd_extractor.py` recovers the forward and discount factor from put-call parity, fits total variance over log-moneyness, reconstructs a smooth call curve, and differentiates that curve to obtain a risk-neutral density. This is substantially more robust than differentiating raw option prices.

This component remains part of the system.

It should continue to provide:

* Risk-neutral probability density.
* Forward.
* Implied distribution standard deviation.
* Skew.
* Excess kurtosis.
* Terminal strike probabilities.
* Option-chain quality metrics.
* Physical-versus-risk-neutral richness measurements.

### 2.2 Option candidate generation

`spread_selector.py` provides a uniform leg representation and enumerates defined-risk and directional structures. Candidate payoffs, Greeks, maximum loss, expected value, probability of profit, liquidity, wall safety, gamma safety, and touch safety are calculated consistently.

This component remains the candidate generator.

Its ranking responsibility will gradually move to a candidate-value model, but candidate generation and payoff mathematics should remain centralized here or in a dedicated successor module.

### 2.3 Independent gate and selector results

`decision_engine.py` intentionally runs the gate and selector independently. It records the would-be candidate on no-trade evaluations so that settlement can later determine whether the gate helped or hurt.

This behavior must not be removed.

Prediction Engine V2 must continue to distinguish:

* No candidate existed.
* A candidate existed but failed selector constraints.
* A candidate existed but failed a structural gate.
* A candidate existed but failed risk-management approval.
* A candidate passed but was not executed.
* A candidate was executed.
* A candidate would have performed well or poorly in hindsight.

### 2.4 Hard risk controls

The current gate architecture separates hard vetoes from weighted setup scores.

Hard vetoes should remain outside the statistical model when they represent an explicit operating restriction, such as:

* No new positions after a configured lockout time.
* No trading during a prohibited catalyst window.
* Maximum daily risk reached.
* Invalid or stale market data.
* Missing chain.
* Broken option-arbitrage checks.
* Liquidity below a minimum executable threshold.
* Broker or system state unavailable.

Market hypotheses such as "short gamma always forbids premium selling" should be treated differently. They may remain conservative guardrails initially, but they must be journaled and empirically tested rather than permanently treated as unquestionable facts.

### 2.5 Audit trail and champion/challenger process

The repository already supports candidate configurations, a live champion, per-regime overrides, audit metadata, and human-reviewed promotion. Only the promotion workflow is intended to modify the live champion.

Prediction Engine V2 must integrate with this framework rather than introducing an untracked model file that silently becomes live.

---

## 3. Problems to Solve

### 3.1 False effective sample size

The current directional readout evaluates every settled tick with a resolved direction against the move from that tick to settlement.

A prediction at 10:00 and a prediction at 10:01 share almost the entire future path. They are not independent experiments.

Consequences:

* Confidence intervals are too narrow.
* Statistical significance is overstated.
* Hyperparameter searches can exploit a small number of strongly trending days.
* One market event can contribute hundreds of apparently successful predictions.
* A result based on 5,000 ticks may represent only 15–20 independent sessions.

The primary reporting unit must therefore be the trading session.

Tick-level observations may still be used for training, but:

* Cross-validation grouping must be by session.
* Confidence intervals must be bootstrapped by session.
* Promotion rules must include a minimum number of independent sessions.
* Performance must be reported both per tick and per session.
* No train/test fold may split a session.

### 3.2 Tick-index walk-forward boundaries

The current walk-forward fold builder divides the timeline by observation index.

That allows:

* Morning ticks from a session to enter warm-up.
* Afternoon ticks from the same session to enter test.
* Model state from the same day to influence both sides.
* Labels with common terminal settlement to straddle boundaries.

Walk-forward folds must be built from complete session-date groups.

### 3.3 Silently swallowed failures

The current fold runner catches broad exceptions during both warm-up and test execution and continues without recording the error.

This can create survivorship bias because difficult or malformed observations can disappear without affecting the reported metrics.

Every failed tick must create a structured error record. A fold must be rejected when failure coverage exceeds an allowed threshold.

### 3.4 Prediction and policy are mixed together

The current decision matrix simultaneously expresses:

* A belief about future price behavior.
* A belief about volatility.
* A strategy preference.
* A position-size preference.
* A strike-placement instruction.

These are different concerns.

Prediction Engine V2 must separate:

1. Forecast: What is likely to happen?
2. Policy: Given that forecast, which strategy is appropriate?
3. Candidate ranking: Which exact spread offers the best net utility?
4. Risk control: Is the exposure allowed?
5. Sizing: How much risk should be allocated?

### 3.5 Directional density circularity

The live loop currently applies a signed drift to the physical density when a directional structure has already been selected. The drift magnitude depends on the routed direction and conviction.

This creates a circular process:

1. Rules select a bullish direction.
2. The physical distribution is shifted bullish.
3. Bullish candidates receive higher expected values.
4. Those expected values appear to validate the bullish decision.

The selected structure must never determine the physical forecast used to justify that structure.

The physical forecast must be generated independently from the feature state. The policy should consume the forecast, not create it.

### 3.6 Feature normalization mixes timeframes

The matrix scaler updates statistics using a feature name that is shared across native timeframes.

For example, a one-minute EMA slope and a daily EMA slope can influence the same scale estimate even though their distributions are fundamentally different.

The current observation is also entered into the scaler before its standardized value is calculated, creating a small contemporaneous influence.

Each native feature must be normalized by:

```
feature name + timeframe + optional time-of-day bucket
```

The observation must be scored against historical state before it updates that state.

### 3.7 Lifetime normalization adapts too slowly

The existing ScaleBook uses accumulated Welford statistics.

Lifetime statistics are useful for stable engineering measurements but are often inappropriate for market data because:

* Volatility regimes shift.
* Intraday liquidity changes.
* Market microstructure changes.
* High-volatility periods can permanently widen the scale.
* Old behavior retains the same weight as recent behavior.

V2 should use rolling or exponentially decayed robust statistics.

### 3.8 Terminal touch approximation is insufficient

The RND uses approximately twice the terminal beyond-strike probability as a touch estimate and explicitly labels this a sanity bound rather than a barrier model.

That approximation is insufficient for:

* Short-strike stop-outs.
* Intraday wall tests.
* Touch-and-recovery behavior.
* First-passage order.
* Candidate-level maximum adverse excursion.
* Position-management rules.

A calibrated path model is required.

### 3.9 Midpoint P&L is not executable P&L

Candidate prices are calculated from option mids. Liquidity is then applied as a score multiplier.

A liquidity multiplier does not convert midpoint EV into executable EV.

V2 must calculate explicit:

* Midpoint price.
* Natural price.
* Expected fill price.
* Conservative fill price.
* Estimated fees.
* Expected exit slippage.
* Stop slippage.
* Quote-age penalty.

### 3.10 Candidate ranking double-counts correlated risks

The current ranking formula multiplies EV per risk by liquidity, wall, gamma, touch, and family multipliers.

Many of these inputs are correlated:

* EV already reflects terminal tail outcomes.
* Touch probability and gamma safety both penalize downside path risk.
* Wall safety and touch safety may be different representations of the same geometry.
* Family weights may duplicate characteristics already expressed through maximum loss and liquidity.

The existing formula should remain as a baseline, but a candidate-level model must learn which characteristics predict net realized P&L.

---

## 4. V2 Goals and Non-Goals

### 4.1 Goals

Prediction Engine V2 will:

1. Produce calibrated probabilities and conditional return forecasts.
2. Predict multiple horizons rather than only settlement direction.
3. Estimate path-dependent outcomes.
4. Rank exact option candidates by net expected utility.
5. Use realistic fill assumptions.
6. Validate by independent sessions.
7. Preserve all no-trade observations.
8. Quantify uncertainty.
9. Maintain deterministic replay.
10. Remain human-reviewable.
11. Support shadow operation before any production effect.
12. Preserve current hard operational safeguards.
13. Integrate with the existing champion/challenger process.
14. Allow the current rules engine to remain available as a baseline and fallback.

### 4.2 Non-goals

The initial V2 implementation will not:

* Place live orders.
* Use a deep neural network.
* Depend on alternative data that is not available historically.
* Automatically promote a model into production.
* Optimize solely for total P&L.
* Remove all rules at once.
* Assume midpoint fills.
* Treat synthetic-market results as proof of live edge.
* Use future-revised market data.
* Train on observations whose source timestamps occur after the prediction timestamp.

---

## 5. Target Architecture

```
Live and recorded market data
        |
        v
AsOfSnapshotBuilder
        |
        +--> raw feature snapshot
        +--> quality and staleness metadata
        +--> chain snapshot
        |
        v
FeaturePipeline
        |
        +--> raw features
        +--> lagged robust standardized features
        +--> missingness indicators
        |
        v
PredictionService
        |
        +--> direction probabilities
        +--> return quantiles
        +--> realized-move forecast
        +--> range-survival probabilities
        +--> wall/flip touch probabilities
        +--> uncertainty
        |
        v
PhysicalDistributionBuilder
        |
        +--> distribution independent of selected trade
        |
        v
PolicyRouter
        |
        +--> permitted structure families
        +--> directional side
        +--> policy confidence
        |
        v
CandidateGenerator
        |
        +--> all feasible candidate structures
        |
        v
ExecutionCostModel
        |
        +--> expected executable entry/exit prices
        |
        v
CandidateValueModel
        |
        +--> expected net P&L
        +--> probability of profit
        +--> expected shortfall
        +--> fill uncertainty
        +--> expected utility
        |
        v
HardRiskGate
        |
        v
PositionSizer
        |
        v
TradeDecision
        |
        v
Journal + prediction audit + settlement
```

### 5.1 Separation of responsibilities

**Forecast layer**

Answers:

* What is the expected return?
* What is the probability of moving up or down?
* What is the likely realized move?
* What is the probability of touching each structural level?
* What is the probability that a candidate range survives?
* How uncertain is the model?

**Policy layer**

Answers:

* Is premium selling, directional debit, or long volatility appropriate?
* Which structure families may be considered?
* Is the forecast actionable?
* Is there enough confidence to trade?

**Candidate layer**

Answers:

* Which exact strikes and widths produce the best net utility?
* What price is realistically fillable?
* What is the candidate's downside distribution?
* How much capital is consumed?

**Risk layer**

Answers:

* Is the trade allowed?
* Does it violate daily or portfolio constraints?
* Is the market data reliable enough?
* Is the position too large?
* Is the trade too late?
* Is a prohibited event active?

---

## 6. Core Data Contract: PredictionBundle

Create:

```
prediction/contracts.py
```

### 6.1 Data class

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PredictionBundle:
    snapshot_id: str
    ts: str
    session_date: str
    symbol: str
    # Direction probabilities
    p_up_5m: Optional[float] = None
    p_up_15m: Optional[float] = None
    p_up_30m: Optional[float] = None
    p_up_60m: Optional[float] = None
    p_up_close: Optional[float] = None
    # Continuous return forecasts
    expected_return_15m: Optional[float] = None
    expected_return_30m: Optional[float] = None
    expected_return_60m: Optional[float] = None
    expected_return_close: Optional[float] = None
    return_q10_30m: Optional[float] = None
    return_q50_30m: Optional[float] = None
    return_q90_30m: Optional[float] = None
    return_q10_close: Optional[float] = None
    return_q50_close: Optional[float] = None
    return_q90_close: Optional[float] = None
    # Volatility and range
    expected_realized_move_30m: Optional[float] = None
    expected_realized_move_close: Optional[float] = None
    p_range_survive_15m: Optional[float] = None
    p_range_survive_30m: Optional[float] = None
    p_range_survive_60m: Optional[float] = None
    p_range_survive_close: Optional[float] = None
    # Structural barrier events
    p_touch_call_wall_30m: Optional[float] = None
    p_touch_put_wall_30m: Optional[float] = None
    p_touch_gamma_flip_30m: Optional[float] = None
    p_touch_call_wall_close: Optional[float] = None
    p_touch_put_wall_close: Optional[float] = None
    p_cross_gamma_flip_close: Optional[float] = None
    # First-passage ordering
    p_call_wall_first: Optional[float] = None
    p_put_wall_first: Optional[float] = None
    p_neither_wall_close: Optional[float] = None
    # Model quality
    uncertainty: Optional[float] = None
    data_quality: Optional[float] = None
    feature_coverage: Optional[float] = None
    feature_version: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, float | str] = field(default_factory=dict)
```

### 6.2 Contract rules

Every probability must:

* Be between 0.0 and 1.0.
* Be None when the required inputs are unavailable.
* Include a model version.
* Be generated only from information available at or before ts.
* Be calibrated on training data that excludes the current session.
* Be reproducible in replay mode.

Every return must:

* Be represented as a decimal return.
* State whether it is log return or simple return.
* Use one convention throughout the repository.
* Be None when the horizon extends beyond market close.

Recommended convention:

```
log_return = ln(future_price / current_price)
```

### 6.3 Independence rule

PredictionBundle must not receive:

* Selected structure.
* Selected family.
* Selected strikes.
* Policy conviction.
* Gate result.
* Candidate score.

The forecast must be created before the policy and candidate-selection stages.

---

## 7. Structural Market State Contract

Create:

```
prediction/structural_state.py
```

```python
@dataclass(frozen=True)
class StructuralState:
    ts: str
    spot: float
    net_gex_oi: Optional[float]
    gamma_flip_oi: Optional[float]
    call_wall_oi: Optional[float]
    put_wall_oi: Optional[float]
    net_gex_volume: Optional[float]
    gamma_flip_volume: Optional[float]
    call_wall_volume: Optional[float]
    put_wall_volume: Optional[float]
    net_gex_hybrid: Optional[float]
    gamma_flip_hybrid: Optional[float]
    call_wall_hybrid: Optional[float]
    put_wall_hybrid: Optional[float]
    gex_concentration: Optional[float]
    wall_stability: Optional[float]
    flip_velocity: Optional[float]
    call_wall_velocity: Optional[float]
    put_wall_velocity: Optional[float]
    quality_score: float
    source_ages: dict[str, float]
```

The policy may consume StructuralState. The statistical forecast may also use its fields as features.

No single GEX calculation should be silently treated as ground truth.

---

## 8. Dataset Construction Specification

Create:

```
prediction/dataset.py
prediction/labels.py
prediction/asof.py
prediction/storage.py
```

### 8.1 Observation unit

One feature observation represents:

```
symbol + session_date + decision timestamp
```

Default timestamp cadence:

* One observation per minute.
* Optional decision cadence of 5 minutes for live policy.
* One-minute observations remain available for model training and barrier labels.

Each observation receives a stable snapshot_id:

```
SHA256(symbol | normalized timestamp | feature version | source sequence)
```

### 8.2 Session identity

Use exchange-local session dates in America/New_York.

The dataset builder must distinguish:

* Regular trading hours.
* Early-close sessions.
* Holidays.
* Missing sessions.
* Data-provider outages.

Session metadata must include:

```
session_open
session_close
is_early_close
minutes_since_open
minutes_to_close
day_of_week
```

### 8.3 As-of rules

A feature may be included only when its source timestamp is less than or equal to the observation timestamp.

Examples:

* A one-minute bar ending at 10:01 may be used for a 10:01 observation.
* A bar ending at 10:02 may not.
* An option quote timestamped 10:01:07 may not be used for a 10:01:00 observation.
* Open interest must carry the publication date actually available at that time.
* Scheduled catalyst information may be used only if the event schedule was known before the observation.
* End-of-day official settlement data is label-only.
* Revised economic data is not permitted unless the historical first-release value is available.

### 8.4 Feature table

Create a canonical training table with at least:

```
snapshot_id
symbol
session_date
ts
minutes_since_open
minutes_to_close
spot
feature_version
raw_features_json
standardized_features_json
missingness_json
source_ages_json
quality_json
```

For efficient research, materialize a columnar version in Parquet.

Recommended location:

```
data/derived/features/version=<feature_version>/session_date=YYYY-MM-DD/*.parquet
```

### 8.5 Preserve both raw and standardized values

Do not train exclusively on the 0–100 matrix.

Store:

* Raw feature.
* Standardized feature.
* Missingness flag.
* Data age.
* Source-quality score.

This allows the model to determine whether the hand-authored transform is useful.

### 8.6 Feature groups

**Price geometry**

* Distance to session VWAP.
* Distance to rolling VWAP.
* VWAP slope.
* Session range position.
* Overnight gap.
* Distance from open.
* Distance from prior close.
* Distance from prior day high and low.
* Distance from overnight high and low.
* Distance from gamma flip.
* Distance from call and put walls.
* Distance normalized by expected remaining move.

**Trend and momentum**

* ADX.
* Positive and negative DI.
* DI spread.
* EMA slopes by timeframe.
* RSI by timeframe.
* Return over 1, 3, 5, 15, 30, and 60 minutes.
* Trend cleanliness.
* Fast-minus-slow direction composite.
* Fast-composite velocity.
* Slow-composite velocity.
* Fast/slow crossover state.

**Volatility**

* Realized volatility by horizon.
* Realized-volatility percentile.
* Short/long realized-volatility ratio.
* Bollinger width and expansion.
* Keltner width and trend strength.
* Donchian width and breakout.
* Implied remaining move.
* Realized move already consumed.
* VIX9D/VIX.
* VIX/VIX3M.
* VVIX level and change.
* Risk-neutral variance.
* Physical forecast variance.
* Implied-versus-realized variance ratio.

**Dealer structure**

* OI GEX.
* Volume GEX proxy.
* Hybrid GEX.
* GEX percentile.
* GEX concentration by strike.
* Flip level.
* Wall levels.
* Wall concentration.
* Flip velocity.
* Wall velocity.
* Distance from flip and walls.
* Structural disagreement across GEX variants.

**Order flow**

* Real signed-volume CVD where available.
* CVD slope.
* CVD acceleration.
* Put/call volume ratio.
* Volume/open-interest ratio.
* Quote imbalance where available.
* Bid/ask spread state.
* Trade aggressor imbalance where available.
* $TICK and TRIN where available.

**Breadth**

* RSP versus SPY divergence.
* Sector alignment.
* Top-ten pressure.
* Advance/decline metrics where available.
* Equal-weight versus cap-weight momentum.

**Time context**

* Minute of session.
* Sine/cosine encoding of time.
* Day of week.
* Expiration type.
* Early-close indicator.
* Scheduled catalyst proximity.
* Minutes to catalyst.

**Data quality**

* Chain age.
* Bar age.
* Number of usable strikes.
* RND arbitrage violation.
* Quote coverage.
* Feature coverage.
* Feed source.
* Provider failover state.

### 8.7 Missingness

Missing values must not be silently replaced with neutral values for model training.

For every nullable feature:

```
feature_value
feature_is_missing
feature_age
```

The rules engine may continue graceful degradation, but the statistical model must know whether a neutral-looking value was observed or imputed.

---

## 9. Label Specification

### 9.1 Continuous forward-return labels

For each horizon h in:

```
5, 15, 30, 60 minutes and session close
```

calculate:

```
forward_return_h = log(mid_or_spot_at_t_plus_h / spot_at_t)
```

Use:

* The first valid underlying midpoint at or after the horizon boundary.
* A maximum tolerance of one base bar.
* None when no valid observation exists.
* No horizon that extends past the session close.

### 9.2 Direction labels

Store the raw return and multiple classification labels.

**Binary raw-direction label**

```
up_h = 1 if forward_return_h > 0 else 0
```

This is useful for comparability but may overemphasize economically meaningless moves.

**Actionable-direction label**

```
threshold_h = max(
    configured_min_return,
    implied_remaining_move * configured_fraction,
    estimated_round_trip_cost_equivalent
)

direction_h =
    +1 if forward_return_h > threshold_h
    -1 if forward_return_h < -threshold_h
     0 otherwise
```

Initial defaults:

```
configured_min_return = 0.0002
configured_fraction = 0.05
```

These are research defaults, not permanent truths.

### 9.3 Maximum favorable and adverse excursion

For each horizon:

```
up_mfe_h = log(max_high_between_t_and_h / spot_t)
up_mae_h = log(min_low_between_t_and_h / spot_t)
down_mfe_h = -up_mae_h
down_mae_h = -up_mfe_h
```

These labels support:

* Target placement.
* Stop placement.
* Candidate path risk.
* Directional strategy evaluation.

### 9.4 Volatility labels

Calculate:

```
realized_variance_h
realized_volatility_h
absolute_return_h
high_low_range_h
maximum_intrahorizon_move_h
```

The primary remaining-session target should be:

```
remaining_realized_move = max(
    abs(high_remaining / spot_t - 1),
    abs(low_remaining / spot_t - 1)
)
```

### 9.5 Wall and flip labels

Structural levels must be frozen at their values at observation time.

For each horizon:

```
touch_call_wall_h
touch_put_wall_h
touch_gamma_flip_h
cross_gamma_flip_h
call_wall_first_h
put_wall_first_h
neither_wall_h
time_to_call_wall
time_to_put_wall
time_to_flip
```

Do not use future-updated wall or flip levels when creating these labels.

### 9.6 Range-survival labels

For a generic structural range:

```
lower_bound = put wall or proposed lower short strike
upper_bound = call wall or proposed upper short strike
```

Define:

```
range_survive_h = 1
if every underlying bar from t through h remains strictly within the bounds
else 0
```

Maintain separate labels for:

* Wall-channel survival.
* Candidate-short-strike survival.
* Candidate-breakeven survival.

### 9.7 First-passage labels

For each target/stop pair:

```
target_first
stop_first
neither
time_to_first_event
```

When the same bar contains both levels:

1. Use higher-frequency data when available.
2. Otherwise mark ambiguous_same_bar.
3. Do not assume the favorable outcome.
4. Conservative economic evaluation should assign the adverse event first.

### 9.8 Candidate outcome labels

For every generated candidate, store:

* P&L at settlement using midpoint entry.
* P&L at settlement using expected fill.
* P&L at settlement using conservative fill.
* P&L under the configured exit policy.
* Maximum favorable P&L during holding period.
* Maximum adverse P&L during holding period.
* Whether a stop would trigger.
* Whether a target would trigger.
* Which triggered first.
* Time in trade.
* Maximum capital usage.
* Realized fees.
* Estimated slippage.
* Net P&L.
* Return on risk.
* Return on capital.
* Expected shortfall contribution.

---

## 10. Feature Normalization Specification

Create:

```
prediction/scalers.py
```

### 10.1 Required keying

Native features must use:

```
key = f"{feature_name}:{timeframe}"
```

Optional time-of-day normalization:

```
key = f"{feature_name}:{timeframe}:{time_bucket}"
```

Snapshot features use:

```
key = feature_name
```

### 10.2 Score-before-update rule

Correct order:

```
state = scaler.get_state(key)
standardized = scaler.transform(value, state)
scaler.update(key, value)
```

Incorrect order:

```
scaler.update(key, value)
standardized = scaler.transform(value)
```

The current observation must not influence the historical scale used to score itself.

### 10.3 Robust exponentially weighted scaler

Recommended state:

```python
@dataclass
class RobustScaleState:
    n: int
    ew_mean: float
    ew_var: float
    ew_abs_dev: float
    lower_clip: float
    upper_clip: float
    last_updated: str
```

Initial implementation may use exponentially weighted mean and variance.

Preferred later implementation:

* Exponentially weighted median approximation.
* Exponentially weighted median absolute deviation.
* Winsorization at historical robust bounds.

### 10.4 Half-life defaults

Research defaults:

```
1m and 5m features: 5 trading sessions
15m and 30m features: 10 sessions
1h features: 20 sessions
4h and 1d features: 60 sessions
snapshot dealer/vol features: 20 sessions
```

These defaults must remain configurable and subject to walk-forward validation.

### 10.5 Cold start

Before minimum history:

* Use the current fixed prior.
* Record scale_source = "prior".
* Ramp reliability based on sample count.
* Do not represent an unwarmed estimate as a strongly held neutral value.
* Preserve the existing persisted-state behavior.

### 10.6 Scaler persistence

Persist:

* State version.
* Feature version.
* Last source timestamp.
* Key-specific state.
* Hash of scaler configuration.

Reject incompatible state instead of silently loading it under a different feature definition.

---

## 11. Model Suite

Add scikit-learn as the initial model dependency.

Do not begin with deep learning. The number of independent market sessions will be much smaller than the number of tick rows.

### 11.1 DirectionModel

Create:

```
prediction/models/direction.py
```

**Targets**

Separate models for:

* up_5m
* up_15m
* up_30m
* up_60m
* up_close

**Baseline model**

Elastic-net logistic regression.

Initial hyperparameters:

```
penalty = elasticnet
solver = saga
C grid = [0.01, 0.05, 0.1, 0.5, 1.0]
l1_ratio grid = [0.0, 0.25, 0.5, 0.75, 1.0]
class_weight = balanced or None, selected in inner validation
```

**Challenger**

HistGradientBoostingClassifier.

Initial search:

```
learning_rate = [0.02, 0.05, 0.1]
max_leaf_nodes = [7, 15, 31]
max_depth = [2, 3, None]
min_samples_leaf = [50, 100, 250]
l2_regularization = [0.0, 0.1, 1.0, 10.0]
```

**Output**

```
raw probability
calibrated probability
decision threshold
model uncertainty
```

**Required baselines**

Compare against:

* Always-up base-rate prediction.
* Previous-return sign.
* Existing 58/42 direction composite.
* Existing bull/bear rule.
* Random or climatology probability.

### 11.2 ReturnQuantileModel

Create:

```
prediction/models/return_quantiles.py
```

Use quantile regression for:

```
q10
q50
q90
```

Targets:

* 30-minute return.
* 60-minute return.
* Return to close.

Initial model:

```
HistGradientBoostingRegressor(loss="quantile")
```

Requirements:

* Quantiles must be monotonically ordered after prediction.
* Apply rearrangement when q10 > q50 or q50 > q90.
* Evaluate pinball loss and interval coverage.
* Record coverage separately by regime and time of day.

### 11.3 VolatilityModel

Create:

```
prediction/models/volatility.py
```

Targets:

* Future realized variance.
* Maximum remaining move.
* Absolute return.
* Remaining-session high-low range.

Use log-transformed positive targets:

```
target = log(realized_measure + epsilon)
```

Output:

* Expected realized move.
* Quantile range.
* Uncertainty.
* Ratio of forecast realized move to implied remaining move.

### 11.4 RangeSurvivalModel

Create:

```
prediction/models/range_survival.py
```

Predict:

* Wall-channel survival.
* Candidate-short-strike survival.
* Candidate-breakeven survival.

Horizons:

```
15, 30, 60 minutes and close
```

Inputs must include:

* Distance to each boundary.
* Forecast volatility.
* Time remaining.
* GEX state.
* Wall stability.
* Current trend and flow.
* Expected move consumed.
* Barrier width normalized by volatility.

### 11.5 BarrierTouchModel

Create:

```
prediction/models/barrier_touch.py
```

Predict:

* Call wall touched before close.
* Put wall touched before close.
* Gamma flip crossed.
* Call wall first versus put wall first.
* Candidate stop before target.

This model may initially be a supervised classifier and later incorporate path-simulation outputs as features.

### 11.6 CandidateValueModel

Create:

```
prediction/models/candidate_value.py
```

One row represents one candidate at one snapshot.

**Inputs**

* PredictionBundle outputs.
* Candidate family.
* Leg geometry.
* Width.
* Debit or credit.
* Expected fill.
* Mid-to-natural spread.
* Maximum loss.
* Capital.
* Delta.
* Gamma.
* Theta.
* Vega where available.
* Probability of profit under the physical forecast.
* Predicted path-touch probability.
* Wall distances.
* Flip distances.
* Time remaining.
* Model uncertainty.
* Data quality.
* Current heuristic score.

**Targets**

Primary:

```
net realized P&L under configured policy
```

Secondary:

```
probability of positive net P&L
expected shortfall
maximum adverse excursion
stop-out probability
```

**Initial model**

Start with:

* Elastic-net regression for net P&L.
* Logistic regression for probability of profit.
* Quantile regression for downside P&L.
* Gradient-boosted challengers.

**Grouping requirement**

Candidates from the same:

```
snapshot_id
session_date
```

must never be divided between training and test.

**Ranking output**

```python
@dataclass(frozen=True)
class CandidateForecast:
    candidate_id: str
    expected_net_pnl: float
    p_profit: float
    pnl_q10: float
    pnl_q50: float
    pnl_q90: float
    expected_shortfall: float
    fill_uncertainty: float
    model_uncertainty: float
    utility_score: float
```

### 11.7 Probability calibration

Create:

```
prediction/calibration.py
```

Default:

* Sigmoid/Platt calibration.

Use isotonic calibration only when:

* Training sample is sufficiently large.
* There are enough independent sessions.
* Inner validation shows stable improvement.
* Calibration does not become stepwise and unstable.

Calibration must occur within training data only.

Never fit a calibrator on the final test or holdout period.

---

## 12. Physical Distribution V2

Create:

```
prediction/physical_distribution.py
```

### 12.1 New input object

```python
@dataclass(frozen=True)
class PhysicalForecast:
    expected_return: float
    return_q10: float
    return_q50: float
    return_q90: float
    expected_realized_move: float
    volatility_scale: float
    skew_adjustment: float
    uncertainty: float
    model_version: str
```

### 12.2 Construction requirements

The physical density must be created from:

* Risk-neutral density shape.
* Independently predicted mean return.
* Independently predicted realized variance.
* Optional independently predicted skew adjustment.
* Time remaining.

It must not depend on:

* Routed structure.
* Candidate family.
* Candidate direction.
* Gate outcome.
* Hand-authored conviction size.

### 12.3 Initial transformation

Phase-one implementation may:

1. Extract the RND.
2. Center it.
3. Scale deviations to match forecast physical standard deviation.
4. Shift the mean to match predicted return.
5. Renormalize.
6. Clip numerical negatives.
7. Report moments and transformation quality.

Conceptually:

```
transformed_price = predicted_mean_price + scale * (
    rnd_price - rnd_mean_price
)
```

The transformed density must then be interpolated back onto the standard grid and normalized.

### 12.4 Uncertainty treatment

When prediction uncertainty is high:

* Blend the physical forecast toward the risk-neutral distribution.
* Reduce directional mean shift.
* Widen prediction intervals.
* Decrease policy actionability.
* Never increase confidence because the model is uncertain.

Example:

```
confidence_weight = 1.0 - uncertainty
physical_pdf = (
    confidence_weight * forecast_pdf
    + uncertainty * risk_neutral_pdf
)
```

### 12.5 Removal of circular tilt

Deprecate:

```
RNDConfig.dir_drift_frac
```

Migration sequence:

1. Keep current tilt available behind prediction.use_legacy_directional_tilt.
2. Run V2 physical forecast in shadow mode.
3. Compare EV calibration.
4. Disable legacy tilt after V2 passes promotion criteria.
5. Remove the field only after backward-compatible config migration.

---

## 13. Execution Cost Model

Create:

```
execution_cost.py
```

### 13.1 Price definitions

For every multi-leg candidate calculate:

```
mid_price
natural_price
expected_fill_price
conservative_fill_price
```

Credit trade:

```
expected_credit = mid_credit - fill_fraction * half_spread_cost
```

Debit trade:

```
expected_debit = mid_debit + fill_fraction * half_spread_cost
```

Where fill_fraction is between:

```
0.0 = midpoint
1.0 = full natural-price concession
```

### 13.2 Initial fill model

Before real fill records exist, use configurable priors based on:

* Number of legs.
* Relative spread.
* Option price.
* Time of day.
* Chain age.
* Quote depth if available.
* Structure family.
* Underlying volatility.

Suggested conservative defaults:

```
single-leg liquid option: 0.35 of midpoint-to-natural concession
two-leg vertical: 0.50
four-leg structure: 0.65
late-day or stale quote: additional penalty
```

These are operational priors only.

### 13.3 Empirical fill model

Paper and manual executions should record:

```
decision timestamp
submission timestamp
fill timestamp
quoted bid/ask by leg
strategy midpoint
strategy natural price
limit price
fill price
partial fills
cancellation
broker fees
```

Train:

```
fill_fraction ~ structure + spread + price + time + volatility + quote age
```

### 13.4 Exit costs

Calculate expected costs for:

* Profit-taking exits.
* Stop-loss exits.
* Close-to-expiration exits.
* Manual emergency exits.
* Multi-leg legging.
* Assignment avoidance.

### 13.5 Candidate evaluation

All V2 economic labels and predictions must use:

```
net executable P&L
```

Midpoint P&L remains a diagnostic only.

---

## 14. Candidate Ranking Policy

### 14.1 Baseline ranker

Preserve the current multiplicative score as:

```
legacy_candidate_score
```

The current system ranks debit candidates using EV per risk, liquidity, and family weight, and credit candidates using additional wall, gamma, and touch multipliers.

### 14.2 V2 utility function

Recommended initial utility:

```
utility = (
    expected_net_pnl
    - lambda_shortfall * expected_shortfall
    - lambda_fill * fill_uncertainty
    - lambda_model * model_uncertainty
    - lambda_capital * capital_penalty
)
```

Where:

```
expected_shortfall = expected loss in the worst configured tail
capital_penalty = capital consumed / portfolio risk budget
```

Do not use maximum Sharpe or maximum expected P&L alone.

### 14.3 Risk constraints before ranking

Candidates must first satisfy minimum constraints:

* Quote quality.
* Maximum quote age.
* Minimum expected executable premium.
* Maximum defined loss.
* Maximum capital.
* Minimum open-interest or volume coverage where required.
* No invalid arbitrage.
* No unsupported family.
* No prohibited undefined risk.
* No daily risk violation.

### 14.4 Family routing

The policy may define eligible families, but the candidate ranker decides among exact candidates within those families.

During shadow research, also generate candidates outside the routed family and settle them as counterfactuals. This measures whether the router adds value.

### 14.5 Required ranking diagnostics

For each snapshot report:

* Top candidate under legacy score.
* Top candidate under V2 utility.
* Rank disagreement.
* Expected net P&L difference.
* Realized winner in hindsight.
* Top-one uplift versus random feasible candidate.
* Top-one uplift versus legacy ranker.
* Within-snapshot Spearman rank correlation.

---

## 15. Path Model Specification

Create:

```
prediction/path_model.py
```

### 15.1 Baseline

Retain mc.py as a structured baseline. Its own documentation correctly states that its regime coefficients are guesses requiring journal calibration.

### 15.2 Empirical residual bootstrap

V2 should simulate paths using historical return blocks.

**Conditioning variables**

* Minute of session.
* Volatility quantile.
* GEX sign.
* Distance to gamma flip.
* Trend state.
* Expected-move-consumed bucket.
* Catalyst state.
* Day type.
* Remaining time.

**Sampling process**

1. Select historical sessions matching the conditioning state.
2. Extract one-minute standardized residual blocks.
3. Sample contiguous blocks of 5–15 minutes.
4. Rescale blocks to the current predicted volatility.
5. Add the independently forecast mean return.
6. Stitch blocks until the horizon.
7. Calculate target, stop, wall, flip, and range events.

Block sampling is required to preserve:

* Serial correlation.
* Volatility clustering.
* Momentum bursts.
* Mean-reversion runs.
* Intraday seasonality.

### 15.3 Outputs

```
P(target before stop)
P(stop before target)
P(neither)
P(call wall touched)
P(put wall touched)
P(flip crossed)
P(range survives)
distribution of terminal price
distribution of maximum adverse excursion
distribution of maximum favorable excursion
distribution of candidate P&L
```

### 15.4 Calibration

Compare predicted barrier probabilities with realized frequencies using:

* Brier score.
* Brier skill.
* Reliability bins.
* Calibration intercept.
* Calibration slope.
* Time-to-event error.
* Survival curves.

---

## 16. GEX Measurement Research Program

The repository already acknowledges that open-interest-based GEX is stale intraday and may miss same-day 0DTE positioning.

### 16.1 GEX variants

Implement parallel providers:

**Variant A: OI-only 0DTE**

Current baseline.

**Variant B: OI with nearest weekly expirations**

Include:

* Same-day expiration.
* Nearest one or two weekly expirations.
* Configurable time-decay weighting.

**Variant C: Intraday volume-weighted gamma proxy**

Approximate same-day positioning using:

* Contract volume.
* Option delta/gamma.
* Trade-side inference when available.
* Put/call sign scenarios.
* Decay by time since trade where available.

**Variant D: Hybrid**

Blend OI and volume proxy according to:

* Time of day.
* Volume/OI ratio.
* Feed quality.
* Estimated confidence.

### 16.2 GEX output contract

Every variant must output:

```
net_gex
gamma_flip
call_wall
put_wall
gex_concentration
wall_concentration
quality_score
assumption_set
source_age
```

### 16.3 Journal all variants

All variants should initially remain observation-only.

Record:

```
gex_oi_*
gex_weekly_*
gex_volume_*
gex_hybrid_*
```

### 16.4 Evaluation targets

Evaluate each GEX variant against:

* Directional continuation.
* Range survival.
* Call-wall touch.
* Put-wall touch.
* Flip crossing.
* Realized volatility.
* Candidate P&L.
* Gate effectiveness.

Do not select the winning variant solely by total trading P&L.

### 16.5 Sign-scenario testing

Because dealer positioning assumptions are uncertain, calculate at least:

* Standard dealer-short-customer-long convention.
* Alternative same-day put-flow convention.
* Confidence-weighted blended convention.

Disagreement itself may be predictive and should be stored as a feature.

---

## 17. Regime-System Consolidation

The current live path uses both the classifier and matrix-routing systems. The classifier creates stand-down and veto state, while the matrix separately chooses execution regime, context regime, direction, and structure.

### 17.1 Transitional approach

Do not delete either system initially.

Create:

```
policy/router.py
```

with two implementations:

```python
class LegacyMatrixPolicy:
    ...

class PredictionPolicy:
    ...
```

Run both in shadow mode.

### 17.2 Unified policy input

```python
@dataclass(frozen=True)
class PolicyInput:
    predictions: PredictionBundle
    structural_state: StructuralState
    operational_risk_state: dict
    legacy_regime_state: Optional[object]
    legacy_matrix_intent: Optional[object]
```

### 17.3 Unified policy output

```python
@dataclass(frozen=True)
class PolicyDecision:
    action: str
    direction: str
    eligible_families: tuple[str, ...]
    confidence: float
    uncertainty: float
    size_cap: float
    hard_vetoes: tuple[str, ...]
    rationale: tuple[str, ...]
    policy_version: str
```

### 17.4 Example policy logic

**Premium selling**

Require:

* Predicted range survival above threshold.
* Forecast realized move below implied remaining move.
* Candidate-level net EV positive.
* Model uncertainty below limit.
* No operational hard veto.

**Directional debit**

Require:

* Calibrated direction probability above threshold.
* Expected return exceeds execution-cost hurdle.
* Favorable return quantile profile.
* Candidate net EV positive.
* Model uncertainty below limit.

**Long volatility**

Require:

* Forecast realized move meaningfully exceeds implied move.
* Expansion probability high.
* Expected path movement can overcome debit and execution costs.

**No trade**

Use when:

* Forecasts conflict.
* Model uncertainty is high.
* Data quality is low.
* Predicted edge is below costs.
* Candidate ranking is unstable.
* No candidate passes risk requirements.

### 17.5 Legacy fallback

When V2 prediction is unavailable:

* Fall back to the legacy policy.
* Mark the decision as fallback_legacy.
* Never silently substitute legacy output as though it came from V2.

---

## 18. Validation Framework

Create:

```
validation/session_folds.py
validation/nested_walk_forward.py
validation/bootstrap.py
validation/model_metrics.py
validation/economic_metrics.py
```

### 18.1 Session-grouped folds

Input observations must first be grouped by complete session_date.

No session may occur in more than one of:

* Training.
* Calibration.
* Validation.
* Test.
* Final holdout.

### 18.2 Expanding walk-forward

Example:

```
Fold 1
Train: sessions 1–40
Embargo: session 41
Test: sessions 42–51

Fold 2
Train: sessions 1–51
Embargo: session 52
Test: sessions 53–62
```

### 18.3 Rolling walk-forward

Example:

```
Fold 1
Train: sessions 1–40
Embargo: session 41
Test: sessions 42–51

Fold 2
Train: sessions 11–50
Embargo: session 51
Test: sessions 52–61
```

### 18.4 Purge and embargo

Default:

```
purge = all observations with labels that overlap the test boundary
embargo = one complete session
```

A longer embargo may be configured for persistent adaptive-state experiments.

### 18.5 Nested tuning

Outer folds measure performance.

Inner folds choose:

* Hyperparameters.
* Feature subsets.
* Probability calibration method.
* Decision thresholds.
* Utility penalties.

The outer test fold must not influence any selection.

### 18.6 Final untouched holdout

The existing optimizer supports a holdout, but it currently divides the timeline by observations.

V2 must reserve the final configured fraction of sessions, not ticks.

Default:

```
20% of sessions
```

The holdout is evaluated:

* Once for the selected challenger.
* Once for the current champion.
* Without further parameter changes.

### 18.7 Error policy

Replace broad silent exception handling with:

```python
@dataclass(frozen=True)
class TickFailure:
    ts: str
    session_date: str
    stage: str
    exception_type: str
    message: str
    traceback_hash: str
    data_quality: dict
```

A fold is invalid when:

```
failed test observations > 1%
or
failed sessions > 0
unless explicitly categorized as provider-unavailable sessions
```

All failures must appear in the validation report.

### 18.8 Session bootstrap

Confidence intervals must sample complete sessions with replacement.

Default:

```
1,000 bootstrap replications
95% interval
```

For candidate ranking, bootstrap both:

* Session-level economic results.
* Session-level rank uplift.

### 18.9 Direction metrics

Report:

* Brier score.
* Brier skill versus base rate.
* Log loss.
* ROC AUC.
* Precision-recall AUC.
* Calibration intercept.
* Calibration slope.
* Reliability bins.
* Hit rate at actionable confidence.
* Average signed forward return.
* Results by time of day.
* Results by regime.
* Results by direction.
* Results by confidence bucket.
* Session-level consistency.

The repository already exposes directional hit rate and Brier skill, and its current readiness checks distinguish prediction quality from profitability. V2 should preserve those metrics but calculate confidence and promotion status using independent sessions.

### 18.10 Quantile metrics

Report:

* Pinball loss for each quantile.
* 10–90 interval coverage.
* Interval width.
* Median absolute error.
* Bias.
* Coverage by volatility regime.
* Coverage by time of day.

### 18.11 Range and barrier metrics

Report:

* Brier score.
* Brier skill.
* Reliability bins.
* False-safe rate.
* False-danger rate.
* Target-before-stop accuracy.
* Survival calibration.
* Time-to-event error.

For premium selling, false-safe predictions deserve explicit weighting because they expose the portfolio to asymmetric loss.

### 18.12 Candidate-model metrics

Report:

* Mean net P&L of top-ranked candidate.
* Mean net P&L of legacy-ranked candidate.
* Mean net P&L of random feasible candidate.
* Top-one uplift.
* Spearman rank correlation within each snapshot.
* Probability calibration.
* EV calibration.
* Mean EV error.
* Median EV error.
* Expected shortfall accuracy.
* Maximum drawdown.
* Profit factor.
* Return on risk.
* Return on capital.
* Number of independent sessions.
* Percentage of profitable test folds.
* Performance by family.
* Performance by time of day.
* Performance by regime.

### 18.13 Economic metrics must be secondary to forecast validity

A model must not be promoted merely because it made money over a small test.

Required hierarchy:

1. Data integrity.
2. Out-of-sample probability or regression skill.
3. Calibration.
4. Stability across sessions.
5. Net economic uplift.
6. Risk-adjusted performance.

---

## 19. Model Registry and Reproducibility

Create:

```
prediction/registry.py
models/
```

### 19.1 Model metadata

Every saved model must include:

```
model_id
model_type
target
horizon
feature_version
training_start
training_end
training_sessions
calibration_sessions
code_commit
data_hash
configuration_hash
hyperparameters
outer_fold_metrics
holdout_metrics
calibration_metrics
created_at
author
status
```

### 19.2 Model statuses

```
research
shadow
candidate
pending_review
champion
rejected
archived
```

### 19.3 Serialization

Use:

* joblib for scikit-learn models.
* JSON for metadata.
* SHA256 hash for each binary.
* Atomic writes.
* Explicit schema version.

### 19.4 Compatibility checks

Live inference must fail closed when:

* Feature version mismatches.
* Required feature is unavailable and the model does not support missingness.
* Model hash fails.
* Metadata is missing.
* Model target is incompatible.
* Model was trained on a future date relative to replay.
* Scaler state version is incompatible.

---

## 20. Journal and Storage Changes

### 20.1 Keep the existing evaluations table

Do not break the current journal or no-trade settlement path.

### 20.2 Add prediction tables

**feature_snapshots**

```sql
CREATE TABLE feature_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_date TEXT NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    standardized_json TEXT,
    missingness_json TEXT,
    source_ages_json TEXT,
    quality_json TEXT
);
```

**prediction_outputs**

```sql
CREATE TABLE prediction_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    model_group_version TEXT NOT NULL,
    predictions_json TEXT NOT NULL,
    uncertainty REAL,
    generated_at TEXT NOT NULL,
    mode TEXT NOT NULL
);
```

**candidate_snapshots**

```sql
CREATE TABLE candidate_snapshots (
    candidate_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    family TEXT NOT NULL,
    legs_json TEXT NOT NULL,
    quote_json TEXT,
    geometry_json TEXT,
    legacy_metrics_json TEXT,
    execution_estimate_json TEXT,
    prediction_json TEXT
);
```

**candidate_outcomes**

```sql
CREATE TABLE candidate_outcomes (
    candidate_id TEXT PRIMARY KEY,
    settled INTEGER NOT NULL DEFAULT 0,
    settlement_price REAL,
    pnl_mid REAL,
    pnl_expected_fill REAL,
    pnl_conservative REAL,
    pnl_policy REAL,
    mfe REAL,
    mae REAL,
    target_hit INTEGER,
    stop_hit INTEGER,
    first_event TEXT,
    outcome_json TEXT
);
```

**model_registry**

```sql
CREATE TABLE model_registry (
    model_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**validation_failures**

```sql
CREATE TABLE validation_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    session_date TEXT,
    ts TEXT,
    stage TEXT,
    exception_type TEXT,
    message TEXT,
    traceback_hash TEXT,
    context_json TEXT
);
```

### 20.3 Link current evaluations

Add nullable columns or use signals_json for transitional linkage:

```
snapshot_id
prediction_model_group
policy_version
candidate_model_version
execution_model_version
legacy_policy_action
v2_policy_action
```

Long-term, explicit columns are preferable to critical provenance hidden in flexible JSON.

---

## 21. Configuration Specification

Create:

```
configs/prediction_v2.yaml
```

Example:

```yaml
prediction:
  enabled: true
  mode: shadow
  feature_version: v2.0.0
  model_registry_dir: models
  horizons_minutes:
    - 5
    - 15
    - 30
    - 60
    - close
  direction:
    champion_model_id: null
    actionable_probability: 0.58
    max_uncertainty: 0.35
  range:
    champion_model_id: null
    minimum_survival_probability: 0.62
  volatility:
    champion_model_id: null
    minimum_rv_iv_edge: 0.08
  physical_distribution:
    use_v2: true
    legacy_directional_tilt: false
    uncertainty_blend: true
  execution:
    model_id: null
    default_fill_fraction:
      single_leg: 0.35
      vertical: 0.50
      four_leg: 0.65
    stale_quote_penalty_seconds: 5
    conservative_mode: true
  candidate_ranker:
    model_id: null
    lambda_shortfall: 0.50
    lambda_fill: 0.25
    lambda_model: 0.25
    lambda_capital: 0.10

validation:
  fold_unit: session
  mode: expanding
  outer_folds: 5
  inner_folds: 3
  embargo_sessions: 1
  holdout_session_fraction: 0.20
  bootstrap_replications: 1000
  max_failed_observation_fraction: 0.01

promotion:
  min_independent_sessions: 40
  min_outer_test_sessions: 20
  require_positive_brier_skill: true
  require_positive_candidate_uplift: true
  require_holdout_non_degradation: true
  require_human_review: true
```

Thresholds in the policy section are starting research values. They must not be represented as proven edge.

---

## 22. Promotion Criteria

A challenger may become eligible for human review only when all applicable requirements pass.

### 22.1 Data requirements

* At least 40 independent sessions overall.
* At least 20 sessions across outer test folds.
* Final holdout contains at least 10 sessions.
* No session split across folds.
* No unresolved data-integrity flags.
* Test observation failure rate below 1%.
* No silent exceptions.
* Model and data hashes recorded.

These are minimums. More independent sessions are strongly preferred.

### 22.2 Direction-model requirements

* Brier skill greater than zero overall.
* Brier skill non-negative in the majority of outer folds.
* Better log loss than the base-rate model.
* Calibration slope within a proposed initial range of 0.75–1.25.
* Calibration intercept within a proposed initial range of -0.05–0.05.
* Actionable-confidence predictions produce positive average signed return after estimated costs.
* No severe side imbalance where one direction carries all apparent skill.

### 22.3 Range-model requirements

* Positive Brier skill.
* False-safe rate no worse than legacy.
* Predicted high-survival bins show monotonically higher realized survival.
* Candidate premium strategies show positive net uplift after costs.

### 22.4 Candidate-ranker requirements

* Top-ranked candidate beats the legacy ranker on mean net P&L.
* Uplift appears in a majority of outer folds.
* Expected shortfall does not materially worsen.
* Holdout performance does not collapse.
* Rank uplift is not concentrated in one session or one structure family.
* Candidate EV calibration improves or remains within a predefined tolerance.

### 22.5 Holdout requirement

The challenger must be evaluated once on the untouched holdout.

Any configuration change after viewing holdout results creates a new challenger and a new holdout protocol. The same holdout must not be repeatedly optimized against.

### 22.6 Human approval

Promotion remains manual.

The reviewer must receive:

* Data coverage.
* Fold definitions.
* Prediction metrics.
* Calibration charts.
* Economic metrics.
* Failure report.
* Regime breakdown.
* Time-of-day breakdown.
* Champion comparison.
* Holdout comparison.
* Known limitations.
* Artifact and configuration hashes.

---

## 23. Deployment Modes

### 23.1 Research

* Offline datasets.
* No live inference.
* No policy effect.
* Full model experimentation.

### 23.2 Shadow

* Live or replay inference.
* Predictions journaled.
* Legacy system remains authoritative.
* V2 candidates and decisions settled hypothetically.
* No user alert based solely on V2.

### 23.3 Advisory

* V2 output shown alongside legacy output.
* User sees disagreements.
* Legacy remains final unless manually overridden.
* No automatic order submission.

### 23.4 Champion

* V2 policy may become authoritative.
* Legacy output remains journaled as a counterfactual.
* Hard risk gates remain active.
* Automatic rollback conditions remain active.

---

## 24. Rollback Rules

Automatically revert V2 to advisory or shadow mode when:

* Model artifact cannot load.
* Feature-version mismatch occurs.
* Data coverage falls below threshold.
* Calibration degrades materially over a rolling session window.
* Prediction distribution shifts beyond control limits.
* Candidate EV error becomes materially biased.
* Fill estimates become systematically optimistic.
* Provider failover removes critical inputs.
* Exception rate exceeds threshold.
* Model outputs become constant or saturated.
* Champion underperforms legacy beyond configured risk tolerance.

Rollback must not delete the model or its audit record.

---

## 25. Monitoring

### 25.1 Daily monitoring

Report:

* Sessions and ticks captured.
* Missing-data rate.
* Feed source.
* Prediction coverage.
* Probability distribution.
* Model uncertainty.
* Direction calibration update.
* Range calibration update.
* Candidate EV bias.
* Fill estimation error.
* V2 versus legacy disagreement.
* Decision funnel.
* Errors and uptime gaps.

### 25.2 Weekly monitoring

Report:

* Rolling Brier skill.
* Rolling calibration slope and intercept.
* Signed-return edge.
* Candidate top-one uplift.
* Net P&L after estimated costs.
* Maximum drawdown.
* Expected-shortfall realization.
* Results by regime.
* Results by time of day.
* Results by family.
* GEX-variant comparison.
* Feature drift.
* Prediction drift.
* Champion degradation flags.

### 25.3 Drift detection

Monitor:

* Population stability index.
* Wasserstein distance.
* Missingness-rate drift.
* Mean and variance drift.
* Probability-distribution drift.
* Calibration drift.
* Feature-importance instability.
* GEX-variant disagreement.

Drift is a warning, not automatic proof that a model is invalid. It should trigger review or reduced size.

---

## 26. Pull Request Plan

The changes should be delivered in small, testable pull requests.

---

### PR 1 — Session-Safe Validation

**Objective**

Eliminate invalid train/test boundaries and silent validation failures.

**Files**

Modify:

```
walk_forward.py
optimizer.py
validation_pipeline.py
journal.py
```

Add:

```
validation/session_folds.py
validation/bootstrap.py
tests/test_session_folds.py
tests/test_walk_forward_failures.py
tests/test_session_bootstrap.py
```

**Requirements**

* Build folds from complete session dates.
* Add configurable embargo sessions.
* Purge overlapping labels.
* Divide holdout by sessions.
* Record all tick failures.
* Reject invalid folds.
* Add session-bootstrap confidence intervals.
* Keep legacy tick-level metrics for diagnostics.
* Report number of independent sessions prominently.

**Acceptance criteria**

* No session appears in both training and test.
* No session is split.
* Holdout contains only complete sessions.
* Injected tick failure appears in validation report.
* Excessive failures invalidate the fold.
* Existing deterministic replays remain reproducible.
* Existing tests remain green except intentionally updated validation assertions.

---

### PR 2 — Scaler Repair and Feature Provenance

**Objective**

Make feature normalization lagged, timeframe-specific, robust, and auditable.

**Files**

Modify:

```
mtf_matrix.py
regime_classifier.py
unified_loop.py
```

Add:

```
prediction/scalers.py
prediction/feature_provenance.py
tests/test_scaler_timeframe_keys.py
tests/test_scaler_score_before_update.py
tests/test_scaler_persistence.py
tests/test_feature_provenance.py
```

**Requirements**

* Native scales keyed by feature and timeframe.
* Score observation before updating scale state.
* Add exponentially decayed scale implementation.
* Preserve fixed priors for cold start.
* Persist state version and configuration hash.
* Record scale source and reliability.

**Acceptance criteria**

* One-minute feature updates do not alter daily-feature state.
* Current observation does not influence its own standardized score.
* Restart reproduces the same next-tick score.
* Incompatible state fails safely.
* Legacy scaler remains available behind configuration during transition.

---

### PR 3 — Canonical Dataset and Labels

**Objective**

Create leakage-safe feature and outcome datasets.

**Files**

Add:

```
prediction/asof.py
prediction/dataset.py
prediction/labels.py
prediction/storage.py
prediction/contracts.py
tests/test_asof_builder.py
tests/test_forward_labels.py
tests/test_barrier_labels.py
tests/test_candidate_labels.py
```

Modify:

```
chain_store.py
journal.py
unified_loop.py
```

**Requirements**

* Stable snapshot IDs.
* As-of source checks.
* Multi-horizon return labels.
* MFE and MAE labels.
* Volatility labels.
* Wall and flip labels.
* First-passage labels.
* Candidate outcome records.
* Parquet export.
* SQLite audit linkage.

**Acceptance criteria**

* Future bars and quotes are rejected.
* Levels are frozen at observation time.
* Ambiguous same-bar target/stop events are marked.
* Horizons past close become missing.
* Rebuilding from identical recordings creates identical hashes.

---

### PR 4 — Probabilistic Baseline Models

**Objective**

Replace the fixed direction composite with calibrated baseline forecasts in shadow mode.

**Files**

Add:

```
prediction/models/base.py
prediction/models/direction.py
prediction/models/return_quantiles.py
prediction/models/volatility.py
prediction/calibration.py
prediction/registry.py
prediction/training.py
tests/test_direction_model.py
tests/test_calibration.py
tests/test_model_registry.py
tests/test_grouped_training.py
```

**Requirements**

* Elastic-net direction models.
* Quantile return models.
* Volatility model.
* Grouped session validation.
* Probability calibration.
* Model metadata and hashes.
* PredictionBundle output.
* Legacy-composite baseline comparison.

**Acceptance criteria**

* No session leakage.
* Model predictions are deterministic given artifact and features.
* Probability outputs remain within bounds.
* Quantiles are ordered.
* Calibration uses training data only.
* Shadow predictions are written to the journal.
* No live policy effect.

---

### PR 5 — Independent Physical Distribution

**Objective**

Remove policy-created directional EV.

**Files**

Add:

```
prediction/physical_distribution.py
tests/test_physical_distribution.py
tests/test_physical_distribution_independence.py
```

Modify:

```
unified_loop.py
rnd_extractor.py
decision_engine.py
```

**Requirements**

* Build physical density from PredictionBundle.
* Match forecast mean and variance.
* Blend toward RND under uncertainty.
* Keep richness measurement independent.
* Add legacy-tilt compatibility flag.
* Record physical-distribution moments.

**Acceptance criteria**

* Changing routed structure does not change the physical density.
* Identical PredictionBundle produces identical density.
* Density integrates to one.
* Mean and variance match forecast within tolerance.
* High uncertainty moves distribution toward RND.
* Legacy and V2 EV can be compared in shadow mode.

---

### PR 6 — Execution Cost Model

**Objective**

Move from midpoint economics to executable economics.

**Files**

Add:

```
execution_cost.py
prediction/models/fill.py
tests/test_execution_cost.py
tests/test_fill_monotonicity.py
```

Modify:

```
spread_selector.py
journal.py
backtest.py
walk_forward.py
```

**Requirements**

* Mid, natural, expected, and conservative prices.
* Fees.
* Quote-age penalties.
* Structure-specific fill priors.
* Entry and exit costs.
* Net P&L labels.
* Paper/manual fill capture.

**Acceptance criteria**

* Expected credit never exceeds midpoint credit.
* Expected debit is never below midpoint debit.
* Conservative price is no better than expected price.
* Wider spreads produce worse expected fills.
* Older quotes do not improve expected fill.
* Economic metrics default to net expected-fill P&L.

---

### PR 7 — Range and Barrier Models

**Objective**

Replace terminal touch approximations with calibrated path-relevant probabilities.

**Files**

Add:

```
prediction/models/range_survival.py
prediction/models/barrier_touch.py
prediction/path_model.py
tests/test_range_model.py
tests/test_barrier_model.py
tests/test_path_bootstrap.py
```

Modify:

```
spread_selector.py
mc.py
```

**Requirements**

* Wall-touch probabilities.
* Flip-cross probabilities.
* Candidate-range survival.
* Target-before-stop probabilities.
* Residual block-bootstrap simulator.
* Existing MC retained as baseline.

**Acceptance criteria**

* Probabilities are calibrated out of sample.
* Same-bar ambiguity is conservatively handled.
* Path model preserves contiguous return blocks.
* Candidate touch risk uses V2 probability when available.
* Legacy reflection approximation remains available as fallback.

---

### PR 8 — Candidate Value Model

**Objective**

Learn candidate ranking from executable outcomes.

**Files**

Add:

```
prediction/models/candidate_value.py
prediction/candidate_dataset.py
prediction/candidate_ranker.py
tests/test_candidate_grouping.py
tests/test_candidate_ranker.py
tests/test_candidate_utility.py
```

Modify:

```
spread_selector.py
decision_engine.py
unified_loop.py
journal.py
```

**Requirements**

* Generate shadow candidate set.
* Store candidate-level features.
* Predict expected net P&L.
* Predict probability of profit.
* Predict downside quantile.
* Calculate utility.
* Compare V2 and legacy rankers.

**Acceptance criteria**

* Candidates from one snapshot remain in one fold.
* Utility decreases when expected shortfall increases.
* Utility decreases when fill or model uncertainty increases.
* V2 ranking runs in shadow mode first.
* Legacy ranking remains authoritative until promotion.

---

### PR 9 — GEX Variants

**Objective**

Measure which dealer-positioning representation actually predicts path behavior.

**Files**

Add:

```
gex/base.py
gex/oi.py
gex/weekly.py
gex/volume_proxy.py
gex/hybrid.py
gex/contracts.py
tests/test_gex_variants.py
```

Modify:

```
composite_feed.py
gate_scorer.py
unified_loop.py
journal.py
```

**Requirements**

* Parallel GEX calculations.
* Quality metadata.
* Observation-only journaling.
* Variant comparison report.
* No immediate gate authority.

**Acceptance criteria**

* All variants use the same output contract.
* Missing volume data does not contaminate OI calculations.
* Disagreement is journaled.
* No new variant affects policy before passing promotion criteria.

---

### PR 10 — Prediction Policy and Regime Consolidation

**Objective**

Separate prediction from policy and gradually replace the duplicate regime decision paths.

**Files**

Add:

```
policy/contracts.py
policy/legacy_matrix.py
policy/prediction_policy.py
policy/router.py
tests/test_prediction_policy.py
tests/test_policy_fallback.py
tests/test_policy_independence.py
```

Modify:

```
decision_matrix.py
regime_classifier.py
unified_loop.py
decision_engine.py
```

**Requirements**

* PolicyInput and PolicyDecision contracts.
* Legacy and V2 policies run together.
* Explicit fallback behavior.
* V2 policy consumes PredictionBundle.
* Structural and operational hard vetoes remain separate.
* Disagreement journaled.

**Acceptance criteria**

* Forecast output does not depend on policy selection.
* Policy can be replayed from stored PredictionBundle.
* Legacy fallback is explicit.
* Shadow comparison produces complete provenance.
* Promotion changes one configuration pointer, not code.

---

## 27. Test Matrix

**Unit tests**

* Feature source timestamp checks.
* Session grouping.
* Purge and embargo.
* Scaler key construction.
* Score-before-update.
* Density normalization.
* Probability bounds.
* Quantile ordering.
* Cost monotonicity.
* Candidate utility monotonicity.
* Barrier labeling.
* Model artifact hash.
* Configuration compatibility.
* Explicit fallback.
* Missing-data behavior.

**Integration tests**

* Recorded session to feature dataset.
* Dataset to trained model.
* Model to PredictionBundle.
* PredictionBundle to physical density.
* Prediction and structural state to policy.
* Policy to candidate set.
* Candidate set to execution estimates.
* Candidate ranker to TradeDecision.
* TradeDecision to journal.
* Settlement to validation report.

**Regression tests**

* Legacy output remains unchanged when V2 is disabled.
* Legacy replay remains deterministic.
* Existing journal records remain readable.
* Existing champion configs remain valid.
* Legacy directional tilt remains available during migration.
* Existing no-trade settlement behavior remains intact.

**Leakage tests**

* Future bar insertion does not change an earlier feature snapshot.
* Future quote insertion does not change an earlier chain snapshot.
* Future session does not change an earlier fold's trained model.
* Test-session outcome does not alter scaler state used at test entry.
* Calibration set does not overlap test.
* Candidates from one snapshot never cross folds.

**Failure tests**

* Corrupt model.
* Corrupt scaler state.
* Stale chain.
* Missing bars.
* Missing model feature.
* Provider outage.
* Unsupported schema version.
* Invalid quantile output.
* Density normalization failure.
* Excessive fold errors.

---

## 28. Definition of Done

Prediction Engine V2 is complete when:

1. Training and validation operate on full session groups.
2. No validation exception is silently swallowed.
3. Session-level confidence intervals are reported.
4. Feature normalization is timeframe-specific and lagged.
5. A canonical as-of dataset can be rebuilt deterministically.
6. Multi-horizon direction, return, volatility, range, and barrier labels exist.
7. A calibrated PredictionBundle is generated in replay and live shadow mode.
8. The physical distribution is independent of routed structure.
9. Executable fill estimates replace midpoint economics in primary reports.
10. Candidate-level outcomes are stored.
11. V2 candidate ranking is evaluated against the legacy ranker.
12. GEX variants are journaled and compared.
13. Prediction and policy are separate modules.
14. Model artifacts are versioned, hashed, and auditable.
15. Champion promotion remains human-controlled.
16. V2 can automatically fall back without losing auditability.
17. The legacy engine remains available for direct comparison.
18. A challenger passes out-of-sample forecast, calibration, economic, and holdout criteria before becoming authoritative.

---

## 29. Immediate Implementation Priority

The correct implementation order is:

1. Session-safe validation.
2. Scaler repair.
3. Canonical as-of dataset and labels.
4. Execution-cost accounting.
5. Calibrated baseline predictions.
6. Independent physical density.
7. Range and barrier predictions.
8. Candidate-value model.
9. GEX-variant arbitration.
10. Policy consolidation.

Do not begin by adding more indicators or increasing model complexity.

A simple regularized model evaluated correctly is more valuable than a sophisticated model evaluated on leaking, highly correlated, midpoint-priced observations.

---

## 30. Final Engineering Principle

Every component must answer one of four questions:

```
What will probably happen?
What trade best expresses that forecast after costs?
Is the risk allowed?
Did the prediction and decision work out of sample?
```

No component should answer all four at once.

The system's strongest current characteristic is its measurement and audit philosophy. Prediction Engine V2 should build on that strength by making the forecasting layer empirical, probabilistic, independently validated, and economically executable.
