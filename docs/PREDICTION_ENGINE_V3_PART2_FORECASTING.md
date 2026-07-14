# Prediction Engine V3 — Part 2 of 3

**Structural-State Forecasting, Mixture-of-Experts, Competing Risks, and Advanced Path Simulation**

Repository: DGator86/0DTE  
Dependency: Prediction Engine V3 Part 1 completed  
Status: Implementation specification  
Audience: Coding agent, quantitative developer, model-validation reviewer  
Required implementation mode: Research and shadow only  
Live trading effect: None until explicit human promotion  
Suggested destination: `docs/PREDICTION_ENGINE_V3_PART2_FORECASTING.md`

⸻

## 1. Executive objective

Part 2 must improve the system’s actual market forecasts after Part 1 establishes statistically valid cross-fitting, calibration, uncertainty estimation, and model evaluation.

Part 2 introduces:

1. A canonical expanded dealer-structure state.
2. Probabilistic rather than hard regime classification.
3. Regime-specialized forecast experts.
4. Discrete-time competing-risk models for target and stop events.
5. Conformalized return distributions.
6. State-conditioned empirical path simulation.
7. A forecast ensemble that combines independent model families.

The existing block-bootstrap path simulator must remain a baseline because it preserves contiguous historical return behavior, including:

* Serial correlation.
* Volatility clustering.
* Momentum bursts.
* Mean-reversion runs.
* Intraday return-block structure.

Part 2 must improve the conditioning and forecasting around that simulator rather than replacing it with a pure Gaussian process.

⸻

## 2. Scope

### 2.1 Included

Part 2 includes:

* Structural-state schema expansion.
* GEX variant reconciliation.
* Structural feature engineering.
* Regime-label construction.
* Multiclass regime probability estimation.
* Regime mixture-of-experts.
* Competing-risk target/stop models.
* Expanded return quantile forecasts.
* Session-grouped conformal calibration.
* State-conditioned residual-block simulation.
* Forecast-level ensemble weighting.
* Shadow-mode integration.
* Persistence, diagnostics, and tests.

### 2.2 Excluded

Part 2 does not include:

* Live order placement.
* Champion promotion.
* Candidate pairwise ranking.
* Empirical fill-probability modeling.
* Trade/no-trade meta-labeling.
* Dynamic production model replacement.
* Reinforcement learning.
* Deep neural networks.
* Automated capital allocation.
* Removal of legacy rules.
* Automatic modification of hard risk controls.

Those belong to Part 3 or later research.

⸻

## 3. Mandatory design rules

### 3.1 GEX is structural evidence, not unquestionable truth

No single GEX calculation may be treated as ground truth.

The system must preserve parallel structural measurements whenever they are available:

* Open-interest GEX.
* Volume-based GEX proxy.
* Hybrid GEX.
* Weekly-expiration GEX.
* Same-day-expiration GEX.
* Strike concentration.
* Gamma-flip position.
* Call-wall and put-wall positions.
* Wall stability.
* Structural disagreement.

The model may learn which variants have predictive value. It may not silently replace every variant with one preferred calculation.

### 3.2 Regimes are probabilities

Do not force every observation into exactly one unquestioned market state.

The regime model must produce a probability distribution such as:

```
P(long_gamma_pin)       = 0.51
P(short_gamma_trend)    = 0.18
P(flip_transition)      = 0.24
P(volatility_expansion) = 0.07
```

The probabilities must sum to one within numerical tolerance.

The dominant regime is a convenience label only. Downstream forecast blending must use the complete probability distribution.

### 3.3 Experts require sufficient independent sessions

A regime-specific expert may train only when its training history meets configured minimum-support requirements.

Initial research defaults:

```
Minimum sessions with nonzero regime support: 40
Minimum effective weighted sessions:          20
Minimum labeled observations:                 500
```

These values must remain configurable.

When support is inadequate:

* Use the global model.
* Record the fallback.
* Increase uncertainty.
* Do not fabricate a regime-specific forecast.

### 3.4 Paths must remain empirical

Do not replace the block-bootstrap simulator with a pure Gaussian simulation.

The Gaussian or Ornstein-Uhlenbeck Monte Carlo remains:

* A diagnostic.
* A baseline.
* An emergency fallback.

The empirical residual block-bootstrap remains the primary path simulator.

### 3.5 Forecasts remain independent of candidate selection

The following must be produced before exact candidates are selected:

* Regime probabilities.
* Direction probabilities.
* Return distribution.
* Expected realized move.
* Barrier probabilities.
* Competing-risk event probabilities.
* Path simulations.
* Forecast uncertainty.

Forecast models may not receive:

* Candidate family.
* Candidate legs.
* Selected strikes.
* Candidate rank.
* Policy choice.
* Gate result.
* Human decision.
* Future candidate P&L.

### 3.6 Hard operational vetoes remain separate

The statistical forecast may model market behavior during unusual or prohibited conditions for research purposes.

The following remain deterministic restrictions:

* Stale data.
* Missing chain.
* Invalid option surface.
* Prohibited catalyst.
* Daily risk limit.
* Portfolio exposure limit.
* System outage.
* Broker unavailable.
* Entry lockout.
* Insufficient liquidity.

⸻

## 4. Target architecture

```
As-of-safe market and chain data
                |
                v
Expanded StructuralState
                |
                v
Canonical feature snapshot
                |
                +------------------------+
                |                        |
                v                        v
      Regime probability model     Global forecast models
                |                        |
                v                        |
      Regime-specialized experts         |
                |                        |
                +-----------+------------+
                            |
                            v
                 Mixture-of-experts
                            |
              +-------------+--------------+
              |             |              |
              v             v              v
      Return distribution  Competing risk  Volatility forecast
              |             model
              +-------------+--------------+
                            |
                            v
          State-conditioned empirical path simulator
                            |
                            v
                  Forecast ensemble
                            |
                            v
                PredictionBundle V3
                            |
                            v
               Shadow persistence and audit
```

⸻

## 5. Canonical structural-state model

### 5.1 New module

Create:

`prediction/structural_state.py`

Modify:

`policy/contracts.py`  
`unified_loop.py`  
`gate_scorer.py`  
`prediction/dataset.py`  
`prediction/contracts.py`  
`prediction/storage.py`

### 5.2 Version

`STRUCTURAL_STATE_VERSION = "v3.0.0"`

### 5.3 Required contract

See `prediction/structural_state.py` (`StructuralState`).

⸻

## 6–8. Structural features, builder, compatibility

See implementation in `prediction/structural_state.py`:

* GEX disagreement / sign agreement (§6.1)
* Concentration / HHI (§6.2)
* Flip / wall velocity (§6.3–6.4)
* Structural stability (§6.5)
* Expected-move normalization (§6.6)
* `StructuralStateBuilder` (§7)
* Compatibility properties with hybrid → OI → volume fallback (§8)

Missing variants remain missing. Do not substitute zero for unavailable levels.

⸻

## 9–51. Remaining Part 2 scope

Subsequent PRs implement regime labels, regime MoE, mixture experts, expanded return distributions, conformal calibration, competing risks, path model V3, forecast ensemble, and shadow integration per the original Part 2 handoff.

### Implementation sequence

| PR | Scope |
|----|-------|
| PR 7 | Structural State V3 (this PR) |
| PR 8 | Regime labels |
| PR 9 | Regime probability model |
| PR 10 | Mixture-of-experts |
| PR 11 | Expanded return distributions |
| PR 12 | Conformal calibration |
| PR 13 | Competing-risk models |
| PR 14 | Path Model V3 |
| PR 15 | Forecast ensemble |
| PR 16 | Part 2 shadow integration |

### Coding-agent execution directive

1. Implement pull requests in the specified order.
2. Inspect existing modules and tests before changing contracts.
3. Preserve backward compatibility unless this specification explicitly changes it.
4. Keep all statistical behavior in research or shadow mode.
5. Do not alter live order-routing behavior.
6. Do not promote any model automatically.
7. Do not replace empirical path simulation with Gaussian-only simulation.
8. Do not use candidate information in forecasting.
9. Do not use future-updated structural levels in labels or predictors.
10. Do not treat missing structural values as zero.
11. Do not train unsupported regime experts.
12. Do not use test sessions for calibration.
13. Do not swallow critical-path errors without structured records.
14. Seed all stochastic behavior deterministically.
15. Add migrations before writing new data.
16. Run the entire test suite after every pull request.
17. Record test commands and outcomes in each pull-request description.
18. Document every deviation from this specification.
19. Stop and fail closed when model artifacts or schemas are incompatible.
20. Update this handoff document when implementation details materially change.

Part 2 is not complete because the new models execute. It is complete only when their outputs are reproducible, calibrated, path coherent, uncertainty aware, support limited, independently validated by session, and safely integrated in shadow mode.

### Implementation notes (PR 7)

* Legacy `policy.contracts.StructuralState` remains the simplified OI policy view.
* V3 → legacy conversion is explicit via `to_legacy_policy_state()`.
* Live gates continue to use `MarketSnapshot` OI fields unchanged.
* `structural_states` table is idempotent (`CREATE TABLE IF NOT EXISTS`).


⸻

## 9. Regime-label construction

Create `prediction/regime_labels.py`.

Regime labels describe **future** price behavior for training. They must not be defined only by current GEX sign.

### 9.3 Regime classes

```
REGIME_CLASSES = (
    "long_gamma_pin",
    "short_gamma_trend",
    "flip_transition",
    "volatility_expansion",
)
```

### 9.4–11 Label rules (summary)

* **long_gamma_pin** — contained move, low directional efficiency, repeated reversion, inside frozen channel.
* **short_gamma_trend** — high directional efficiency, material move, shallow pullbacks.
* **flip_transition** — frozen flip crossed with meaningful two-sided occupation.
* **volatility_expansion** — move exceeds expected; substantial two-sided excursion; low directional cleanliness.

Precedence (mutually exclusive): volatility_expansion → flip_transition → short_gamma_trend → long_gamma_pin.

Ambiguous observations may remain unlabeled (`regime_label = None`). Store independent component flags for multilabel research.

⸻

## 12–15. Regime probability model and mixture-of-experts

Create `prediction/models/regime_moe.py` and `prediction/models/mixture_experts.py`.

* Baseline: multinomial logistic regression; challenger: HistGradientBoostingClassifier.
* One-vs-rest cross-fitted calibration; probabilities sum to 1 within 1e-6.
* Experts: global + one per regime; shrink toward global when support is limited (`shrinkage_sessions=40`).
* Blend with the full regime probability vector — never hard-route on dominant regime alone.

⸻

## 16–18. Return distributions and conformal calibration

Expanded quantile grid: 0.05 … 0.95. Create `prediction/conformal.py` with session-grouped split conformal. OOD observations must not claim coverage guarantees.

⸻

## 19–23. Competing-risk barrier model

Create `prediction/event_dataset.py` and `prediction/models/competing_risk.py`.

Hazards: target / stop / none per discrete future minute. Survival and cumulative incidence must satisfy:

```
p_target_first + p_stop_first + p_neither ≈ 1
```

Same-bar ambiguity: adverse-first; never assume favorable first.

⸻

## 24–31. Path Model V3

Modify `prediction/path_model.py` (`PATH_MODEL_VERSION = "v3.0.0"`).

State-conditioned residual block bootstrap with:

* Nearest-neighbor sampling kernel
* Source-session weight cap (default 10%)
* Explicit backoff hierarchy (levels 0–6)
* Deterministic seeds from snapshot_id + version + horizon + config hash
* Gaussian/OU only as labeled fallback (level 6)

⸻

## 32–33. Forecast ensemble

Create `prediction/ensemble.py`. Weights from historical OOS loss only. Max component weight 60%. Missing / failed components explicit. Legacy Monte Carlo must not silently dominate.

⸻

## 34–37. Bundle, storage, registry, shadow

`PredictionBundle` gains optional Part 2 fields (regime / return distributions / competing risk / path / ensemble). Storage tables: `structural_states`, `regime_outputs`, `competing_risk_outputs`, `path_forecasts`, `ensemble_outputs`. Shadow integration must isolate failures without crashing the legacy loop.

⸻

## 39–50. Tests and acceptance

Required test modules are listed in the original Part 2 handoff (§39). Acceptance requires all Part 1 + Part 2 tests green, no live API dependence, no unseeded stochastic tests, and no candidate information in forecasts.

### Deviations log

* PR 7 stores the full Part 2 handoff in condensed form in this file; detailed contracts live in the implementing modules.
* Legacy `policy.contracts.StructuralState` retains zero-default fields for gate compatibility; V3 conversion is explicit.
