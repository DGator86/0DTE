# Prediction Engine V3 — Part 1 of 3

**Validation Integrity, Cross-Fitting, and Live Uncertainty**

Repository: DGator86/0DTE  
Document date: July 13, 2026  
Status: Implementation specification  
Audience: Coding agent, quantitative developer, model-validation reviewer  
Required implementation mode: Research and shadow only  
Live trading effect: None until explicit human promotion  

⸻

## 1. Executive objective

Part 1 must make the existing Prediction Engine statistically trustworthy before additional forecasting complexity is introduced.

The current repository already contains:

* Session-grouped, embargoed walk-forward evaluation.
* Calibrated direction models.
* Multi-horizon return quantile models.
* Candidate-level expected P&L and probability-of-profit models.
* A model registry with versioned artifacts and human-controlled statuses.
* A shadow pipeline that journals predictions without routing live orders.

The directional model currently supports elastic-net logistic regression and histogram gradient boosting, with probability calibration performed on an internal session-based calibration slice. The candidate-value model predicts expected net P&L, probability of positive P&L, and P&L quantiles.

However, the next generation must correct three weaknesses:

1. Hyperparameter selection and probability calibration are not fully separated.
2. Candidate expected-P&L hyperparameters are selected using in-sample error.
3. Model uncertainty is mostly represented by a global scalar derived from calibration skill rather than observation-specific uncertainty.

Part 1 must fix those weaknesses without changing any live trade decision.

⸻

## 2. Mandatory architectural invariants

### 2.1 No future leakage

No model input may use information timestamped after the prediction timestamp.

All folds must remain time ordered and grouped by complete market session. A session must never appear on both sides of a train, calibration, validation, or test boundary.

### 2.2 Forecast-policy separation

Forecast models may not receive:

* Selected structure.
* Selected option family.
* Selected strikes.
* Gate result.
* Policy direction.
* Candidate score.
* Human decision.
* Future fill or outcome information.

Policy may consume predictions, but it may not write information back into the forecast.

### 2.3 Hard risk controls remain deterministic

The following remain outside all statistical models:

* Stale or invalid market data.
* Missing option chain.
* Broken arbitrage checks.
* Prohibited catalyst window.
* Maximum daily loss.
* Maximum portfolio exposure.
* Broker or system unavailable.
* Entry lockout time.
* Insufficient executable liquidity.

### 2.4 Fail closed

When a required artifact, schema, feature version, timestamp, or data-quality requirement is invalid, the V3 path must return an unavailable or abstain result. It must not silently substitute a model or claim a valid forecast.

### 2.5 Deterministic replay

Given identical recorded inputs, feature version, label version, model artifacts, configuration, and random seed, the system must produce identical predictions, uncertainty outputs, rankings, and audit records.

### 2.6 No production authority

All work in Part 1 must operate under `research` / `shadow`. No Part 1 component may become champion automatically.

⸻

## 3. Target architecture

```
Canonical session-grouped dataset
            |
            v
Outer walk-forward folds
            |
            v
Inner hyperparameter folds
            |
            v
Cross-fitted raw predictions
            |
            +--> independent probability calibration
            |
            +--> out-of-fold residuals
            |
            +--> uncertainty estimators
            |
            v
Final model trained on eligible historical sessions
            |
            v
Shadow PredictionBundle
            |
            +--> component uncertainty
            +--> composite uncertainty
            +--> out-of-distribution score
            +--> feature and data quality
            |
            v
PredictionStore + ModelRegistry
```

⸻

## 4. Work package 1.1 — Cross-fitting framework

Module: `prediction/crossfit.py`

See module contracts: `FoldDefinition`, `CrossFitResult`, `NestedCrossFitConfig`,
`build_nested_session_folds`, `crossfit_classifier`, `crossfit_regressor`.

Fold behavior:

1. Sort unique session dates.
2. Construct expanding, time-ordered outer folds.
3. Reserve complete sessions for outer testing.
4. Apply whole-session embargoes.
5. Inside each outer training window, construct inner folds for hyperparameter selection.
6. Select hyperparameters using only inner out-of-fold results.
7. Generate raw predictions for held-out sessions.
8. Never use the final outer test sessions for tuning or calibration.
9. Return explicit row indices and session membership for audit.

Classification primary selection metric: **log loss**.  
Regression selection: MAE / Huber / median AE / bias / tail underprediction (not sole MSE).

Candidate grouping: outer key `session_date`; non-splittable group `snapshot_id`.

⸻

## 5. Work package 1.2 — Independent probability calibration

Required sequence:

1. Select hyperparameters using inner OOF scores.
2. Generate cross-fitted raw probabilities across eligible training sessions.
3. Fit the probability calibrator on those cross-fitted probabilities.
4. Evaluate the calibrated system on untouched outer test sessions.
5. Train the final base estimator on all eligible historical sessions.
6. Attach the calibrator fitted from training-only cross-fitted predictions.

Retain sigmoid/Platt default; isotonic only when sample/session minimums are met and
compared via nested/cross-fitted evaluation (not same-label fit comparison).

⸻

## 6. Work package 1.3 — Fix candidate-value model selection

Replace in-sample expected-P&L parameter selection with session-grouped cross-fitting.

Initial models: ElasticNet, HuberRegressor, HistGradientBoostingRegressor (sklearn only).

Selection score:

```
oof_huber_loss + 0.25 * abs(oof_bias) + 0.25 * downside_underprediction_penalty
```

Profit-probability head uses cross-fitting + independent calibration.  
Quantile heads evaluated OOF (pinball, coverage, width, downside miss, crossing).

⸻

## 7. Work package 1.4 — Observation-specific uncertainty

Modules: `prediction/uncertainty.py`, `prediction/ood.py`, `prediction/session_bootstrap.py`

Components (0 = low uncertainty, 1 = max): ensemble, conformal, OOD, calibration,
data quality, model age → composite via reweighted mean (missing ≠ zero).

Part 1 effects: PredictionBundle / journal / shadow confidence / `ABSTAIN_SHADOW` only.
Must not alter legacy live trade behavior.

⸻

## 8–11. Bundle, storage, registry, session bootstrap

See implementation in:

* `prediction/contracts.py` — V3 optional uncertainty/OOD fields
* `prediction/storage.py` — `model_evaluations`, `uncertainty_outputs`
* `prediction/registry.py` — `SCHEMA_VERSION = 2` with v1 read-only compatibility
* `prediction/session_bootstrap.py` — session-level CIs (tick bootstrap prohibited)

⸻

## 12. Required tests

* `tests/test_crossfit.py`
* `tests/test_nested_session_folds.py`
* `tests/test_crossfit_calibration.py`
* `tests/test_candidate_value_no_insample_selection.py`
* `tests/test_uncertainty.py`
* `tests/test_ood.py`
* `tests/test_session_bootstrap.py`
* `tests/test_registry_v2.py`
* `tests/test_storage_v3_part1.py`
* `tests/test_prediction_bundle_v3.py`

Leakage, determinism, and monotonic uncertainty tests are mandatory.

⸻

## 13. Acceptance criteria

* No in-sample hyperparameter selection in candidate-value training.
* All probability calibrators use cross-fitted or independently held-out predictions.
* Outer test sessions untouched until final evaluation.
* Session-grouped confidence intervals produced.
* Legacy decisions unchanged; V3 shadow only; registry mismatch fails closed.
* All existing + new Part 1 tests pass offline with explicit seeds.

⸻

## 14. Implementation sequence (PRs)

1. Cross-fitting primitives  
2. Direction-model cross-fitting + independent calibrator  
3. Candidate-value correction  
4. Observation-specific uncertainty  
5. Registry and persistence  
6. Part 1 integration  

No PR may combine live policy promotion with Part 1 implementation.
