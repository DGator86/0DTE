"""
prediction
==========
Prediction Engine V2 components (see docs/PREDICTION_ENGINE_V2_HANDOFF.md).

Currently implemented:
  scalers   — per-feature-and-timeframe, exponentially decayed, lagged
              (score-before-update) standardization scales with versioned
              persistence (PR 2).
  contracts — PredictionBundle, the forecast data contract (PR 3).
  asof      — point-in-time source rules: future bars/quotes are rejected,
              missing values stay missing (PR 3).
  labels    — multi-horizon forward-return / excursion / volatility /
              wall-flip / first-passage / range-survival / candidate-outcome
              labels with frozen observation-time levels (PR 3).
  dataset   — stable snapshot ids, session identity, observation rows, and
              the offline recording -> dataset builder (PR 3).
  storage   — SQLite dataset tables + Parquet export + deterministic
              dataset hashing (PR 3).
  models/   — calibrated elastic-net direction models, quantile return
              models, and the realized-move volatility model (PR 4).
  calibration — sigmoid/Platt (default) and gated isotonic probability
              calibration, fitted on training data only (PR 4).
  registry  — versioned, hashed, fail-closed joblib model artifacts with
              §19 metadata and status vocabulary (PR 4).
  training  — session-grouped, embargoed walk-forward training with naive
              baselines, PredictionBundle assembly, and journaled shadow
              inference with zero policy effect (PR 4).
  physical_distribution — independent physical density from
              PredictionBundle / PhysicalForecast: center, scale, shift,
              uncertainty-blend toward RND; replaces the circular
              dir_drift_frac tilt (PR 5).
"""
