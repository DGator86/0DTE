"""
prediction
==========
Prediction Engine V2 components (see docs/PREDICTION_ENGINE_V2_HANDOFF.md).

Currently implemented (PR 2 — Scaler Repair):
  scalers — per-feature-and-timeframe, exponentially decayed, lagged
            (score-before-update) standardization scales with versioned
            persistence.
"""
