"""
prediction.models
=================
Prediction Engine V2 model suite (docs/PREDICTION_ENGINE_V2_HANDOFF.md §11).

Currently implemented (PR 4 — probabilistic baselines):
  base             — feature vectorization (value + explicit missingness
                     columns), shared fit/predict plumbing, determinism.
  direction        — calibrated elastic-net logistic direction models per
                     horizon, with an optional gradient-boosted challenger
                     and the required naive baselines.
  return_quantiles — q10/q50/q90 gradient-boosted quantile regression with
                     monotone rearrangement, pinball loss and coverage.
  volatility       — log-target realized-move regressor with quantile range
                     and forecast/implied ratio.
  fill             — structure-specific fill-fraction priors for the
                     execution-cost model (PR 6); empirical trainer later.

No deep learning by design: the number of independent market sessions is far
smaller than the number of tick rows.
"""
