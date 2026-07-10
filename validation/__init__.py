"""
validation
==========
Session-safe validation utilities (Prediction Engine V2, PR 1 — see
docs/PREDICTION_ENGINE_V2_HANDOFF.md §18).

The unit of statistical evidence in this system is the trading SESSION, not
the tick: two predictions one minute apart share virtually the entire future
price path and are not independent experiments. Everything in this package
therefore groups, folds, and resamples by complete session dates.

Modules
-------
session_folds  — group timestamps by exchange session; build walk-forward
                 folds from complete sessions with a purge/embargo gap;
                 split holdouts by sessions rather than tick counts.
bootstrap      — session-level bootstrap confidence intervals.
"""
