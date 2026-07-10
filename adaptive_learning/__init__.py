"""
adaptive_learning
=================
Offline Adaptive Learning Engine (ALE) for the 0DTE pipeline.

The package closes the loop journal -> diagnose -> hypothesize -> optimize
(holdout mandatory) -> stability-check -> promotion report -> human-approved
champion config. It NEVER modifies live trading parameters automatically:
the live engine loads one static configs/champion.json at startup and the
only code path that writes that file is the explicit human approval CLI
(python3 -m adaptive_learning.promoter --approve <config_id>).

Modules
-------
  config_store  champion/challenger config records, shared override applier
  diagnostics   failure-mode detection from the journal + report history
  hypothesis    diagnosis -> targeted parameter search spaces
  stability     parameter fold-stability and feature stability scoring
  feature_lab   Spearman / mutual information / permutation importance
  promoter      rule-based promotion checks + human approve/reject CLI
  reports       promotion_candidate reports (validation_reports + files)
  learner       orchestration: run_learning_cycle / run_daily / run_weekly

Intentionally no eager imports here: learner pulls in the full engine stack
(optimizer -> walk_forward -> unified_loop) and config_store is imported by
optimizer itself, so the package root must stay import-cycle free.

NOT financial advice.
"""
