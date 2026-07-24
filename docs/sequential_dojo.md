# Anchored sequential dojo — prequential curriculum

`sequential_dojo.py` is the governance spine that keeps accelerated training
from becoming backtest overfitting. It answers one question, honestly:

> **Does learning from Days 1…t improve the *first-pass* result on Day t+1?**

## The one rule

Score Day *t* **before** learning from it. The only number that represents
genuine learning is the **prequential** (predict-then-learn) score — the
system's first-pass result on a blind day, using only information available
through the prior sessions. Re-running a day until it "wins" just memorizes
that day's answer key; the Dojo docs already say replaying adds no new
information. This module makes that rule structural.

## What it measures

For each session *t* in chronological order, leak-free (each session is scored
warmed only on the sessions before it, via `walk_forward`'s
`initial_warm_sessions` — the same session-fold machinery the rest of the
stack uses; 0DTE sessions settle same-day, so no embargo gap is needed):

- **prequential J(champion, t)** — the carried-in learned state on blind day t
- **prequential J(baseline, t)** — the untrained baseline on the same blind day
- **forward transfer** `FT_t = J(champion, t) − J(baseline, t)`

`FT_t > 0` on average ⇒ the curriculum teaches generalized behavior. `FT_t ≤ 0`
⇒ the "champion" is overfit and should not size up (the run flags
`no_forward_transfer`).

`retention_forgetting(...)` scores a fixed panel of earlier sessions under
champion-before vs candidate-after a learning step and returns the
anti-forgetting penalty `F = mean max(0, J_before − J_after)` — the gate a
learner-curriculum uses so a Day-t update can't silently degrade Day-1
behavior.

Sessions in the **sealed** set are removed from the curriculum entirely, so
nothing here ever scores or learns against them — a benchmark that stays
sealed.

## Usage

```bash
python3 sequential_dojo.py \
    --db /var/lib/zerodte/shadow.db \
    --record-dir /var/lib/zerodte/ticks \
    --configs-dir /var/lib/zerodte/configs \
    --sealed 2026-07-04,2026-07-18 \
    --min-warm 3
```

Run it with the current champion to verify it beats baseline on blind
sessions (positive forward transfer) rather than being memorized. The report
persists as a `validation_reports` row (`report_type='sequential_dojo'`).

## Scope — Stage 1 (this module) and what's deferred

This is the prequential **measurement + governance spine**. It reuses the
existing session-fold walk-forward, the composite objective, and the champion
config; it does **not** yet implement the full design's later stages, which
are deliberately separate:

- **Learner curriculum** — the per-day candidate update (replay-mixture batch:
  current + recent + hard-failure + regime-balanced + synthetic), gated on the
  `retention_forgetting` penalty, stability, and a synthetic-holdout pass,
  evolving the champion day over day. The spine and the retention gate exist;
  wiring the learner into the loop is the next stage.
- **Intraday forecast-vintage grading** — freezing the MTF forecast cone at
  each decision timestamp and grading every horizon (direction, quantile
  coverage, trajectory, barrier touch, regime sequence, revision quality) as
  the recorded future resolves, with successive vintages per target. This ties
  into the prediction store and the cone work (`forecast_stabilizer`).
- **Three-memory store** — model memory (weights/calibration), structured
  agent lessons (hypotheses), and episodic replay (hard cases) as first-class
  persisted stores.
- **Selection / abstention regret** — grading the chosen option structure
  against best-available / no-trade / deterministic baseline along the realized
  path, so a good price forecast that doesn't support a good trade is scored
  as such.

The guardrail that motivates the whole design stays fixed: **encounter Day t
blind → score it → learn from it → verify retention and robustness → encounter
Day t+1 blind.** Never "inspect Day t's answer until it looks profitable."
