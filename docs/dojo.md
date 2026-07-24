# The Dojo — matrix-style training for the 0DTE / SPY-DER stack

> "I know kung fu." — the goal, made measurable.

`dojo.py` compresses months of market experience into one run: it replays
everything shadow mode has recorded, runs one adaptive-learning cycle on it,
and then spars the full pipeline against a combinatoric catalog of
Markov-generated market universes so its behavior is measured in situations
the live tape has never shown it. The result is a persisted **dojo report**
rendered by the dashboard's **Dojo** tab — which the Vercel deployment serves
through the same read-only `/api/*` proxy as every other tab.

## Quick start (VPS)

```bash
cd /opt/zerodte
venv/bin/python dojo.py \
    --db /var/lib/zerodte/shadow.db \
    --record-dir /var/lib/zerodte/ticks \
    --configs-dir /var/lib/zerodte/configs \
    --reports-dir /var/lib/zerodte/reports/dojo
```

Scheduled weekly (Saturday 15:00 ET, before the Sunday learning cycle):

```bash
sudo cp deploy/zerodte-dojo.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zerodte-dojo.timer
```

Open the dashboard (VPS or Vercel) → **Dojo** tab. The tab dot lights amber
when the latest run flagged a weak archetype.

## The three phases

### 1. Recorded tape — the real-data baseline
Session-unit walk-forward (`walk_forward.py`) over every tick shadow mode has
recorded via `chain_store`, plus full-window calibration (directional hit
rate, Brier skill, EV bias). Reports `insufficient_data` honestly until at
least 3 sessions / 100 ticks exist — it never fabricates a baseline.

### 2. Adaptive learner — one full learning cycle
`adaptive_learning.learner.run_learning_cycle(mode="dojo")`: diagnose journal
failure modes → generate hypotheses → optimize with a **mandatory holdout** →
stability analysis → stage a candidate as `pending_review`. The dojo never
touches `champion.json`; promotion stays a human decision via the promoter
CLI (the report links the pending candidate).

### 3. Universe sparring — the full universe simulator
`matrix_universe.py` generates markets from stacked Markov chains:

```
market archetypes (day-scale chain)
  └─ conditions → intraday regimes (minute-scale chain)
        └─ conditions → per-variable chains (GEX, realized vol, VRP, skew,
                        drift — each with its own RNG stream + OU relaxation)
              └─ drive → price path, repriced 0DTE chain, settlement
```

- **8 archetypes**: `calm_pin`, `grind_up`, `grind_down`, `range_chop`,
  `vol_expansion`, `squeeze_melt_up`, `crash`, `gap_shock`
- **5 regimes**: `pin`, `drift_up`, `drift_down`, `compression`, `breakout`
- **Combinatoric lattice**: archetype × persistence tilt × vol multiplier =
  72 seeded, fully deterministic universes (`--full-lattice` runs them all;
  the default samples `--universes` per generation)
- **Evolution**: each generation re-weights sampling toward the archetypes
  the pipeline scored worst on — it spars hardest where it is weakest — and
  from generation 1 onward applies seeded Dirichlet jitter to both
  transition layers so no two generations replay identical dynamics
- **Attribution**: every generated minute is labeled (archetype, regime), so
  the report shows a **robustness matrix** (P&L / win rate per archetype;
  directional hit is charged to each universe's start archetype) and a
  **situation coverage map** — reported both as generated environment
  minutes and as the tick-stride subset the pipeline actually evaluated
  (the flags and dashboard use the evaluated counts)

## Reading the report

- **Robustness matrix** — a red `mean/sess` row is a market type where the
  pipeline loses money *in a world built from its own thesis*. That earns a
  `weak_archetype:<name>` flag and the amber tab dot.
- **Coverage map** — cells at `·` are situations the pipeline has not yet
  evaluated; the next generations fill them in.
- **Flags** — `no_recorded_tape`, `weak_archetype:*`,
  `promotion_pending_review`, `uncovered_situations`.

Reports persist to `journal.validation_reports` (`report_type='dojo'`,
served at `/api/dojo`) and as JSON under `reports/dojo/`.

## Honest limits (read this)

- The universe simulator is built from the system's **own thesis** (dealer
  gamma couples to price). Surviving all 72 universes does not prove live
  edge; **failing one is a real, attributable weakness.** The asymmetry is
  the point.
- Replaying the recorded tape more times adds no information — the holdout
  discipline in the learner (and the walk-forward embargo) exists precisely
  so accelerated training doesn't become memorization.
- The readiness gates (directional hit ≥ 52% over ≥ 100 resolved ticks on
  REAL data, Brier skill ≥ 0) still decide sizing. The dojo accelerates
  evaluation and hardening; it cannot substitute for live calendar
  diversity.
- The smile is a single-parameter linear-in-log-moneyness slope (no wings,
  no curvature), so the gym cannot surface wing- or convexity-shaped failure
  modes. The slope **is** now direction-coherent — an up-move (drift_up or an
  up-resolving breakout) bids the calls, a down-move steepens the puts, and
  breakout direction is biased by archetype (crash breaks down, squeeze
  breaks up) — but shape realism beyond the slope is future work.
- The transition matrices, OU parameters, and target levels are hand-tuned
  from the system's thesis, not estimated from the joint distribution of real
  SPY 0DTE sessions. Calibrating them from recorded data is the natural next
  step, with one caveat: the simulator's regimes (`pin`/`drift_*`/…) are a
  distinct taxonomy from the live Legacy/V3 regime labels, so real-data
  calibration needs a labeling bridge first — it is not a drop-in.
