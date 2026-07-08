# Feature Impact Testing Workflow

The standard process for evaluating **any new or modified feature in
`mtf_matrix.py`** (and, by extension, any config-level change worth measuring).
A feature earns a permanent registry entry or a regime-blend weight only after
it has been through this workflow; until then it is an experiment.

## The standard workflow

| Step | Action | Tool / Script | Output |
|---|---|---|---|
| 1 | Toggle the feature on/off via config | `configs/*.yaml` (`mtf.disabled_vars`) + `config_loader.py` | Controlled comparison |
| 2 | Run backtest with feature ON vs OFF | `backtest.py` (driven by step 5's script) | Quick delta |
| 3 | Run walk-forward on both versions | `walk_forward.py` (driven by step 5's script) | Out-of-sample consistency |
| 4 | Run shadow runner with the feature enabled | `shadow_runner.py` | Real-market behavior |
| 5 | Generate the Feature Impact Report | `scripts/feature_impact.py` | Standardized analysis (Markdown) |
| 6 | Log the report to the journal | `--db` flag (calls `journal.log_validation_report`, type `feature_impact`) | Persistent record, visible in the dashboard's Validation tab |
| 7 | Review and decide (Keep / Modify / Remove) | Manual + Validation tab | Documented decision (`--notes`) |

Steps 2, 3, 5 and 6 are one command:

```bash
python scripts/feature_impact.py \
  --feature bollinger_keltner_donchian_v1 \
  --baseline-config configs/baseline.yaml \
  --new-config configs/with_channels.yaml \
  --recorded /var/lib/zerodte/ticks \
  --output reports/feature_impact_$(date +%F).md \
  --db /var/lib/zerodte/shadow.db \
  --notes "channel vars v1 evaluation"
```

Omit `--recorded` to run on the coupled synthetic world (plumbing check only —
decisions should be made on recorded real ticks).

## What the report contains

- Delta in key metrics: Sharpe, win rate, mean P&L per trade, total P&L,
  max drawdown, EV accuracy
- Change in gate effectiveness (`gate_edge` = taken mean − blocked mean) and
  gate pass rate
- Per-regime impact (mean P&L of taken trades per GEX regime)
- Walk-forward consistency (profitable folds, mean out-of-sample Sharpe)
- Signal activity / noise: trade-count and gate-pass-rate changes (a large
  activity jump without a quality improvement usually means added noise)
- A clear recommendation tier: **Strong Positive / Positive / Neutral /
  Negative**, from a net-improvement vote across the scored metrics

RAS action quality is judged in step 4: run the shadow runner with the feature
enabled and watch `ras_evaluations` / the Trade Journal tab before making the
feature permanent.

## Defining a comparison pair

A config overlay describes a *delta* from the dataclass defaults:

```yaml
# configs/my_experiment.yaml
name: my_experiment
description: what this tests and why
mtf:
  disabled_vars: [some_var, another_var]   # switch matrix variables OFF
overrides:                                  # optional dataclass overrides
  gate.max_adx: 22.5
  classifier.min_dominant_confidence: 55
```

`configs/baseline.yaml` (channels OFF) and `configs/with_channels.yaml`
(everything ON) are the first real pair: the Bollinger/Keltner/Donchian
volatility-channel evaluation.

For a NEW variable the natural pair is: baseline = the new variable listed in
`disabled_vars`, variant = empty overlay (variable active).

## Rules of the road

- New `mtf_matrix.py` variables go through this workflow **before** they get a
  `_REGIME_DEF` weight or any gate/veto power (the same admission rule as
  `signals_json` observation-only signals — see `journal.py`).
- Prefer recorded ticks over synthetic; if only synthetic evidence exists, say
  so in `--notes` and re-run when enough sessions are recorded.
- Every evaluation is logged to `validation_reports` — the Validation dashboard
  tab is the single source of truth for what was tested and what was decided.
