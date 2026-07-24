# Forecast stabilizer — whiplash control for the cone

`forecast_stabilizer.py` is the piece the sigma cone was missing: it takes a
noisy per-tick forecast target and produces a stable one, without the lag that
naïve smoothing introduces at exactly the wrong moment. It separates the three
concerns a trading forecast has to keep apart.

## The problem

A raw per-tick forecast (the median close/target the cone points at) jumps
every bar. Showing it unfiltered makes the cone whip around; a plain EMA lags
when the market genuinely turns. The fix is not "smooth harder" — it is to
distinguish *underlying forecast change*, *presentation-layer stabilization*,
and *a true regime break*.

## The four mechanisms

1. **Confidence-weighted inertia.** `y_hat_t = (1-α)·y_hat_{t-1} + α·y_raw`,
   with `α = clip(α_base · C · R, α_min, α_max)` where `C` is model confidence
   and `R` is regime-change intensity, both in `[0,1]`. Low confidence in a
   stable regime → the target creeps; high confidence during a regime change →
   it moves fast. `α_min > 0` means it never fully freezes.

2. **Adaptive deadband.** The visible target only moves when
   `|y_hat_t − y_hat_{t-1}| > k·σ_short`. The threshold scales with
   short-horizon volatility, so a 10-cent wiggle is ignored in a choppy tape
   and acted on in a dead-calm one.

3. **Directional hysteresis.** The deadband is scaled by the transition the
   raw move implies relative to the current lean (target above/below spot):
   `continue` (extend the lean) is cheapest, `neutralize` (pull back toward
   spot) costs more, `reverse` (cross spot) costs the most. Bar noise can't
   flip the cone bullish/bearish. The classification is reported even when the
   move is held, so a caller can see *why* it held.

4. **Structural-break override.** A genuine break — VWAP loss with volume,
   gamma-flip breach + failed reclaim, wall breakout with acceptance, abrupt
   vol expansion, order-flow reversal, correlation shock, macro release —
   bypasses the deadband and hysteresis and snaps the target toward the raw
   reading. Smoothing must not be slow and stupid exactly when the market
   changes.

## API

```python
from forecast_stabilizer import ForecastStabilizer, StabilizerConfig, BreakSignals

st = ForecastStabilizer(StabilizerConfig())     # one per session
st.reset()                                        # at each session open

r = st.update(
    raw_target=743.20, spot=742.00, sigma_short=0.25,
    confidence=0.61, regime_change_intensity=0.4,
    breaks=BreakSignals(vwap_loss_confirmed=False),
)
r.target        # the stabilized target to render
r.changed       # did the visible target move?
r.transition    # "seed" | "continue" | "neutralize" | "reverse"
r.override      # active break signals, empty if none
r.to_dict()     # JSON for the live state / dashboard
```

Pure, deterministic, no numpy/IO — safe on the live path and trivially tested
(`tests/test_forecast_stabilizer.py`).

## Where it fits (Stage 2)

This module is the foundation; the **cone visualization rebuild** (next PR)
consumes `r.target` as the stabilized centerline and layers the existing V3
outputs around it — the asymmetric quantile fan (`return_distribution` /
`return_quantiles`), barrier-touch probabilities (`models/barrier_touch`),
remaining-session high/low (`models/range_survival`), and regime-mixture
confidence (`models/regime_moe`). The stabilizer lands first so the surfaced
cone is stable from day one. Live wiring (extracting `σ_short`, `C`, `R`, and
the break signals from each tick's snapshot) happens in that Stage-2 PR, next
to the consumer.
