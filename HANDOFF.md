# 0DTE GEX System — Claude Code Handoff

A GEX-driven 0DTE options system for SPY/XSP. The thesis: intraday 0DTE price is
dominated by dealer gamma hedging, so the edge is **structural** (regime + dealer
positioning), not indicator-based. The system measures the risk-neutral density,
detects the dealer/volatility regime, picks a structure, gates it, sizes it, and
journals **every** evaluation — trades *and* no-trades — so the gates can later be
proven against realized P&L.

**Operating model:** notification + manual execution (Robinhood / broker), not
auto-execution. The pipeline produces a *ticket*; a human places it.

> Not financial advice. This is decision-support tooling.

---

## 1. Environment

- Python 3.11+
- Deps: `numpy`, `scipy`, `pandas` (`pip install numpy scipy pandas`)
- Stdlib: `sqlite3`, `dataclasses`, `zoneinfo`
- No network, no API keys required to run the demos. Each module has a
  `__main__` demo: `python3 <module>.py`.

Quick verify (from the directory containing all modules):
```bash
for m in rnd_extractor gate_scorer spread_selector decision_engine journal \
         orchestrator regime_classifier mtf_matrix decision_matrix resample \
         live_feed_adapter; do python3 -c "import $m" && echo "ok $m"; done
```

---

## 2. The two tracks (read this first)

The system is **two partially-overlapping subsystems** that share `rnd_extractor`
and `spread_selector`. They were originally separate loops; **`unified_loop.py`
(`UnifiedOrchestrator`) now combines them into one tick** — Track A RND feeds the
matrix, Track B routes the structure, Track A fills the strikes, and everything
lands in one journal. `shadow_runner.py` drives that loop live against
`composite_feed.build_default_feed`. The per-track orchestrators below remain as
standalone harnesses for testing each track in isolation.

### Track A — Premium engine + measurement (the original spine)
```
MarketSnapshot (gate_scorer)  +  ChainSnapshot (rnd_extractor)
   -> rnd_extractor.extract_rnd / compute_edge        (risk-neutral density + edge)
   -> spread_selector.select_spreads                  (EV-ranked structure + strikes)
   -> gate_scorer.evaluate                            (hard gates + confidence)
   -> decision_engine.decide                          (compose -> TradeDecision)
   -> journal.log / settle_session                    (SQLite; trades AND no-trades)
   driven by orchestrator.Orchestrator (tick loop + settlement)
```
This track produces **fillable tickets** (concrete strikes, EV, max_loss, Kelly
size) and is the part with the measurement loop.

### Track B — Regime routing (the multi-timeframe layer)
```
raw bars (RawBars)  +  dealer/vol snapshot
   -> resample.build_mtf_input                        (bars -> per-TF native features)
   -> mtf_matrix.build_matrix / regime_rows           (110-var standardized matrix)
   -> decision_matrix.decide_from_matrix              (27-cell table -> TradeIntent)
   driven by live_feed_adapter.PipelineOrchestrator (run_once + route_ticket)
   (regime_classifier.py is the deterministic classifier behind the same idea)
```
This track decides **what kind of trade and how big** (structure family, direction,
conviction) but **stops short of strikes** — `route_ticket` names the engine
(`premium_selector` / `directional_selector`) without calling it, because no option
chain flows through the adapter yet.

**The seam:** Track B should route INTO Track A. `route_ticket` should, for premium
structures, call `spread_selector.select_spreads` with a chain and return the real
`SpreadCandidate`. See §6.

---

## 3. Module reference

| Module | Purpose | Key entry points | Status |
|---|---|---|---|
| `rnd_extractor.py` | Breeden-Litzenberger risk-neutral density from the 0DTE chain; physical-vs-RN edge (the `richness`/variance-ratio signal). | `extract_rnd`, `compute_edge`, `RiskNeutralDensity`, `EdgeReport` | Validated (round-trips a known density) |
| `gate_scorer.py` | Hard pre-trade gates (GEX regime, flip, term structure, ADX, catalyst) + weighted 0–100 confidence -> Kelly fraction. | `evaluate`, `MarketSnapshot`, `GateConfig`, `Decision` | Validated |
| `spread_selector.py` | Generates every defined-risk structure (+ optional naked/CSP), prices vs physical density, ranks by risk-adjusted EV. | `select_spreads`, `SpreadCandidate`, `GammaContext`, `SelectorConfig` | Validated |
| `decision_engine.py` | Pure composition: runs gate + selector independently, ANDs them, captures the would-be candidate on no-trades. | `decide`, `TradeDecision` | Validated |
| `journal.py` | SQLite persistence; settlement fills realized P&L for trades AND no-trades; `gate_effectiveness()`. | `Journal`, `log`, `settle_session`, `gate_effectiveness` | Validated |
| `orchestrator.py` | Track-A tick loop: feed -> engine -> journal; `settle()` post-close. `DataFeed` protocol. | `Orchestrator`, `DataFeed`, `SyntheticFeed` | Validated |
| `regime_classifier.py` | Deterministic regime classifier with adaptive `ScaleBook`, engine-specific vetoes, information gain. | `RegimeClassifier.classify`, `RegimeState` | Validated |
| `mtf_matrix.py` | Multi-timeframe standardized feature matrix (native vs snapshot vars) + per-TF regime rows. | `build_matrix`, `regime_rows`, `MTFInput` | Validated |
| `decision_matrix.py` | 27-cell (exec × context × direction) -> structure/conviction/size; dealer vetoes override. | `decide_from_matrix`, `DECISION_TABLE`, `TradeIntent` | Validated (hardened, see §5) |
| `resample.py` | Raw bars -> per-timeframe indicators (ADX/RSI/EMA/BB/RV/CVD/...) -> `MTFInput.native`. | `build_mtf_input`, `RawBars` | Validated |
| `live_feed_adapter.py` | Vendor-agnostic feed adapter; drives Track B standalone; `route_ticket`. `CSVBarFeed` + `SyntheticFeed`. | `PipelineOrchestrator`, `MarketSnapshot`, `DataFeed`, `CSVBarFeed` | Legacy harness; live path is `unified_loop` |
| `unified_loop.py` | **The** live tick loop: Track A RND + realized-vol physical pdf -> matrix -> Track B routing -> Track A fill -> risk -> journal. | `UnifiedOrchestrator`, `TickSnapshot`, `TickResult` | Validated (seam tests) |
| `shadow_runner.py` | Drives `UnifiedOrchestrator` live in no-order mode; auto-settle 4:15 ET; paper broker + dashboard state. | `ShadowRunner`, `--report` | Works |
| `composite_feed.py` | Live `DataFeed` with provider failover (Tradier -> Tastytrade -> Massive); Yahoo backstops bars/settlement only. | `build_default_feed` | Works |
| `chain_store.py` | Record live ticks (market+chain+incremental bars+settlements) to gzipped JSONL; replay them as a `DataFeed` — the missing piece for REAL-data walk-forward. `shadow_runner` records by default. | `ChainRecorder`, `RecordedFeed` | Validated (round-trip test) |
| `synthetic_world.py` | COUPLED synthetic market: GEX regime drives price dynamics, chains reprice off the live path each tick, settlement = the path's close. Makes prediction measurable in backtests (the frozen-chain `SyntheticUnifiedFeed` cannot). | `CoupledSyntheticFeed`, `WorldConfig` | Validated |

### Dependency graph
```
rnd_extractor ──┬─> spread_selector ──┐
                └─> (edge)            ├─> decision_engine ──> orchestrator ──> journal
gate_scorer ─────────────────────────┘
                                        (Track A)

resample ──> mtf_matrix ──> decision_matrix ──> live_feed_adapter   (Track B)
regime_classifier  (standalone; same idea as decision_matrix's regime layer)
```
`mtf_matrix.py` has **no external deps** beyond stdlib `math`. `resample.py` needs
`pandas`. `rnd_extractor.py` needs `scipy`.

---

## 4. Conventions (follow these when extending)

- **Standardization:** every feature maps to 0–100. Helpers (`clip100`, `P`, `S`,
  `N`) live in `mtf_matrix.py` / `regime_classifier.py`. Magnitude/level vars use
  100 = strong; directional vars use `S()` with 50 = neutral.
- **No-trades are first-class.** Never drop a no-trade. `decision_engine` always
  records the would-be candidate; `journal.settle_session` fills its *hypothetical*
  realized P&L. This is the entire measurement thesis — `gate_effectiveness()`
  compares trades taken vs trades the gate blocked.
- **Hard gates are multiplicative, scores are additive.** A veto (short gamma,
  catalyst, below flip) zeroes a structure regardless of how good its score is.
  Keep that separation; do not let a confidence score outvote a regime fact.
- **`physical_pdf` is one object.** Pass the *same* callable to `compute_edge`
  and `select_spreads` — it's the single source of truth for the edge measure.
  `UnifiedOrchestrator.tick` builds it per tick from **EWMA realized vol of the
  1-min bars** (`rnd_extractor.ewma_realized_vol` +
  `physical_pdf_from_realized_vol`: the RND's shape, rescaled to the realized-vol
  forecast). The static `vol_risk_premium` haircut inside `compute_edge` is a
  last-resort fallback only — with it, the variance ratio (and thus `richness`)
  is a constant by construction, so never rely on it in a live path.
- **Adaptive scales (TODO, important).** Most `*_scale` constants in `mtf_matrix.py`
  and `decision_matrix`/`resample` are fixed priors. They should be sourced from
  `regime_classifier.ScaleBook` (trailing distributions) so a feature reads ~50 at
  its own recent median. Until then, cross-variable comparisons are approximate.

---

## 5. What's validated vs known issues

**Validated (with tests in-session):**
- RND extractor round-trips a known skewed density (forward, std, skew, area, arb).
- Selector enforces "highest EV per unit of tail risk" (passed over a 4× higher
  EV/risk condor whose shorts were undefended).
- Naked/CSP families: correct stop-defined / price-to-zero risk accounting; hard
  gating (long-gamma + defended wall + high GEX rank); off by default; gap-risk
  haircut so defined structures outrank naked.
- Journal settlement + `gate_effectiveness` produce the trades-vs-blocked readout.
- Regime classifier separates opposite regimes and routes the permitted engine;
  information gain spikes on regime flips.
- Full stack imports clean; both tracks run end to end (verified this handoff).

**Known issues / honest gaps:**
1. ~~`route_ticket` is a router, not a filler~~ **Superseded by `unified_loop.py`:**
   the unified tick passes Track B's structure/direction into
   `decision_engine.decide`, which fills concrete strikes via `spread_selector`.
   `live_feed_adapter.route_ticket` remains a stub but is not on the live path.
2. **Two `MarketSnapshot` types** still exist (`gate_scorer` vs
   `live_feed_adapter`), but only `gate_scorer.MarketSnapshot` is on the live
   path (`unified_loop.TickSnapshot`). The adapter's type is legacy/test-only.
3. ~~Two orchestrators~~ **Resolved:** `unified_loop.UnifiedOrchestrator` is the
   one loop (regime-route -> select -> gate -> risk -> journal), driven live by
   `shadow_runner.py`. `orchestrator.py` / `live_feed_adapter.PipelineOrchestrator`
   are per-track harnesses.
4. **Fixed scales** (see §4) — not yet adaptive.
4b. **OI-based GEX is stale intraday.** The gamma map weights by open interest,
   which updates overnight; most 0DTE volume opens and closes intraday without
   printing to OI, and the customer-long-puts sign convention is most contested
   precisely for 0DTE flow. Flip/walls/regime and the dealer vetoes all sit on
   this map. Mitigations to evaluate via the journal (not by fiat): include
   front weekly expiries, and/or a parallel intraday volume-weighted gamma
   proxy, then let `gate_effectiveness()` arbitrate which flip/walls predict.
5. **CVD proxy** in `resample.py` returns 0 when closes sit mid-bar (only bites the
   synthetic generator; real bars are fine). Prefer a real signed-volume feed.
6. **`tick_two_sided` needs a real $TICK feed**; absent it the cell is `None` (by
   design, drops out).
7. ~~Directional engine not built~~ **Wrong — and now actually unblocked.**
   `spread_selector` has had the debit families (`long_call_spread`,
   `long_put_spread`, `long_call`, `long_put`, `long_strangle`, backspreads)
   all along; what kept the directional engine dead in the live loop was:
   - `decision_engine.decide` ANDed the **premium-selling** gate (no trend,
     above flip, strong long gamma) against every structure — the exact tape
     a debit trade wants is the tape that gate forbids. Fixed:
     `gate_scorer.evaluate(structure_class="directional")` applies only the
     universal stops (catalyst, late lockout) and scores trend quality
     (`score_directional`: ADX presence, sign-aligned flow, dealer
     amplification, vol value, timing — weights in `GateConfig`, journal-
     calibratable).
   - `regime_classifier` veto names (`short_gamma_regime`/`below_gamma_flip`)
     never matched `decision_matrix`'s premium-veto set (`short_gamma`/
     `below_flip`), so the intended credit→debit flip was dead code. Fixed
     with `NO_PREMIUM_VETOES` accepting both conventions.
   - A drift-less physical density prices every debit at EV≤0 by
     construction. A resolved directional intent now tilts the realized-vol
     density (`dir_drift_frac` × conviction × phys std, in `RNDConfig`) for
     the fill only; the richness measurement stays drift-less.

**Bug fixed this session:** `decision_matrix._dominant` crashed (`None > float`) on
short history / session open when a timeframe basket had no computable regime.
Now guarded: ignores `None` regimes, and `decide_from_matrix` stands down cleanly
when a basket is undefined. (This is the version in this handoff.)

**Adaptive state now persists.** The |GEX| percentile window
(`gex_window.GexRankWindow`: abs-magnitude, multi-day horizon, neutral 0.5
until 30 samples, shared by all three feeds) and both ScaleBooks (matrix +
classifier, via `UnifiedOrchestrator(state_path=...)`) survive restarts as
JSON next to the journal DB. Before this, every deploy cold-started the gate
(`gex_pct_rank` pinned at an extreme) and washed the direction bias out to
neutral.

---

## 6. Next tasks (prioritized)

Done since this handoff was written: the Track B → Track A seam, the unified
orchestrator (`unified_loop.py`), and the live `DataFeed`
(`composite_feed.build_default_feed`: Tradier → Tastytrade → Massive failover,
Yahoo quarantined to bars/settlement). Remaining, in order:

1. **Run shadow mode, then calibrate.** Journal every tick (no execution) for a
   few weeks, then regress realized P&L on the score components (`journal`
   `component_correlations` / `gate_effectiveness`) to set the weights, the
   `naked_gap_multiplier`, and the `mc.MCConfig` knobs from data instead of
   priors. Don't build new modules before this loop closes.
   **Predictive-power readouts now exist and gate readiness:**
   `journal.directional_accuracy()` (bias vs realized move, scored on EVERY
   settled tick incl. no-trades), `journal.prob_calibration()` (Brier + skill
   + reliability bins), and `journal.calibration()` (the readout `mc.py`
   promises). The dashboard readiness checklist blocks on them: directional
   hit >= 52% over >= 100 resolved-bias ticks, Brier skill >= 0, |EV bias|
   <= $0.10/share. Profitability without prediction is luck; these force the
   distinction before sizing up.
   **Real-data walk-forward is now buildable:** `shadow_runner` records every
   tick via `chain_store.ChainRecorder` (default `<db_dir>/ticks`, ~1 MB/day);
   after a few weeks, `RecordedFeed(dir)` + `run_walk_forward` is an
   out-of-sample test on actual markets. `optimizer.OptimizerConfig
   (holdout_frac=0.2)` keeps a final untouched window the search never sees.
2. **Arbitrate the GEX measurement** (§5 item 4b) with the shadow journal:
   OI-only vs front-weeklies-included vs intraday volume-weighted proxy.
3. **Adaptive scales — partially done.** `build_matrix` already routes through
   `ScaleBook` in the live loop and the books now persist across restarts;
   `decision_matrix`/`resample` fixed priors remain for their residual
   constants.
4. **Calibrate the directional priors.** The directional path is now live end
   to end (see §5 item 7): `GateConfig.w_dir_*`, `dir_adx_floor/full`, and
   `RNDConfig.dir_drift_frac` are structured guesses. The journal settles
   every directional would-be candidate — regress and adjust from data.

---

## 7. How to run things

```bash
# Track A — premium selection on a synthetic chain
python3 spread_selector.py

# Track A — RND extractor self-validation (round-trip)
python3 rnd_extractor.py

# Track A — journaling tick loop + settlement + gate effectiveness
python3 orchestrator.py        # (uses its SyntheticFeed)

# Track B — full regime-routing pipeline from synthetic bars
python3 live_feed_adapter.py   # prints the matrix + routed ticket

# Track B — the multi-timeframe matrix alone
python3 mtf_matrix.py
python3 decision_matrix.py     # full 27-cell table + live decision

# Regime classifier across opposite regimes
python3 regime_classifier.py
```

For local replay with real data, use `live_feed_adapter.CSVBarFeed` with a
1-minute OHLCV CSV (`timestamp, open, high, low, close, volume`; optional
`signed_volume`, `tick`).

---

## 8. Safety / guardrails already in the code

- Catalyst is a hard stop (blocks every engine). Short gamma / below flip / term
  backwardation block premium selling but permit directional/vol engines.
- Naked structures are **off by default**, gated to strong-long-gamma + defended
  wall + high GEX rank, capped at a small size, and haircut for gap risk in ranking.
- The selector's core rule is risk-adjusted, not raw EV: it will refuse a
  higher-EV trade whose tail is undefended.
- Everything is decision-support for manual execution; nothing here places orders.
