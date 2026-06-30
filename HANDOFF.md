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
and `spread_selector` but currently have **separate snapshot types and separate
orchestrators**. Unifying them is the main outstanding integration work (§6).

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
| `live_feed_adapter.py` | Vendor-agnostic feed adapter; drives Track B; `route_ticket`. `CSVBarFeed` + `SyntheticFeed`. | `PipelineOrchestrator`, `MarketSnapshot`, `DataFeed`, `CSVBarFeed` | Works; routing is a stub (§6) |

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
- **`physical_pdf` is one object.** When wiring the Monte-Carlo close distribution,
  pass the *same* callable to `compute_edge` and `select_spreads`. It's the single
  source of truth for the edge measure.
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
1. **`route_ticket` is a router, not a filler** (Track B). It names the engine but
   doesn't call `spread_selector`, because `FeedSnapshot` carries no option chain.
   This is the #1 next task (§6).
2. **Two `MarketSnapshot` types.** `gate_scorer.MarketSnapshot` (rich, Track A) and
   `live_feed_adapter.MarketSnapshot` (lighter, Track B) overlap but differ. Unify
   or adapter-map them.
3. **Two orchestrators.** `orchestrator.Orchestrator` (journaling) and
   `live_feed_adapter.PipelineOrchestrator` (regime routing) are separate. The
   target is one loop: regime-route -> select -> gate -> journal.
4. **Fixed scales** (see §4) — not yet adaptive.
5. **CVD proxy** in `resample.py` returns 0 when closes sit mid-bar (only bites the
   synthetic generator; real bars are fine). Prefer a real signed-volume feed.
6. **`tick_two_sided` needs a real $TICK feed**; absent it the cell is `None` (by
   design, drops out).
7. **Directional engine not built.** Track B emits LCS/LC/LP/STG/BKS tickets, but
   only the premium families have a selector. `spread_selector` is credit-only.

**Bug fixed this session:** `decision_matrix._dominant` crashed (`None > float`) on
short history / session open when a timeframe basket had no computable regime.
Now guarded: ignores `None` regimes, and `decide_from_matrix` stands down cleanly
when a basket is undefined. (This is the version in this handoff.)

---

## 6. Next tasks (prioritized)

1. **Close the Track B → Track A seam (highest value).**
   - Add a `chain: ChainSnapshot` field to `live_feed_adapter.FeedSnapshot`, sourced
     from the options feed.
   - In `PipelineOrchestrator.route_ticket`, when the decision is a premium family,
     build the `GammaContext` from the snapshot, run `extract_rnd` + `compute_edge`
     + `select_spreads`, and return the winning `SpreadCandidate` (concrete strikes,
     credit, max_loss, Kelly size × `size_mult`). That turns a *named* trade into a
     *fillable* one.
2. **Unify the snapshot + orchestrator.** One `MarketSnapshot` (or a clean adapter),
   one tick loop that does regime-route -> select -> gate -> journal, so Track B's
   no-trades and would-be tickets land in the same SQLite journal as Track A.
3. **Wire the live `DataFeed`.** Implement `snapshot()` against the real vendor
   (Massive REST + S3 flat files per the existing infra): populate `RawBars`, the
   dealer/vol snapshot (from the GEX + RND modules), and the option chain.
4. **Adaptive scales.** Route `mtf_matrix`/`decision_matrix`/`resample` standardizer
   scales through `regime_classifier.ScaleBook`.
5. **Run shadow mode, then calibrate.** Journal every tick (no execution) for a few
   weeks, then regress realized P&L on the score components (`journal`
   `component_correlations` / `gate_effectiveness`) to set the weights and the
   `naked_gap_multiplier` from data instead of priors.
6. **Build the directional selector** (debit spreads / convex longs / strangles) so
   the LCS/LC/LP/STG/BKS cells become fillable, mirroring `spread_selector`'s
   structure (leg-based payoff, EV vs physical density, defined risk).

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
