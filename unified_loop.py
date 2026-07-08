"""
unified_loop.py
===============
Single tick loop combining Track B (regime routing) and Track A (premium engine).

Per tick:
  1. Track A RND first — extract_rnd + compute_edge from the chain (if present).
     RND-derived richness/skew/kurtosis are injected into the mtf_snapshot dict
     so the matrix sees them as SNAPSHOT variables.
  2. Track B — resample bars -> build_matrix -> regime_classifier.classify ->
     decide_from_matrix. Produces a TradeIntent (structure family, conviction,
     size_mult) and a RegimeState (dominant_regime, permitted_engine, stand_down).
  3. Combine — if regime stands down, or TradeIntent is NT, log a NO_TRADE row
     and return. Otherwise run Track A decide() for a concrete SpreadCandidate.
  4. Size — final_size_mult = intent.size_mult. Track A's gate and selector
     veto independently; the regime multiplier scales the position on top.
  5. Journal every tick (trade and no-trade), because no-trades are first-class.

DataFeed protocol (superset of both prior orchestrator protocols):
    snapshot(now: datetime) -> Optional[TickSnapshot]
    settlement_price(session_date: str) -> Optional[float]

TickSnapshot bundles everything both tracks need in one place:
    market: gate_scorer.MarketSnapshot   (has .mtf_snapshot() + .dealer_vetoes())
    bars:   resample.RawBars             (Track B indicator computation)
    chain:  Optional[ChainSnapshot]      (Track A options pricing; None = no data yet)

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from gate_scorer import MarketSnapshot
from rnd_extractor import (
    ChainSnapshot, extract_rnd, compute_edge, RNDConfig,
    ewma_realized_vol, physical_pdf_from_realized_vol,
)
from decision_engine import decide, EngineConfig, TradeDecision
from resample import RawBars, build_mtf_input
from mtf_matrix import build_matrix, regime_rows
from decision_matrix import decide_from_matrix, TradeIntent
from regime_classifier import RegimeClassifier, RegimeState, ClassifierContext, ClassifierConfig, ScaleBook
from regime_alignment import (
    PositionContext, RASConfig, RASResult, compute_ras, ras_to_signals,
)
from journal import Journal
from market_dynamics import DynamicsWindow, session_open_from_bars
from risk_manager import RiskManager

ET = ZoneInfo("America/New_York")

log = logging.getLogger("unified_loop")


# --------------------------------------------------------------------------- #
# Unified tick bundle                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class TickSnapshot:
    market: MarketSnapshot
    bars: RawBars
    chain: Optional[ChainSnapshot] = None


@dataclass
class TickResult:
    ts: dt.datetime
    regime: RegimeState
    intent: TradeIntent
    decision: Optional[TradeDecision]
    final_size_mult: float      # intent.size_mult, 0 if regime stand_down
    vetoes: list
    snapshot: Optional[TickSnapshot] = None   # live market data for paper marking
    ras_results: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# DataFeed protocol                                                            #
# --------------------------------------------------------------------------- #
class DataFeed(Protocol):
    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]: ...
    def settlement_price(self, session_date: str) -> Optional[float]: ...


# --------------------------------------------------------------------------- #
# Unified Orchestrator                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class UnifiedOrchestrator:
    feed: DataFeed
    journal: Optional[Journal] = None
    engine_cfg: Optional[EngineConfig] = None
    classifier_cfg: Optional[ClassifierConfig] = None
    physical_pdf: Optional[Callable] = None     # callable(grid)->density for Track A
    risk_manager: Optional[RiskManager] = None
    state_path: Optional[str] = None            # persist adaptive scales across restarts
    ras_cfg: Optional[RASConfig] = None

    def __post_init__(self):
        self._classifier = RegimeClassifier(
            cfg=self.classifier_cfg or ClassifierConfig()
        )
        self._prev_std: Optional[dict] = None   # for information-gain computation
        self._matrix_scale_book = ScaleBook()   # adaptive scales for MTF matrix variables
        self._ticks_since_save = 0
        # dealer-surface / vol-state derivatives (observation-only signals)
        dyn_path = None
        if self.state_path:
            import os
            dyn_path = os.path.join(os.path.dirname(self.state_path) or ".",
                                    "dynamics_state.json")
        self._dynamics = DynamicsWindow(path=dyn_path)
        self._load_state()

    # -- adaptive-state persistence -------------------------------------------
    # The ScaleBooks ARE the system's memory of what "normal" looks like; if
    # they die with the process, every restart re-runs the cold start where
    # slope/flow variables read ~50 and the direction bias washes out to
    # neutral. Best-effort JSON: corrupt or missing state just re-warms.
    def _load_state(self) -> None:
        if not self.state_path:
            return
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            self._matrix_scale_book.load_dict(data.get("matrix_scales", {}))
            self._classifier.scales.load_dict(data.get("classifier_scales", {}))
        except Exception:
            pass

    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            import os
            import tempfile
            payload = {
                "matrix_scales": self._matrix_scale_book.to_dict(),
                "classifier_scales": self._classifier.scales.to_dict(),
            }
            directory = os.path.dirname(self.state_path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".adaptive_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self.state_path)
        except Exception:
            pass                                 # never let persistence break a tick

    def _compute_ras(self, regime_state: RegimeState, intent: TradeIntent,
                     market: MarketSnapshot,
                     position_contexts: Optional[list]) -> list:
        if not position_contexts:
            return []
        cfg = self.ras_cfg or RASConfig()
        results: list[RASResult] = []
        for ctx in position_contexts:
            try:
                results.append(compute_ras(
                    regime_state, intent, market, ctx, cfg=cfg))
            except Exception as exc:
                # Broken evaluations must be visible, not silent: an
                # unmonitored position looks exactly like a healthy one.
                log.warning("RAS evaluation failed for position %s: %s",
                            ctx.position_id, exc)
        return results

    def _journal_ras(self, now: dt.datetime, ras_results: list) -> None:
        """One ras_evaluations row per open position per tick. Best-effort:
        journaling must never break a tick."""
        if self.journal is None or not ras_results:
            return
        session_date = now.astimezone(ET).date().isoformat()
        for ras in ras_results:
            try:
                self.journal.log_ras(now.isoformat(), session_date, ras)
            except Exception as exc:
                log.warning("RAS journaling failed for position %s: %s",
                            ras.position_id, exc)

    @staticmethod
    def _signals_with_ras(signals: dict, ras_results: list) -> tuple[dict, Optional[str]]:
        if not ras_results:
            return signals, (json.dumps({k: round(v, 6) for k, v in signals.items()
                                        if isinstance(v, (int, float))})
                             if signals else None)
        merged = dict(signals)
        # Flatten only the WORST-scoring position: with several open positions
        # the ras_* keys would otherwise overwrite each other arbitrarily, and
        # the minimum score is the correlation-relevant health signal anyway.
        # Full per-position detail lands in journal.ras_evaluations.
        worst = min(ras_results, key=lambda r: r.score)
        merged.update(ras_to_signals(worst))
        signals_json = json.dumps(
            {k: (v if isinstance(v, str) else round(v, 6))
             for k, v in merged.items()
             if isinstance(v, (int, float, str))}
        ) if merged else None
        return merged, signals_json

    def tick(self, now: dt.datetime,
             position_contexts: Optional[list[PositionContext]] = None
             ) -> Optional[TickResult]:
        snap = self.feed.snapshot(now)
        if snap is None:
            return None

        cfg = self.engine_cfg or EngineConfig()

        # ---- Track A RND (feeds both regime and selector) ----
        # One physical density per tick, shared by compute_edge and decide()
        # (single source of truth). Priority: injected callable > realized-vol
        # squeeze of the RND (from the tick's own bars) > static VRP haircut
        # inside compute_edge. Without the realized-vol step the variance
        # ratio — and thus `richness` — is a constant by construction.
        rnd = edge = None
        sigma_rv = None
        phys_pdf = self.physical_pdf
        if snap.chain is not None:
            try:
                rnd = extract_rnd(snap.chain, cfg.rnd)
                if phys_pdf is None:
                    sigma_rv = _safe_realized_sigma(snap.bars, cfg.rnd)
                    if sigma_rv is not None:
                        phys_pdf = physical_pdf_from_realized_vol(rnd, sigma_rv, cfg.rnd)
                edge = compute_edge(rnd, snap.chain, cfg.rnd,
                                    physical_pdf=phys_pdf)
            except Exception:
                pass

        # ---- Build mtf snapshot, inject RND-derived vars ----
        snap_dict = snap.market.mtf_snapshot()
        if edge is not None:
            snap_dict["richness"] = edge.richness_signal
        if rnd is not None:
            try:
                snap_dict["skew_dir"] = rnd.skew()
                snap_dict["tail_heaviness"] = rnd.excess_kurtosis()
            except Exception:
                pass

        # ---- Observation-only orthogonal signals (admission pipeline) ----
        # Dealer-surface derivatives + expected-move-consumed from the
        # dynamics window; flow/breadth extras from the feed. They render in
        # the matrix and land in signals_json for component_correlations to
        # score — nothing downstream gates or vetoes on them yet.
        m = snap.market
        signals: dict = {}
        try:
            sess_open = (session_open_from_bars(snap.bars, now)
                         if snap.bars is not None else None)
            signals = self._dynamics.update(
                now.timestamp(), spot=m.spot, gamma_flip=m.gamma_flip,
                call_wall=m.call_wall, put_wall=m.put_wall, net_gex=m.net_gex,
                straddle_be=m.straddle_breakeven, session_open=sess_open,
            )
        except Exception:
            signals = {}
        for k in ("pcr_volume", "volume_oi_ratio", "rsp_spy_div",
                  "sector_align", "top10_pressure"):
            v = getattr(m, k, None)
            if isinstance(v, (int, float)) and math.isfinite(v):
                signals[k] = v
        snap_dict.update(signals)
        # signals_json finalized after RAS merge below

        # ---- Track B: regime classifier ----
        clf_ctx = ClassifierContext(market=snap.market, rnd=rnd, edge=edge)
        regime_state = self._classifier.classify(clf_ctx, self._prev_std)
        self._prev_std = regime_state.standardized

        # periodic flush of adaptive scales (cheap; ~every 10 minutes at 60s ticks)
        self._ticks_since_save += 1
        if self._ticks_since_save >= 10:
            self._save_state()
            self._ticks_since_save = 0

        # ---- Track B: matrix + decision routing ----
        mtf_in = build_mtf_input(snap.bars, snap_dict)
        mat_rows = build_matrix(mtf_in, self._matrix_scale_book)
        regimes = regime_rows(mat_rows)
        intent = decide_from_matrix(mat_rows, regimes, vetoes=regime_state.vetoes)

        # Observation-only regime time series for the dashboard (chart shading
        # + quadrant view): the continuous direction-bias value (0-100, 50 =
        # neutral) and the dominant regime's confidence. Journaled in
        # signals_json so no schema change and zero gate/veto power.
        if isinstance(intent.bias_value, (int, float)) and math.isfinite(intent.bias_value):
            signals["regime_bias_value"] = float(intent.bias_value)
        dom_conf = regime_state.confidences.get(regime_state.dominant_regime)
        if isinstance(dom_conf, (int, float)) and math.isfinite(dom_conf):
            signals["regime_dominant_conf"] = float(dom_conf)

        ras_results = self._compute_ras(
            regime_state, intent, snap.market, position_contexts)
        self._journal_ras(now, ras_results)
        signals, signals_json = self._signals_with_ras(signals, ras_results)

        # ---- Stand-down: regime unstable or NT cell ----
        if regime_state.stand_down or intent.decision.structure == "NT":
            result = TickResult(
                ts=now, regime=regime_state, intent=intent,
                decision=None, final_size_mult=0.0,
                vetoes=regime_state.vetoes, snapshot=snap,
                ras_results=ras_results,
            )
            if self.journal:
                self.journal.log(_no_trade_row(snap.market, intent, regime_state,
                                               direction=intent.decision.direction,
                                               signals_json=signals_json))
            return result

        # ---- Track A: full engine (requires chain) ----
        # A resolved directional intent carries a drift belief. Encode it as a
        # tilt of the same realized-vol density (fraction of phys std, scaled
        # by conviction) so debit structures aren't priced against a density
        # that says the market goes nowhere. The tick-level edge/richness above
        # stays drift-less — variance, not direction, is that measurement.
        decision = None
        if snap.chain is not None:
            decide_pdf = phys_pdf
            if (self.physical_pdf is None and rnd is not None and sigma_rv is not None
                    and intent.decision.structure in DIRECTIONAL_TILT_STRUCTURES):
                sign = 1.0 if intent.decision.direction == "call" else -1.0
                tilt = sign * cfg.rnd.dir_drift_frac * intent.size_mult
                tilted = physical_pdf_from_realized_vol(rnd, sigma_rv, cfg.rnd,
                                                        drift_std_frac=tilt)
                if tilted is not None:
                    decide_pdf = tilted
            decision = decide(snap.market, snap.chain, cfg,
                              physical_pdf=decide_pdf,
                              target_structure=intent.decision.structure,
                              direction=intent.decision.direction)
            # ---- Risk gate (optional, applied before journaling) ----
            if (self.risk_manager is not None
                    and decision.decision == "TRADE"
                    and decision.candidate is not None):
                session_date = now.astimezone(ET).date().isoformat()
                rcheck = self.risk_manager.check(decision.candidate, session_date)
                if not rcheck.approved:
                    decision = dataclasses.replace(
                        decision,
                        decision="NO_TRADE",
                        no_trade_reason="risk:" + ",".join(rcheck.vetoes),
                    )
                else:
                    self.risk_manager.record_trade(decision.candidate, session_date)
            if self.journal:
                row = decision.as_row()
                row["signals_json"] = signals_json
                self.journal.log(row)
        else:
            # No chain yet — log intent as a no-trade stub for calibration
            if self.journal:
                self.journal.log(_no_trade_row(snap.market, intent, regime_state,
                                               reason="no_chain",
                                               direction=intent.decision.direction,
                                               signals_json=signals_json))

        # size_mult from Track B scales the Track A position
        final_size = intent.size_mult if (decision is not None
                                          and decision.decision == "TRADE") else 0.0

        return TickResult(
            ts=now, regime=regime_state, intent=intent,
            decision=decision,
            final_size_mult=round(final_size, 2),
            vetoes=regime_state.vetoes, snapshot=snap,
            ras_results=ras_results,
        )

    def run_replay(self, timestamps: Sequence[dt.datetime]) -> list[TickResult]:
        out = []
        for t in timestamps:
            r = self.tick(t)
            if r is not None:
                out.append(r)
        return out

    def run_live(self, interval_seconds: int, until: dt.datetime,
                 clock=None) -> list[TickResult]:
        if clock is None:
            clock = lambda: dt.datetime.now(ET)
        out = []
        while clock() < until:
            r = self.tick(clock())
            if r is not None:
                out.append(r)
            time.sleep(interval_seconds)
        return out

    def settle(self, session_date: str) -> int:
        self._save_state()                       # end-of-day flush of adaptive scales
        if self.journal is None:
            return 0
        price = self.feed.settlement_price(session_date)
        if price is None:
            return 0
        return self.journal.settle_session(session_date, price)


# --------------------------------------------------------------------------- #
# Default physical density: realized vol from the tick's own bars             #
# --------------------------------------------------------------------------- #
# Debit structures whose fill should be priced against the drift-tilted
# density (the resolved bias IS the drift belief). STG is direction-"both"
# long vol — it gets the drift-less density like everything else.
DIRECTIONAL_TILT_STRUCTURES = frozenset({"LCS", "LPS", "LC", "LP", "BKS"})


def _safe_realized_sigma(bars: Optional[RawBars], cfg: RNDConfig) -> Optional[float]:
    """EWMA realized vol from 1-min bars; None (never raises) when too thin."""
    try:
        return ewma_realized_vol(bars.ts, bars.close, cfg)
    except Exception:
        return None


def _realized_vol_pdf(rnd, bars: RawBars, cfg: RNDConfig):
    """
    EWMA realized vol from the 1-min bars, imposed on the RND's shape.
    Returns None (never raises) when the bar history is too thin or degenerate,
    letting compute_edge fall back to the static VRP haircut.
    """
    sigma = _safe_realized_sigma(bars, cfg)
    if sigma is None:
        return None
    try:
        return physical_pdf_from_realized_vol(rnd, sigma, cfg)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Row builder for no-trade / no-chain ticks                                   #
# --------------------------------------------------------------------------- #
def _no_trade_row(market: MarketSnapshot, intent: TradeIntent,
                  regime: RegimeState, reason: str = "",
                  direction: str = "", signals_json=None) -> dict:
    now = market.now
    session_date = now.astimezone(ET).date().isoformat()
    gex_regime = "long" if market.net_gex > 0 else ("short" if market.net_gex < 0 else "flat")
    zg = market.spot - market.gamma_flip
    no_reason = reason or ("regime_nt" if intent.decision.structure == "NT"
                           else f"stand_down:{regime.dominant_regime}")
    return {
        "session_date": session_date,
        "ts": now.isoformat(),
        "spot": market.spot,
        "net_gex": market.net_gex,
        "gex_regime": gex_regime,
        "gex_pct_rank": market.gex_pct_rank,
        "zero_gamma_dist": zg,
        "zero_gamma_dist_pct": zg / market.spot,
        "adx": market.adx,
        "call_wall": market.call_wall,
        "put_wall": market.put_wall,
        "selected_family": (intent.decision.structure
                            if intent.decision.structure != "NT" else None),
        "short_strikes": None, "long_strikes": None, "legs_json": None,
        "credit": None, "candidate_score": None, "ev": None,
        "max_loss": None, "ev_per_risk": None,
        "theta": None, "gamma": None,
        "prob_profit": None, "prob_touch_short": None,
        "liquidity_score": None, "wall_safety": None,
        "gamma_safety": None, "touch_safety": None,
        "gate_pass": 0, "gate_score": 0.0,
        "gate_failed": json.dumps([no_reason]),
        "veto_reasons": json.dumps(intent.vetoes),
        "decision": "NO_TRADE",
        "no_trade_reason": no_reason,
        "was_traded": 0,
        "candidate_present": 0,
        "regime_direction": direction or intent.decision.direction,
        "signals_json": signals_json,
    }


# --------------------------------------------------------------------------- #
# Synthetic feed for replay / tests                                            #
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticUnifiedFeed:
    """
    Builds a multi-day bar stream and a static market snapshot.
    Optionally injects a ChainSnapshot at every tick for Track A testing.
    """
    days: int = 20
    seed: int = 7
    base_spot: float = 600.0
    settle: float = 600.0
    chain: Optional[ChainSnapshot] = None       # inject a fixed chain for seam testing
    _raw: RawBars = field(init=False)
    _market: MarketSnapshot = field(init=False)
    _ts_iter: object = field(init=False)

    def __post_init__(self):
        from resample import _synth_bars
        self._raw = _synth_bars(days=self.days, seed=self.seed)
        spot = float(self._raw.close[-1])
        self._market = MarketSnapshot(
            spot=spot, net_gex=4.0e9, gamma_flip=spot - 6.0,
            call_wall=spot + 5.0, put_wall=spot - 5.0, gex_pct_rank=0.86,
            vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
            straddle_breakeven=4.0, expected_range=3.2,
            adx=13.0, rsi=51.0, bb_width=1.4, bb_width_baseline=2.0,
            vwap=spot, vwap_reversion_count=3,
            tick_abs_mean=480.0, cvd_slope=0.02,
            now=dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET),
            has_catalyst=False,
        )
        # Walk through timestamps one tick at a time
        self._idx = 0

    def snapshot(self, now: dt.datetime) -> Optional[TickSnapshot]:
        i = self._idx + 1
        if i > len(self._raw.close):
            return None
        self._idx = i
        # rolling bar window up to current bar
        bars = RawBars(
            ts=self._raw.ts[:i], open=self._raw.open[:i], high=self._raw.high[:i],
            low=self._raw.low[:i], close=self._raw.close[:i], volume=self._raw.volume[:i],
        )
        # update spot from last close
        import dataclasses
        market = dataclasses.replace(self._market,
                                     spot=float(self._raw.close[i - 1]),
                                     now=now)
        return TickSnapshot(market=market, bars=bars, chain=self.chain)

    def settlement_price(self, session_date: str) -> Optional[float]:
        return self.settle


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from journal import Journal

    # ---- no-chain run (Track B only, no options data) ----
    print("=== Unified loop — no chain (regime routing only) ===")
    feed = SyntheticUnifiedFeed(days=5)
    jrn = Journal(":memory:")
    orch = UnifiedOrchestrator(feed=feed, journal=jrn)

    start = dt.datetime(2026, 6, 26, 9, 30, tzinfo=ET)
    ticks = [start + dt.timedelta(minutes=i) for i in range(5 * 390)]
    results = orch.run_replay(ticks)

    trades = [r for r in results if r.decision is not None and r.decision.decision == "TRADE"]
    standed = [r for r in results if r.final_size_mult == 0.0]
    print(f"  {len(results)} ticks  |  {len(trades)} TRADE  |  {len(standed)} stand-down/NT")
    if results:
        last = results[-1]
        print(f"  last tick: regime={last.regime.dominant_regime} "
              f"engine={last.regime.permitted_engine} "
              f"struct={last.intent.decision.structure} "
              f"size_mult={last.final_size_mult}")

    # ---- with chain (full Track A seam) ----
    print("\n=== Unified loop — with chain (full Track A seam) ===")
    from rnd_extractor import ChainSnapshot, ChainQuote, _bs_call_fwd
    spot0 = 600.0
    T0, r0 = 4.0 / (24 * 365), 0.05
    DF0 = math.exp(-r0 * T0)
    F0 = spot0 * math.exp(r0 * T0)
    qs = []
    for K in np.arange(spot0 - 15, spot0 + 16, 1.0):
        k = math.log(K / F0)
        s = max(0.0050 - 0.030 * k, 0.0008)
        cm = _bs_call_fwd(F0, K, s) * DF0
        pm = max(cm - DF0 * (F0 - K), 0.0)
        cm = max(cm, 0.0)
        h = 0.01 + 0.002 * max(cm, pm)
        qs.append(ChainQuote(float(K), max(cm - h, 0), cm + h,
                             max(pm - h, 0), pm + h))
    chain = ChainSnapshot(qs, spot=spot0, t_years=T0, r=r0)

    feed2 = SyntheticUnifiedFeed(days=5, chain=chain)
    jrn2 = Journal(":memory:")
    orch2 = UnifiedOrchestrator(feed=feed2, journal=jrn2)
    ticks2 = [start + dt.timedelta(minutes=i) for i in range(20)]
    results2 = orch2.run_replay(ticks2)
    trades2 = [r for r in results2 if r.decision is not None and r.decision.decision == "TRADE"]
    print(f"  20 ticks  |  {len(trades2)} TRADE decisions from Track A")
    if trades2:
        d = trades2[0].decision
        print(f"  first trade: {d.candidate.family if d.candidate else 'no candidate'} "
              f"gate={'PASS' if d.gate_pass else 'FAIL'} "
              f"size_mult={trades2[0].final_size_mult}")

    eff = jrn2.gate_effectiveness()
    print(f"\n  journal: {eff['trades_taken']['n']} taken, "
          f"{eff['blocked_by_gate']['n']} blocked by gate")
