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
     Prediction Engine V2 / PR 10: PolicyRouter also runs LegacyMatrixPolicy +
     PredictionPolicy in shadow (legacy authoritative until mode=champion).
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
from types import SimpleNamespace
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
from volatility_channel_features import channel_features_from_bars

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
    # Prediction Engine V2 / PR 9: raw option rows for parallel GEX variants.
    # Observation-only — when absent, variant journaling is skipped.
    # Feeds may attach these without changing MarketSnapshot policy fields.
    option_rows: Optional[list] = None
    weekly_option_rows: Optional[list] = None
    gex_feed_source: str = ""


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
    # Observation signals (policy / phys / gex / v2) for live_state + dashboard.
    signals: dict = field(default_factory=dict)
    # MTF sigma-cone panes for the Prediction tab (also journaled in store).
    sigma_cones: Optional[dict] = None
    # Part 3 shadow decision payload for live_state / dashboard assessment.
    part3: Optional[dict] = None
    # Parallel paper fills: list of {track, candidate, size_mult, ...} for
    # legacy / v2 / v3. PaperBroker opens each track independently.
    paper_intents: list = field(default_factory=list)
    # Unified decision stack (UNIFIED handoff §11.2) — optional until fully wired.
    legacy_decision: Optional[object] = None
    v3_decision: Optional[object] = None
    authoritative_decision: Optional[object] = None
    authority_source: Optional[str] = None
    deployment_id: Optional[str] = None
    fallback_used: bool = False
    candidate_universe_summary: Optional[dict] = None


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
    # Separate ledger for the V3 candidate dual-paper account (candidate mode).
    # When None, portfolio_risk_fn falls back to risk_manager for both accounts.
    candidate_risk_manager: Optional[RiskManager] = None
    state_path: Optional[str] = None            # persist adaptive scales across restarts
    ras_cfg: Optional[RASConfig] = None
    # Static per-dominant-regime engine deltas from the champion config
    # (adaptive_learning.config_store schema). Resolved ONCE at construction —
    # nothing adaptive happens at tick time, the live engine stays
    # deterministic. Keys: classifier regime names (or "unknown"); values:
    # dot-notation engine overrides plus the special "size_mult".
    regime_overrides: Optional[dict] = None
    # Transition flag (Prediction Engine V2, PR 2): the matrix scale book is
    # the per-feature-AND-timeframe, exponentially decayed, lagged
    # RobustScaleBook by default. Set True to fall back to the legacy
    # name-only Welford ScaleBook (update-before-score) for comparison.
    use_legacy_scaler: bool = False
    # Canonical dataset capture (Prediction Engine V2, PR 3): when set, every
    # tick writes one feature_snapshots row (raw features + missingness +
    # quality) keyed by the same snapshot_id that lands on the journal row —
    # the audit linkage between evaluations and the V2 training dataset.
    # Observation-only; a failed write never breaks a tick.
    prediction_store: Optional[object] = None
    symbol: str = "SPY"
    # Physical-density migration (Prediction Engine V2, PR 5 / §12.5).
    # True (default): keep the legacy dir_drift_frac tilt of the realized-vol
    # density when the router emits a directional debit — the pre-V2 path.
    # False: when a PhysicalForecast is available, price candidates against
    # the independent V2 density instead. Richness measurement always stays
    # on the drift-less realized-vol density either way.
    use_legacy_directional_tilt: bool = True
    # Optional independent forecast for the V2 physical density. Either a
    # static PhysicalForecast, a PredictionBundle (lifted via
    # forecast_from_bundle), or a callable(snap, signals, intent) ->
    # PhysicalForecast | PredictionBundle | None. Never receives the selected
    # candidate or gate result — callers must not close that loop.
    physical_forecast: Optional[object] = None
    physical_forecast_provider: Optional[Callable] = None
    # Candidate-value shadow ranker (Prediction Engine V2, PR 8 / §14).
    # When set, every tick with a chain runs V2 utility ranking on the
    # evaluated candidate set, journals diagnostics into signals_json, and
    # optionally persists candidate snapshots. Legacy `decision.candidate`
    # remains authoritative until RankerConfig.mode == "champion".
    candidate_value_model: Optional[object] = None
    candidate_ranker_cfg: Optional[object] = None
    #
    # Prediction policy / regime consolidation (PR 10 / §17). Promotion is
    # this single pointer: legacy | shadow | champion. Shadow (default) keeps
    # the matrix authoritative while journaling V2 disagreement.
    policy_mode: str = "shadow"
    policy_router_cfg: Optional[object] = None
    # Optional PredictionBundle for PredictionPolicy. Prefer a dedicated
    # provider so physical_forecast can stay a PhysicalForecast. Callable
    # signature: (snap, signals, intent, regime_state) -> PredictionBundle|None.
    prediction_bundle: Optional[object] = None
    prediction_bundle_provider: Optional[Callable] = None
    # Unified decision stack (UNIFIED handoff). When set, tick() can produce
    # UnifiedDecisionRecord alongside the legacy path. Legacy remains
    # authoritative until deployment mode says otherwise.
    decision_stack: Optional[object] = None
    deployment_bundle: Optional[object] = None

    def __post_init__(self):
        self._classifier = RegimeClassifier(
            cfg=self.classifier_cfg or ClassifierConfig()
        )
        self._prev_std: Optional[dict] = None   # for information-gain computation
        # adaptive scales for MTF matrix variables
        if self.use_legacy_scaler:
            self._matrix_scale_book = ScaleBook()
        else:
            from prediction.scalers import RobustScaleBook
            self._matrix_scale_book = RobustScaleBook()
        self._ticks_since_save = 0
        self._bias_side: Optional[int] = None   # fast-vs-slow composite side, for crossovers
        self._snap_seq = 0                      # per-session source sequence for snapshot ids
        self._snap_session: Optional[str] = None
        # dealer-surface / vol-state derivatives (observation-only signals)
        dyn_path = None
        if self.state_path:
            import os
            dyn_path = os.path.join(os.path.dirname(self.state_path) or ".",
                                    "dynamics_state.json")
        self._dynamics = DynamicsWindow(path=dyn_path)
        # Pre-resolve one (EngineConfig, size_mult) per overridden regime so a
        # bad champion file fails at startup, not mid-session.
        self._regime_cfg: dict[str, tuple[EngineConfig, float]] = {}
        if self.regime_overrides:
            from adaptive_learning.config_store import (
                engine_cfg_for_regime, validate_regime_overrides)
            validate_regime_overrides(self.regime_overrides)
            base = self.engine_cfg or EngineConfig()
            for regime in self.regime_overrides:
                self._regime_cfg[regime] = engine_cfg_for_regime(
                    base, self.regime_overrides, regime)
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
            # V2 matrix scales live under a separate versioned key so legacy
            # name-only Welford state can never be reinterpreted as
            # per-timeframe state (RobustScaleBook.load_dict additionally
            # rejects any version/config-hash mismatch and re-warms).
            matrix_key = ("matrix_scales" if self.use_legacy_scaler
                          else "matrix_scales_v2")
            self._matrix_scale_book.load_dict(data.get(matrix_key, {}))
            self._classifier.scales.load_dict(data.get("classifier_scales", {}))
        except Exception:
            pass

    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            import os
            import tempfile
            matrix_key = ("matrix_scales" if self.use_legacy_scaler
                          else "matrix_scales_v2")
            payload = {
                matrix_key: self._matrix_scale_book.to_dict(),
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

    def _bias_cross(self, fast: Optional[float],
                    slow: Optional[float]) -> Optional[float]:
        """Fast/slow direction-composite crossover detector with hysteresis.

        Returns +1.0 the tick the fast composite crosses above the slow one
        (the V-bottom signature: short timeframes turning before the
        session-anchored context), -1.0 crossing below, None otherwise.
        A 1-point deadband on the gap prevents chatter when the composites
        are riding on top of each other.
        """
        if fast is None or slow is None:
            return None
        gap = fast - slow
        side = 1 if gap >= 1.0 else (-1 if gap <= -1.0 else None)
        if side is None:                     # inside deadband: hold prior side
            return None
        prev = self._bias_side
        self._bias_side = side
        if prev is not None and side != prev:
            return float(side)
        return None

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
        # Keep internal keys (e.g. _snapshot_id) in the in-memory dict for
        # providers, but never journal them.
        if not ras_results:
            public = {k: v for k, v in signals.items()
                      if not str(k).startswith("_")}
            return signals, (json.dumps({k: (v if isinstance(v, str) else round(v, 6))
                                        for k, v in public.items()
                                        if isinstance(v, (int, float, str))})
                             if public else None)
        merged = dict(signals)
        # Flatten only the WORST-scoring position: with several open positions
        # the ras_* keys would otherwise overwrite each other arbitrarily, and
        # the minimum score is the correlation-relevant health signal anyway.
        # Full per-position detail lands in journal.ras_evaluations.
        worst = min(ras_results, key=lambda r: r.score)
        merged.update(ras_to_signals(worst))
        public = {k: v for k, v in merged.items() if not str(k).startswith("_")}
        signals_json = json.dumps(
            {k: (v if isinstance(v, str) else round(v, 6))
             for k, v in public.items()
             if isinstance(v, (int, float, str))}
        ) if public else None
        return merged, signals_json

    @staticmethod
    def _attach_forecast_signals(signals: dict, bundle) -> None:
        """Journal PredictionBundle fields for the dashboard V2 forecast panel.

        Flat ``v2_fc_*`` keys survive even when the separate prediction_store
        path is not mounted on the dashboard process.
        """
        if bundle is None or not isinstance(signals, dict):
            return
        keys = (
            "p_up_30m", "p_up_close", "expected_return_30m",
            "return_q10_30m", "return_q50_30m", "return_q90_30m",
            "p_range_survive_30m", "expected_realized_move_30m",
            "p_touch_call_wall_30m", "p_touch_put_wall_30m",
            "uncertainty", "data_quality", "feature_coverage",
        )
        for k in keys:
            v = getattr(bundle, k, None)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                signals[f"v2_fc_{k}"] = float(v)
        mv = getattr(bundle, "model_versions", None) or {}
        ver = mv.get("bundle") or mv.get("group") or ""
        if ver:
            signals["v2_fc_model_version"] = str(ver)
        signals["v2_fc_mode"] = "shadow"

    def _resolve_physical_forecast(self, snap, signals: dict, intent) -> Optional[object]:
        """
        Resolve an independent PhysicalForecast for this tick.

        Sources, in order: this tick's cached PredictionBundle (from policy
        dual-run), physical_forecast_provider (callable), then the static
        physical_forecast field. Accepts a PhysicalForecast or a
        PredictionBundle (lifted via forecast_from_bundle). Returns None when
        nothing usable is available — the tick then falls back to the legacy
        / realized-vol path. The provider must NOT close the policy loop: it
        may read snap/signals/intent features, but the density builder itself
        never sees structure/direction/conviction.
        """
        from prediction.contracts import PredictionBundle
        from prediction.physical_distribution import (
            PhysicalForecast, forecast_from_bundle)

        # Prefer the bundle already resolved for policy this tick — avoids a
        # second provider call / duplicate prediction_outputs row.
        cached = getattr(self, "_tick_prediction_bundle", None)
        if isinstance(cached, PredictionBundle):
            return forecast_from_bundle(cached)

        raw = None
        if self.physical_forecast_provider is not None:
            try:
                raw = self.physical_forecast_provider(snap, signals, intent)
            except Exception as exc:
                log.warning("physical_forecast_provider failed: %s", exc)
                raw = None
        if raw is None:
            raw = self.physical_forecast
        if raw is None:
            return None
        if isinstance(raw, PhysicalForecast):
            return raw
        if isinstance(raw, PredictionBundle):
            return forecast_from_bundle(raw)
        if isinstance(raw, dict):
            # allow a plain dict shaped like PhysicalForecast
            try:
                return PhysicalForecast(**{k: raw[k] for k in (
                    "expected_return", "return_q10", "return_q50", "return_q90",
                    "expected_realized_move", "volatility_scale",
                    "skew_adjustment", "uncertainty", "model_version")
                    if k in raw})
            except (TypeError, ValueError) as exc:
                log.warning("invalid physical_forecast dict: %s", exc)
                return None
        log.warning("unsupported physical_forecast type: %s", type(raw).__name__)
        return None

    def _build_v2_physical_result(self, snap, signals: dict, intent, rnd, cfg):
        """Build V2 physical density and journal ``phys_v2_*`` moments.

        Observation-only: safe to run on stand-down / NT ticks so the V2 tab
        stays populated while legacy is not trading.
        """
        forecast = self._resolve_physical_forecast(snap, signals, intent)
        if forecast is None or rnd is None:
            return None
        try:
            from prediction.physical_distribution import build_physical_density
            v2_result = build_physical_density(
                rnd, forecast,
                scale_min=cfg.rnd.rv_scale_min,
                scale_max=cfg.rnd.rv_scale_max)
            signals["phys_v2_mean"] = v2_result.moments.get("mean")
            signals["phys_v2_std"] = v2_result.moments.get("std")
            signals["phys_v2_var_ratio"] = v2_result.moments.get("var_ratio")
            signals["phys_v2_uncertainty"] = forecast.uncertainty
            signals["phys_v2_expected_return"] = forecast.expected_return
            signals["phys_v2_model_version"] = forecast.model_version
            return v2_result
        except Exception as exc:
            log.warning("V2 physical density failed: %s", exc)
            return None

    def _run_v2_shadow_ranking(
            self, snap, signals: dict, snapshot_id: str, decide_pdf, cfg,
            decision=None, pin_active: bool = False):
        """Observation-only candidate ranking (works without a live TRADE).

        Returns the (possibly annotated) decision, or the input decision when
        ranking is skipped / fails.
        """
        if self.candidate_value_model is None or snap.chain is None:
            return decision
        try:
            from prediction.candidate_ranker import (
                RankerConfig, run_shadow_ranking,
            )
            from spread_selector import select_spreads, GammaContext

            rcfg = self.candidate_ranker_cfg or RankerConfig()
            shadow_cands = list(getattr(self, "_tick_shared_cands", None) or [])
            if not shadow_cands and decision is not None:
                shadow_cands = list(decision.all_candidates or [])
            # Prefer the shared tick universe; only regenerate when absent.
            if not shadow_cands and (
                    getattr(rcfg, "shadow_all_families", True)):
                try:
                    rnd_s = extract_rnd(snap.chain, cfg.rnd)
                    edge_s = compute_edge(
                        rnd_s, snap.chain, cfg.rnd,
                        physical_pdf=decide_pdf)
                    ctx_s = GammaContext.from_market_snapshot(
                        snap.market, pin_active=pin_active)
                    sel_s = select_spreads(
                        snap.chain, rnd_s, edge_s, ctx_s, cfg.selector,
                        physical_pdf=decide_pdf, target_families=None)
                    shadow_cands = list(
                        sel_s.all_candidates or sel_s.ranked or [])
                except Exception as exc:
                    log.warning("shadow candidate gen failed: %s", exc)
            if not shadow_cands:
                return decision
            self._tick_shared_cands = list(shadow_cands)
            self._tick_shadow_cands = list(shadow_cands)
            mkt = snap.market
            minutes_left = None
            try:
                minutes_left = getattr(mkt, "minutes_to_close", None)
            except Exception:
                minutes_left = None
            ranking = run_shadow_ranking(
                shadow_cands,
                self.candidate_value_model,
                snapshot_id=snapshot_id,
                spot=float(mkt.spot),
                call_wall=float(mkt.call_wall),
                put_wall=float(mkt.put_wall),
                gamma_flip=float(mkt.gamma_flip),
                minutes_to_close=minutes_left,
                net_gex=float(mkt.net_gex),
                cfg=rcfg,
                store=self.prediction_store,
            )
            signals.update(ranking.signals())
            self._tick_shadow_cands = list(shadow_cands)
            self._tick_shadow_forecasts = dict(ranking.forecasts or {})
            # Part 3 shadow decision (observation-only) for dashboard assessment.
            try:
                from prediction.part3_shadow import build_part3_live_payload
                ts_iso = getattr(snap, "ts", None)
                if hasattr(ts_iso, "isoformat"):
                    ts_iso = ts_iso.isoformat()
                else:
                    ts_iso = str(signals.get("ts") or snapshot_id)
                hard = list(getattr(self, "_tick_hard_vetoes", None) or [])
                for v in (signals.get("_hard_vetoes") or []):
                    hard.append(str(v))
                # Also surface regime / risk veto strings already on signals
                for key in ("stand_down_reason", "no_trade_reason"):
                    if signals.get(key):
                        hard.append(str(signals[key]))
                # Dedupe while preserving order
                seen = set()
                hard_u = []
                for h in hard:
                    if h and h not in seen:
                        seen.add(h)
                        hard_u.append(h)
                self._tick_part3 = build_part3_live_payload(
                    snapshot_id=snapshot_id,
                    ts=str(ts_iso),
                    symbol=str(getattr(self, "symbol", "SPY") or "SPY"),
                    candidates=shadow_cands,
                    forecasts=ranking.forecasts,
                    signals=signals,
                    hard_vetoes=hard_u,
                    mode=str(getattr(self, "policy_mode", "shadow") or "shadow"),
                    store=self.prediction_store,
                    direction=str(signals.get("v2_policy_direction")
                                  or signals.get("policy_direction")
                                  or "unknown"),
                )
            except Exception as exc:
                log.warning("Part 3 shadow payload failed: %s", exc)
                self._tick_part3 = {
                    "note": f"part3 shadow failed: {type(exc).__name__}: {exc}",
                    "shadow_label": "SHADOW — not an executed order",
                    "mode": str(getattr(self, "policy_mode", "shadow") or "shadow"),
                    "component_errors": {"part3": f"{type(exc).__name__}: {exc}"},
                }
            # Annotate the live (legacy) candidate with V2 utility when present.
            if decision is not None and decision.candidate is not None:
                live_id = None
                for c in shadow_cands:
                    if (c.family == decision.candidate.family
                            and c.short_strikes
                            == decision.candidate.short_strikes
                            and c.long_strikes
                            == decision.candidate.long_strikes):
                        live_id = (
                            getattr(c, "candidate_id", None)
                            or getattr(c, "v2_candidate_id", None)
                            or getattr(c, "_v2_candidate_id", None)
                        )
                        break
                if live_id and live_id in ranking.forecasts:
                    fc = ranking.forecasts[live_id]
                    decision = dataclasses.replace(
                        decision,
                        candidate=dataclasses.replace(
                            decision.candidate,
                            v2_utility_score=fc.utility_score,
                            v2_candidate_id=live_id,
                        ),
                    )
            return decision
        except Exception as exc:
            log.warning("V2 candidate shadow ranking failed: %s", exc)
            return decision

    def _pick_shadow_candidate(self, candidate_id: Optional[str] = None,
                               family: Optional[str] = None,
                               structure_code: Optional[str] = None):
        """Pick a concrete SpreadCandidate from this tick's shadow set.

        When ``candidate_id`` is supplied, resolve **exactly** that identity
        (``candidate_id`` / ``v2_candidate_id`` / ``_v2_candidate_id`` aliases).
        Never fall back by family — an unresolved explicit ID returns ``None``
        so the caller can mark the decision UNAVAILABLE rather than paper-
        trading a substitute.
        """
        cands = list(getattr(self, "_tick_shadow_cands", None) or [])
        if not cands:
            return None
        if candidate_id is not None and str(candidate_id).strip():
            want = str(candidate_id).strip()
            for c in cands:
                aliases = (
                    getattr(c, "candidate_id", None),
                    getattr(c, "v2_candidate_id", None),
                    getattr(c, "_v2_candidate_id", None),
                )
                if any(a is not None and str(a) == want for a in aliases):
                    return c
            return None
        if structure_code:
            try:
                from spread_selector import STRUCTURE_TO_FAMILIES
                fams = set(STRUCTURE_TO_FAMILIES.get(structure_code) or ())
            except Exception:
                fams = set()
            if fams:
                for c in cands:
                    if getattr(c, "family", None) in fams:
                        return c
        if family:
            for c in cands:
                if getattr(c, "family", None) == family:
                    return c
        # Prefer highest V2 utility when available
        forecasts = getattr(self, "_tick_shadow_forecasts", None) or {}
        best, best_u = None, None
        for c in cands:
            cid = (
                getattr(c, "candidate_id", None)
                or getattr(c, "_v2_candidate_id", None)
            )
            fc = forecasts.get(cid) if cid else None
            u = float(getattr(fc, "utility_score", None)
                      or getattr(c, "v2_utility_score", None)
                      or getattr(c, "score", None)
                      or float("-inf"))
            if best is None or u > best_u:
                best, best_u = c, u
        return best

    def _ensure_shared_candidate_universe(
            self, snap, decide_pdf, cfg, *, pin_active: bool = False,
            snapshot_id: str = "", now: Optional[dt.datetime] = None) -> list:
        """
        Generate the tick's candidate set exactly once (all families).

        Legacy decide() and the unified V3 stack both consume this immutable
        list — never re-run select_spreads for the same tick.
        """
        existing = list(getattr(self, "_tick_shared_cands", None) or [])
        if existing:
            return existing
        if snap is None or getattr(snap, "chain", None) is None:
            self._tick_shared_cands = []
            return []
        try:
            from spread_selector import select_spreads, GammaContext
            rnd_s = extract_rnd(snap.chain, cfg.rnd)
            edge_s = compute_edge(
                rnd_s, snap.chain, cfg.rnd, physical_pdf=decide_pdf)
            ctx_s = GammaContext.from_market_snapshot(
                snap.market, pin_active=pin_active)
            sel = select_spreads(
                snap.chain, rnd_s, edge_s, ctx_s, cfg.selector,
                physical_pdf=decide_pdf, target_families=None)
            cands = list(sel.all_candidates or sel.ranked or [])
        except Exception as exc:
            log.warning("shared candidate universe generation failed: %s", exc)
            cands = []
        # Stamp stable candidate ids (canonical + v2 aliases)
        if snapshot_id and cands:
            try:
                from prediction.candidate_universe import stamp_candidate_id
                for c in cands:
                    stamp_candidate_id(c, snapshot_id)
            except Exception:
                pass
        self._tick_shared_cands = cands
        self._tick_shadow_cands = list(cands)
        if snapshot_id and cands:
            try:
                from prediction.candidate_universe import build_candidate_universe
                ts_iso = (now.astimezone(ET).isoformat()
                          if now is not None else snapshot_id)
                self._tick_shared_universe = build_candidate_universe(
                    snapshot_id=snapshot_id,
                    generated_at=ts_iso,
                    candidates=cands,
                    generator_config={
                        "selector_min_ev": getattr(cfg.selector, "min_ev", None),
                        "shared": True,
                    },
                )
            except Exception as exc:
                log.warning("build_candidate_universe failed: %s", exc)
                self._tick_shared_universe = None
        else:
            self._tick_shared_universe = None
        return cands

    def _resolve_structure_candidate(
            self, snap, decide_pdf, cfg, *, structure: str, direction: str,
            pin_active: bool, density_mode: str = "vrp",
            density_moments: Optional[dict] = None):
        """Run Track-A decide() for a specific structure → executable candidate."""
        if snap is None or snap.chain is None:
            return None
        if not structure or structure in ("NT", "none", "NONE"):
            return None
        try:
            from decision_engine import decide
            shared = list(getattr(self, "_tick_shared_cands", None) or [])
            dec = decide(
                snap.market, snap.chain, cfg,
                physical_pdf=decide_pdf,
                target_structure=structure,
                direction=direction or "both",
                physical_density_mode=density_mode,
                physical_moments=density_moments,
                pin_active=pin_active,
                precomputed_candidates=shared if shared else None)
            if dec is None or dec.candidate is None:
                return None
            # Prediction-engine paper fills: require gate pass so we don't
            # open garbage, but allow through when gate wasn't evaluated.
            if getattr(dec, "gate_pass", True) is False:
                return None
            return dec
        except Exception as exc:
            log.warning("paper candidate resolve failed (%s/%s): %s",
                        structure, direction, exc)
            return None

    def _build_paper_intents(
            self, *, snap, signals: dict, intent, regime_state,
            decision, decide_pdf, cfg, pin_active: bool,
            density_mode: str, density_moments: Optional[dict],
            final_size_mult: float,
            matrix_stand_down: bool) -> list:
        """
        Build parallel paper intents for legacy / V2 / V3.

        - legacy: matrix intent → Track A decide (independent of champion override)
        - v2: PredictionPolicy TRADE → decide with V2 structure/direction
        - v3: Part 3 TRADE → selected shadow candidate

        Session warmup is enforced in PaperBroker. Regime stand-down blocks
        legacy only — V2/V3 still fill when their engines say TRADE so the
        tracks can be compared.
        """
        intents: list = []
        if snap is None or snap.chain is None:
            return intents

        from decision_stack.authority import coerce_size_mult

        # ---- LEGACY (matrix) ----
        raw_struct = getattr(getattr(intent, "decision", None), "structure", None)
        raw_dir = getattr(getattr(intent, "decision", None), "direction", None) or "both"
        legacy_ok = (
            not matrix_stand_down
            and raw_struct not in (None, "NT", "none", "NONE")
        )
        if legacy_ok:
            # Prefer the live decision candidate when it already matches matrix
            leg_cand = None
            if (decision is not None and getattr(decision, "decision", None) == "TRADE"
                    and getattr(decision, "gate_pass", False)
                    and decision.candidate is not None
                    and str(getattr(self, "policy_mode", "shadow")).lower() != "champion"):
                leg_cand = decision.candidate
                gate_kelly = getattr(decision, "gate_kelly", None)
                gate_score = getattr(decision, "gate_score", None)
            else:
                leg_dec = self._resolve_structure_candidate(
                    snap, decide_pdf, cfg,
                    structure=str(raw_struct), direction=str(raw_dir),
                    pin_active=pin_active, density_mode=density_mode,
                    density_moments=density_moments)
                leg_cand = getattr(leg_dec, "candidate", None) if leg_dec else None
                gate_kelly = getattr(leg_dec, "gate_kelly", None) if leg_dec else None
                gate_score = getattr(leg_dec, "gate_score", None) if leg_dec else None
            if leg_cand is not None:
                size = coerce_size_mult(
                    getattr(intent, "size_mult", None), default=1.0)
                if size <= 0.0:
                    pass  # zero size = no position
                else:
                    intents.append({
                        "track": "legacy",
                        "candidate": leg_cand,
                        "size_mult": size,
                        "gate_kelly": gate_kelly,
                        "gate_score": gate_score,
                        "structure": raw_struct,
                        "direction": raw_dir,
                        "reason": "matrix_intent",
                    })

        # ---- V2 (PredictionPolicy) ----
        v2_action = str(signals.get("v2_policy_action") or "").upper()
        v2_struct = signals.get("v2_policy_structure")
        v2_dir = signals.get("v2_policy_direction") or "both"
        if v2_action == "TRADE" and v2_struct not in (None, "NT", "none", "NONE", ""):
            v2_mode = density_mode
            v2_moments = density_moments
            # Prefer V2 physical density when available on signals
            if signals.get("phys_density_mode") == "v2":
                v2_mode = "v2"
            v2_dec = self._resolve_structure_candidate(
                snap, decide_pdf, cfg,
                structure=str(v2_struct), direction=str(v2_dir),
                pin_active=pin_active, density_mode=v2_mode,
                density_moments=v2_moments)
            v2_cand = getattr(v2_dec, "candidate", None) if v2_dec else None
            if v2_cand is None:
                v2_cand = self._pick_shadow_candidate(
                    family=signals.get("v2_top_family"),
                    structure_code=str(v2_struct),
                    candidate_id=signals.get("v2_top_candidate_id"))
            if v2_cand is not None:
                size = coerce_size_mult(
                    signals.get("policy_size_cap")
                    if signals.get("policy_size_cap") is not None
                    else signals.get("v2_policy_size_cap"),
                    default=1.0,
                )
                if size > 0.0:
                    intents.append({
                        "track": "v2",
                        "candidate": v2_cand,
                        "size_mult": size,
                        "gate_kelly": getattr(v2_dec, "gate_kelly", None) if v2_dec else None,
                        "gate_score": getattr(v2_dec, "gate_score", None) if v2_dec else None,
                        "structure": v2_struct,
                        "direction": v2_dir,
                        "reason": "prediction_policy",
                    })

        # ---- V3 (unified stack / Part 3 TradeDecisionV3) ----
        # Prefer the post-stack unified V3 decision so paper intents are built
        # AFTER authority resolution — never from a stale pre-stack summary.
        v3 = getattr(self, "_tick_unified_v3", None) or {}
        part3 = getattr(self, "_tick_part3", None) or {}
        ds = part3.get("decision_summary") or {}
        v3_action = str(
            v3.get("final_action")
            or part3.get("final_action")
            or ds.get("statistical_action")
            or ds.get("action")
            or ""
        ).upper()
        if v3_action == "TRADE":
            cid = (
                v3.get("candidate_id")
                or part3.get("selected_candidate_id")
                or ds.get("selected_candidate_id")
            )
            # Explicit ID required for V3 paper — never family-fallback.
            if not cid:
                log.warning(
                    "V3 TRADE without selected_candidate_id; skip paper intent")
            else:
                v3_cand = self._pick_shadow_candidate(candidate_id=str(cid))
                if v3_cand is None:
                    log.warning(
                        "V3 selected candidate_id=%s unresolved in shared "
                        "universe; skip paper intent (no family substitute)",
                        cid)
                else:
                    resolved_id = (
                        getattr(v3_cand, "candidate_id", None)
                        or getattr(v3_cand, "v2_candidate_id", None)
                        or getattr(v3_cand, "_v2_candidate_id", None)
                    )
                    if str(resolved_id) != str(cid):
                        log.error(
                            "V3 candidate identity mismatch: selected=%s "
                            "resolved=%s; refusing paper intent",
                            cid, resolved_id)
                    else:
                        auth = getattr(self, "_tick_authoritative", None) or {}
                        if auth.get("size_mult") is not None:
                            size = coerce_size_mult(auth.get("size_mult"),
                                                    default=0.0)
                        else:
                            size = coerce_size_mult(final_size_mult,
                                                    default=1.0)
                        if size <= 0.0:
                            log.info(
                                "V3 TRADE suppressed: size_mult=%.4f <= 0",
                                size)
                        else:
                            sel_eval = (
                                (getattr(self, "_tick_part3", None) or {})
                                .get("unified", {})
                                .get("selected_candidate_evaluation")
                            ) or {}
                            diag = sel_eval.get("diagnostics") or {}
                            exec_est = {
                                "fill_probability": sel_eval.get(
                                    "fill_probability"),
                                "expected_fill_price": sel_eval.get(
                                    "expected_fill_price"),
                                "conservative_fill_price": sel_eval.get(
                                    "conservative_fill_price"),
                                "expected_order_value": sel_eval.get(
                                    "expected_order_value"),
                                "expected_concession": sel_eval.get(
                                    "expected_concession"),
                                "fees": sel_eval.get("fees"),
                                "entry_fees": diag.get("entry_fees"),
                                "expected_exit_fees": diag.get(
                                    "expected_exit_fees"),
                                "mid_credit": diag.get("mid_credit"),
                            }
                            intents.append({
                                "track": "v3",
                                "candidate": v3_cand,
                                "size_mult": size,
                                "gate_kelly": None,
                                "gate_score": None,
                                "structure": (
                                    v3.get("structure") or ds.get("family")),
                                "direction": (
                                    v3.get("direction")
                                    or ds.get("direction")
                                    or signals.get("v2_policy_direction")),
                                "candidate_id": cid,
                                "v3_action": v3_action,
                                "reason": "unified_v3",
                                "execution_estimate": exec_est,
                            })

        # Each paper account records into its own RiskManager on open.
        for intent in intents:
            intent["risk_record"] = True

        return intents

    def _resolve_prediction_bundle(self, snap, signals: dict, intent,
                                   regime_state) -> Optional[object]:
        """
        Resolve a PredictionBundle for PredictionPolicy (PR 10).

        Order: prediction_bundle_provider, prediction_bundle, then a
        PredictionBundle already sitting on physical_forecast /
        physical_forecast_provider (without lifting to PhysicalForecast).
        Never invents a neutral bundle — missing means explicit fallback.
        """
        from prediction.contracts import PredictionBundle

        raw = None
        if self.prediction_bundle_provider is not None:
            try:
                raw = self.prediction_bundle_provider(
                    snap, signals, intent, regime_state)
            except Exception as exc:
                log.warning("prediction_bundle_provider failed: %s", exc)
                raw = None
        if raw is None:
            raw = self.prediction_bundle
        if raw is None and self.physical_forecast_provider is not None:
            try:
                cand = self.physical_forecast_provider(snap, signals, intent)
                if isinstance(cand, PredictionBundle):
                    raw = cand
            except Exception:
                pass
        if raw is None and isinstance(self.physical_forecast, PredictionBundle):
            raw = self.physical_forecast
        if raw is None:
            return None
        if isinstance(raw, PredictionBundle):
            return raw
        if isinstance(raw, dict):
            try:
                return PredictionBundle.from_dict(raw)
            except (TypeError, ValueError) as exc:
                log.warning("invalid prediction_bundle dict: %s", exc)
                return None
        log.warning("unsupported prediction_bundle type: %s", type(raw).__name__)
        return None

    def _route_policy(self, snap, signals: dict, intent, regime_state):
        """
        Dual-run legacy + V2 policy (PR 10). Returns PolicyRouteResult or
        None on hard failure (tick continues on the matrix path alone).
        """
        try:
            from policy.contracts import PolicyInput, StructuralState
            from policy.router import PolicyRouter, PolicyRouterConfig

            bundle = self._resolve_prediction_bundle(
                snap, signals, intent, regime_state)
            self._tick_prediction_bundle = bundle
            self._attach_forecast_signals(signals, bundle)
            implied = signals.get("implied_move")
            if implied is None:
                # Prefer dollar/spot move fractions — never chan_bb_width
                # (that is a 0-1 percentile rank, not a move).
                spot = float(getattr(snap.market, "spot", 0.0) or 0.0)
                for attr in ("expected_range", "straddle_breakeven"):
                    v = getattr(snap.market, attr, None)
                    if (isinstance(v, (int, float)) and math.isfinite(float(v))
                            and float(v) > 0 and spot > 0):
                        implied = float(v) / spot
                        break
            hard_vetoes = list(regime_state.vetoes or [])
            # Mirror the Track A session-warmup hard gate into policy so V2
            # dual-run also stands down until entry open.
            try:
                from gate_scorer import GateConfig
                entry_open = (self.engine_cfg.gate.morning_entry_time
                              if self.engine_cfg is not None
                              else GateConfig().morning_entry_time)
            except Exception:
                from gate_scorer import GateConfig
                entry_open = GateConfig().morning_entry_time
            in_warmup = snap.market.et_time() < entry_open
            if in_warmup and "session_warmup" not in hard_vetoes:
                hard_vetoes.append("session_warmup")
            op_risk = {
                "hard_vetoes": hard_vetoes,
                "stand_down": bool(regime_state.stand_down) or in_warmup,
                "implied_remaining_move": implied,
                "session_warmup": in_warmup,
                "pin_active": bool(signals.get("pin_active")),
            }
            pin = PolicyInput(
                predictions=bundle,
                structural_state=StructuralState.from_market(snap.market),
                operational_risk_state=op_risk,
                legacy_regime_state=regime_state,
                legacy_matrix_intent=intent,
            )
            cfg = self.policy_router_cfg
            if cfg is None:
                cfg = PolicyRouterConfig(mode=self.policy_mode)
            elif getattr(cfg, "mode", None) is None:
                cfg.mode = self.policy_mode
            return PolicyRouter(cfg).route(pin)
        except Exception as exc:
            log.warning("policy router failed: %s", exc)
            # Champion mode must not silently fail-open: emit an explicit
            # fallback_legacy route so the journal records the failure.
            if str(self.policy_mode).lower() == "champion":
                try:
                    from policy.contracts import (
                        PolicyInput, StructuralState, SOURCE_FALLBACK_LEGACY,
                    )
                    from policy.legacy_matrix import (
                        LegacyMatrixPolicy, intent_to_decision,
                    )
                    from policy.router import PolicyRouteResult
                    legacy = intent_to_decision(intent, regime_state=regime_state)
                    fallback = intent_to_decision(
                        intent, regime_state=regime_state,
                        source=SOURCE_FALLBACK_LEGACY)
                    fallback = dataclasses.replace(
                        fallback,
                        rationale=(f"router exception: {exc}",
                                   *tuple(fallback.rationale or ())),
                    )
                    return PolicyRouteResult(
                        mode="champion",
                        authoritative=fallback,
                        legacy=legacy,
                        v2=None,
                        disagreement=False,
                        fallback_used=True,
                        diagnostics={"v2_unavailable_reason":
                                     f"router_exception:{type(exc).__name__}"},
                    )
                except Exception as exc2:
                    log.warning("champion fallback also failed: %s", exc2)
            return None

    def _emit_and_settle_sigma_cones(self, now: dt.datetime, snapshot_id: str,
                                     snap, signals: dict) -> None:
        """
        Build MTF sigma cones, journal them, settle due bands vs true spot.

        Observation-only: never gates. Failures are logged and skipped.
        """
        try:
            from prediction.sigma_cone import (
                build_mtf_cones, cones_to_signals, cones_live_summary,
            )
            from prediction.dataset import session_metadata

            market = snap.market
            spot = float(getattr(market, "spot", 0.0) or 0.0)
            if spot <= 0:
                self._tick_sigma_cones = None
                return

            meta = session_metadata(now)
            session_date = (meta.get("session_date")
                            or now.astimezone(ET).date().isoformat())
            mtc = meta.get("minutes_to_close")

            cones = build_mtf_cones(
                snapshot_id=snapshot_id,
                ts=now,
                session_date=session_date,
                spot=spot,
                market=market,
                signals=signals,
                minutes_to_close=mtc,
            )
            signals.update(cones_to_signals(cones))
            self._tick_sigma_cones = cones_live_summary(cones)

            store = self.prediction_store
            if store is not None and hasattr(store, "log_sigma_cones"):
                store.log_sigma_cones(cones)
                now_et = now if now.tzinfo else now.replace(tzinfo=ET)
                now_iso = now_et.astimezone(ET).isoformat()
                store.settle_sigma_cones(now_iso, spot, realized_ts=now_iso)
        except Exception as exc:
            log.warning("sigma cone emit/settle failed: %s", exc)
            self._tick_sigma_cones = None

    def _next_snapshot_id(self, now: dt.datetime) -> str:
        """Stable per-tick observation id (PR 3): SHA256 of symbol |
        normalized ET timestamp | feature version | per-session sequence."""
        from prediction.dataset import FEATURE_VERSION, make_snapshot_id
        session_date = now.astimezone(ET).date().isoformat()
        if session_date != self._snap_session:
            self._snap_session = session_date
            self._snap_seq = 0
        seq = self._snap_seq
        self._snap_seq += 1
        return make_snapshot_id(self.symbol, now, FEATURE_VERSION, seq)

    def _log_feature_snapshot(self, now: dt.datetime, snapshot_id: str,
                              snap: TickSnapshot, snap_dict: dict,
                              signals: dict, mat_rows: list) -> None:
        """Best-effort canonical feature_snapshots row (PR 3).

        Train/serve parity: capture the same as-of market + MTF snapshot
        features as offline `prediction.dataset._tick_features`. Routing /
        policy / RAS diagnostics stay in journal signals_json only — they
        must not enter the training feature set.
        """
        if self.prediction_store is None:
            return
        try:
            from prediction.asof import AsOfFeatureBuilder, bars_asof
            from prediction.dataset import build_observation
            from prediction.inference import live_feature_row
            b = AsOfFeatureBuilder(observation_ts=now)
            # Pre-routing model features only (aligned with offline builder).
            for name, v in live_feature_row(snap, signals).items():
                b.add(name, v, source_ts=now)
            # Also capture raw snap_dict keys not already present.
            for name, v in snap_dict.items():
                if name not in b.features:
                    b.add(name, v, source_ts=now)
            # Native multi-TF indicators (same as offline _tick_features).
            if snap.bars is not None and len(snap.bars.ts):
                try:
                    safe_bars = bars_asof(snap.bars, now)
                    if len(safe_bars.ts):
                        mtf = build_mtf_input(safe_bars, {})
                        for name, per_tf in mtf.native.items():
                            for tf, v in per_tf.items():
                                b.add(f"{name}:{tf}", v, source_ts=now)
                except Exception:
                    pass
            built = b.build()
            standardized = {}
            for row in mat_rows:
                for tf, score in row.scores.items():
                    if score is not None:
                        standardized[f"{row.variable}:{tf}"] = score
            obs = build_observation(
                self.symbol, now, snap.market.spot,
                features=built["features"],
                standardized=standardized,
                missingness=built["missingness"],
                source_ages=built["source_ages"],
                quality={"feature_coverage": built["coverage"],
                         "has_chain": snap.chain is not None},
            )
            # keep the journal-linked id (seq already advanced this tick)
            obs = dataclasses.replace(obs, snapshot_id=snapshot_id)
            self.prediction_store.log_feature_snapshot(obs)
        except Exception as exc:
            log.warning("feature snapshot logging failed: %s", exc)


    def _legacy_decision_dict(self, decision, intent, snapshot_id: str,
                              final_size_mult: float) -> dict:
        """Map TradeDecision / stand-down into UnifiedDecisionStack legacy dict."""
        if decision is None:
            return {
                "action": "NO_EDGE",
                "candidate_id": None,
                "structure": getattr(getattr(intent, "decision", None),
                                     "structure", None),
                "direction": getattr(getattr(intent, "decision", None),
                                     "direction", None),
                "size_mult": float(final_size_mult or 0.0),
            }
        action = (
            "TRADE" if (
                getattr(decision, "decision", None) == "TRADE"
                and getattr(decision, "gate_pass", False)
            ) else "NO_EDGE"
        )
        cand = getattr(decision, "candidate", None)
        cid = None
        structure = None
        if cand is not None:
            cid = (getattr(cand, "candidate_id", None)
                   or getattr(cand, "v2_candidate_id", None)
                   or getattr(cand, "_v2_candidate_id", None))
            structure = getattr(cand, "family", None)
            if not cid:
                try:
                    from prediction.candidate_universe import make_candidate_id
                    from prediction.candidate_universe import _legs_from
                    cid = make_candidate_id(
                        snapshot_id,
                        family=str(structure or "unknown"),
                        legs=_legs_from(cand),
                    )
                    try:
                        setattr(cand, "candidate_id", cid)
                    except Exception:
                        pass
                except Exception:
                    cid = None
        return {
            "action": action,
            "candidate_id": cid,
            "structure": structure or getattr(
                getattr(intent, "decision", None), "structure", None),
            "direction": (
                getattr(decision, "direction", None)
                or getattr(getattr(intent, "decision", None), "direction", None)),
            "size_mult": float(final_size_mult or 0.0),
        }

    def _deployment_mode(self) -> str:
        dep = getattr(self, "deployment_bundle", None)
        if dep is not None:
            return str(getattr(dep, "mode", "shadow") or "shadow").lower()
        return str(getattr(self, "policy_mode", "shadow") or "shadow").lower()

    def _unified_unavailable_fields(
            self, *, reason: str, decision=None, intent=None,
            final_size_mult: float = 0.0, snapshot_id: str = "",
            exc: Optional[BaseException] = None,
            now: Optional[dt.datetime] = None) -> dict:
        """
        Fail-closed TickResult unified fields for candidate/champion.

        Always routes through resolve_authority() — never a hand-rolled
        fallback. Persists a UnifiedDecisionRecord when a store is available.
        """
        from decision_stack.authority import resolve_authority
        from decision_stack.contracts import UnifiedDecisionRecord

        mode = self._deployment_mode()
        dep = getattr(self, "deployment_bundle", None)
        dep_id = str(getattr(dep, "deployment_id", "") or "") if dep else ""
        cfg_hash = str(
            getattr(dep, "configuration_hash", "") or "") if dep else ""
        fallback = str(
            getattr(dep, "fallback_policy", "abstain") if dep else "abstain"
        ).lower()
        legacy = self._legacy_decision_dict(
            decision, intent, snapshot_id, final_size_mult)
        reasons = [reason]
        if exc is not None:
            reasons.append(f"{type(exc).__name__}:{exc}")

        v3 = {
            "statistical_action": "UNAVAILABLE",
            "final_action": "UNAVAILABLE",
            "candidate_id": None,
            "structure": None,
            "direction": None,
            "reasons": tuple(reasons),
        }
        auth = resolve_authority(
            mode=mode,
            legacy_decision=legacy,
            v3_decision=v3,
            hard_vetoes=(),
            fallback_policy=fallback,
            legacy_size_mult=float(legacy.get("size_mult") or 0.0),
            v3_size_mult=0.0,
        )
        ts_iso = ""
        session_date = ""
        if now is not None:
            ts_iso = now.astimezone(ET).isoformat()
            session_date = now.astimezone(ET).date().isoformat()
        record = UnifiedDecisionRecord(
            snapshot_id=str(snapshot_id or ""),
            ts=ts_iso,
            session_date=session_date,
            symbol=str(getattr(self, "symbol", "SPY") or "SPY"),
            deployment_id=dep_id,
            deployment_mode=str(mode),
            authority_source=auth.authority_source,
            legacy_action=str(legacy.get("action") or "NO_EDGE"),
            legacy_candidate_id=legacy.get("candidate_id"),
            legacy_structure=legacy.get("structure"),
            legacy_direction=legacy.get("direction"),
            legacy_size_mult=float(legacy.get("size_mult") or 0.0),
            v3_statistical_action="UNAVAILABLE",
            v3_final_action="UNAVAILABLE",
            selected_candidate_id=auth.selected_candidate_id,
            final_action=auth.final_action,
            final_structure=auth.final_structure,
            final_direction=auth.final_direction,
            final_size_mult=auth.final_size_mult,
            reasons=tuple(auth.reasons) + tuple(reasons),
            fallback_used=auth.fallback_used,
            fallback_reason=auth.fallback_reason or reason,
            configuration_hash=cfg_hash,
            diagnostics={"unavailable_reason": reason},
        )
        # Persist fail-closed record so it does not disappear from the journal.
        store = getattr(self, "prediction_store", None)
        stack = getattr(self, "decision_stack", None)
        try:
            if stack is not None and getattr(stack, "persist_fn", None):
                stack.persist_fn(record, snapshot=None, universe=None,
                                 forecast=None, v3_result=None)
            elif store is not None:
                from decision_stack.persistence import persist_unified_decision
                persist_unified_decision(store, record)
        except Exception as persist_exc:
            log.warning("unavailable decision persist failed: %s", persist_exc)

        part3_payload = {
            **(getattr(self, "_tick_part3", None) or {}),
            "unified": record.to_dict(),
            "statistical_action": "UNAVAILABLE",
            "final_action": "UNAVAILABLE",
            "selected_candidate_id": auth.selected_candidate_id,
            "authority_source": auth.authority_source,
            "deployment_id": dep_id,
            "deployment_mode": mode,
            "fallback_used": auth.fallback_used,
            "reasons": list(record.reasons),
            "mode": mode,
            "shadow_label": f"{mode.upper()} — UNAVAILABLE ({reason})",
        }
        self._tick_part3 = part3_payload
        self._tick_unified_v3 = v3
        self._tick_authoritative = {
            "final_action": auth.final_action,
            "selected_candidate_id": auth.selected_candidate_id,
            "structure": auth.final_structure,
            "direction": auth.final_direction,
            "size_mult": auth.final_size_mult,
        }
        return {
            "legacy_decision": legacy,
            "v3_decision": v3,
            "authoritative_decision": self._tick_authoritative,
            "authority_source": auth.authority_source,
            "deployment_id": dep_id,
            "fallback_used": auth.fallback_used,
            "candidate_universe_summary": None,
            "part3": part3_payload,
        }

    def _evaluate_unified_stack(
            self, *, snap, now, snapshot_id: str, signals: dict,
            intent, regime_state, decision, final_size_mult: float,
            decide_pdf, cfg) -> dict:
        """
        Run UnifiedDecisionStack when configured. Builds one CanonicalSnapshot
        and one CandidateUniverse from the shared select_spreads enumeration.

        Returns TickResult unified-field kwargs. In candidate/champion mode,
        failures never return {} — they return UNAVAILABLE (or legacy fallback
        when fallback_policy=legacy).
        """
        stack = getattr(self, "decision_stack", None)
        if stack is None:
            return {}
        mode = self._deployment_mode()
        strict = mode in ("candidate", "champion")

        try:
            from prediction.canonical_snapshot import (
                build_canonical_snapshot,
                compute_source_ages_seconds,
                extract_source_timestamps,
            )
            from prediction.candidate_universe import build_candidate_universe
            from prediction.inference import live_feature_row
        except Exception as exc:
            log.warning("unified stack imports failed: %s", exc)
            if strict:
                return self._unified_unavailable_fields(
                    reason="unified_import_failed", decision=decision,
                    intent=intent, final_size_mult=final_size_mult,
                    snapshot_id=snapshot_id, exc=exc, now=now)
            return {}

        try:
            row = live_feature_row(snap, signals)
            ts_iso = now.astimezone(ET).isoformat()
            session_date = now.astimezone(ET).date().isoformat()
            src_ts = extract_source_timestamps(
                now_iso=ts_iso,
                bars=getattr(snap, "bars", None),
                chain=getattr(snap, "chain", None),
                market=getattr(snap, "market", None),
                signals=signals,
            )
            ages = compute_source_ages_seconds(ts_iso, src_ts)
            canon = build_canonical_snapshot(
                symbol=str(getattr(self, "symbol", "SPY") or "SPY"),
                ts=ts_iso,
                session_date=session_date,
                market=snap.market,
                bars=getattr(snap, "bars", None),
                chain=getattr(snap, "chain", None),
                raw_features=row,
                snapshot_id=snapshot_id,
                quality={
                    "data_quality": (
                        float(signals["data_quality"])
                        if signals.get("data_quality") is not None
                        else 0.85
                    ),
                    "feature_coverage": min(1.0, len(row) / 40.0),
                },
                source_timestamps=src_ts,
                source_ages_seconds=ages,
            )

            # One immutable shared universe for V1 + V3 (built earlier in tick).
            universe = getattr(self, "_tick_shared_universe", None)
            cands = list(getattr(self, "_tick_shared_cands", None) or [])
            if not cands:
                cands = list(getattr(self, "_tick_shadow_cands", None) or [])
            if universe is None:
                universe = build_candidate_universe(
                    snapshot_id=snapshot_id,
                    generated_at=ts_iso,
                    candidates=cands,
                    generator_config={
                        "selector_min_ev": getattr(cfg.selector, "min_ev", None),
                        "shared": True,
                    },
                )
                self._tick_shared_universe = universe

            # Capture universe for the stack without replacing a caller-provided fn.
            stack.candidate_universe_fn = (
                lambda snapshot, forecast=None, _u=universe: _u)

            # Portfolio risk is account-scoped. record_trade deferred to paper open.
            def _portfolio_risk(candidate_id: str, session_date: str,
                                account: str = "v3"):
                rm = self.risk_manager
                if str(account).lower() in ("v3", "candidate"):
                    rm = getattr(self, "candidate_risk_manager", None) or rm
                if rm is None:
                    return ()
                cand = self._pick_shadow_candidate(candidate_id=candidate_id)
                if cand is None:
                    return ("risk:candidate_unresolved",)
                # Prefer execution-adjusted max loss when V3 economics exist.
                risk_cand = cand
                try:
                    sel = (
                        (getattr(self, "_tick_part3", None) or {})
                        .get("unified", {})
                        .get("selected_candidate_evaluation")
                    ) or {}
                    if (str(sel.get("candidate_id") or "") == str(candidate_id)
                            and sel.get("expected_fill_price") is not None):
                        mid = (sel.get("diagnostics") or {}).get("mid_credit")
                        if mid is None:
                            mid = getattr(cand, "credit", None)
                        exp = float(sel["expected_fill_price"])
                        base_ml = float(getattr(cand, "max_loss", 0.0) or 0.0)
                        if mid is not None:
                            # Worse fill → higher max loss for credit structures.
                            adj = base_ml + max(0.0, float(mid) - exp)
                            risk_cand = SimpleNamespace(
                                family=getattr(cand, "family", None),
                                max_loss=adj,
                                gamma=getattr(cand, "gamma", 0.0),
                                capital=getattr(cand, "capital", adj),
                            )
                except Exception:
                    risk_cand = cand
                rcheck = rm.check(risk_cand, session_date)
                if rcheck.approved:
                    return ()
                return tuple(f"risk:{v}" for v in (rcheck.vetoes or []))

            stack.portfolio_risk_fn = _portfolio_risk

            legacy = self._legacy_decision_dict(
                decision, intent, snapshot_id, final_size_mult)
            hard = tuple(getattr(regime_state, "vetoes", None) or ())
            for key in ("hard_veto", "op_risk_vetoes"):
                extra = signals.get(key)
                if isinstance(extra, (list, tuple)):
                    hard = hard + tuple(str(x) for x in extra)

            record = stack.evaluate(
                canon,
                legacy_decision=legacy,
                hard_vetoes=hard,
            )

            part3_payload = getattr(self, "_tick_part3", None)
            if record is not None:
                part3_payload = {
                    **(part3_payload or {}),
                    "unified": record.to_dict(),
                    "statistical_action": record.v3_statistical_action,
                    "final_action": record.v3_final_action,
                    "selected_candidate_id": record.v3_candidate_id,
                    "authority_source": record.authority_source,
                    "deployment_id": record.deployment_id,
                    "deployment_mode": record.deployment_mode,
                    "fallback_used": record.fallback_used,
                    "hard_vetoes": list(record.hard_vetoes),
                    "reasons": list(record.reasons),
                    "shadow_label": (
                        f"{record.deployment_mode.upper()} — "
                        f"authority={record.authority_source}"),
                    "mode": record.deployment_mode,
                    # Keep decision_summary in sync for paper intents / dashboards
                    "decision_summary": {
                        **((part3_payload or {}).get("decision_summary") or {}),
                        "action": record.v3_final_action,
                        "statistical_action": record.v3_statistical_action,
                        "selected_candidate_id": record.v3_candidate_id,
                        "family": record.v3_structure,
                        "direction": record.v3_direction,
                        "reasons": list(record.reasons),
                    },
                }
                self._tick_part3 = part3_payload

            v3 = {
                "statistical_action": record.v3_statistical_action,
                "final_action": record.v3_final_action,
                "candidate_id": record.v3_candidate_id,
                "structure": record.v3_structure,
                "direction": record.v3_direction,
            }
            auth = {
                "final_action": record.final_action,
                "selected_candidate_id": record.selected_candidate_id,
                "structure": record.final_structure,
                "direction": record.final_direction,
                "size_mult": record.final_size_mult,
            }
            self._tick_unified_v3 = v3
            self._tick_authoritative = auth
            return {
                "legacy_decision": legacy,
                "v3_decision": v3,
                "authoritative_decision": auth,
                "authority_source": record.authority_source,
                "deployment_id": record.deployment_id,
                "fallback_used": record.fallback_used,
                "candidate_universe_summary": universe.to_dict(),
                "part3": part3_payload,
            }
        except Exception as exc:
            log.warning("unified decision stack failed: %s", exc)
            if strict:
                return self._unified_unavailable_fields(
                    reason="unified_stack_exception", decision=decision,
                    intent=intent, final_size_mult=final_size_mult,
                    snapshot_id=snapshot_id, exc=exc, now=now)
            return {}

    def tick(self, now: dt.datetime,
             position_contexts: Optional[list[PositionContext]] = None
             ) -> Optional[TickResult]:
        snap = self.feed.snapshot(now)
        if snap is None:
            return None

        self._tick_prediction_bundle = None
        self._tick_part3 = None
        self._tick_shadow_cands = []
        self._tick_shared_cands = []
        self._tick_shared_universe = None
        self._tick_shadow_forecasts = {}
        self._tick_paper_intents = []
        self._tick_unified_v3 = None
        self._tick_authoritative = None
        snapshot_id = self._next_snapshot_id(now)
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

        # ---- GEX variants (Prediction Engine V2 / PR 9, observation-only) ----
        # Parallel OI / weekly / volume / hybrid panels. Never overwrite
        # MarketSnapshot.net_gex / walls / flip — those remain the OI baseline
        # that gates and the selector consume until promotion.
        if getattr(snap, "option_rows", None):
            try:
                from gex.base import compute_all_variants
                from prediction.dataset import session_metadata
                mos = None
                try:
                    mos = session_metadata(now).get("minutes_since_open")
                    if mos is not None:
                        mos = float(mos)
                except Exception:
                    mos = None
                vor = getattr(m, "volume_oi_ratio", None)
                if isinstance(vor, float) and not math.isfinite(vor):
                    vor = None
                bundle = compute_all_variants(
                    spot=float(m.spot),
                    rows_0dte=snap.option_rows,
                    rows_weekly=getattr(snap, "weekly_option_rows", None),
                    source_age=None,
                    minute_of_session=mos,
                    volume_oi_ratio=vor,
                    feed_source=(getattr(snap, "gex_feed_source", "")
                                 or type(self.feed).__name__),
                )
                for k, v in bundle.to_signals_json().items():
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        signals[k] = float(v)
                    elif isinstance(v, str):
                        signals[k] = v
            except Exception as exc:
                log.warning("GEX variant panel failed: %s", exc)

        snap_dict.update(signals)
        # signals_json finalized after RAS merge below

        # ---- Volatility channels (Bollinger / Keltner / Donchian) ----
        # One classifier-TF computation per tick, shared by the classifier
        # (ctx.channel), RAS (via RegimeState.standardized), and the journal
        # (chan_* keys in signals_json for component_correlations).
        channel = channel_features_from_bars(snap.bars)
        for k, v in channel.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                signals[f"chan_{k}"] = float(v)

        # One pin assessment per tick — shared by classifier, matrix, gate,
        # selector, and V2 policy so labels cannot disagree with structure.
        from pin_regime import assess_pin, pin_to_signals
        pin = assess_pin(snap.market, signals=signals, channel=channel)
        signals.update(pin_to_signals(pin))

        # ---- Track B: regime classifier ----
        clf_ctx = ClassifierContext(market=snap.market, rnd=rnd, edge=edge,
                                    channel=channel, pin=pin)
        regime_state = self._classifier.classify(clf_ctx, self._prev_std)
        self._prev_std = regime_state.standardized

        # Champion regime_overrides: swap in the pre-resolved per-regime
        # EngineConfig for everything downstream of classification (gate +
        # selector). The RND extraction above ran on the base config — the
        # regime is unknowable before the classifier has spoken.
        regime_size_mult = 1.0
        if self._regime_cfg:
            key = regime_state.dominant_regime or "unknown"
            if key in self._regime_cfg:
                cfg, regime_size_mult = self._regime_cfg[key]

        # periodic flush of adaptive scales (cheap; ~every 10 minutes at 60s ticks)
        self._ticks_since_save += 1
        if self._ticks_since_save >= 10:
            self._save_state()
            self._ticks_since_save = 0

        # ---- Track B: matrix + decision routing ----
        mtf_in = build_mtf_input(snap.bars, snap_dict)
        mat_rows = build_matrix(mtf_in, self._matrix_scale_book)
        regimes = regime_rows(mat_rows)

        # Raw table lookup (no pin) for counterfactual journaling.
        raw_intent = decide_from_matrix(
            mat_rows, regimes, vetoes=regime_state.vetoes, pin=None)
        intent = decide_from_matrix(
            mat_rows, regimes, vetoes=regime_state.vetoes, pin=pin)
        signals["pin_raw_structure"] = raw_intent.decision.structure
        signals["pin_raw_exec"] = raw_intent.exec_regime
        signals["pin_raw_context"] = raw_intent.context_regime
        if pin.is_pin and raw_intent.decision.structure != intent.decision.structure:
            signals["pin_structure_override"] = 1.0
        else:
            signals["pin_structure_override"] = 0.0

        # Observation-only regime time series for the dashboard (chart shading
        # + quadrant view): the continuous direction-bias value (0-100, 50 =
        # neutral) and the dominant regime's confidence. Journaled in
        # signals_json so no schema change and zero gate/veto power.
        if isinstance(intent.bias_value, (int, float)) and math.isfinite(intent.bias_value):
            signals["regime_bias_value"] = float(intent.bias_value)
        dom_conf = regime_state.confidences.get(regime_state.dominant_regime)
        if isinstance(dom_conf, (int, float)) and math.isfinite(dom_conf):
            signals["regime_dominant_conf"] = float(dom_conf)

        # Raw fast/slow direction composites plus the crossover event. The fast
        # composite is the turn-detection channel (leads the 60%-slow blend at
        # intraday reversals); bias_cross = +/-1 only on the tick where the
        # fast side overtakes/loses the slow side. Observation-only.
        bf, bs = intent.bias_fast, intent.bias_slow
        if isinstance(bf, (int, float)) and math.isfinite(bf):
            signals["bias_fast"] = float(bf)
        if isinstance(bs, (int, float)) and math.isfinite(bs):
            signals["bias_slow"] = float(bs)
        cross = self._bias_cross(
            bf if isinstance(bf, (int, float)) else None,
            bs if isinstance(bs, (int, float)) else None)
        if cross is not None:
            signals["bias_cross"] = cross
            log.info("Direction composite crossover: fast %s slow (fast=%.1f slow=%.1f)",
                     "above" if cross > 0 else "below", bf, bs)

        # Routing provenance for journal.decision_funnel(): what Track B
        # actually routed, whether a dealer veto flipped a credit cell to its
        # debit cousin, and which regime vetoes were active. Without this the
        # journal only sees the FINAL family, so a forced LCS is
        # indistinguishable from a trend-cell LCS — exactly the distinction
        # needed to answer "why is premium not trading?".
        signals["routed_structure"] = intent.decision.structure
        signals["premium_flip"] = 1.0 if "premium veto" in intent.note else 0.0
        if regime_state.vetoes:
            signals["regime_vetoes"] = ",".join(regime_state.vetoes)
        # Warm-up provenance for funnel / validation honesty.
        signals["gex_rank_warm"] = 1.0 if bool(
            getattr(snap.market, "gex_rank_warm", True)) else 0.0
        # Session entry warmup (first ~30m): hard gate blocks new tickets;
        # journal the flag so the funnel/dashboard can show why.
        try:
            from gate_scorer import GateConfig
            entry_open = (self.engine_cfg.gate.morning_entry_time
                          if self.engine_cfg is not None
                          else GateConfig().morning_entry_time)
        except Exception:
            from gate_scorer import GateConfig
            entry_open = GateConfig().morning_entry_time
        signals["session_warmup"] = (
            1.0 if snap.market.et_time() < entry_open else 0.0)
        # Internal: let bundle provider reuse the journal-linked snapshot id.
        signals["_snapshot_id"] = snapshot_id

        ras_results = self._compute_ras(
            regime_state, intent, snap.market, position_contexts)
        self._journal_ras(now, ras_results)
        signals, signals_json = self._signals_with_ras(signals, ras_results)

        # ---- Prediction policy dual-run (PR 10 / §17) ----
        # Shadow (default): legacy matrix remains authoritative; V2 decision
        # and disagreement are journaled. Champion: V2 drives structure /
        # stand-down with explicit fallback_legacy when the bundle is missing.
        policy_route = self._route_policy(snap, signals, intent, regime_state)

        # ---- MTF sigma cones (outward-looking prediction journal) ----
        # Emit after policy so bias/vol signals exist; settle due bands against
        # the true spot on every tick so coverage is measurable.
        self._emit_and_settle_sigma_cones(
            now, snapshot_id, snap, signals)

        live_structure = intent.decision.structure
        live_direction = intent.decision.direction
        live_size_mult = float(intent.size_mult)
        # Matrix stand-down (independent of champion override) — drives legacy paper.
        matrix_stand_down = bool(
            regime_state.stand_down or intent.decision.structure == "NT")
        stand_down_now = bool(matrix_stand_down)
        if policy_route is not None:
            for k, v in policy_route.journal_signals().items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    signals[k] = float(v) if not isinstance(v, bool) else v
                elif isinstance(v, bool):
                    signals[k] = v
                elif isinstance(v, str):
                    signals[k] = v
            # Refresh signals_json so policy provenance lands on the journal.
            signals, signals_json = self._signals_with_ras(signals, ras_results)
            # Single authority: when UnifiedDecisionStack is wired, V2 champion
            # must not mutate live structure / stand-down — the stack owns that.
            apply_v2_champion = (
                str(getattr(policy_route, "mode", "")).lower() == "champion"
                and getattr(self, "decision_stack", None) is None
            )
            if apply_v2_champion:
                auth = policy_route.authoritative
                if auth.action == "NO_TRADE":
                    stand_down_now = True
                    live_structure = "NT"
                    live_direction = "none"
                    live_size_mult = 0.0
                else:
                    stand_down_now = False
                    live_structure = auth.structure_code or "NT"
                    live_direction = auth.direction
                    live_size_mult = float(auth.size_cap)

        # ---- Canonical dataset capture (observation-only, PR 3) ----
        self._log_feature_snapshot(now, snapshot_id, snap, snap_dict,
                                   signals, mat_rows)

        # ---- V2 physical density (observation on every chain tick) ----
        # Run before the stand-down early return so the V2 dashboard tab
        # still shows forecast moments / density while legacy is NT.
        v2_result = None
        if snap.chain is not None and rnd is not None:
            v2_result = self._build_v2_physical_result(
                snap, signals, intent, rnd, cfg)

        # ---- Shared candidate universe (once per tick, before V1 + V3) ----
        # Both legacy decide() and the unified stack consume this immutable set.
        decide_pdf_for_universe = phys_pdf
        if (v2_result is not None and not self.use_legacy_directional_tilt
                and self.physical_pdf is None):
            decide_pdf_for_universe = v2_result.as_callable()
        if snap.chain is not None:
            self._ensure_shared_candidate_universe(
                snap, decide_pdf_for_universe, cfg,
                pin_active=bool(pin.is_pin),
                snapshot_id=snapshot_id, now=now)

        # ---- Stand-down: regime unstable or NT cell (or champion NO_TRADE) ----
        if stand_down_now:
            decide_pdf = phys_pdf
            density_moments_sd = None
            if snap.chain is not None:
                if (v2_result is not None
                        and not self.use_legacy_directional_tilt
                        and self.physical_pdf is None):
                    decide_pdf = v2_result.as_callable()
                    density_moments_sd = v2_result.moments
                    signals["phys_density_mode"] = "v2"
                elif decide_pdf is not None:
                    signals["phys_density_mode"] = (
                        "realized_vol" if phys_pdf is not None else "vrp")
                else:
                    signals["phys_density_mode"] = "vrp"
                self._run_v2_shadow_ranking(
                    snap, signals, snapshot_id, decide_pdf, cfg,
                    decision=None, pin_active=bool(pin.is_pin))
            signals, signals_json = self._signals_with_ras(signals, ras_results)
            pub_signals = {k: v for k, v in signals.items()
                           if not str(k).startswith("_")}
            # Stand-down: overlay HARD_VETO so assessment sees operational veto
            # separately from the statistical Part 3 action.
            if getattr(self, "_tick_part3", None):
                p3 = dict(self._tick_part3)
                ds = dict(p3.get("decision_summary") or {})
                vetoes = [str(v) for v in (regime_state.vetoes or [])]
                if not vetoes:
                    reason = (
                        signals.get("stand_down_reason")
                        or signals.get("no_trade_reason")
                        or f"stand_down:{regime_state.dominant_regime}"
                    )
                    vetoes = [str(reason)]
                ds["hard_vetoes"] = vetoes
                ds["statistical_action"] = (
                    ds.get("statistical_action") or ds.get("action"))
                ds["action"] = "HARD_VETO"
                ds["reasons"] = list(ds.get("reasons") or []) + [
                    f"hard_veto:{v}" for v in vetoes
                ]
                p3["decision_summary"] = ds
                self._tick_part3 = p3
            density_mode_sd = str(signals.get("phys_density_mode") or "vrp")
            # Unified first — paper intents must read post-stack V3 outputs.
            unified_fields = self._evaluate_unified_stack(
                snap=snap, now=now, snapshot_id=snapshot_id, signals=signals,
                intent=intent, regime_state=regime_state, decision=None,
                final_size_mult=0.0, decide_pdf=decide_pdf, cfg=cfg,
            )
            paper_intents = self._build_paper_intents(
                snap=snap, signals=signals, intent=intent,
                regime_state=regime_state, decision=None,
                decide_pdf=decide_pdf,
                cfg=cfg, pin_active=bool(pin.is_pin),
                density_mode=density_mode_sd,
                density_moments=density_moments_sd,
                final_size_mult=0.0,
                matrix_stand_down=True,
            )
            result = TickResult(
                ts=now, regime=regime_state, intent=intent,
                decision=None, final_size_mult=0.0,
                vetoes=regime_state.vetoes, snapshot=snap,
                ras_results=ras_results,
                signals=pub_signals,
                sigma_cones=getattr(self, "_tick_sigma_cones", None),
                part3=unified_fields.get(
                    "part3", getattr(self, "_tick_part3", None)),
                paper_intents=paper_intents,
                legacy_decision=unified_fields.get("legacy_decision"),
                v3_decision=unified_fields.get("v3_decision"),
                authoritative_decision=unified_fields.get(
                    "authoritative_decision"),
                authority_source=unified_fields.get("authority_source"),
                deployment_id=unified_fields.get("deployment_id"),
                fallback_used=bool(unified_fields.get("fallback_used")),
                candidate_universe_summary=unified_fields.get(
                    "candidate_universe_summary"),
            )
            if self.journal:
                row = _no_trade_row(snap.market, intent, regime_state,
                                    direction=live_direction,
                                    signals_json=signals_json)
                row["snapshot_id"] = snapshot_id
                self.journal.log(row)
            return result

        # ---- Track A: full engine (requires chain) ----
        # Physical density for candidate EV. Priority:
        #   1. Injected self.physical_pdf (tests / external override)
        #   2. V2 independent density from PhysicalForecast, when the legacy
        #      tilt flag is off (§12.5 migration)
        #   3. Legacy dir_drift_frac tilt of the realized-vol density when the
        #      router emits a directional debit AND use_legacy_directional_tilt
        #   4. Drift-less realized-vol density (same as richness above)
        # Richness (edge above) ALWAYS stays on the drift-less density — that
        # measurement is variance, not direction, and must remain independent
        # of the routed structure.
        decision = None
        density_mode = "injected" if self.physical_pdf is not None else (
            "realized_vol" if phys_pdf is not None else "vrp")
        density_moments: Optional[dict] = None
        decide_pdf = phys_pdf
        if snap.chain is not None:
            use_v2 = (v2_result is not None
                      and not self.use_legacy_directional_tilt
                      and self.physical_pdf is None)
            if use_v2:
                decide_pdf = v2_result.as_callable()
                density_mode = "v2"
                density_moments = v2_result.moments
            elif (self.physical_pdf is None and rnd is not None
                    and sigma_rv is not None
                    and self.use_legacy_directional_tilt
                    and live_structure in DIRECTIONAL_TILT_STRUCTURES):
                # Legacy circular tilt — kept behind the migration flag.
                sign = 1.0 if live_direction == "call" else -1.0
                tilt = sign * cfg.rnd.dir_drift_frac * live_size_mult
                tilted = physical_pdf_from_realized_vol(
                    rnd, sigma_rv, cfg.rnd, drift_std_frac=tilt)
                if tilted is not None:
                    decide_pdf = tilted
                    density_mode = "legacy_tilt"
                    signals["phys_legacy_tilt"] = tilt

            # Shadow EV comparison: when V2 is available but legacy still
            # prices the live decision (or vice versa), re-price the same
            # candidate set under the other density and journal both EVs.
            pin_active = bool(pin.is_pin)
            shared = list(getattr(self, "_tick_shared_cands", None) or []) or None
            if (v2_result is not None and self.physical_pdf is None
                    and density_mode != "v2"):
                try:
                    shadow = decide(
                        snap.market, snap.chain, cfg,
                        physical_pdf=v2_result.as_callable(),
                        target_structure=live_structure,
                        direction=live_direction,
                        physical_density_mode="v2",
                        physical_moments=v2_result.moments,
                        pin_active=pin_active,
                        precomputed_candidates=shared)
                    if shadow.candidate is not None:
                        signals["phys_v2_shadow_ev"] = shadow.candidate.ev
                        signals["phys_v2_shadow_family"] = shadow.candidate.family
                except Exception as exc:
                    log.warning("V2 shadow EV failed: %s", exc)

            decision = decide(snap.market, snap.chain, cfg,
                              physical_pdf=decide_pdf,
                              target_structure=live_structure,
                              direction=live_direction,
                              physical_density_mode=density_mode,
                              physical_moments=density_moments,
                              pin_active=pin_active,
                              precomputed_candidates=shared)
            signals["phys_density_mode"] = density_mode
            if (decision.candidate is not None
                    and isinstance(decision.candidate.ev, (int, float))):
                signals["phys_live_ev"] = decision.candidate.ev

            # Counterfactual: under pin, journal the path we did NOT take.
            try:
                from decision_matrix import PREMIUM_STRUCTURES
                raw_struct = raw_intent.decision.structure
                if pin_active and live_structure in PREMIUM_STRUCTURES:
                    if raw_struct not in PREMIUM_STRUCTURES and raw_struct != "NT":
                        cf = decide(
                            snap.market, snap.chain, cfg,
                            physical_pdf=decide_pdf,
                            target_structure=raw_struct,
                            direction=raw_intent.decision.direction,
                            physical_density_mode=density_mode,
                            pin_active=False,
                            precomputed_candidates=shared)
                        if cf.candidate is not None:
                            signals["cf_debit_ev"] = cf.candidate.ev
                            signals["cf_debit_family"] = cf.candidate.family
                            signals["cf_debit_structure"] = raw_struct
                elif pin_active and live_structure not in PREMIUM_STRUCTURES:
                    cf = decide(
                        snap.market, snap.chain, cfg,
                        physical_pdf=decide_pdf,
                        target_structure="IF",
                        direction="both",
                        physical_density_mode=density_mode,
                        pin_active=True,
                        precomputed_candidates=shared)
                    if cf.candidate is not None:
                        signals["cf_premium_ev"] = cf.candidate.ev
                        signals["cf_premium_family"] = cf.candidate.family
                        signals["cf_premium_structure"] = "IF"
            except Exception as exc:
                log.warning("pin counterfactual failed: %s", exc)

            # ---- V2 candidate-value shadow ranking (PR 8 / §14) ----
            annotated = self._run_v2_shadow_ranking(
                snap, signals, snapshot_id, decide_pdf, cfg,
                decision=decision, pin_active=pin_active)
            if annotated is not None:
                decision = annotated

            # Re-finalize signals_json after density provenance is attached.
            signals, signals_json = self._signals_with_ras(signals, ras_results)
            # ---- Risk gate (legacy journal only) ----
            # Portfolio RiskManager.check/record_trade for the *authoritative*
            # candidate runs via UnifiedDecisionStack.portfolio_risk_fn /
            # paper open. Here we only annotate the legacy decision for the
            # journal — never record_trade (that would desync state from V3).
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
            if self.journal:
                row = decision.as_row()
                row["signals_json"] = signals_json
                row["snapshot_id"] = snapshot_id
                self.journal.log(row)
        else:
            # No chain yet — log intent as a no-trade stub for calibration
            if self.journal:
                row = _no_trade_row(snap.market, intent, regime_state,
                                    reason="no_chain",
                                    direction=live_direction,
                                    signals_json=signals_json)
                row["snapshot_id"] = snapshot_id
                self.journal.log(row)

        # size_mult from Track B (or champion policy size_cap) scales the
        # Track A position; the champion's per-regime size_mult (if any)
        # scales on top.
        final_size = (live_size_mult * regime_size_mult
                      if (decision is not None
                          and decision.decision == "TRADE") else 0.0)

        pub_signals = {k: v for k, v in signals.items()
                       if not str(k).startswith("_")}
        # Unified evaluation before paper intents so V3 track uses the
        # post-stack decision, not a stale Part 3 summary.
        unified_fields = self._evaluate_unified_stack(
            snap=snap, now=now, snapshot_id=snapshot_id, signals=signals,
            intent=intent, regime_state=regime_state, decision=decision,
            final_size_mult=final_size, decide_pdf=decide_pdf, cfg=cfg,
        )
        paper_intents = self._build_paper_intents(
            snap=snap, signals=signals, intent=intent,
            regime_state=regime_state, decision=decision,
            decide_pdf=decide_pdf,
            cfg=cfg, pin_active=bool(pin.is_pin),
            density_mode=density_mode,
            density_moments=density_moments,
            final_size_mult=final_size,
            matrix_stand_down=matrix_stand_down,
        )
        return TickResult(
            ts=now, regime=regime_state, intent=intent,
            decision=decision,
            final_size_mult=round(final_size, 2),
            vetoes=regime_state.vetoes, snapshot=snap,
            ras_results=ras_results,
            signals=pub_signals,
            sigma_cones=getattr(self, "_tick_sigma_cones", None),
            part3=unified_fields.get("part3",
                                     getattr(self, "_tick_part3", None)),
            paper_intents=paper_intents,
            legacy_decision=unified_fields.get("legacy_decision"),
            v3_decision=unified_fields.get("v3_decision"),
            authoritative_decision=unified_fields.get("authoritative_decision"),
            authority_source=unified_fields.get("authority_source"),
            deployment_id=unified_fields.get("deployment_id"),
            fallback_used=bool(unified_fields.get("fallback_used")),
            candidate_universe_summary=unified_fields.get(
                "candidate_universe_summary"),
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
        n = 0
        if self.journal is not None:
            price = self.feed.settlement_price(session_date)
            if price is not None:
                n = self.journal.settle_session(session_date, price)
                # Force-settle any remaining cone bands for the session against
                # the official settlement price so the prediction journal closes.
                store = self.prediction_store
                if store is not None and hasattr(store, "settle_sigma_cones"):
                    try:
                        # Far-future ISO so every unsettled band for the day is due.
                        far = f"{session_date}T23:59:59-04:00"
                        store.settle_sigma_cones(far, float(price),
                                                 realized_ts=far)
                    except Exception as exc:
                        log.warning("sigma cone EOD settle failed: %s", exc)
        return n


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
