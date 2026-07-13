"""
prediction/inference.py
=======================
Live V2 inference helpers for shadow parallel operation.

Provides:
  * live_feature_row — train/serve-safe feature capture (pre-routing)
  * heuristic_bundle_from_tick — usable PredictionBundle without trained models
  * HeuristicCandidateValueModel — utility ranking from legacy EV/safety fields
  * make_bundle_provider / make_physical_forecast_provider — UnifiedOrchestrator hooks

Observation / shadow only until policy_mode=champion. NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from prediction.contracts import PredictionBundle
from prediction.dataset import FEATURE_VERSION, session_metadata
from prediction.models.candidate_value import (
    CANDIDATE_VALUE_VERSION, CandidateForecast,
)

ET = ZoneInfo("America/New_York")
HEURISTIC_BUNDLE_VERSION = "v2-heuristic-bundle-v1"
HEURISTIC_RANKER_VERSION = "v2-heuristic-ranker-v1"


def live_feature_row(snap, signals: Optional[dict] = None) -> dict:
    """
    Model-feature row aligned with offline `_tick_features` (market + MTF
    snapshot only). Routing / policy / RAS keys must NOT enter here.
    """
    signals = signals or {}
    market = snap.market
    row: dict = {}
    try:
        snap_dict = market.mtf_snapshot()
        for name, v in snap_dict.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                row[name] = float(v)
    except Exception:
        pass
    for name in ("spot", "net_gex", "gamma_flip", "call_wall", "put_wall",
                 "gex_pct_rank", "vix9d", "vix", "vix3m", "vvix",
                 "straddle_breakeven", "expected_range", "adx", "rsi",
                 "bb_width", "vwap", "tick_abs_mean", "cvd_slope"):
        v = getattr(market, name, None)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            row[name] = float(v)
    # Observation-only dynamics that are as-of safe (no routing provenance).
    for name in ("emc", "flip_velocity", "wall_velocity_call", "wall_velocity_put",
                 "gex_velocity", "move_consumed"):
        v = signals.get(name)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            row[name] = float(v)
    return row


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _implied_move_frac(market) -> Optional[float]:
    """Remaining-session implied move as a fraction of spot."""
    spot = float(getattr(market, "spot", 0.0) or 0.0)
    if spot <= 0:
        return None
    for attr in ("expected_range", "straddle_breakeven"):
        v = getattr(market, attr, None)
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0:
            return float(v) / spot
    return None


def heuristic_bundle_from_tick(
    snap,
    signals: Optional[dict] = None,
    *,
    snapshot_id: str,
    symbol: str = "SPY",
) -> PredictionBundle:
    """
    Build a usable PredictionBundle from live market + matrix bias signals so
    PredictionPolicy can dual-run without trained artifacts.

    Direction comes from bias_value / bias_fast (0-100, 50=neutral).
    Range survival is a soft prior from GEX rank + ADX + wall cushion.
    Realized move is proxied from EWMA-ish bb_width / straddle when available.
    """
    signals = signals or {}
    market = snap.market
    now = market.now
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    meta = session_metadata(now)
    session_date = meta.get("session_date") or now.astimezone(ET).date().isoformat()

    bias = signals.get("regime_bias_value")
    if not isinstance(bias, (int, float)) or not math.isfinite(float(bias)):
        bias = signals.get("bias_fast")
    if not isinstance(bias, (int, float)) or not math.isfinite(float(bias)):
        bias = 50.0
    bias = float(bias)
    # Map 0-100 bias to P(up): 50 -> 0.50, 100 -> ~0.75, 0 -> ~0.25
    p_up = _clip01(0.5 + (bias - 50.0) / 200.0)

    implied = _implied_move_frac(market)
    # Realized-move prior: bb_width is absolute; prefer expected_range fraction.
    realized = implied
    bb = getattr(market, "bb_width", None)
    spot = float(getattr(market, "spot", 0.0) or 0.0)
    if isinstance(bb, (int, float)) and spot > 0 and math.isfinite(float(bb)):
        bb_frac = float(bb) / spot
        if bb_frac > 0:
            realized = bb_frac if realized is None else 0.5 * realized + 0.5 * bb_frac

    adx = float(getattr(market, "adx", 0.0) or 0.0)
    gex_rank = float(getattr(market, "gex_pct_rank", 0.5) or 0.5)
    # High GEX + low ADX → higher range survival; trending/short-gamma → lower.
    range_p = _clip01(0.45 + 0.35 * gex_rank - 0.01 * max(adx - 15.0, 0.0))
    if float(getattr(market, "net_gex", 0.0) or 0.0) < 0:
        range_p = _clip01(range_p - 0.15)

    # Expected log return from directional bias (small).
    exp_ret = (p_up - 0.5) * 2.0 * (realized or 0.004)
    q_spread = (realized or 0.004) * 1.28  # ~80% interval under normal prior
    q10 = exp_ret - q_spread
    q50 = exp_ret
    q90 = exp_ret + q_spread

    # Uncertainty rises when warm-up / missing chain / extreme ADX.
    warm = bool(getattr(market, "gex_rank_warm", True))
    has_chain = snap.chain is not None
    uncertainty = 0.35
    if not warm:
        uncertainty += 0.15
    if not has_chain:
        uncertainty += 0.20
    if adx >= 30:
        uncertainty += 0.10
    uncertainty = _clip01(uncertainty)
    coverage = 0.85 if has_chain else 0.55
    if not warm:
        coverage = max(0.4, coverage - 0.15)

    return PredictionBundle(
        snapshot_id=snapshot_id,
        ts=now.astimezone(ET).isoformat(),
        session_date=session_date,
        symbol=symbol,
        p_up_15m=p_up,
        p_up_30m=p_up,
        p_up_60m=p_up,
        p_up_close=p_up,
        expected_return_15m=exp_ret * 0.5,
        expected_return_30m=exp_ret,
        expected_return_60m=exp_ret * 1.2,
        expected_return_close=exp_ret * 1.4,
        return_q10_30m=q10,
        return_q50_30m=q50,
        return_q90_30m=q90,
        return_q10_close=q10 * 1.2,
        return_q50_close=q50 * 1.2,
        return_q90_close=q90 * 1.2,
        expected_realized_move_30m=realized,
        expected_realized_move_close=(
            realized * 1.3 if realized is not None else None),
        p_range_survive_15m=range_p,
        p_range_survive_30m=range_p,
        p_range_survive_60m=_clip01(range_p - 0.05),
        p_range_survive_close=_clip01(range_p - 0.08),
        uncertainty=uncertainty,
        data_quality=coverage,
        feature_coverage=coverage,
        feature_version=FEATURE_VERSION,
        model_versions={"bundle": HEURISTIC_BUNDLE_VERSION},
        diagnostics={"source": "heuristic", "bias_value": bias},
    )


def make_bundle_provider(
    *,
    symbol: str = "SPY",
    group=None,
    store=None,
) -> Callable:
    """
    Provider for UnifiedOrchestrator.prediction_bundle_provider.

    Signature: (snap, signals, intent, regime_state) -> PredictionBundle|None
    Uses trained PredictionModelGroup when provided; otherwise heuristic.
    """
    def provider(snap, signals, intent, regime_state):
        from prediction.training import build_prediction_bundle

        now = snap.market.now
        if now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        # Prefer orchestrator-assigned snapshot id when journaled on signals.
        snapshot_id = None
        if isinstance(signals, dict):
            snapshot_id = signals.get("_snapshot_id")
        if not snapshot_id:
            from prediction.dataset import make_snapshot_id
            snapshot_id = make_snapshot_id(symbol, now, FEATURE_VERSION, 0)
        meta = session_metadata(now)
        session_date = (meta.get("session_date")
                        or now.astimezone(ET).date().isoformat())

        if group is not None:
            row = live_feature_row(snap, signals)
            # Structural walls for range survival.
            structural = {
                "spot": float(snap.market.spot),
                "put_wall": float(snap.market.put_wall),
                "call_wall": float(snap.market.call_wall),
                "net_gex": float(snap.market.net_gex),
                "adx": float(snap.market.adx),
                "cvd_slope": float(snap.market.cvd_slope),
                "minutes_to_close": meta.get("minutes_to_close"),
            }
            bundle = build_prediction_bundle(
                group, row,
                snapshot_id=str(snapshot_id),
                ts=now.astimezone(ET).isoformat(),
                session_date=session_date,
                symbol=symbol,
                quality={"feature_coverage": 0.8},
                structural=structural,
            )
        else:
            bundle = heuristic_bundle_from_tick(
                snap, signals, snapshot_id=str(snapshot_id), symbol=symbol)

        if store is not None:
            try:
                import datetime as _dt
                store.log_prediction(
                    bundle.snapshot_id,
                    bundle.model_versions.get("group", HEURISTIC_BUNDLE_VERSION),
                    bundle.to_dict(),
                    uncertainty=bundle.uncertainty,
                    generated_at=_dt.datetime.now(tz=ET).isoformat(),
                    mode="shadow",
                )
            except Exception:
                pass
        return bundle

    return provider


def make_physical_forecast_provider(bundle_provider: Callable) -> Callable:
    """Lift PredictionBundle → PhysicalForecast for density construction."""
    def provider(snap, signals, intent):
        # Bundle provider needs regime_state; pass a minimal stand-in.
        regime = getattr(intent, "_regime_state", None)
        bundle = bundle_provider(snap, signals, intent, regime)
        if bundle is None:
            return None
        from prediction.physical_distribution import forecast_from_bundle
        return forecast_from_bundle(bundle)
    return provider


@dataclass
class HeuristicCandidateValueModel:
    """
    Unfitted candidate ranker that scores from legacy EV / safety fields so
    V2 utility ranking can run in parallel without a trained model.
    Duck-types CandidateValueModel.predict().
    """
    metadata: dict = field(default_factory=lambda: {
        "model_version": HEURISTIC_RANKER_VERSION,
        "source": "heuristic",
    })
    fitted: bool = True

    def predict(
        self,
        rows: Sequence[dict],
        *,
        candidate_ids: Sequence[str],
        fill_uncertainty: Optional[Sequence[float]] = None,
        capital: Optional[Sequence[float]] = None,
        utility_fn: Optional[Callable] = None,
    ) -> list:
        fills = list(fill_uncertainty or [0.3] * len(rows))
        caps = list(capital or [0.0] * len(rows))
        out = []
        for i, row in enumerate(rows):
            ev = float(row.get("ev") or 0.0)
            max_loss = float(row.get("max_loss") or 0.0) or 1.0
            p_profit = _clip01(float(row.get("prob_profit") or 0.5))
            touch = float(row.get("prob_touch_short") or 0.0)
            shortfall = max(0.0, max_loss * touch)
            q10 = ev - max_loss * 0.5
            q50 = ev
            q90 = ev + abs(float(row.get("credit") or 0.0))
            fill_u = float(fills[i]) if i < len(fills) else 0.3
            model_u = 0.40  # heuristic uncertainty
            fc = CandidateForecast(
                candidate_id=str(candidate_ids[i]),
                expected_net_pnl=ev,
                p_profit=p_profit,
                pnl_q10=q10,
                pnl_q50=q50,
                pnl_q90=q90,
                expected_shortfall=shortfall,
                fill_uncertainty=fill_u,
                model_uncertainty=model_u,
                utility_score=0.0,
                model_version=HEURISTIC_RANKER_VERSION,
            )
            if utility_fn is not None:
                util = float(utility_fn(fc, capital=float(caps[i] if i < len(caps) else 0.0)))
            else:
                util = ev - 0.5 * shortfall - 0.25 * fill_u - 0.25 * model_u
            out.append(CandidateForecast(
                candidate_id=fc.candidate_id,
                expected_net_pnl=fc.expected_net_pnl,
                p_profit=fc.p_profit,
                pnl_q10=fc.pnl_q10,
                pnl_q50=fc.pnl_q50,
                pnl_q90=fc.pnl_q90,
                expected_shortfall=fc.expected_shortfall,
                fill_uncertainty=fc.fill_uncertainty,
                model_uncertainty=fc.model_uncertainty,
                utility_score=util,
                model_version=fc.model_version,
            ))
        return out
