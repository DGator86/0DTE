"""SPY-DER price prediction — the agent reading the chart like a trader.

Produces a structured, drawable forecast from the same market context the
dashboard chart shows (GEX walls, gamma flip, VWAP, expected range, momentum).
The result is attached to the ``spy_der`` parallel-track payload under
``prediction`` and rendered on the SPY-DER tab's chart (directional target, a
confidence cone into the close, and key levels).

Source precedence:
  1. The SPY-DER package's own forecaster (``spy_der.integrations.zerodte.
     predict_shadow_tick``) when installed on the VPS — Grok-driven.
  2. A deterministic "trader" model here as a fail-closed fallback, so the tab
     draws a real prediction even before the SPY-DER package is deployed.

NOT financial advice. No real orders are placed.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("spy_der_predict")

PREDICTION_SCHEMA = "spy_der.prediction.v1"


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if v == v else None  # reject NaN
    except (TypeError, ValueError):
        return None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _expected_band(spot: float, market: Any) -> float:
    """Half-width of the expected move; the cone scales off this."""
    band = _f(getattr(market, "expected_range", None))
    if not band or band <= 0:
        be = _f(getattr(market, "straddle_breakeven", None))
        if be is not None:
            band = abs(be - spot)
    if not band or band <= 0:
        band = spot * 0.004
    return min(band, spot * 0.05)


def deterministic_prediction(market: Any, *, now_iso: str = "") -> Optional[dict]:
    """Trader-style forecast from the live market snapshot.

    Reads dealer positioning (call/put walls, gamma flip, net GEX), trend/where
    price sits vs VWAP, and momentum (RSI, CVD slope) to produce a directional
    bias, an end-of-session target, a confidence cone, and the key levels it is
    watching. Deterministic and side-effect free.
    """
    spot = _f(getattr(market, "spot", None))
    if spot is None or spot <= 0:
        return None
    band = _expected_band(spot, market)

    vwap = _f(getattr(market, "vwap", None))
    call_wall = _f(getattr(market, "call_wall", None))
    put_wall = _f(getattr(market, "put_wall", None))
    gamma_flip = _f(getattr(market, "gamma_flip", None))
    net_gex = _f(getattr(market, "net_gex", None))
    gex_rank = _f(getattr(market, "gex_pct_rank", None)) or 0.5
    rsi = _f(getattr(market, "rsi", None))
    adx = _f(getattr(market, "adx", None))
    cvd_slope = _f(getattr(market, "cvd_slope", None))

    # --- directional score in ~[-1, 1]: each driver is a weighted vote. ---
    votes: list[tuple[str, float]] = []
    if vwap is not None:
        votes.append(("vwap", _clip((spot - vwap) / band, -1, 1) * 0.9))
    if gamma_flip is not None:
        votes.append(("gamma_flip", _clip((spot - gamma_flip) / band, -1, 1) * 0.6))
    if rsi is not None:
        votes.append(("rsi", _clip((rsi - 50.0) / 25.0, -1, 1) * 0.7))
    if cvd_slope is not None:
        votes.append(("cvd", _clip(cvd_slope * 4.0, -1, 1) * 0.5))
    score = sum(v for _, v in votes)
    denom = sum(abs(w) for w in (0.9, 0.6, 0.7, 0.5)) or 1.0
    score = _clip(score / denom, -1.0, 1.0)

    # Positive net GEX => dealers dampen moves (pinning): shrink the drift.
    pinning = 1.0
    if net_gex is not None:
        pinning = 0.55 if net_gex > 0 else 1.15
    drift = score * band * 0.6 * pinning
    target = spot + drift
    # Walls act as magnets/caps for a 0DTE session.
    if call_wall is not None and target > call_wall:
        target = spot + (call_wall - spot) * 0.85
    if put_wall is not None and target < put_wall:
        target = spot - (spot - put_wall) * 0.85

    # --- confidence: conviction * trend strength * positioning clarity ---
    conviction = abs(score)
    trend = _clip((adx - 12.0) / 25.0, 0.0, 1.0) if adx is not None else 0.4
    positioning = _clip(gex_rank, 0.0, 1.0)
    confidence = _clip(0.20 + 0.45 * conviction + 0.20 * trend + 0.15 * positioning, 0.15, 0.9)

    # Cone tightens with confidence; never narrower than half the expected move.
    cone = band * (1.30 - 0.55 * confidence)
    target_low = target - cone
    target_high = target + cone

    if score > 0.18:
        bias = "bullish"
    elif score < -0.18:
        bias = "bearish"
    else:
        bias = "neutral"

    key_levels = []
    if call_wall is not None:
        key_levels.append({"price": round(call_wall, 2), "label": "Call wall", "kind": "resistance"})
    if put_wall is not None:
        key_levels.append({"price": round(put_wall, 2), "label": "Put wall", "kind": "support"})
    if gamma_flip is not None:
        key_levels.append({"price": round(gamma_flip, 2), "label": "γ-flip", "kind": "pivot"})
    if vwap is not None:
        key_levels.append({"price": round(vwap, 2), "label": "VWAP", "kind": "pivot"})

    drivers = []
    if vwap is not None:
        drivers.append(f"{'above' if spot >= vwap else 'below'} VWAP")
    if gamma_flip is not None:
        drivers.append(f"{'above' if spot >= gamma_flip else 'below'} γ-flip")
    if net_gex is not None:
        drivers.append("positive GEX (pinning)" if net_gex > 0 else "negative GEX (unstable)")
    if rsi is not None:
        drivers.append(f"RSI {rsi:.0f}")
    rationale = (
        f"{bias.capitalize()} into the close: spot {spot:.2f} "
        + (", ".join(drivers) if drivers else "limited context")
        + f". Target {target:.2f} (±{cone:.2f}), confidence {confidence:.0%}."
    )

    return {
        "schema": PREDICTION_SCHEMA,
        "source": "deterministic",
        "bias": bias,
        "spot_at_pred": round(spot, 2),
        "target": round(target, 2),
        "target_low": round(target_low, 2),
        "target_high": round(target_high, 2),
        "band": round(band, 3),
        "horizon": "eod",
        "confidence": round(confidence, 3),
        "key_levels": key_levels,
        "drivers": [k for k, _ in votes],
        "rationale": rationale,
        "generated_at": now_iso,
    }


def predict_spy_der_tick(market: Any, *, now_iso: str = "",
                         decision: Any = None) -> Optional[dict]:
    """Resolve a SPY-DER price prediction, package-first then deterministic.

    The SPY-DER package may expose ``predict_shadow_tick`` returning either a
    ready dict or an object with ``as_dict()``; any failure falls back to the
    deterministic trader model so the chart always has something to draw.
    """
    try:
        from spy_der.integrations.zerodte import predict_shadow_tick  # type: ignore
    except Exception:
        predict_shadow_tick = None  # package absent or older — use fallback

    if predict_shadow_tick is not None:
        try:
            out = predict_shadow_tick(market=market, now_iso=now_iso, decision=decision)
            if out is not None:
                payload = out.as_dict() if hasattr(out, "as_dict") else dict(out)
                payload.setdefault("schema", PREDICTION_SCHEMA)
                payload.setdefault("source", "spy_der")
                return payload
        except Exception as exc:  # never let a model failure break the tick
            log.warning("spy_der predict_shadow_tick failed, using fallback: %s", exc)

    return deterministic_prediction(market, now_iso=now_iso)
