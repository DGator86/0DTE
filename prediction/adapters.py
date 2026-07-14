"""
prediction/adapters.py
======================
Typed adapters between trained V3 model contracts and the decision stack.

Never use permissive getattr() chains that silently invent 0.5 / EV-as-utility.

NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from prediction.models.candidate_value import CandidateForecastV3
from prediction.models.fill_probability import (
    fill_features_from_attempt,
)


class AdapterError(RuntimeError):
    """Trained model output could not be adapted to the decision contract."""


def adapt_candidate_forecast_v3(forecast: Any) -> dict:
    """
    Map CandidateForecastV3 (or compatible) onto CandidateEvaluation fields.

    Required: p_profit, utility_score, expected_net_pnl, quantile ladder, ES.
    """
    if isinstance(forecast, CandidateForecastV3):
        fc = forecast
    elif hasattr(forecast, "p_profit") and hasattr(forecast, "utility_score"):
        # Duck-typed but require the real field names — no p_positive_pnl alias.
        fc = forecast
    else:
        raise AdapterError(
            f"expected CandidateForecastV3-like object with p_profit/"
            f"utility_score, got {type(forecast).__name__}")

    p_profit = getattr(fc, "p_profit", None)
    util = getattr(fc, "utility_score", None)
    if p_profit is None or util is None:
        raise AdapterError(
            "CandidateForecast missing required p_profit / utility_score")

    return {
        "candidate_id": str(getattr(fc, "candidate_id", "") or ""),
        "expected_net_pnl": float(getattr(fc, "expected_net_pnl", 0.0) or 0.0),
        "p_positive_pnl": float(p_profit),
        "absolute_utility": float(util),
        "expected_shortfall": float(
            getattr(fc, "expected_shortfall", 0.0) or 0.0),
        "pnl_quantiles": {
            "q05": float(getattr(fc, "pnl_q05", 0.0) or 0.0),
            "q10": float(getattr(fc, "pnl_q10", 0.0) or 0.0),
            "q25": float(getattr(fc, "pnl_q25", 0.0) or 0.0),
            "q50": float(getattr(fc, "pnl_q50", 0.0) or 0.0),
            "q75": float(getattr(fc, "pnl_q75", 0.0) or 0.0),
            "q90": float(getattr(fc, "pnl_q90", 0.0) or 0.0),
            "q95": float(getattr(fc, "pnl_q95", 0.0) or 0.0),
        },
        "model_versions": {
            "candidate_value": str(
                getattr(fc, "model_version", "candidate_forecast_v3")),
        },
        "diagnostics": dict(getattr(fc, "diagnostics", {}) or {}),
    }


def candidate_value_rows(
    candidates: Sequence[Any],
) -> tuple[list[dict], list[str]]:
    """Build feature rows + ids for CandidateValueModel.predict_v3."""
    rows: list[dict] = []
    ids: list[str] = []
    for c in candidates:
        if isinstance(c, dict):
            d = dict(c)
            cid = str(d.get("candidate_id") or d.get("v2_candidate_id") or "")
        else:
            d = {
                "family": getattr(c, "family", None),
                "ev": getattr(c, "ev", None),
                "credit": getattr(c, "credit", None),
                "max_loss": getattr(c, "max_loss", None),
                "prob_profit": getattr(c, "prob_profit", None),
                "capital": getattr(c, "capital", None),
                "score": getattr(c, "score", None),
            }
            cid = str(
                getattr(c, "candidate_id", None)
                or getattr(c, "v2_candidate_id", None)
                or "")
        rows.append(d)
        ids.append(cid)
    return rows, ids


def fill_attempt_features_from_candidate(
    *,
    candidate: Any,
    mid_credit: float,
    natural_credit: float,
    family: str,
    n_legs: int,
    quote_age_seconds: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    realized_volatility: Optional[float] = None,
    implied_remaining_move: Optional[float] = None,
    data_quality: Optional[float] = None,
    requested_quantity: float = 1.0,
) -> dict:
    """
    Build a fill-attempt feature row matching training `_features_from_record`.

    Uses the shared fill_features_from_attempt builder so train/serve names
    and units stay identical.
    """
    abs_spread = abs(float(mid_credit) - float(natural_credit))
    rel = abs_spread / max(abs(float(mid_credit)), 1e-9)
    is_credit = float(mid_credit) >= 0.0
    attempt = {
        "n_legs": n_legs,
        "side": "credit" if is_credit else "debit",
        "mid_credit_at_submit": float(mid_credit),
        "natural_credit_at_submit": float(natural_credit),
        "limit_credit": float(natural_credit),  # natural = executable limit
        "relative_spread": float(rel),
        "absolute_spread": float(abs_spread),
        "option_price_scale": abs(float(mid_credit)),
        "quote_age_seconds": float(quote_age_seconds or 0.0),
        "minutes_to_close": float(minutes_to_close or 0.0),
        "realized_volatility": realized_volatility,
        "implied_remaining_move": implied_remaining_move,
        "data_quality": data_quality,
        "replacement_count": 0.0,
        "requested_quantity": float(requested_quantity),
        "family": family,
    }
    # Attach optional candidate diagnostics without breaking schema
    if isinstance(candidate, dict):
        attempt.setdefault(
            "option_price_scale",
            abs(float(candidate.get("credit") or mid_credit)))
    return fill_features_from_attempt(attempt)


__all__ = [
    "AdapterError",
    "adapt_candidate_forecast_v3",
    "candidate_value_rows",
    "fill_attempt_features_from_candidate",
    "fill_features_from_attempt",
]
