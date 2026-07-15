"""
prediction/adapters.py
======================
Typed adapters between trained V3 model contracts and the decision stack.

Never use permissive getattr() chains that silently invent 0.5 / EV-as-utility.
Candidate-value serving must use the same feature builder as training.

NOT financial advice.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

from prediction.candidate_dataset import build_candidate_feature_row
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
    Missing required fields raise AdapterError — never invent zero risk.
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

    required = (
        "p_profit", "utility_score", "expected_net_pnl", "expected_shortfall",
        "pnl_q05", "pnl_q10", "pnl_q25", "pnl_q50",
        "pnl_q75", "pnl_q90", "pnl_q95",
    )
    missing = [n for n in required if getattr(fc, n, None) is None]
    if missing:
        raise AdapterError(
            "CandidateForecast missing required fields: "
            + ",".join(missing))

    return {
        "candidate_id": str(getattr(fc, "candidate_id", "") or ""),
        "expected_net_pnl": float(fc.expected_net_pnl),
        "p_positive_pnl": float(fc.p_profit),
        "absolute_utility": float(fc.utility_score),
        "expected_shortfall": float(fc.expected_shortfall),
        "pnl_quantiles": {
            "q05": float(fc.pnl_q05),
            "q10": float(fc.pnl_q10),
            "q25": float(fc.pnl_q25),
            "q50": float(fc.pnl_q50),
            "q75": float(fc.pnl_q75),
            "q90": float(fc.pnl_q90),
            "q95": float(fc.pnl_q95),
        },
        "model_versions": {
            "candidate_value": str(
                getattr(fc, "model_version", "candidate_forecast_v3")),
        },
        "diagnostics": dict(getattr(fc, "diagnostics", {}) or {}),
    }


def candidate_feature_schema_hash(rows: Sequence[Mapping]) -> str:
    """Deterministic hash of the union of feature keys (matches training)."""
    keys = sorted({k for r in rows for k in r.keys()})
    return hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def verify_candidate_feature_schema(
    rows: Sequence[Mapping],
    *,
    expected_hash: Optional[str] = None,
    trained_feature_names: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Return trained feature names missing from **any** serving row.

    Checks each row individually — a batch-union check would pass when only
    some candidates carry a required key (silent median imputation).
    """
    if not rows:
        return list(trained_feature_names or []) or ["<empty_rows>"]
    if trained_feature_names:
        missing: set[str] = set()
        for r in rows:
            present = set(r.keys())
            for n in trained_feature_names:
                if n not in present:
                    missing.add(n)
        return sorted(missing)
    if expected_hash:
        # Hash path still uses the union (single schema fingerprint).
        got = candidate_feature_schema_hash(rows)
        if got != expected_hash:
            return [f"schema_hash:{got}!={expected_hash}"]
    return []


def candidate_value_rows(
    candidates: Sequence[Any],
    *,
    snapshot_id: str = "",
    spot: Optional[float] = None,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    net_gex: Optional[float] = None,
    bundle: Optional[dict] = None,
    data_quality: Optional[float] = None,
) -> tuple[list[dict], list[str]]:
    """
    Build feature rows + ids for CandidateValueModel.predict_v3.

    Uses ``build_candidate_feature_row`` — the same builder training uses —
    so train/serve feature names stay aligned for SpreadCandidate objects.
    Dict candidates already shaped as feature rows are passed through.
    """
    rows: list[dict] = []
    ids: list[str] = []
    for c in candidates:
        if isinstance(c, dict) and "legs" not in c and "family" in c:
            # Already a feature-ish dict (training-shaped) — keep as-is.
            d = dict(c)
            cid = str(d.get("candidate_id") or d.get("v2_candidate_id") or "")
            rows.append(d)
            ids.append(cid)
            continue
        # SpreadCandidate (or dict with legs) → canonical feature builder
        if isinstance(c, dict):
            # Minimal duck type for build_candidate_feature_row
            from types import SimpleNamespace
            legs_raw = c.get("legs") or ()
            legs = []
            for lg in legs_raw:
                if isinstance(lg, dict):
                    legs.append(SimpleNamespace(
                        strike=float(lg.get("strike") or 0),
                        kind=str(lg.get("kind") or lg.get("right") or "P"),
                        qty=int(lg.get("qty") or 1),
                    ))
                else:
                    legs.append(lg)
            cand_obj = SimpleNamespace(
                legs=legs,
                family=c.get("family"),
                credit=c.get("credit"),
                max_loss=c.get("max_loss"),
                capital=c.get("capital"),
                theta=c.get("theta"),
                gamma=c.get("gamma"),
                prob_profit=c.get("prob_profit"),
                prob_touch_short=c.get("prob_touch_short"),
                distance_to_wall=c.get("distance_to_wall"),
                liquidity_score=c.get("liquidity_score"),
                wall_safety=c.get("wall_safety"),
                gamma_safety=c.get("gamma_safety"),
                touch_safety=c.get("touch_safety"),
                score=c.get("score"),
                ev=c.get("ev"),
                ev_per_risk=c.get("ev_per_risk"),
                execution=c.get("execution"),
            )
            cid = str(c.get("candidate_id") or c.get("v2_candidate_id") or "")
        else:
            cand_obj = c
            cid = str(
                getattr(c, "candidate_id", None)
                or getattr(c, "v2_candidate_id", None)
                or "")
        spot_v = float(spot) if spot is not None else float(
            getattr(cand_obj, "spot", None) or 0.0) or 1.0
        d = build_candidate_feature_row(
            cand_obj,
            snapshot_id=snapshot_id,
            spot=spot_v,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_flip=gamma_flip,
            minutes_to_close=minutes_to_close,
            net_gex=net_gex,
            bundle=bundle,
            data_quality=data_quality,
        )
        # Preserve family string for diagnostics (not always in feature row)
        if getattr(cand_obj, "family", None) and "family" not in d:
            d["family"] = getattr(cand_obj, "family")
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
        # Preserve unknown quote age as None — never coerce to fresh (0.0).
        "quote_age_seconds": (
            None if quote_age_seconds is None else float(quote_age_seconds)),
        "minutes_to_close": (
            None if minutes_to_close is None else float(minutes_to_close)),
        "realized_volatility": realized_volatility,
        "implied_remaining_move": implied_remaining_move,
        "data_quality": data_quality,
        "replacement_count": 0.0,
        "requested_quantity": float(requested_quantity),
        "family": family,
    }
    if isinstance(candidate, dict):
        attempt.setdefault(
            "option_price_scale",
            abs(float(candidate.get("credit") or mid_credit)))
    return fill_features_from_attempt(attempt)


__all__ = [
    "AdapterError",
    "adapt_candidate_forecast_v3",
    "candidate_feature_schema_hash",
    "candidate_value_rows",
    "fill_attempt_features_from_candidate",
    "fill_features_from_attempt",
    "verify_candidate_feature_schema",
]
