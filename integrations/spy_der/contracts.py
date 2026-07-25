"""Versioned 0DTE ↔ SPY-DER packet schemas (dict-level, no spy_der imports).

Schemas mirror SPY-DER ``spy_der.contracts.integration`` (PR #41 / #43):
  zerodte.spyder.market.v1
  zerodte.spyder.outcome.v1
  spyder.dashboard.v1
  spyder.decision.request.v1
  spyder.decision.response.v1
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

MARKET_PACKET_SCHEMA = "zerodte.spyder.market.v1"
OUTCOME_PACKET_SCHEMA = "zerodte.spyder.outcome.v1"
DASHBOARD_SCHEMA = "spyder.dashboard.v1"
DECISION_REQUEST_SCHEMA = "spyder.decision.request.v1"
DECISION_RESPONSE_SCHEMA = "spyder.decision.response.v1"

PARALLEL_TRACK_ID = "spy_der"
PARALLEL_TRACK_LABEL = "SPY-DER"

DEFAULT_STATE_ROOT = "/var/lib/spy-der"
DEFAULT_LIVE_STATE = "/var/lib/spy-der/live_state.json"
DEFAULT_DOJO_LATEST = "/var/lib/spy-der/reports/dojo/latest.json"
DEFAULT_EXPERIENCE_ROOT = "/var/lib/zerodte/spyder_experience"


def _dec(value: Any) -> str:
    return str(Decimal(str(value)))


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _iso_dt(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def candidate_view_to_dict(
    *,
    candidate_id: str,
    family: str = "unknown",
    direction: str = "both",
    maximum_loss: Any = 1,
    capital_required: Any = None,
    geometry_hash: str = "",
    expiration: Any = None,
    mid_price: Any = None,
    fill_probability: float = 1.0,
    utility: Optional[float] = None,
    v3_rank: Optional[int] = None,
    hard_vetoed: bool = False,
    session_date: Any = None,
) -> dict[str, Any]:
    cid = str(candidate_id)
    exp = expiration if expiration is not None else session_date
    if exp is None:
        exp = date.today()
    return {
        "candidate_id": cid,
        "family": str(family or "unknown"),
        "direction": str(direction or "both"),
        "maximum_loss": _dec(maximum_loss if maximum_loss is not None else 1),
        "capital_required": _dec(
            capital_required if capital_required is not None else maximum_loss or 1
        ),
        "geometry_hash": str(geometry_hash or f"sha256:{cid}"),
        "expiration": _iso_date(exp),
        "mid_price": _dec(mid_price) if mid_price is not None else None,
        "fill_probability": float(fill_probability),
        "utility": float(utility) if utility is not None else None,
        "v3_rank": int(v3_rank) if v3_rank is not None else None,
        "hard_vetoed": bool(hard_vetoed),
    }


def build_market_packet(
    *,
    snapshot_id: str,
    session_date: Any,
    symbol: str = "SPY",
    underlying_price: Any,
    data_quality: float = 1.0,
    forecast_uncertainty: float = 0.0,
    hard_vetoes: tuple[str, ...] | list[str] = (),
    forecast: Optional[dict[str, Any]] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
    track_record: Optional[dict[str, Any]] = None,
    generated_at: Any = None,
    risk_max_size_scalar: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": MARKET_PACKET_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "session_date": _iso_date(session_date),
        "symbol": str(symbol or "SPY"),
        "underlying_price": _dec(underlying_price),
        "data_quality": float(data_quality),
        "forecast_uncertainty": float(forecast_uncertainty),
        "hard_vetoes": [str(v) for v in (hard_vetoes or ())],
        "forecast": dict(forecast or {}),
        "candidates": list(candidates or []),
        "track_record": dict(track_record or {}),
        "generated_at": _iso_dt(generated_at),
        "risk_max_size_scalar": float(risk_max_size_scalar),
    }


def build_outcome_packet(
    *,
    snapshot_id: str,
    session_date: Any,
    symbol: str = "SPY",
    candidate_id: Optional[str],
    action: str,
    realized_pnl: Any = None,
    settled: bool = False,
    labels: Optional[dict[str, Any]] = None,
    settled_at: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": OUTCOME_PACKET_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "session_date": _iso_date(session_date),
        "symbol": str(symbol or "SPY"),
        "candidate_id": str(candidate_id) if candidate_id is not None else None,
        "action": str(action or "UNKNOWN"),
        "realized_pnl": _dec(realized_pnl) if realized_pnl is not None else None,
        "settled": bool(settled),
        "labels": dict(labels or {}),
        "settled_at": _iso_dt(settled_at),
    }


def build_decision_request(
    market: dict[str, Any], *, request_id: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": DECISION_REQUEST_SCHEMA,
        "request_id": str(request_id or ""),
        "market": market,
    }


def validate_market_packet(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("market packet must be an object")
    if data.get("schema_version") != MARKET_PACKET_SCHEMA:
        raise ValueError(
            f"expected {MARKET_PACKET_SCHEMA}, got {data.get('schema_version')!r}"
        )
    if not data.get("snapshot_id"):
        raise ValueError("snapshot_id required")
    if "session_date" not in data:
        raise ValueError("session_date required")
    if "underlying_price" not in data:
        raise ValueError("underlying_price required")
    cands = data.get("candidates")
    if cands is not None and not isinstance(cands, list):
        raise ValueError("candidates must be a list")
    return data


def validate_outcome_packet(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("outcome packet must be an object")
    if data.get("schema_version") != OUTCOME_PACKET_SCHEMA:
        raise ValueError(
            f"expected {OUTCOME_PACKET_SCHEMA}, got {data.get('schema_version')!r}"
        )
    if not data.get("snapshot_id"):
        raise ValueError("snapshot_id required")
    return data


def validate_dashboard_packet(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("dashboard packet must be an object")
    if data.get("schema_version") != DASHBOARD_SCHEMA:
        raise ValueError(
            f"expected {DASHBOARD_SCHEMA}, got {data.get('schema_version')!r}"
        )
    if not data.get("generated_at"):
        raise ValueError("generated_at required")
    return data


def dashboard_to_parallel_payload(decision: dict[str, Any]) -> dict[str, Any]:
    """Map spyder.dashboard.v1 fields onto the existing parallel-panel shape."""
    return {
        "track": PARALLEL_TRACK_ID,
        "label": PARALLEL_TRACK_LABEL,
        "source": str(decision.get("provider") or "spy_der"),
        "mode": str(decision.get("mode") or "shadow"),
        "action": str(decision.get("action") or "ABSTAIN"),
        "structure": decision.get("structure"),
        "direction": decision.get("direction"),
        "candidate_id": decision.get("candidate_id"),
        "size_cap": float(decision.get("size_scalar") or 0.0),
        "confidence": float(decision.get("confidence") or 0.0),
        "uncertainty": float(decision.get("uncertainty") or 1.0),
        "rationale": str(decision.get("rationale") or ""),
        "reason_codes": list(decision.get("reason_codes") or []),
        "model_id": str(
            decision.get("trader_model")
            or decision.get("model_id")
            or ""
        ),
        "available": bool(decision.get("available", True)),
        "snapshot_id": str(decision.get("snapshot_id") or ""),
        "symbol": str(decision.get("symbol") or "SPY"),
    }


def shadow_candidate_to_view(cand: Any, *, index: int, session_date: Any) -> dict[str, Any]:
    """Convert a legacy shadow candidate object into MarketCandidateView dict."""
    cid = str(
        getattr(cand, "candidate_id", None)
        or getattr(cand, "id", None)
        or f"shadow-{index}"
    )
    mid = getattr(cand, "credit", None)
    if mid is None:
        mid = getattr(cand, "mid_price", None)
    util = getattr(cand, "ev_per_risk", None)
    if util is None:
        util = getattr(cand, "ev", None)
    max_loss = getattr(cand, "max_loss", None) or getattr(cand, "maximum_loss", None) or 1
    return candidate_view_to_dict(
        candidate_id=cid,
        family=str(getattr(cand, "family", None) or "unknown"),
        direction=str(getattr(cand, "direction", None) or "both"),
        maximum_loss=max_loss,
        capital_required=getattr(cand, "capital_required", None),
        geometry_hash=str(getattr(cand, "geometry_hash", None) or ""),
        expiration=getattr(cand, "expiration", None),
        mid_price=mid,
        utility=float(util) if util is not None else None,
        v3_rank=index + 1,
        session_date=session_date,
    )
