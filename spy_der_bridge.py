"""Soft bridge from 0DTE shadow loop into SPY-DER AI decisions.

When the `spy_der` package is installed in the VPS venv, each tick asks
SPY-DER to choose among shadow candidates for the `spy_der` paper track
and parallel panel. When unavailable, the track reports `UNAVAILABLE`
and no paper fills are opened.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

log = logging.getLogger("spy_der_bridge")

PARALLEL_TRACK_ID = "spy_der"
PARALLEL_TRACK_LABEL = "SPY-DER"


@dataclass(frozen=True)
class BridgeDecision:
    action: str
    candidate_id: Optional[str]
    size_scalar: float
    structure: Optional[str]
    direction: Optional[str]
    confidence: float
    uncertainty: float
    rationale: str
    reason_codes: tuple[str, ...]
    provider: str
    model_id: str
    available: bool

    def as_parallel_payload(self) -> dict[str, Any]:
        return {
            "track": PARALLEL_TRACK_ID,
            "label": PARALLEL_TRACK_LABEL,
            "source": self.provider,
            "mode": "shadow",
            "action": self.action,
            "structure": self.structure,
            "direction": self.direction,
            "candidate_id": self.candidate_id,
            "size_cap": self.size_scalar,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "rationale": self.rationale,
            "reason_codes": list(self.reason_codes),
            "model_id": self.model_id,
            "available": self.available,
        }


def spy_der_available() -> bool:
    try:
        import spy_der.integrations.zerodte  # noqa: F401
        return True
    except Exception:
        return False


def decide_spy_der_tick(
    *,
    snapshot_id: str,
    symbol: str,
    session_date: date,
    underlying_price: float,
    shadow_candidates: list[Any],
    now: datetime,
    hard_vetoes: tuple[str, ...] = (),
    track_record: Optional[dict] = None,
) -> BridgeDecision:
    """Ask SPY-DER AI for a shadow decision; fail closed to ABSTAIN.

    ``track_record`` is the spy_der paper track's own realized history
    (journal_insights.track_feedback) — the learning feedback loop. It is
    forwarded only when the installed spy_der package supports it, so old
    package versions keep working.
    """
    if not spy_der_available():
        return BridgeDecision(
            action="UNAVAILABLE",
            candidate_id=None,
            size_scalar=0.0,
            structure=None,
            direction=None,
            confidence=0.0,
            uncertainty=1.0,
            rationale="spy_der package not installed in VPS venv",
            reason_codes=("spy_der_not_installed",),
            provider="none",
            model_id="",
            available=False,
        )
    try:
        from spy_der.integrations.zerodte import (
            ShadowCandidateView,
            decide_shadow_tick,
        )
    except Exception as exc:
        log.warning("spy_der import failed: %s", exc)
        return BridgeDecision(
            action="UNAVAILABLE",
            candidate_id=None,
            size_scalar=0.0,
            structure=None,
            direction=None,
            confidence=0.0,
            uncertainty=1.0,
            rationale=f"import_error:{type(exc).__name__}",
            reason_codes=("spy_der_import_error",),
            provider="none",
            model_id="",
            available=False,
        )

    views: list[Any] = []
    for i, cand in enumerate(shadow_candidates or []):
        try:
            cid = str(
                getattr(cand, "candidate_id", None)
                or getattr(cand, "id", None)
                or f"shadow-{i}"
            )
            family = str(getattr(cand, "family", None) or "unknown")
            direction = str(getattr(cand, "direction", None) or "both")
            max_loss = Decimal(str(getattr(cand, "max_loss", None) or 1))
            capital = Decimal(str(getattr(cand, "capital_required", None) or max_loss))
            geom = str(getattr(cand, "geometry_hash", None) or f"sha256:{cid}")
            exp = getattr(cand, "expiration", None)
            if not isinstance(exp, date):
                exp = session_date
            mid = getattr(cand, "credit", None)
            if mid is None:
                mid = getattr(cand, "mid_price", None)
            mid_d = Decimal(str(mid)) if mid is not None else None
            util = getattr(cand, "ev_per_risk", None)
            if util is None:
                util = getattr(cand, "ev", None)
            views.append(
                ShadowCandidateView(
                    candidate_id=cid,
                    family=family,
                    direction=direction,
                    maximum_loss=max_loss,
                    capital_required=capital,
                    geometry_hash=geom,
                    expiration=exp,
                    mid_price=mid_d,
                    utility=float(util) if util is not None else None,
                    v3_rank=i + 1,
                )
            )
        except Exception as exc:
            log.debug("skip shadow candidate %s: %s", i, exc)

    kwargs: dict[str, Any] = {}
    if track_record:
        import inspect
        try:
            if "track_record" in inspect.signature(decide_shadow_tick).parameters:
                kwargs["track_record"] = track_record
        except (TypeError, ValueError):
            pass

    try:
        decision = decide_shadow_tick(
            snapshot_id=snapshot_id or "snap-unknown",
            symbol=symbol or "SPY",
            session_date=session_date,
            underlying_price=Decimal(str(underlying_price)),
            candidates=views,
            now=now,
            hard_vetoes=hard_vetoes,
            **kwargs,
        )
        return BridgeDecision(
            action=decision.action,
            candidate_id=decision.candidate_id,
            size_scalar=float(decision.size_scalar),
            structure=decision.structure,
            direction=decision.direction,
            confidence=float(decision.confidence),
            uncertainty=float(decision.uncertainty),
            rationale=decision.rationale,
            reason_codes=tuple(decision.reason_codes),
            provider=decision.provider,
            model_id=decision.model_id,
            available=True,
        )
    except Exception as exc:
        log.warning("spy_der decide failed: %s", exc)
        return BridgeDecision(
            action="ABSTAIN",
            candidate_id=None,
            size_scalar=0.0,
            structure=None,
            direction=None,
            confidence=0.0,
            uncertainty=1.0,
            rationale=f"bridge_error:{type(exc).__name__}:{exc}",
            reason_codes=("spy_der_bridge_error",),
            provider="spy_der",
            model_id="",
            available=True,
        )
