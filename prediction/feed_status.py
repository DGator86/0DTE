"""
prediction/feed_status.py
=========================
Per-source feed freshness contracts (unified integration handoff §6.1 / PR B).

Overall LIVE is allowed only when every *required* source is valid and fresh.
The dashboard must not infer this from truthiness — it reads documented
fields only (serialized in a later PR).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence

FeedStatusCode = Literal[
    "LIVE",
    "DELAYED",
    "STALE",
    "MISSING",
    "INVALID",
    "FALLBACK",
]

OverallFeedStatus = Literal[
    "LIVE",
    "DEGRADED",
    "STALE",
    "MISSING",
    "INVALID",
]

# Default required sources for a production tick (handoff §6.1).
DEFAULT_REQUIRED_SOURCES: tuple[str, ...] = (
    "spot",
    "bars",
    "option_chain",
    "settlement",
)

# Delayed-but-usable band: within freshness limit but past a soft delay fraction.
_DEFAULT_DELAY_FRACTION = 0.5


@dataclass(frozen=True)
class FeedStatus:
    source: str
    provider: Optional[str]
    observed_at: Optional[str]
    received_at: Optional[str]
    age_seconds: Optional[float]
    freshness_limit_seconds: float
    status: FeedStatusCode
    required: bool
    error_code: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "provider": self.provider,
            "observed_at": self.observed_at,
            "received_at": self.received_at,
            "age_seconds": self.age_seconds,
            "freshness_limit_seconds": self.freshness_limit_seconds,
            "status": self.status,
            "required": self.required,
            "error_code": self.error_code,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "FeedStatus":
        return cls(
            source=str(d["source"]),
            provider=(None if d.get("provider") is None else str(d["provider"])),
            observed_at=(None if d.get("observed_at") is None
                         else str(d["observed_at"])),
            received_at=(None if d.get("received_at") is None
                         else str(d["received_at"])),
            age_seconds=(None if d.get("age_seconds") is None
                         else float(d["age_seconds"])),
            freshness_limit_seconds=float(d["freshness_limit_seconds"]),
            status=str(d["status"]),  # type: ignore[arg-type]
            required=bool(d.get("required", True)),
            error_code=(None if d.get("error_code") is None
                        else str(d["error_code"])),
            diagnostics=dict(d.get("diagnostics") or {}),
        )


def classify_feed_status(
    *,
    age_seconds: Optional[float],
    freshness_limit_seconds: float,
    present: bool = True,
    valid: bool = True,
    is_fallback: bool = False,
    delay_fraction: float = _DEFAULT_DELAY_FRACTION,
) -> FeedStatusCode:
    """
    Derive a status code from age / presence / validity.

    Missing values stay missing — absence is MISSING, not age=0 LIVE.
    """
    if not present:
        return "MISSING"
    if not valid:
        return "INVALID"
    if is_fallback:
        return "FALLBACK"
    if age_seconds is None:
        # Present but age unknown — not LIVE; treat as DELAYED for safety.
        return "DELAYED"
    limit = float(freshness_limit_seconds)
    age = float(age_seconds)
    if age < 0:
        return "INVALID"
    if age > limit:
        return "STALE"
    soft = max(0.0, min(1.0, float(delay_fraction))) * limit
    if age > soft:
        return "DELAYED"
    return "LIVE"


def build_feed_status(
    *,
    source: str,
    freshness_limit_seconds: float,
    provider: Optional[str] = None,
    observed_at: Optional[str] = None,
    received_at: Optional[str] = None,
    age_seconds: Optional[float] = None,
    required: bool = True,
    present: bool = True,
    valid: bool = True,
    is_fallback: bool = False,
    error_code: Optional[str] = None,
    diagnostics: Optional[dict] = None,
    delay_fraction: float = _DEFAULT_DELAY_FRACTION,
    status: Optional[FeedStatusCode] = None,
) -> FeedStatus:
    """Construct a FeedStatus, classifying unless an explicit status is given."""
    code = status or classify_feed_status(
        age_seconds=age_seconds,
        freshness_limit_seconds=freshness_limit_seconds,
        present=present,
        valid=valid,
        is_fallback=is_fallback,
        delay_fraction=delay_fraction,
    )
    diag = dict(diagnostics or {})
    if age_seconds is None and present and code == "DELAYED":
        diag.setdefault("note", "age_unknown")
    return FeedStatus(
        source=str(source),
        provider=provider,
        observed_at=observed_at,
        received_at=received_at,
        age_seconds=(None if age_seconds is None else float(age_seconds)),
        freshness_limit_seconds=float(freshness_limit_seconds),
        status=code,
        required=bool(required),
        error_code=error_code,
        diagnostics=diag,
    )


def overall_feed_status(
    statuses: Mapping[str, FeedStatus] | Sequence[FeedStatus],
) -> OverallFeedStatus:
    """
    Aggregate required sources into an overall status.

    LIVE: every required source is LIVE.
    DEGRADED: required sources are usable (DELAYED or FALLBACK) but not all LIVE.
    STALE / MISSING / INVALID: at least one required source has that status
    (INVALID > MISSING > STALE precedence when multiple apply).
    """
    if isinstance(statuses, Mapping):
        items = list(statuses.values())
    else:
        items = list(statuses)
    required = [s for s in items if s.required]
    if not required:
        return "MISSING"

    codes = {s.status for s in required}
    if "INVALID" in codes:
        return "INVALID"
    if "MISSING" in codes:
        return "MISSING"
    if "STALE" in codes:
        return "STALE"
    if codes <= {"LIVE"}:
        return "LIVE"
    # DELAYED / FALLBACK (and LIVE mix) → degraded but usable
    if codes <= {"LIVE", "DELAYED", "FALLBACK"}:
        return "DEGRADED"
    return "DEGRADED"
