"""
execution/fill_records.py
=========================
Fill-attempt recording for empirical execution models
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §11–§12).

Provenance rules:
  * Simulated fills must not be labeled broker_actual.
  * Midpoint diagnostics must not be labeled filled.
  * Advisory candidates never submitted: source=hypothetical, filled=False.
  * Unfilled / cancelled / rejected attempts remain stored evidence.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

FILL_RECORD_VERSION = "v3.0.0"

ALLOWED_SOURCES = frozenset({
    "paper",
    "manual_paper",
    "broker_actual",
    "user_confirmed",
    "hypothetical",
    "rejected",
    "cancelled",
    "expired_unfilled",
    "advisory",
})

ALLOWED_MODES = frozenset({
    "research", "shadow", "advisory", "candidate", "champion", "paper",
})


@dataclass(frozen=True)
class FillRecord:
    fill_record_id: str
    snapshot_id: str
    candidate_id: str
    session_date: str
    decision_ts: str
    submitted_ts: str
    resolved_ts: Optional[str]
    symbol: str
    family: str
    side: str
    n_legs: int
    limit_credit: float
    mid_credit_at_submit: float
    natural_credit_at_submit: float
    relative_spread: float
    absolute_spread: float
    option_price_scale: float
    quote_age_seconds: float
    minutes_to_close: float
    realized_volatility: Optional[float] = None
    implied_remaining_move: Optional[float] = None
    dominant_regime: Optional[str] = None
    data_quality: Optional[float] = None
    replacement_count: int = 0
    replacement_prices: tuple = ()
    filled: bool = False
    partial_fill: bool = False
    filled_quantity: int = 0
    requested_quantity: int = 0
    seconds_to_first_fill: Optional[float] = None
    seconds_to_complete_fill: Optional[float] = None
    fill_credit: Optional[float] = None
    fill_fraction: Optional[float] = None
    fill_fraction_raw: Optional[float] = None
    fill_fraction_clipped: Optional[float] = None
    cancelled: bool = False
    expired_unfilled: bool = False
    rejected: bool = False
    source: str = "paper"
    mode: str = "shadow"
    version: str = FILL_RECORD_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["replacement_prices"] = list(self.replacement_prices)
        d["diagnostics"] = dict(self.diagnostics)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FillRecord":
        payload = dict(d)
        rp = payload.get("replacement_prices") or ()
        payload["replacement_prices"] = tuple(rp)
        payload.setdefault("diagnostics", {})
        # Drop unknown keys for forward compatibility
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(**payload)


def fill_fraction(
    mid_credit: float,
    natural_credit: float,
    fill_credit: float,
    *,
    epsilon: float = 1e-9,
    side: str = "credit",
) -> tuple:
    """
    Concession from midpoint toward natural.

    Returns (raw, clipped) where clipped is in [0, 1] for modeling and
    raw retains anomalies (better-than-mid or worse-than-natural).

    Credit: 0 = mid, 1 = natural.
    Debit: same interpretation via adverse interpolation.
    """
    mid = float(mid_credit)
    nat = float(natural_credit)
    fill = float(fill_credit)
    # Signed credit convention: works for credit and debit structures.
    # 0 = midpoint, 1 = full natural concession.
    denom = mid - nat
    if abs(denom) < epsilon:
        raw = 0.0
    else:
        raw = (mid - fill) / denom
    clipped = float(min(max(raw, 0.0), 1.0))
    return float(raw), clipped


def validate_fill_record(rec: FillRecord) -> None:
    """Raise ValueError on provenance / timestamp / identity violations."""
    if not rec.fill_record_id:
        raise ValueError("fill_record_id required")
    if not rec.snapshot_id or not rec.candidate_id:
        raise ValueError("snapshot_id and candidate_id required")
    if rec.source not in ALLOWED_SOURCES:
        raise ValueError(f"unknown fill source: {rec.source}")
    if rec.mode not in ALLOWED_MODES:
        raise ValueError(f"unknown fill mode: {rec.mode}")
    # Provenance: simulated must not claim broker_actual
    if rec.source == "broker_actual" and rec.diagnostics.get("simulated"):
        raise ValueError("simulated fill cannot be source=broker_actual")
    # Midpoint diagnostic must not be labeled filled
    if rec.diagnostics.get("midpoint_diagnostic") and rec.filled:
        raise ValueError("midpoint diagnostic cannot be labeled filled")
    # Hypothetical / advisory never submitted
    if rec.source in ("hypothetical", "advisory"):
        if rec.filled:
            raise ValueError(
                f"source={rec.source} must have filled=False")
    # Timestamps: decision <= submitted; resolved >= submitted when present
    if rec.decision_ts and rec.submitted_ts and rec.decision_ts > rec.submitted_ts:
        raise ValueError("decision_ts must be <= submitted_ts")
    if rec.resolved_ts is not None and rec.submitted_ts:
        if rec.resolved_ts < rec.submitted_ts:
            raise ValueError("resolved_ts must be >= submitted_ts")
    if rec.filled and rec.fill_credit is None:
        raise ValueError("filled records require fill_credit")
    if rec.partial_fill and rec.filled_quantity <= 0:
        raise ValueError("partial_fill requires filled_quantity > 0")


def enrich_fill_fractions(rec: FillRecord) -> FillRecord:
    """Attach raw/clipped fill fractions when a fill_credit is present."""
    if rec.fill_credit is None:
        return rec
    raw, clipped = fill_fraction(
        rec.mid_credit_at_submit,
        rec.natural_credit_at_submit,
        rec.fill_credit,
        side=rec.side,
    )
    d = rec.to_dict()
    d["fill_fraction_raw"] = raw
    d["fill_fraction_clipped"] = clipped
    d["fill_fraction"] = clipped
    return FillRecord.from_dict(d)
