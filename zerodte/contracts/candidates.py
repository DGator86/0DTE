"""Canonical candidate geometry and executable-economics contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


CANDIDATE_SCHEMA = "candidate.v1"


@dataclass(frozen=True, order=True)
class CandidateLeg:
    option_type: str
    strike: float
    quantity: int
    expiration: str

    def __post_init__(self) -> None:
        kind = self.option_type.upper()
        if kind not in {"C", "P"}:
            raise ValueError("option_type must be C or P")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero")
        if not self.expiration:
            raise ValueError("expiration is required")
        object.__setattr__(self, "option_type", kind)


@dataclass(frozen=True)
class CandidateEconomics:
    entry_price: float
    expected_fill_price: float
    fill_probability: float
    max_profit: float
    max_loss: float
    expected_value: float
    expected_utility: float
    probability_profit: float
    probability_touch: float | None = None
    cvar_95: float | None = None
    liquidity_score: float = 0.0
    data_quality: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.fill_probability <= 1.0:
            raise ValueError("fill_probability must be within [0, 1]")
        if self.max_loss <= 0:
            raise ValueError("candidate maximum loss must be finite and positive")
        if not 0.0 <= self.probability_profit <= 1.0:
            raise ValueError("probability_profit must be within [0, 1]")
        if self.probability_touch is not None and not 0.0 <= self.probability_touch <= 1.0:
            raise ValueError("probability_touch must be within [0, 1]")
        if not 0.0 <= self.liquidity_score <= 1.0:
            raise ValueError("liquidity_score must be within [0, 1]")
        if not 0.0 <= self.data_quality <= 1.0:
            raise ValueError("data_quality must be within [0, 1]")


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    snapshot_id: str
    family: str
    direction: str
    legs: tuple[CandidateLeg, ...]
    economics: CandidateEconomics
    schema_version: str = CANDIDATE_SCHEMA
    hard_vetoes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if not self.family:
            raise ValueError("family is required")
        if not self.legs:
            raise ValueError("at least one option leg is required")
        expirations = {leg.expiration for leg in self.legs}
        if len(expirations) != 1:
            raise ValueError("all candidate legs must share one expiration")
        object.__setattr__(self, "legs", tuple(self.legs))
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    @property
    def selectable(self) -> bool:
        return not self.hard_vetoes


def make_candidate_id(
    snapshot_id: str,
    family: str,
    legs: tuple[CandidateLeg, ...] | list[CandidateLeg],
) -> str:
    """Create a stable ID so every policy evaluates the same candidate object."""
    normalized = [
        {
            "option_type": leg.option_type,
            "strike": round(float(leg.strike), 6),
            "quantity": int(leg.quantity),
            "expiration": leg.expiration,
        }
        for leg in sorted(tuple(legs))
    ]
    payload = {
        "snapshot_id": snapshot_id,
        "family": family,
        "legs": normalized,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"cand_{digest[:20]}"
