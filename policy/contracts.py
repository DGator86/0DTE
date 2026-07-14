"""
policy/contracts.py
===================
Unified policy I/O contracts
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17.2–17.3, PR 10).

Policy consumes a PredictionBundle; it must never write back into the
forecast. Structural and operational hard vetoes remain separate inputs
so they cannot be silently absorbed into model probabilities.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from prediction.contracts import PredictionBundle


class PolicyMode(str, Enum):
    """Promotion pointer — change this, not call sites (§17 / PR 10)."""
    LEGACY = "legacy"
    SHADOW = "shadow"
    CHAMPION = "champion"


# Provenance tags for PolicyDecision.source (§17.5).
SOURCE_LEGACY = "legacy"
SOURCE_V2 = "v2"
SOURCE_FALLBACK_LEGACY = "fallback_legacy"


@dataclass(frozen=True)
class StructuralState:
    """
    Legacy simplified dealer / wall geometry for policy I/O.

    V3 Part 2 expands structure in `prediction.structural_state.StructuralState`.
    Conversion from V3 → this legacy view must be explicit via
    `prediction.structural_state.StructuralState.to_legacy_policy_state()`.
    Live gates continue to read MarketSnapshot OI fields unchanged.
    """
    spot: float = 0.0
    net_gex: float = 0.0
    gamma_flip: float = 0.0
    call_wall: float = 0.0
    put_wall: float = 0.0
    gex_pct_rank: float = 0.5
    notes: str = ""

    @classmethod
    def from_market(cls, market: object) -> "StructuralState":
        """Lift from gate_scorer.MarketSnapshot (duck-typed)."""
        return cls(
            spot=float(getattr(market, "spot", 0.0) or 0.0),
            net_gex=float(getattr(market, "net_gex", 0.0) or 0.0),
            gamma_flip=float(getattr(market, "gamma_flip", 0.0) or 0.0),
            call_wall=float(getattr(market, "call_wall", 0.0) or 0.0),
            put_wall=float(getattr(market, "put_wall", 0.0) or 0.0),
            gex_pct_rank=float(getattr(market, "gex_pct_rank", 0.5) or 0.5),
        )

    @classmethod
    def from_v3_structural(cls, v3_state: object) -> "StructuralState":
        """Explicit V3 → legacy conversion (docs Part 2 §8)."""
        to_legacy = getattr(v3_state, "to_legacy_policy_state", None)
        if callable(to_legacy):
            return to_legacy()
        raise TypeError(
            "from_v3_structural requires prediction.structural_state."
            "StructuralState (got %r)" % (type(v3_state).__name__,))

    def to_dict(self) -> dict:
        return {
            "spot": self.spot,
            "net_gex": self.net_gex,
            "gamma_flip": self.gamma_flip,
            "call_wall": self.call_wall,
            "put_wall": self.put_wall,
            "gex_pct_rank": self.gex_pct_rank,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PolicyInput:
    """§17.2 — everything policy may read; nothing it may write back."""
    predictions: Optional[PredictionBundle]
    structural_state: StructuralState
    operational_risk_state: dict = field(default_factory=dict)
    legacy_regime_state: Optional[object] = None
    legacy_matrix_intent: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "predictions": (self.predictions.to_dict()
                            if self.predictions is not None else None),
            "structural_state": self.structural_state.to_dict(),
            "operational_risk_state": dict(self.operational_risk_state),
            "legacy_regime_state": None,   # opaque; not serialized
            "legacy_matrix_intent": None,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """§17.3 — unified policy output + explicit provenance (§17.5)."""
    action: str                              # TRADE | NO_TRADE
    direction: str                           # call | put | both | none
    eligible_families: tuple[str, ...]
    confidence: float
    uncertainty: float
    size_cap: float
    hard_vetoes: tuple[str, ...]
    rationale: tuple[str, ...]
    policy_version: str
    # Provenance — required by §17.5; not in the sketch but acceptance-
    # critical so silent legacy substitution is impossible.
    source: str = SOURCE_LEGACY
    # Routing convenience for the live selector (maps to STRUCTURE_TO_FAMILIES
    # keys). Empty when action is NO_TRADE.
    structure_code: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "direction": self.direction,
            "eligible_families": list(self.eligible_families),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "size_cap": self.size_cap,
            "hard_vetoes": list(self.hard_vetoes),
            "rationale": list(self.rationale),
            "policy_version": self.policy_version,
            "source": self.source,
            "structure_code": self.structure_code,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyDecision":
        return cls(
            action=str(d["action"]),
            direction=str(d.get("direction", "none")),
            eligible_families=tuple(d.get("eligible_families") or ()),
            confidence=float(d.get("confidence", 0.0)),
            uncertainty=float(d.get("uncertainty", 1.0)),
            size_cap=float(d.get("size_cap", 0.0)),
            hard_vetoes=tuple(d.get("hard_vetoes") or ()),
            rationale=tuple(d.get("rationale") or ()),
            policy_version=str(d.get("policy_version", "")),
            source=str(d.get("source", SOURCE_LEGACY)),
            structure_code=str(d.get("structure_code", "")),
        )
