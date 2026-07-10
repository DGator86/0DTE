"""
gex/contracts.py
================
Shared GEX variant output contract
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §16.2, PR 9).

Every provider (OI / weekly / volume / hybrid) emits the same GEXSnapshot.
Variants are observation-only until promotion — the live MarketSnapshot
continues to carry the OI baseline for gates and ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GexAssumption(str, Enum):
    """§16.5 sign-scenario vocabulary."""
    DEALER_SHORT_CALLS_LONG_PUTS = "dealer_short_calls_long_puts"  # baseline
    SAME_DAY_PUT_FLOW_ALT = "same_day_put_flow_alt"
    CONFIDENCE_BLEND = "confidence_blend"


class GexVariantId(str, Enum):
    OI = "oi"
    WEEKLY = "weekly"
    VOLUME = "volume"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class GEXSnapshot:
    """§16.2 — one variant at one tick."""
    net_gex: float
    gamma_flip: float
    call_wall: float
    put_wall: float
    gex_concentration: float
    wall_concentration: float
    quality_score: float
    assumption_set: GexAssumption
    source_age: Optional[float] = None
    variant: GexVariantId = GexVariantId.OI
    net_ratio: Optional[float] = None
    gex_pct_rank: Optional[float] = None
    missing_volume: bool = False
    n_contracts: int = 0
    n_expirations: int = 1
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "net_gex": self.net_gex,
            "gamma_flip": self.gamma_flip,
            "call_wall": self.call_wall,
            "put_wall": self.put_wall,
            "gex_concentration": self.gex_concentration,
            "wall_concentration": self.wall_concentration,
            "quality_score": self.quality_score,
            "assumption_set": self.assumption_set.value,
            "source_age": self.source_age,
            "variant": self.variant.value,
            "net_ratio": self.net_ratio,
            "gex_pct_rank": self.gex_pct_rank,
            "missing_volume": self.missing_volume,
            "n_contracts": self.n_contracts,
            "n_expirations": self.n_expirations,
            "notes": self.notes,
        }

    @property
    def is_finite(self) -> bool:
        import math
        return (math.isfinite(self.net_gex)
                and math.isfinite(self.gamma_flip)
                and math.isfinite(self.call_wall)
                and math.isfinite(self.put_wall))


@dataclass(frozen=True)
class GexDisagreement:
    """Structural disagreement across variants (§16.5 / §25.3)."""
    flip_spread: float
    wall_spread_call: float
    wall_spread_put: float
    net_gex_sign_disagree: bool
    net_gex_range_bn: float
    n_finite_variants: int

    def to_signals(self) -> dict:
        return {
            "gex_disagree_flip_spread": float(self.flip_spread),
            "gex_disagree_wall_call": float(self.wall_spread_call),
            "gex_disagree_wall_put": float(self.wall_spread_put),
            "gex_disagree_sign": 1.0 if self.net_gex_sign_disagree else 0.0,
            "gex_disagree_net_gex_range": float(self.net_gex_range_bn),
            "gex_disagree_n_variants": float(self.n_finite_variants),
        }


@dataclass(frozen=True)
class GexVariantBundle:
    """Parallel variant panel for one tick (observation-only)."""
    spot: float
    authoritative: GexVariantId
    oi: GEXSnapshot
    weekly: Optional[GEXSnapshot] = None
    volume: Optional[GEXSnapshot] = None
    hybrid: Optional[GEXSnapshot] = None
    disagreement: Optional[GexDisagreement] = None
    feed_source: str = ""

    def snapshots(self) -> list:
        out = [self.oi]
        for s in (self.weekly, self.volume, self.hybrid):
            if s is not None:
                out.append(s)
        return out

    def to_signals_json(self) -> dict:
        """Flatten to gex_{variant}_{field} keys for journal admission."""
        out: dict = {
            "gex_authoritative": self.authoritative.value,
        }
        for snap in self.snapshots():
            prefix = f"gex_{snap.variant.value}"
            out[f"{prefix}_net_gex"] = float(snap.net_gex) if _finite(snap.net_gex) else None
            out[f"{prefix}_gamma_flip"] = float(snap.gamma_flip) if _finite(snap.gamma_flip) else None
            out[f"{prefix}_call_wall"] = float(snap.call_wall) if _finite(snap.call_wall) else None
            out[f"{prefix}_put_wall"] = float(snap.put_wall) if _finite(snap.put_wall) else None
            out[f"{prefix}_gex_concentration"] = float(snap.gex_concentration)
            out[f"{prefix}_wall_concentration"] = float(snap.wall_concentration)
            out[f"{prefix}_quality_score"] = float(snap.quality_score)
            out[f"{prefix}_assumption"] = snap.assumption_set.value
            if snap.source_age is not None:
                out[f"{prefix}_source_age"] = float(snap.source_age)
            if snap.net_ratio is not None and _finite(snap.net_ratio):
                out[f"{prefix}_net_ratio"] = float(snap.net_ratio)
            if snap.missing_volume:
                out[f"{prefix}_missing_volume"] = 1.0
            out[f"{prefix}_n_contracts"] = float(snap.n_contracts)
            out[f"{prefix}_n_expirations"] = float(snap.n_expirations)
        # Drop Nones (signals serializer skips non-primitive; cleaner without)
        out = {k: v for k, v in out.items() if v is not None}
        if self.disagreement is not None:
            out.update(self.disagreement.to_signals())
        if self.feed_source:
            out["gex_feed_source"] = self.feed_source
        return out


def _finite(x) -> bool:
    import math
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# Fields journaled per variant (for docs / comparison reports).
GEX_SIGNAL_FIELDS = (
    "net_gex", "gamma_flip", "call_wall", "put_wall",
    "gex_concentration", "wall_concentration", "quality_score",
)
