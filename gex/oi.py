"""
gex/oi.py
=========
Variant A — OI-only 0DTE GEX (current baseline)
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §16.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from gex.base import WeightedContract, empty_snapshot, snapshot_from_contracts
from gex.contracts import GEXSnapshot, GexAssumption, GexVariantId


@dataclass
class OiGexConfig:
    assumption: GexAssumption = GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS
    min_oi: int = 1


class OiGexProvider:
    """Open-interest weighted GEX — must match spy0dte.build_gamma_map."""
    variant = GexVariantId.OI

    def __init__(self, cfg: Optional[OiGexConfig] = None):
        self.cfg = cfg or OiGexConfig()

    def compute(
        self,
        *,
        spot: float,
        rows: Sequence,
        source_age: Optional[float] = None,
        **_,
    ) -> GEXSnapshot:
        contracts = []
        n_valid = 0
        for r in rows:
            oi = int(getattr(r, "oi", 0) or 0)
            gamma = float(getattr(r, "gamma", 0.0) or 0.0)
            side = getattr(r, "side", "")
            strike = float(getattr(r, "strike", 0.0))
            if oi < self.cfg.min_oi or side not in ("call", "put"):
                continue
            n_valid += 1
            contracts.append(WeightedContract(
                side=side, strike=strike, gamma=gamma, weight=float(oi)))
        if not contracts:
            return empty_snapshot(
                spot, variant=self.variant, assumption=self.cfg.assumption,
                notes="no oi-weighted contracts")
        # Quality: fraction of input rows that contributed
        n_in = max(len(rows), 1)
        quality = min(1.0, n_valid / n_in) * (1.0 if n_valid >= 4 else 0.5)
        return snapshot_from_contracts(
            contracts, spot,
            variant=self.variant,
            assumption=self.cfg.assumption,
            quality_score=quality,
            source_age=source_age,
            n_expirations=1,
            notes="oi_0dte",
        )
