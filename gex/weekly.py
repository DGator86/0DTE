"""
gex/weekly.py
=============
Variant B — OI with nearest weekly expirations + DTE decay
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §16.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from gex.base import WeightedContract, empty_snapshot, snapshot_from_contracts
from gex.contracts import GEXSnapshot, GexAssumption, GexVariantId


@dataclass
class WeeklyGexConfig:
    assumption: GexAssumption = GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS
    max_weeklies: int = 2
    decay: Literal["sqrt_dte", "linear_dte", "none"] = "sqrt_dte"
    min_oi: int = 1


def decay_weight(dte_days: float, mode: str) -> float:
    """Time-decay weight for a contract's OI contribution."""
    dte = max(float(dte_days), 0.0)
    if mode == "none":
        return 1.0
    if mode == "linear_dte":
        # 0DTE = 1.0, 7 DTE ≈ 0.5, 14 DTE ≈ 0.0 floor
        return max(0.05, 1.0 - dte / 14.0)
    # sqrt_dte (default): slower decay — 0DTE=1, 1DTE≈0.71, 4DTE=0.5
    return 1.0 / math.sqrt(1.0 + dte)


class WeeklyGexProvider:
    variant = GexVariantId.WEEKLY

    def __init__(self, cfg: Optional[WeeklyGexConfig] = None):
        self.cfg = cfg or WeeklyGexConfig()

    def compute(
        self,
        *,
        spot: float,
        rows: Sequence,
        source_age: Optional[float] = None,
        **_,
    ) -> GEXSnapshot:
        """
        `rows` may mix expirations. Each row may carry `dte_days` (float).
        When all rows are 0DTE (no dte_days / all 0), this reduces to OI
        with a note that weeklies were unavailable.
        """
        # Cap to nearest weeklies by unique dte buckets
        dtes = sorted({round(float(getattr(r, "dte_days", 0.0) or 0.0), 4)
                       for r in rows})
        # Keep 0DTE + up to max_weeklies positive DTEs
        positive = [d for d in dtes if d > 0]
        keep = {0.0} | set(positive[:self.cfg.max_weeklies])

        contracts = []
        expirations = set()
        for r in rows:
            dte = float(getattr(r, "dte_days", 0.0) or 0.0)
            if round(dte, 4) not in keep and dte not in keep:
                # allow exact 0
                if dte > 0 and dte not in positive[:self.cfg.max_weeklies]:
                    continue
            oi = int(getattr(r, "oi", 0) or 0)
            side = getattr(r, "side", "")
            if oi < self.cfg.min_oi or side not in ("call", "put"):
                continue
            w = float(oi) * decay_weight(dte, self.cfg.decay)
            contracts.append(WeightedContract(
                side=side,
                strike=float(getattr(r, "strike", 0.0)),
                gamma=float(getattr(r, "gamma", 0.0) or 0.0),
                weight=w,
                dte_days=dte,
            ))
            expirations.add(round(dte, 4))

        if not contracts:
            return empty_snapshot(
                spot, variant=self.variant, assumption=self.cfg.assumption,
                notes="no weekly-weighted contracts")

        n_exp = max(len(expirations), 1)
        # Quality scales with how many expirations we actually used
        requested = 1 + self.cfg.max_weeklies
        quality = min(1.0, n_exp / requested)
        if n_exp == 1 and all(
                float(getattr(r, "dte_days", 0.0) or 0.0) == 0 for r in rows):
            notes = "weekly_unavailable_0dte_only"
            quality *= 0.7
        else:
            notes = f"weekly_n_exp={n_exp}_decay={self.cfg.decay}"

        return snapshot_from_contracts(
            contracts, spot,
            variant=self.variant,
            assumption=self.cfg.assumption,
            quality_score=quality,
            source_age=source_age,
            n_expirations=n_exp,
            notes=notes,
        )
