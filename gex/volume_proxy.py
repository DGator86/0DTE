"""
gex/volume_proxy.py
===================
Variant C — intraday volume-weighted gamma proxy
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §16.1 / §16.5).

Uses contract volume instead of OI. When volume is missing/zero, returns a
zero-quality empty snapshot — NEVER falls back to OI (PR 9 acceptance:
missing volume must not contaminate OI calculations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from gex.base import WeightedContract, empty_snapshot, snapshot_from_contracts
from gex.contracts import GEXSnapshot, GexAssumption, GexVariantId


@dataclass
class VolumeProxyConfig:
    assumption: GexAssumption = GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS
    alt_assumption: GexAssumption = GexAssumption.SAME_DAY_PUT_FLOW_ALT
    min_volume: int = 1
    compute_sign_scenarios: bool = True


class VolumeProxyProvider:
    variant = GexVariantId.VOLUME

    def __init__(self, cfg: Optional[VolumeProxyConfig] = None):
        self.cfg = cfg or VolumeProxyConfig()

    def compute(
        self,
        *,
        spot: float,
        rows: Sequence,
        source_age: Optional[float] = None,
        **_,
    ) -> GEXSnapshot:
        total_vol = sum(int(getattr(r, "volume", 0) or 0) for r in rows)
        if total_vol <= 0:
            # Fail closed: do not substitute OI.
            return empty_snapshot(
                spot, variant=self.variant, assumption=self.cfg.assumption,
                notes="missing_volume", missing_volume=True)

        contracts = []
        for r in rows:
            vol = int(getattr(r, "volume", 0) or 0)
            side = getattr(r, "side", "")
            if vol < self.cfg.min_volume or side not in ("call", "put"):
                continue
            contracts.append(WeightedContract(
                side=side,
                strike=float(getattr(r, "strike", 0.0)),
                gamma=float(getattr(r, "gamma", 0.0) or 0.0),
                weight=float(vol),
            ))
        if not contracts:
            return empty_snapshot(
                spot, variant=self.variant, assumption=self.cfg.assumption,
                notes="no volume-weighted contracts", missing_volume=True)

        primary = snapshot_from_contracts(
            contracts, spot,
            variant=self.variant,
            assumption=self.cfg.assumption,
            quality_score=min(1.0, total_vol / 5000.0),
            source_age=source_age,
            missing_volume=False,
            notes="volume_proxy",
        )

        if not self.cfg.compute_sign_scenarios:
            return primary

        # §16.5: also compute alt sign; if they disagree, lower quality and
        # mark assumption as confidence blend (levels stay on baseline).
        alt = snapshot_from_contracts(
            contracts, spot,
            variant=self.variant,
            assumption=self.cfg.alt_assumption,
            quality_score=primary.quality_score,
            source_age=source_age,
            notes="volume_proxy_alt",
        )
        sign_disagree = (
            primary.is_finite and alt.is_finite
            and (primary.net_gex > 0) != (alt.net_gex > 0)
        )
        if sign_disagree:
            return GEXSnapshot(
                net_gex=primary.net_gex,
                gamma_flip=primary.gamma_flip,
                call_wall=primary.call_wall,
                put_wall=primary.put_wall,
                gex_concentration=primary.gex_concentration,
                wall_concentration=primary.wall_concentration,
                quality_score=max(0.1, primary.quality_score * 0.7),
                assumption_set=GexAssumption.CONFIDENCE_BLEND,
                source_age=source_age,
                variant=self.variant,
                net_ratio=primary.net_ratio,
                missing_volume=False,
                n_contracts=primary.n_contracts,
                notes="volume_proxy_sign_disagree",
            )
        return primary
