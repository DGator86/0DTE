"""
gex/hybrid.py
=============
Variant D — blend OI / weekly / volume by time-of-day, volume/OI ratio,
and feed quality (docs/PREDICTION_ENGINE_V2_HANDOFF.md §16.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from gex.contracts import GEXSnapshot, GexAssumption, GexVariantId


@dataclass
class HybridConfig:
    open_minutes_oi_heavy: float = 30.0
    vol_oi_ratio_threshold: float = 0.15
    min_volume_quality: float = 0.3
    min_weekly_quality: float = 0.2


def _blend_levels(parts: list[tuple[GEXSnapshot, float]]) -> GEXSnapshot:
    """Quality-weighted blend of finite snapshots' levels."""
    usable = [(s, w) for s, w in parts
              if s is not None and s.is_finite and w > 0 and s.quality_score > 0]
    if not usable:
        # Fall back to first snapshot even if empty
        s0 = parts[0][0]
        return GEXSnapshot(
            net_gex=s0.net_gex, gamma_flip=s0.gamma_flip,
            call_wall=s0.call_wall, put_wall=s0.put_wall,
            gex_concentration=s0.gex_concentration,
            wall_concentration=s0.wall_concentration,
            quality_score=0.0,
            assumption_set=GexAssumption.CONFIDENCE_BLEND,
            variant=GexVariantId.HYBRID,
            notes="hybrid_no_usable_inputs",
        )
    wsum = sum(w * s.quality_score for s, w in usable)
    if wsum <= 0:
        wsum = sum(w for _, w in usable)

    def wavg(attr: str) -> float:
        return sum(getattr(s, attr) * w * max(s.quality_score, 1e-6)
                   for s, w in usable) / wsum

    net = wavg("net_gex")
    flip = wavg("gamma_flip")
    cw = wavg("call_wall")
    pw = wavg("put_wall")
    gex_c = wavg("gex_concentration")
    wall_c = wavg("wall_concentration")
    q = min(1.0, sum(s.quality_score * w for s, w in usable) / sum(w for _, w in usable))
    return GEXSnapshot(
        net_gex=round(net, 6),
        gamma_flip=round(flip, 4),
        call_wall=round(cw, 4),
        put_wall=round(pw, 4),
        gex_concentration=round(gex_c, 6),
        wall_concentration=round(wall_c, 6),
        quality_score=round(q, 6),
        assumption_set=GexAssumption.CONFIDENCE_BLEND,
        variant=GexVariantId.HYBRID,
        net_ratio=None,
        n_contracts=sum(s.n_contracts for s, _ in usable),
        n_expirations=max(s.n_expirations for s, _ in usable),
        notes="hybrid_blend",
    )


class HybridGexProvider:
    variant = GexVariantId.HYBRID

    def __init__(self, cfg: Optional[HybridConfig] = None):
        self.cfg = cfg or HybridConfig()

    def compute(
        self,
        *,
        spot: float,
        oi: GEXSnapshot,
        weekly: Optional[GEXSnapshot] = None,
        volume: Optional[GEXSnapshot] = None,
        minute_of_session: Optional[float] = None,
        volume_oi_ratio: Optional[float] = None,
        feed_source: str = "",
        **_,
    ) -> GEXSnapshot:
        """
        Blend weights:
          * First `open_minutes_oi_heavy` minutes → OI-heavy.
          * High volume/OI ratio + usable volume quality → lean volume.
          * Weekly contributes when quality >= min_weekly_quality.
        """
        w_oi, w_vol, w_weekly = 1.0, 0.0, 0.0
        mos = float(minute_of_session) if minute_of_session is not None else 120.0
        vor = float(volume_oi_ratio) if (
            volume_oi_ratio is not None and math.isfinite(volume_oi_ratio)
        ) else 0.0

        if mos < self.cfg.open_minutes_oi_heavy:
            w_oi, w_vol = 0.85, 0.15
        else:
            w_oi, w_vol = 0.55, 0.45

        if (volume is not None
                and not volume.missing_volume
                and volume.quality_score >= self.cfg.min_volume_quality
                and vor >= self.cfg.vol_oi_ratio_threshold):
            w_vol = max(w_vol, 0.60)
            w_oi = 1.0 - w_vol
        elif volume is None or volume.missing_volume or volume.quality_score < self.cfg.min_volume_quality:
            w_vol = 0.0
            w_oi = 1.0

        if (weekly is not None
                and weekly.quality_score >= self.cfg.min_weekly_quality
                and weekly.is_finite):
            w_weekly = 0.20
            scale = 1.0 - w_weekly
            w_oi *= scale
            w_vol *= scale

        parts = [(oi, w_oi)]
        if volume is not None:
            parts.append((volume, w_vol))
        if weekly is not None:
            parts.append((weekly, w_weekly))

        out = _blend_levels(parts)
        note = (f"hybrid_mos={mos:.0f}_vor={vor:.3f}_"
                f"w=({w_oi:.2f},{w_vol:.2f},{w_weekly:.2f})")
        if feed_source:
            note += f"_feed={feed_source}"
        return GEXSnapshot(
            net_gex=out.net_gex,
            gamma_flip=out.gamma_flip,
            call_wall=out.call_wall,
            put_wall=out.put_wall,
            gex_concentration=out.gex_concentration,
            wall_concentration=out.wall_concentration,
            quality_score=out.quality_score,
            assumption_set=GexAssumption.CONFIDENCE_BLEND,
            variant=GexVariantId.HYBRID,
            net_ratio=out.net_ratio,
            n_contracts=out.n_contracts,
            n_expirations=out.n_expirations,
            notes=note,
        )
