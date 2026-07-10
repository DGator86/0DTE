"""
gex/base.py
===========
Shared GEX aggregation, disagreement, and parallel variant orchestration
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §16, PR 9).

The strike-level engine mirrors spy0dte.build_gamma_map so OI and challengers
share one flip/wall implementation. Policy continues to use the feed's OI
baseline on MarketSnapshot; this module only produces observation panels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from gex.contracts import (
    GEXSnapshot, GexAssumption, GexDisagreement, GexVariantBundle, GexVariantId,
)

MULT = 100  # index/ETF options multiplier (matches spy0dte.MULT)


@dataclass(frozen=True)
class WeightedContract:
    """One contract contribution after provider-specific weighting."""
    side: str          # "call" | "put"
    strike: float
    gamma: float
    weight: float      # oi, volume, or oi*decay
    dte_days: float = 0.0


def signed_dollar_gamma(
    side: str,
    gamma: float,
    weight: float,
    spot: float,
    *,
    assumption: GexAssumption = GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS,
) -> float:
    """
    Dollar gamma contribution for one contract.

    Baseline (dealer-short-customer-long): calls +, puts −.
    Alt same-day put-flow: flip put sign (puts +) to stress-test convention.
    """
    dollar = float(gamma) * float(weight) * MULT * float(spot) * float(spot) * 0.01
    if assumption == GexAssumption.SAME_DAY_PUT_FLOW_ALT:
        # Alternative: treat put flow as dealer-long-puts (sign flip on puts)
        return -dollar if side == "call" else dollar
    # baseline + confidence_blend (blend applied at hybrid layer)
    return dollar if side == "call" else -dollar


def aggregate_strikes(
    contracts: Sequence[WeightedContract],
    spot: float,
    *,
    assumption: GexAssumption = GexAssumption.DEALER_SHORT_CALLS_LONG_PUTS,
) -> tuple:
    """
    Build per-strike signed GEX maps.

    Returns (by_strike, call_g, put_g) in raw dollar-gamma units (not $bn).
    """
    by_strike: dict[float, float] = {}
    call_g: dict[float, float] = {}
    put_g: dict[float, float] = {}
    for c in contracts:
        if c.weight <= 0 or not math.isfinite(c.gamma):
            continue
        signed = signed_dollar_gamma(
            c.side, c.gamma, c.weight, spot, assumption=assumption)
        dollar_abs = abs(signed)
        by_strike[c.strike] = by_strike.get(c.strike, 0.0) + signed
        if c.side == "call":
            call_g[c.strike] = call_g.get(c.strike, 0.0) + dollar_abs
        else:
            put_g[c.strike] = put_g.get(c.strike, 0.0) + dollar_abs
    return by_strike, call_g, put_g


def levels_from_maps(
    by_strike: dict,
    call_g: dict,
    put_g: dict,
    spot: float,
) -> tuple:
    """
    Derive (net_gex_bn, net_ratio, gamma_flip, call_wall, put_wall,
            gex_concentration, wall_concentration) from strike maps.

    Matches spy0dte.build_gamma_map arithmetic for net/flip/walls.
    """
    if not by_strike:
        return (float("nan"), float("nan"), float(spot), float(spot),
                float(spot), 0.0, 0.0)

    net_gex = sum(by_strike.values()) / 1e9
    gross_gex = sum(abs(v) for v in by_strike.values()) / 1e9
    net_ratio = (net_gex / gross_gex) if gross_gex else 0.0

    strikes = sorted(by_strike)
    cum = 0.0
    flip = float(spot)
    prev_k, prev_cum = strikes[0], 0.0
    for k in strikes:
        cum += by_strike[k]
        if prev_cum < 0 <= cum or prev_cum > 0 >= cum:
            span = (cum - prev_cum)
            flip = prev_k + (k - prev_k) * (0 - prev_cum) / span if span else k
            break
        prev_k, prev_cum = k, cum

    calls_above = {k: g for k, g in call_g.items() if k >= spot}
    puts_below = {k: g for k, g in put_g.items() if k <= spot}
    call_wall = (max(calls_above, key=calls_above.get) if calls_above
                 else max(call_g, key=call_g.get) if call_g else spot)
    put_wall = (max(puts_below, key=puts_below.get) if puts_below
                else min(put_g, key=put_g.get) if put_g else spot)

    # Concentration: max |strike| share of gross (HHI-lite)
    if gross_gex > 0:
        gex_concentration = max(abs(v) for v in by_strike.values()) / 1e9 / gross_gex
    else:
        gex_concentration = 0.0
    side_call = sum(call_g.values()) or 1.0
    side_put = sum(put_g.values()) or 1.0
    wall_c = (call_g.get(call_wall, 0.0) / side_call) if call_g else 0.0
    wall_p = (put_g.get(put_wall, 0.0) / side_put) if put_g else 0.0
    wall_concentration = max(wall_c, wall_p)

    return (net_gex, net_ratio, flip, float(call_wall), float(put_wall),
            float(gex_concentration), float(wall_concentration))


def snapshot_from_contracts(
    contracts: Sequence[WeightedContract],
    spot: float,
    *,
    variant: GexVariantId,
    assumption: GexAssumption,
    quality_score: float,
    source_age: Optional[float] = None,
    missing_volume: bool = False,
    n_expirations: int = 1,
    notes: str = "",
    gex_pct_rank: Optional[float] = None,
) -> GEXSnapshot:
    by_strike, call_g, put_g = aggregate_strikes(
        contracts, spot, assumption=assumption)
    (net_gex, net_ratio, flip, call_wall, put_wall,
     gex_c, wall_c) = levels_from_maps(by_strike, call_g, put_g, spot)
    return GEXSnapshot(
        net_gex=round(net_gex, 6) if math.isfinite(net_gex) else float("nan"),
        gamma_flip=round(flip, 4) if math.isfinite(flip) else float(spot),
        call_wall=call_wall,
        put_wall=put_wall,
        gex_concentration=round(gex_c, 6),
        wall_concentration=round(wall_c, 6),
        quality_score=float(max(0.0, min(1.0, quality_score))),
        assumption_set=assumption,
        source_age=source_age,
        variant=variant,
        net_ratio=round(net_ratio, 6) if math.isfinite(net_ratio) else None,
        gex_pct_rank=gex_pct_rank,
        missing_volume=missing_volume,
        n_contracts=len(contracts),
        n_expirations=n_expirations,
        notes=notes,
    )


def empty_snapshot(
    spot: float,
    *,
    variant: GexVariantId,
    assumption: GexAssumption,
    notes: str,
    missing_volume: bool = False,
) -> GEXSnapshot:
    return GEXSnapshot(
        net_gex=float("nan"),
        gamma_flip=float(spot),
        call_wall=float(spot),
        put_wall=float(spot),
        gex_concentration=0.0,
        wall_concentration=0.0,
        quality_score=0.0,
        assumption_set=assumption,
        variant=variant,
        missing_volume=missing_volume,
        n_contracts=0,
        notes=notes,
    )


def compute_disagreement(snapshots: Sequence[GEXSnapshot]) -> GexDisagreement:
    finite = [s for s in snapshots if s.is_finite]
    if len(finite) < 2:
        return GexDisagreement(
            flip_spread=0.0, wall_spread_call=0.0, wall_spread_put=0.0,
            net_gex_sign_disagree=False, net_gex_range_bn=0.0,
            n_finite_variants=len(finite))
    flips = [s.gamma_flip for s in finite]
    calls = [s.call_wall for s in finite]
    puts = [s.put_wall for s in finite]
    nets = [s.net_gex for s in finite]
    signs = {1 if n > 0 else (-1 if n < 0 else 0) for n in nets}
    return GexDisagreement(
        flip_spread=float(max(flips) - min(flips)),
        wall_spread_call=float(max(calls) - min(calls)),
        wall_spread_put=float(max(puts) - min(puts)),
        net_gex_sign_disagree=len(signs - {0}) > 1,
        net_gex_range_bn=float(max(nets) - min(nets)),
        n_finite_variants=len(finite),
    )


def compute_all_variants(
    *,
    spot: float,
    rows_0dte: Sequence,
    rows_weekly: Optional[Sequence] = None,
    now_epoch: Optional[float] = None,
    source_age: Optional[float] = None,
    minute_of_session: Optional[float] = None,
    volume_oi_ratio: Optional[float] = None,
    feed_source: str = "",
    authoritative: GexVariantId = GexVariantId.OI,
) -> GexVariantBundle:
    """
    Run OI / weekly / volume / hybrid providers in parallel.

    `rows_*` are OptionRow-like (side, strike, oi, gamma, volume[, dte_days]).
    Missing weekly/volume data yields quality_score=0 snapshots without
    mutating the OI path.
    """
    from gex.oi import OiGexProvider
    from gex.weekly import WeeklyGexProvider
    from gex.volume_proxy import VolumeProxyProvider
    from gex.hybrid import HybridGexProvider

    oi = OiGexProvider().compute(
        spot=spot, rows=rows_0dte, source_age=source_age)
    weekly = WeeklyGexProvider().compute(
        spot=spot,
        rows=(rows_weekly if rows_weekly is not None else rows_0dte),
        source_age=source_age)
    volume = VolumeProxyProvider().compute(
        spot=spot, rows=rows_0dte, source_age=source_age)
    hybrid = HybridGexProvider().compute(
        spot=spot, oi=oi, weekly=weekly, volume=volume,
        minute_of_session=minute_of_session,
        volume_oi_ratio=volume_oi_ratio,
        feed_source=feed_source)
    disagree = compute_disagreement([oi, weekly, volume, hybrid])
    return GexVariantBundle(
        spot=float(spot),
        authoritative=authoritative,
        oi=oi, weekly=weekly, volume=volume, hybrid=hybrid,
        disagreement=disagree, feed_source=feed_source,
    )
