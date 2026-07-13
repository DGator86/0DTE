"""
pin_regime.py
=============
Detect when spot is pinned near the gamma flip / inside the wall channel.

Used to:
  * Prefer premium-selling (IC / IF / PCS / CCS) over breakout debit cells
  * Soft-exempt short-gamma / below-flip / trending dealer vetoes that would
    otherwise force credit → debit flips on a tape that is not expanding
  * Soften the premium gate + selector short-gamma vetoes so paper / live
    can actually fill those credit tickets

Does NOT require net_gex > 0 — negative GEX with price glued to the flip is
exactly the pattern this module is meant to catch (dealers short, spot still
pinning). Selector's older `_is_pinned` required long GEX; that stays as a
family-weight bonus path, while this assessment drives policy.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PinConfig:
    # |spot - flip| / spot at or below this → flip-proximity component fires
    zg_frac: float = 0.0015          # 0.15%
    # Upgrade IC → IF when even tighter to the flip
    fly_zg_frac: float = 0.0008      # 0.08%
    # Spot must sit between put_wall and call_wall (inclusive ± buffer)
    require_inside_walls: bool = True
    wall_buffer_frac: float = 0.0005
    # Dynamics / channel soft components (optional; missing → ignored)
    max_move_consumed: float = 0.70
    max_donchian_break: float = 65.0
    min_score: float = 0.55


@dataclass(frozen=True)
class PinAssessment:
    is_pin: bool
    prefer_fly: bool
    zg_pct: float
    inside_walls: bool
    score: float
    reasons: tuple


def assess_pin(
    market,
    *,
    signals: Optional[dict] = None,
    channel: Optional[dict] = None,
    cfg: Optional[PinConfig] = None,
) -> PinAssessment:
    """
    Score whether the tape looks pinned.

    `signals` may carry dynamics keys (expected_move_consumed, wall_rupture).
    `channel` may carry chan_* or raw channel feature dicts
    (donchian_breakout_up/down, bb_squeeze).
    """
    cfg = cfg or PinConfig()
    signals = signals or {}
    channel = channel or {}

    spot = float(getattr(market, "spot", 0.0) or 0.0)
    flip = float(getattr(market, "gamma_flip", 0.0) or 0.0)
    call_wall = float(getattr(market, "call_wall", 0.0) or 0.0)
    put_wall = float(getattr(market, "put_wall", 0.0) or 0.0)

    if spot <= 0 or flip <= 0:
        return PinAssessment(
            is_pin=False, prefer_fly=False, zg_pct=0.0,
            inside_walls=False, score=0.0,
            reasons=("invalid_spot_or_flip",),
        )

    zg_pct = (spot - flip) / spot
    zg_abs = abs(zg_pct)

    # Inside the wall channel?
    lo = min(put_wall, call_wall)
    hi = max(put_wall, call_wall)
    buf = spot * cfg.wall_buffer_frac
    inside = True
    if cfg.require_inside_walls and lo > 0 and hi > lo:
        inside = (lo - buf) <= spot <= (hi + buf)
    elif cfg.require_inside_walls and (lo <= 0 or hi <= lo):
        inside = True  # walls missing → don't hard-fail

    reasons = []
    score = 0.0

    # Flip proximity (required core)
    if zg_abs <= cfg.zg_frac:
        # 1.0 at 0, 0.55 at zg_frac edge
        prox = 1.0 - 0.45 * (zg_abs / cfg.zg_frac)
        score += 0.55 * prox
        reasons.append(f"zg={zg_pct:+.4%}")
    else:
        reasons.append(f"zg_wide={zg_pct:+.4%}")

    if inside:
        score += 0.25
        reasons.append("inside_walls")
    else:
        reasons.append("outside_walls")

    # Optional: move not already consumed / no wall rupture
    move_c = _num(signals.get("expected_move_consumed"))
    if move_c is not None:
        if move_c <= cfg.max_move_consumed:
            score += 0.10 * (1.0 - move_c / max(cfg.max_move_consumed, 1e-9))
            reasons.append(f"move_consumed={move_c:.2f}")
        else:
            score -= 0.15
            reasons.append(f"move_expanded={move_c:.2f}")

    rupture = _num(signals.get("wall_rupture"))
    if rupture is not None and rupture > 0:
        score -= 0.20
        reasons.append(f"wall_rupture={rupture:.2f}")

    # Optional: channel — no active Donchian break
    brk = _max_donchian(channel, signals)
    if brk is not None:
        if brk < cfg.max_donchian_break:
            score += 0.10
            reasons.append(f"donchian_quiet={brk:.0f}")
        else:
            score -= 0.20
            reasons.append(f"donchian_break={brk:.0f}")

    score = max(0.0, min(1.0, score))
    is_pin = bool(
        zg_abs <= cfg.zg_frac
        and inside
        and score >= cfg.min_score
    )
    prefer_fly = bool(is_pin and zg_abs <= cfg.fly_zg_frac)

    return PinAssessment(
        is_pin=is_pin,
        prefer_fly=prefer_fly,
        zg_pct=float(zg_pct),
        inside_walls=inside,
        score=float(round(score, 4)),
        reasons=tuple(reasons),
    )


def pin_soft_exempt_vetoes() -> frozenset:
    """Dealer/tape vetoes that pin may soft-exempt for premium selling."""
    return frozenset({
        "short_gamma", "short_gamma_regime",
        "below_flip", "below_gamma_flip",
        "trending",
    })


def pin_to_signals(pin: PinAssessment) -> dict:
    """Flat journal keys for signals_json / live_state."""
    return {
        "pin_active": 1.0 if pin.is_pin else 0.0,
        "pin_prefer_fly": 1.0 if pin.prefer_fly else 0.0,
        "pin_zg_pct": pin.zg_pct,
        "pin_score": pin.score,
        "pin_inside_walls": 1.0 if pin.inside_walls else 0.0,
        "pin_reasons": ",".join(pin.reasons)[:200],
    }


def _num(v) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _max_donchian(channel: dict, signals: dict) -> Optional[float]:
    keys = (
        ("donchian_breakout_up", "donchian_breakout_down"),
        ("chan_donchian_breakout_up", "chan_donchian_breakout_down"),
    )
    vals = []
    for up_k, dn_k in keys:
        for src in (channel, signals):
            u = _num(src.get(up_k))
            d = _num(src.get(dn_k))
            if u is not None:
                vals.append(u)
            if d is not None:
                vals.append(d)
    return max(vals) if vals else None
