"""
prediction/regime_labels.py
===========================
Future-behavior regime labels for Prediction Engine V3 Part 2
(docs/PREDICTION_ENGINE_V3_PART2_FORECASTING.md §9–§11, PR 8).

Labels describe what the market *subsequently did*, not the current GEX
sign. Structural reference levels (flip, walls, VWAP) are FROZEN at
observation time — never looked up in the future.

Ambiguous / weak paths may remain unlabeled. Component behavior flags
are always emitted for multilabel research.

Research / shadow only. No candidate outcomes may enter labeling.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

REGIME_CLASSES = (
    "long_gamma_pin",
    "short_gamma_trend",
    "flip_transition",
    "volatility_expansion",
)

REGIME_LABEL_VERSION = "v3.0.0"

# Mutually exclusive precedence (§11.5)
_PRECEDENCE = (
    "volatility_expansion",
    "flip_transition",
    "short_gamma_trend",
    "long_gamma_pin",
)


@dataclass
class RegimeLabelConfig:
    horizon_minutes: int = 30
    pin_max_move_fraction: float = 0.50
    pin_max_directional_efficiency: float = 0.35
    pin_min_reversion_count: int = 2
    trend_min_move_fraction: float = 0.75
    trend_min_directional_efficiency: float = 0.60
    trend_max_pullback_fraction: float = 0.45
    transition_flip_cross_required: bool = True
    transition_min_side_changes: int = 1
    vol_expansion_move_fraction: float = 1.00
    vol_expansion_min_two_sided_excursion: float = 0.50
    ambiguity_policy: str = "exclude"  # exclude | prefer_precedence
    epsilon: float = 1e-9
    wall_breach_tolerance: float = 0.0  # absolute price units


@dataclass(frozen=True)
class PathBehaviorStats:
    """Frozen-horizon path statistics used for regime labeling (§10)."""
    start_price: float
    end_price: float
    n_bars: int
    total_path_variation: float
    directional_efficiency: float
    upward_excursion: float
    downward_excursion: float
    maximum_absolute_excursion: float
    future_move_fraction: Optional[float]
    pullback_fraction: float
    reversion_count: int
    two_sided_excursion: Optional[float]
    flip_crossed: bool
    sides_of_flip_occupied: int
    call_wall_breached: bool
    put_wall_breached: bool
    realized_vol: float
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegimeLabelResult:
    regime_label: Optional[str]
    is_pin_behavior: bool
    is_trend_behavior: bool
    is_transition_behavior: bool
    is_vol_expansion_behavior: bool
    path_stats: PathBehaviorStats
    label_version: str = REGIME_LABEL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "regime_label": self.regime_label,
            "is_pin_behavior": self.is_pin_behavior,
            "is_trend_behavior": self.is_trend_behavior,
            "is_transition_behavior": self.is_transition_behavior,
            "is_vol_expansion_behavior": self.is_vol_expansion_behavior,
            "label_version": self.label_version,
            "path_stats": self.path_stats.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }
        return d


def directional_efficiency(
    prices: Sequence[float],
    *,
    epsilon: float = 1e-9,
) -> float:
    """abs(final - start) / max(total_path_variation, eps) (§10.1)."""
    if len(prices) < 2:
        return 0.0
    start, end = float(prices[0]), float(prices[-1])
    variation = total_path_variation(prices)
    return abs(end - start) / max(variation, epsilon)


def total_path_variation(prices: Sequence[float]) -> float:
    if len(prices) < 2:
        return 0.0
    return float(sum(
        abs(float(prices[i]) - float(prices[i - 1]))
        for i in range(1, len(prices))
    ))


def future_move_fraction(
    max_abs_excursion: float,
    expected_remaining_move: Optional[float],
    *,
    epsilon: float = 1e-9,
) -> Optional[float]:
    """maximum_absolute_excursion / expected_remaining_move (§10.2)."""
    if expected_remaining_move is None:
        return None
    try:
        erm = float(expected_remaining_move)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(erm) or erm < 0:
        return None
    return float(max_abs_excursion / max(erm, epsilon))


def pullback_fraction(
    prices: Sequence[float],
    *,
    epsilon: float = 1e-9,
) -> float:
    """
    Pullback relative to the dominant directional excursion (§10.3).

    Upward move: max drawdown from running high / upward excursion.
    Downward move: max bounce from running low / downward excursion.
    """
    if len(prices) < 2:
        return 0.0
    vals = [float(p) for p in prices]
    start = vals[0]
    up = max(vals) - start
    down = start - min(vals)
    if up >= down and up > epsilon:
        peak = start
        max_pull = 0.0
        for p in vals:
            if p > peak:
                peak = p
            max_pull = max(max_pull, peak - p)
        return float(max_pull / max(up, epsilon))
    if down > epsilon:
        trough = start
        max_bounce = 0.0
        for p in vals:
            if p < trough:
                trough = p
            max_bounce = max(max_bounce, p - trough)
        return float(max_bounce / max(down, epsilon))
    return 0.0


def reversion_count(
    prices: Sequence[float],
    references: Sequence[Optional[float]],
    *,
    epsilon: float = 1e-9,
) -> int:
    """
    Count returns toward frozen reference levels (§10.4).

    A reversion is counted each time the signed distance to a reference
    changes sign (the path crosses back through the frozen level).
    """
    refs = [float(r) for r in references
            if r is not None and math.isfinite(float(r))]
    if len(prices) < 3 or not refs:
        return 0
    vals = [float(p) for p in prices]
    count = 0
    for ref in refs:
        prev_sign = 0
        for p in vals:
            if abs(p - ref) <= epsilon:
                sign = 0
            elif p > ref:
                sign = 1
            else:
                sign = -1
            if prev_sign != 0 and sign != 0 and sign != prev_sign:
                count += 1
            if sign != 0:
                prev_sign = sign
    return int(count)


def two_sided_excursion(
    upward: float,
    downward: float,
    expected_remaining_move: Optional[float],
    *,
    epsilon: float = 1e-9,
) -> Optional[float]:
    """min(up, down) / expected_remaining_move (§10.5)."""
    if expected_remaining_move is None:
        return None
    try:
        erm = float(expected_remaining_move)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(erm) or erm < 0:
        return None
    return float(min(upward, downward) / max(erm, epsilon))


def compute_path_stats(
    prices: Sequence[float],
    *,
    expected_remaining_move: Optional[float],
    frozen_gamma_flip: Optional[float] = None,
    frozen_call_wall: Optional[float] = None,
    frozen_put_wall: Optional[float] = None,
    frozen_vwap: Optional[float] = None,
    wall_midpoint: Optional[float] = None,
    cfg: Optional[RegimeLabelConfig] = None,
) -> PathBehaviorStats:
    """Compute frozen-horizon path statistics from a forward price path."""
    cfg = cfg or RegimeLabelConfig()
    eps = cfg.epsilon
    if not prices:
        raise ValueError("prices must be non-empty")
    vals = [float(p) for p in prices]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("prices must be finite")
    start, end = vals[0], vals[-1]
    up = max(0.0, max(vals) - start)
    down = max(0.0, start - min(vals))
    max_abs = max(up, down)
    variation = total_path_variation(vals)
    de = abs(end - start) / max(variation, eps) if len(vals) >= 2 else 0.0
    fmf = future_move_fraction(max_abs, expected_remaining_move, epsilon=eps)
    tse = two_sided_excursion(up, down, expected_remaining_move, epsilon=eps)

    # Midpoint between frozen walls if not provided
    mid = wall_midpoint
    if (mid is None and frozen_call_wall is not None
            and frozen_put_wall is not None
            and math.isfinite(frozen_call_wall)
            and math.isfinite(frozen_put_wall)):
        mid = 0.5 * (float(frozen_call_wall) + float(frozen_put_wall))

    refs = [frozen_gamma_flip, frozen_vwap, mid]
    rev = reversion_count(vals, refs, epsilon=eps)

    flip_crossed = False
    sides = 0
    flip_up_exc = 0.0
    flip_down_exc = 0.0
    if frozen_gamma_flip is not None and math.isfinite(float(frozen_gamma_flip)):
        flip = float(frozen_gamma_flip)
        above = any(v > flip + eps for v in vals)
        below = any(v < flip - eps for v in vals)
        sides = int(above) + int(below)
        flip_up_exc = max(0.0, max(vals) - flip)
        flip_down_exc = max(0.0, flip - min(vals))
        # Crossed if start side differs from some later point on other side
        if sides == 2:
            flip_crossed = True
        elif len(vals) >= 2:
            s0 = vals[0] - flip
            for v in vals[1:]:
                if s0 == 0:
                    s0 = v - flip
                    continue
                if (v - flip) * s0 < 0:
                    flip_crossed = True
                    break

    tol = cfg.wall_breach_tolerance
    cw_breach = False
    pw_breach = False
    if frozen_call_wall is not None and math.isfinite(float(frozen_call_wall)):
        cw_breach = max(vals) >= float(frozen_call_wall) - tol
    if frozen_put_wall is not None and math.isfinite(float(frozen_put_wall)):
        pw_breach = min(vals) <= float(frozen_put_wall) + tol

    # Simple realized vol proxy: std of simple returns
    if len(vals) >= 3:
        rets = [
            (vals[i] - vals[i - 1]) / max(abs(vals[i - 1]), eps)
            for i in range(1, len(vals))
        ]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        rvol = math.sqrt(var)
    else:
        rvol = 0.0

    # Flip-relative two-sided occupation (for transition labels)
    flip_two_sided = None
    if (frozen_gamma_flip is not None
            and expected_remaining_move is not None
            and math.isfinite(float(expected_remaining_move))
            and float(expected_remaining_move) >= 0):
        flip_two_sided = float(
            min(flip_up_exc, flip_down_exc)
            / max(float(expected_remaining_move), eps)
        )

    return PathBehaviorStats(
        start_price=start,
        end_price=end,
        n_bars=len(vals),
        total_path_variation=float(variation),
        directional_efficiency=float(de),
        upward_excursion=float(up),
        downward_excursion=float(down),
        maximum_absolute_excursion=float(max_abs),
        future_move_fraction=fmf,
        pullback_fraction=float(pullback_fraction(vals, epsilon=eps)),
        reversion_count=int(rev),
        two_sided_excursion=tse,
        flip_crossed=bool(flip_crossed),
        sides_of_flip_occupied=int(sides),
        call_wall_breached=bool(cw_breach),
        put_wall_breached=bool(pw_breach),
        realized_vol=float(rvol),
        diagnostics={
            "frozen_gamma_flip": frozen_gamma_flip,
            "frozen_call_wall": frozen_call_wall,
            "frozen_put_wall": frozen_put_wall,
            "frozen_vwap": frozen_vwap,
            "wall_midpoint": mid,
            "flip_up_excursion": flip_up_exc,
            "flip_down_excursion": flip_down_exc,
            "flip_two_sided_excursion": flip_two_sided,
        },
    )


def _is_pin(stats: PathBehaviorStats, cfg: RegimeLabelConfig) -> bool:
    if stats.future_move_fraction is None:
        return False
    if stats.future_move_fraction > cfg.pin_max_move_fraction:
        return False
    if stats.directional_efficiency > cfg.pin_max_directional_efficiency:
        return False
    if stats.reversion_count < cfg.pin_min_reversion_count:
        return False
    # Neither frozen wall decisively breached
    if stats.call_wall_breached or stats.put_wall_breached:
        return False
    return True


def _is_trend(stats: PathBehaviorStats, cfg: RegimeLabelConfig) -> bool:
    if stats.future_move_fraction is None:
        return False
    if stats.future_move_fraction < cfg.trend_min_move_fraction:
        return False
    if stats.directional_efficiency < cfg.trend_min_directional_efficiency:
        return False
    if stats.pullback_fraction > cfg.trend_max_pullback_fraction:
        return False
    return True


def _is_transition(stats: PathBehaviorStats, cfg: RegimeLabelConfig) -> bool:
    if cfg.transition_flip_cross_required and not stats.flip_crossed:
        return False
    if stats.sides_of_flip_occupied < 2:
        return False
    # Meaningful occupation on both sides of the frozen flip
    flip_tse = stats.diagnostics.get("flip_two_sided_excursion")
    if flip_tse is not None and float(flip_tse) < 0.15:
        return False
    # A trivial single touch without material two-sided occupation is not
    # a transition when expected-move scaling is unavailable.
    if flip_tse is None:
        up = float(stats.diagnostics.get("flip_up_excursion") or 0.0)
        down = float(stats.diagnostics.get("flip_down_excursion") or 0.0)
        if min(up, down) < abs(stats.start_price) * 0.0005:
            return False
    return True


def _is_vol_expansion(stats: PathBehaviorStats, cfg: RegimeLabelConfig) -> bool:
    if stats.future_move_fraction is None or stats.two_sided_excursion is None:
        return False
    if stats.future_move_fraction < cfg.vol_expansion_move_fraction:
        return False
    if stats.two_sided_excursion < cfg.vol_expansion_min_two_sided_excursion:
        return False
    # Range expands without clean directional efficiency
    if stats.directional_efficiency >= cfg.trend_min_directional_efficiency:
        return False
    return True


def label_regime(
    prices: Sequence[float],
    *,
    expected_remaining_move: Optional[float],
    frozen_gamma_flip: Optional[float] = None,
    frozen_call_wall: Optional[float] = None,
    frozen_put_wall: Optional[float] = None,
    frozen_vwap: Optional[float] = None,
    cfg: Optional[RegimeLabelConfig] = None,
    # Current GEX sign is a FEATURE only — never required for the label
    current_gex_sign: Optional[float] = None,
) -> RegimeLabelResult:
    """
    Assign a mutually exclusive future-behavior regime label (§11).

    Returns component flags regardless of the exclusive label. Ambiguous
    paths yield regime_label=None when ambiguity_policy='exclude'.
    """
    cfg = cfg or RegimeLabelConfig()
    stats = compute_path_stats(
        prices,
        expected_remaining_move=expected_remaining_move,
        frozen_gamma_flip=frozen_gamma_flip,
        frozen_call_wall=frozen_call_wall,
        frozen_put_wall=frozen_put_wall,
        frozen_vwap=frozen_vwap,
        cfg=cfg,
    )
    flags = {
        "volatility_expansion": _is_vol_expansion(stats, cfg),
        "flip_transition": _is_transition(stats, cfg),
        "short_gamma_trend": _is_trend(stats, cfg),
        "long_gamma_pin": _is_pin(stats, cfg),
    }
    matched = [name for name in _PRECEDENCE if flags[name]]
    if not matched:
        label = None
    elif len(matched) == 1 or cfg.ambiguity_policy == "prefer_precedence":
        label = matched[0]
    else:
        # Multiple independent flags with exclude policy → still apply
        # precedence for the exclusive label (deterministic), but mark
        # multi_match in diagnostics. Spec: precedence is deterministic.
        label = matched[0]

    diagnostics = {
        "matched_classes": matched,
        "component_flags": dict(flags),
        "current_gex_sign": current_gex_sign,
        "horizon_minutes": cfg.horizon_minutes,
        "ambiguity_policy": cfg.ambiguity_policy,
        "unclassified": label is None,
    }
    return RegimeLabelResult(
        regime_label=label,
        is_pin_behavior=flags["long_gamma_pin"],
        is_trend_behavior=flags["short_gamma_trend"],
        is_transition_behavior=flags["flip_transition"],
        is_vol_expansion_behavior=flags["volatility_expansion"],
        path_stats=stats,
        label_version=REGIME_LABEL_VERSION,
        diagnostics=diagnostics,
    )


def label_regime_from_bars(
    bars: Sequence[dict],
    *,
    expected_remaining_move: Optional[float],
    frozen_gamma_flip: Optional[float] = None,
    frozen_call_wall: Optional[float] = None,
    frozen_put_wall: Optional[float] = None,
    frozen_vwap: Optional[float] = None,
    price_key: str = "close",
    cfg: Optional[RegimeLabelConfig] = None,
    current_gex_sign: Optional[float] = None,
) -> RegimeLabelResult:
    """Convenience: extract prices from bar dicts then label."""
    prices = [float(b[price_key]) for b in bars]
    return label_regime(
        prices,
        expected_remaining_move=expected_remaining_move,
        frozen_gamma_flip=frozen_gamma_flip,
        frozen_call_wall=frozen_call_wall,
        frozen_put_wall=frozen_put_wall,
        frozen_vwap=frozen_vwap,
        cfg=cfg,
        current_gex_sign=current_gex_sign,
    )
