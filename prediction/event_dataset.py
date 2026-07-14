"""
prediction/event_dataset.py
===========================
Discrete-time event-dataset construction for competing-risk models
(V3 Part 2 §19–§20, PR 13).

Observation-time features and barrier geometry are frozen. Future minutes
only add elapsed-time context. Same-bar ambiguity never defaults to the
favorable event.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class EventDatasetRow:
    snapshot_id: str
    session_date: str
    origin_ts: str
    future_minute: int
    time_fraction: float
    still_at_risk: bool
    event_target: int
    event_stop: int
    event_none: int
    censored: bool
    ambiguous_same_bar: bool
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventDatasetConfig:
    horizon_minutes: int = 30
    same_bar_policy: str = "adverse_first"  # adverse_first | exclude
    epsilon: float = 1e-9


def expand_observation_to_event_rows(
    *,
    snapshot_id: str,
    session_date: str,
    origin_ts: str,
    prices: Sequence[float],
    minutes: Sequence[float],
    target: float,
    stop: float,
    direction: str = "up",
    frozen_features: Optional[dict] = None,
    expected_remaining_move: Optional[float] = None,
    predicted_vol: Optional[float] = None,
    highs: Optional[Sequence[float]] = None,
    lows: Optional[Sequence[float]] = None,
    cfg: Optional[EventDatasetConfig] = None,
) -> list[EventDatasetRow]:
    """
    Expand one observation into discrete-time at-risk rows (§20.1).

    `prices` / `minutes` are the forward path (index 0 = observation or first
    future bar). Optional `highs`/`lows` enable same-bar ambiguity detection;
    when omitted, high=low=close.
    """
    cfg = cfg or EventDatasetConfig()
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be up|down, got {direction!r}")
    if len(prices) != len(minutes):
        raise ValueError("prices/minutes length mismatch")
    if not prices:
        return []
    hi_arr = list(highs) if highs is not None else list(prices)
    lo_arr = list(lows) if lows is not None else list(prices)
    if len(hi_arr) != len(prices) or len(lo_arr) != len(prices):
        raise ValueError("highs/lows length mismatch")

    frozen = dict(frozen_features or {})
    # Geometry features frozen at observation time (§20.3)
    entry = float(prices[0])
    erm = expected_remaining_move
    geom = _geometry_features(
        entry, target, stop, direction, erm, predicted_vol, cfg.epsilon)
    frozen = {**frozen, **geom}

    horizon = int(cfg.horizon_minutes)
    rows: list[EventDatasetRow] = []
    event_happened = False

    for i, (px, m) in enumerate(zip(prices, minutes)):
        fut_min = int(round(float(m))) if i > 0 else 0
        if fut_min > horizon:
            break
        if event_happened:
            break

        hit_target, hit_stop = _bar_hits(
            hi_arr[i], lo_arr[i], target, stop, direction)
        ambiguous = hit_target and hit_stop
        censored = False
        e_target = e_stop = e_none = 0

        if ambiguous:
            if cfg.same_bar_policy == "exclude":
                e_none = 1
            else:
                # adverse_first — stop wins; never assume favorable first
                e_stop = 1
                event_happened = True
        elif hit_stop:
            e_stop = 1
            event_happened = True
        elif hit_target:
            e_target = 1
            event_happened = True
        else:
            e_none = 1
            if fut_min >= horizon:
                censored = True

        feat = {
            **frozen,
            "future_minute": float(fut_min),
            "time_fraction": float(fut_min) / max(horizon, 1),
            "remaining_horizon": float(max(0, horizon - fut_min)),
        }
        rows.append(EventDatasetRow(
            snapshot_id=snapshot_id,
            session_date=session_date,
            origin_ts=origin_ts,
            future_minute=fut_min,
            time_fraction=float(fut_min) / max(horizon, 1),
            still_at_risk=True,
            event_target=int(e_target),
            event_stop=int(e_stop),
            event_none=int(e_none),
            censored=bool(censored and not (e_target or e_stop)),
            ambiguous_same_bar=bool(ambiguous),
            features=feat,
        ))
        if e_target or e_stop:
            event_happened = True

    if rows and not any(r.event_target or r.event_stop for r in rows):
        last = rows[-1]
        if not last.censored and last.future_minute >= horizon:
            rows[-1] = EventDatasetRow(
                **{**last.to_dict(), "censored": True,
                   "features": last.features})
    return rows


def _bar_hits(high, low, target, stop, direction) -> tuple[bool, bool]:
    if direction == "up":
        return high >= target, low <= stop
    return low <= target, high >= stop


def _geometry_features(
    entry: float,
    target: float,
    stop: float,
    direction: str,
    expected_remaining_move: Optional[float],
    predicted_vol: Optional[float],
    epsilon: float,
) -> dict:
    if direction == "up":
        dist_t = target - entry
        dist_s = entry - stop
    else:
        dist_t = entry - target
        dist_s = stop - entry
    out = {
        "target_distance": float(dist_t),
        "stop_distance": float(dist_s),
        "reward_to_risk": float(dist_t / max(dist_s, epsilon)),
    }
    if expected_remaining_move is not None and math.isfinite(
            float(expected_remaining_move)):
        erm = max(float(expected_remaining_move), epsilon)
        out["target_distance_expected_move"] = float(dist_t / erm)
        out["stop_distance_expected_move"] = float(dist_s / erm)
    if predicted_vol is not None and math.isfinite(float(predicted_vol)):
        vol = max(float(predicted_vol), epsilon)
        out["target_distance_vol"] = float(dist_t / vol)
        out["stop_distance_vol"] = float(dist_s / vol)
    return out
