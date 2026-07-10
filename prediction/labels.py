"""
prediction/labels.py
====================
Multi-horizon, strategy-relevant outcome labels for the canonical V2 dataset
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §9).

All labels are computed from ONE session's bars, strictly forward of the
observation timestamp; structural levels (walls, gamma flip) are FROZEN at
their observation-time values — the caller passes them in, this module never
looks levels up in the future. Returns are decimal LOG returns
(ln(future/spot)), the repo-wide convention.

Horizon rules (§9.1):
  * horizons: 5, 15, 30, 60 minutes and session close;
  * terminal price = first bar at/after the horizon boundary, tolerance one
    base bar — otherwise the label is None;
  * a horizon extending past the session close is None (never truncated —
    "30-minute return" measured over 7 minutes is a different quantity).

First-passage rules (§9.7): when one bar contains both the target and the
stop the event is marked ambiguous_same_bar and the CONSERVATIVE reading
assigns the adverse event first; the favorable outcome is never assumed.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from prediction.asof import _to_naive_utc

# Fixed-minute horizons; "close" is handled separately.
HORIZON_MINUTES: dict = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
HORIZONS: tuple = ("5m", "15m", "30m", "60m", "close")

# Actionable-direction research defaults (§9.2) — not permanent truths.
DEFAULT_MIN_RETURN = 0.0002
DEFAULT_MOVE_FRACTION = 0.05


def _ln(a: float, b: float) -> float:
    return math.log(a / b)


def direction_label(forward_return: Optional[float],
                    implied_remaining_move: Optional[float] = None,
                    min_return: float = DEFAULT_MIN_RETURN,
                    move_fraction: float = DEFAULT_MOVE_FRACTION,
                    cost_equivalent: float = 0.0) -> Optional[int]:
    """+1 / -1 / 0 actionable direction; None when the return is unknown."""
    if forward_return is None:
        return None
    threshold = max(min_return,
                    (implied_remaining_move or 0.0) * move_fraction,
                    cost_equivalent)
    if forward_return > threshold:
        return 1
    if forward_return < -threshold:
        return -1
    return 0


def first_passage(highs: Sequence[float], lows: Sequence[float],
                  minutes: Sequence[float],
                  target: float, stop: float,
                  direction: str = "up") -> dict:
    """
    First-passage outcome over a forward bar window (§9.7).

    direction "up":   target above entry (hit when high >= target),
                      stop below (hit when low <= stop).
    direction "down": target below entry (hit when low <= target),
                      stop above (hit when high >= stop).

    Returns:
      first_event               "target" | "stop" | "ambiguous" | "neither"
      first_event_conservative  same, with ambiguous resolved to the ADVERSE
                                event ("stop") — never the favorable one
      ambiguous_same_bar        1 when both levels sit inside one bar
      time_to_first_event       minutes from entry, None when "neither"
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    for hi, lo, m in zip(highs, lows, minutes):
        if direction == "up":
            hit_target = hi >= target
            hit_stop = lo <= stop
        else:
            hit_target = lo <= target
            hit_stop = hi >= stop
        if hit_target and hit_stop:
            return {"first_event": "ambiguous",
                    "first_event_conservative": "stop",
                    "ambiguous_same_bar": 1,
                    "time_to_first_event": float(m)}
        if hit_target:
            return {"first_event": "target",
                    "first_event_conservative": "target",
                    "ambiguous_same_bar": 0,
                    "time_to_first_event": float(m)}
        if hit_stop:
            return {"first_event": "stop",
                    "first_event_conservative": "stop",
                    "ambiguous_same_bar": 0,
                    "time_to_first_event": float(m)}
    return {"first_event": "neither", "first_event_conservative": "neither",
            "ambiguous_same_bar": 0, "time_to_first_event": None}


def range_survival(highs: Sequence[float], lows: Sequence[float],
                   lower: float, upper: float) -> int:
    """1 if every bar stays STRICTLY within (lower, upper), else 0 (§9.6)."""
    for hi, lo in zip(highs, lows):
        if hi >= upper or lo <= lower:
            return 0
    return 1


# --------------------------------------------------------------------------- #
# Session labeler                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SessionLabeler:
    """
    Label observations against ONE session's completed bar path.

    ts must be the session's bar END timestamps (naive UTC, sorted — the
    convention of every feed and resample.py). The last bar defines the
    session close for horizon validity.
    """
    ts: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    base_bar_minutes: int = 1

    def __post_init__(self):
        self.ts = np.asarray(self.ts, dtype="datetime64[ns]")
        if len(self.ts) == 0:
            raise ValueError("SessionLabeler needs at least one bar")
        if np.any(np.diff(self.ts) <= np.timedelta64(0, "ns")):
            raise ValueError("bar timestamps must be strictly increasing")
        self.high = np.asarray(self.high, dtype=float)
        self.low = np.asarray(self.low, dtype=float)
        self.close = np.asarray(self.close, dtype=float)

    # -- internals -----------------------------------------------------------
    def _obs64(self, observation_ts: dt.datetime) -> np.datetime64:
        return np.datetime64(_to_naive_utc(observation_ts))

    def _horizon_end_idx(self, obs: np.datetime64,
                         horizon: str) -> Optional[int]:
        """Index of the terminal bar for a horizon, or None when invalid."""
        session_close = self.ts[-1]
        if horizon == "close":
            return len(self.ts) - 1 if obs < session_close else None
        boundary = obs + np.timedelta64(HORIZON_MINUTES[horizon], "m")
        if boundary > session_close:
            return None                       # horizon extends past the close
        i = int(np.searchsorted(self.ts, boundary, side="left"))
        if i >= len(self.ts):
            return None
        tolerance = np.timedelta64(self.base_bar_minutes, "m")
        if self.ts[i] - boundary > tolerance:
            return None                       # no bar within one base bar
        return i

    def _future_start_idx(self, obs: np.datetime64) -> int:
        """First bar strictly AFTER the observation (bar ends are past data)."""
        return int(np.searchsorted(self.ts, obs, side="right"))

    def _minutes_from(self, obs: np.datetime64, idx: int) -> float:
        return float((self.ts[idx] - obs) / np.timedelta64(1, "m"))

    # -- main entry ------------------------------------------------------------
    def label_observation(self, observation_ts: dt.datetime, spot: float, *,
                          call_wall: Optional[float] = None,
                          put_wall: Optional[float] = None,
                          gamma_flip: Optional[float] = None,
                          implied_remaining_move: Optional[float] = None,
                          min_return: float = DEFAULT_MIN_RETURN,
                          move_fraction: float = DEFAULT_MOVE_FRACTION,
                          cost_equivalent: float = 0.0) -> dict:
        """
        Full label dict for one observation: forward returns, direction,
        MFE/MAE, volatility, wall/flip touches, wall first-passage and
        wall-channel survival. Levels are the caller-frozen observation-time
        values. Any label whose window is invalid is None.
        """
        obs = self._obs64(observation_ts)
        start = self._future_start_idx(obs)
        out: dict = {}

        for h in HORIZONS:
            end = self._horizon_end_idx(obs, h)
            valid = end is not None and end >= start
            key = h

            # ---- terminal return ----
            fwd = _ln(self.close[end], spot) if valid else None
            out[f"fwd_return_{key}"] = fwd
            out[f"up_{key}"] = (1 if fwd > 0 else 0) if fwd is not None else None
            out[f"direction_{key}"] = direction_label(
                fwd, implied_remaining_move, min_return, move_fraction,
                cost_equivalent)

            # ---- path window ----
            if not valid:
                for name in ("up_mfe", "up_mae", "down_mfe", "down_mae",
                             "realized_variance", "realized_volatility",
                             "abs_return", "high_low_range",
                             "max_intrahorizon_move",
                             "touch_call_wall", "touch_put_wall",
                             "touch_gamma_flip", "cross_gamma_flip",
                             "range_survive"):
                    out[f"{name}_{key}"] = None
                continue

            hi = self.high[start:end + 1]
            lo = self.low[start:end + 1]
            cl = self.close[start:end + 1]

            # ---- excursions (§9.3) ----
            up_mfe = _ln(float(hi.max()), spot)
            up_mae = _ln(float(lo.min()), spot)
            out[f"up_mfe_{key}"] = up_mfe
            out[f"up_mae_{key}"] = up_mae
            out[f"down_mfe_{key}"] = -up_mae
            out[f"down_mae_{key}"] = -up_mfe

            # ---- volatility (§9.4) ----
            path = np.concatenate(([spot], cl))
            rets = np.diff(np.log(path))
            var = float(np.sum(rets ** 2))
            out[f"realized_variance_{key}"] = var
            out[f"realized_volatility_{key}"] = math.sqrt(var)
            out[f"abs_return_{key}"] = abs(fwd) if fwd is not None else None
            out[f"high_low_range_{key}"] = _ln(float(hi.max()), float(lo.min()))
            out[f"max_intrahorizon_move_{key}"] = max(abs(up_mfe), abs(up_mae))

            # ---- frozen-level touches (§9.5) ----
            out[f"touch_call_wall_{key}"] = (
                int(bool(np.any(hi >= call_wall))) if call_wall is not None else None)
            out[f"touch_put_wall_{key}"] = (
                int(bool(np.any(lo <= put_wall))) if put_wall is not None else None)
            if gamma_flip is not None:
                out[f"touch_gamma_flip_{key}"] = int(bool(
                    np.any((lo <= gamma_flip) & (hi >= gamma_flip))))
                side0 = 1.0 if spot > gamma_flip else (-1.0 if spot < gamma_flip else 0.0)
                if side0:
                    crossed = bool(np.any(np.sign(cl - gamma_flip) == -side0))
                else:                     # starting AT the flip: any strict side
                    crossed = bool(np.any(cl != gamma_flip))
                out[f"cross_gamma_flip_{key}"] = int(crossed)
            else:
                out[f"touch_gamma_flip_{key}"] = None
                out[f"cross_gamma_flip_{key}"] = None

            # ---- wall-channel survival (§9.6) ----
            out[f"range_survive_{key}"] = (
                range_survival(hi, lo, put_wall, call_wall)
                if call_wall is not None and put_wall is not None else None)

        # ---- remaining-session realized move (§9.4, simple returns) ----
        if start < len(self.ts):
            hi_rem = float(self.high[start:].max())
            lo_rem = float(self.low[start:].min())
            out["remaining_realized_move"] = max(abs(hi_rem / spot - 1.0),
                                                 abs(lo_rem / spot - 1.0))
        else:
            out["remaining_realized_move"] = None

        # ---- wall first-passage to close (§9.5/9.7) ----
        out.update(self._wall_first_passage(obs, start, call_wall, put_wall,
                                            gamma_flip))
        return out

    def _wall_first_passage(self, obs: np.datetime64, start: int,
                            call_wall: Optional[float],
                            put_wall: Optional[float],
                            gamma_flip: Optional[float]) -> dict:
        out = {"call_wall_first": None, "put_wall_first": None,
               "neither_wall": None, "wall_first_ambiguous": None,
               "time_to_call_wall": None, "time_to_put_wall": None,
               "time_to_flip": None}
        hi = self.high[start:]
        lo = self.low[start:]

        if call_wall is not None:
            idx = np.nonzero(hi >= call_wall)[0]
            if len(idx):
                out["time_to_call_wall"] = self._minutes_from(obs, start + int(idx[0]))
        if put_wall is not None:
            idx = np.nonzero(lo <= put_wall)[0]
            if len(idx):
                out["time_to_put_wall"] = self._minutes_from(obs, start + int(idx[0]))
        if gamma_flip is not None:
            idx = np.nonzero((lo <= gamma_flip) & (hi >= gamma_flip))[0]
            if len(idx):
                out["time_to_flip"] = self._minutes_from(obs, start + int(idx[0]))

        if call_wall is not None and put_wall is not None:
            fp = first_passage(hi, lo,
                               [self._minutes_from(obs, start + i)
                                for i in range(len(hi))],
                               target=call_wall, stop=put_wall, direction="up")
            ev = fp["first_event"]
            out["wall_first_ambiguous"] = fp["ambiguous_same_bar"]
            if ev == "ambiguous":
                # both walls inside one bar: never assume an ordering
                out["neither_wall"] = 0
            else:
                out["call_wall_first"] = 1 if ev == "target" else 0
                out["put_wall_first"] = 1 if ev == "stop" else 0
                out["neither_wall"] = 1 if ev == "neither" else 0
        return out


# --------------------------------------------------------------------------- #
# Candidate outcome labels (§9.8)                                              #
# --------------------------------------------------------------------------- #
def _intrinsic(strike: float, kind: str, S: float) -> float:
    return max(S - strike, 0.0) if kind == "C" else max(strike - S, 0.0)


def _structure_value(legs: list, credit: float, S: float) -> float:
    """P&L per share of holding the structure with underlying at S."""
    total = float(credit)
    for lg in legs:
        total += lg["qty"] * _intrinsic(lg["strike"], lg["kind"], S)
    return total


def candidate_outcome_labels(legs: list, credit: float,
                             settle_price: float, *,
                             max_loss: Optional[float] = None,
                             capital: Optional[float] = None,
                             labeler: Optional[SessionLabeler] = None,
                             entry_ts: Optional[dt.datetime] = None,
                             target_pnl: Optional[float] = None,
                             stop_pnl: Optional[float] = None) -> dict:
    """
    Outcome record for one candidate (§9.8). Settlement P&L uses midpoint
    entry economics (pnl_mid); expected/conservative-fill P&L are recorded as
    None until the execution-cost model lands (PR 6) — absent, never faked.

    Path P&L (MFE/MAE, target/stop) is approximated by marking the structure
    intrinsically at each bar's high, low and close — a lower bound on option
    marks (no extrinsic value), documented and consistent across candidates.
    Same-bar target+stop is ambiguous and conservatively resolved to the stop.
    """
    pnl_mid = _structure_value(legs, credit, settle_price)
    out = {
        "settled": 1,
        "settlement_price": float(settle_price),
        "pnl_mid": pnl_mid,
        "pnl_expected_fill": None,       # PR 6 (execution-cost model)
        "pnl_conservative": None,        # PR 6
        "pnl_policy": None,              # PR 6 (exit-policy simulation)
        "mfe": None, "mae": None,
        "target_hit": None, "stop_hit": None,
        "first_event": None, "ambiguous_same_bar": None,
        "time_in_trade_min": None,
        "return_on_risk": (pnl_mid / max_loss
                           if max_loss is not None and max_loss > 0 else None),
        "return_on_capital": (pnl_mid / capital
                              if capital is not None and capital > 0 else None),
    }
    if labeler is None or entry_ts is None:
        return out

    obs = labeler._obs64(entry_ts)
    start = labeler._future_start_idx(obs)
    if start >= len(labeler.ts):
        return out

    marks = []
    for i in range(start, len(labeler.ts)):
        vals = (_structure_value(legs, credit, float(labeler.high[i])),
                _structure_value(legs, credit, float(labeler.low[i])),
                _structure_value(legs, credit, float(labeler.close[i])))
        marks.append((min(vals), max(vals),
                      labeler._minutes_from(obs, i)))

    out["mfe"] = max(m[1] for m in marks)
    out["mae"] = min(m[0] for m in marks)
    out["time_in_trade_min"] = marks[-1][2]

    if target_pnl is not None and stop_pnl is not None:
        out["target_hit"] = int(any(m[1] >= target_pnl for m in marks))
        out["stop_hit"] = int(any(m[0] <= stop_pnl for m in marks))
        out["first_event"] = "neither"
        out["ambiguous_same_bar"] = 0
        for lo_v, hi_v, mins in marks:
            hit_t, hit_s = hi_v >= target_pnl, lo_v <= stop_pnl
            if hit_t and hit_s:
                out["first_event"] = "stop"        # conservative: adverse first
                out["ambiguous_same_bar"] = 1
                out["time_in_trade_min"] = mins
                break
            if hit_t or hit_s:
                out["first_event"] = "target" if hit_t else "stop"
                out["time_in_trade_min"] = mins
                break
    return out
