"""
market_dynamics.py
==================
Time-derivatives of the dealer surface and vol state the system already
measures — the "how is pressure CHANGING" layer on top of "where is it".

0DTE edge is often in the change, not the level: a gamma flip chasing spot,
walls migrating, premium ramping while spot sits still, or the day's expected
move already being spent. All of these are computable from data the tick loop
already holds; this module just remembers the recent past and differentiates.

Every output is OBSERVATION-ONLY on arrival: it flows into the MTF matrix as
an un-consumed row and into the journal's signals_json, where
component_correlations() scores it against realized P&L. Nothing here gates
or vetoes a trade until the data says it should (see journal.py's admission
rule). Thresholds you may be tempted to hard-code belong in that future,
data-earned step — not here.

State persists to JSON (same pattern as gex_window) so restarts don't blind
the derivatives.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DynamicsConfig:
    horizon_min: float = 5.0        # differentiation horizon (minutes)
    max_age_min: float = 90.0       # history kept (minutes)
    rupture_window_min: float = 5.0 # spot must hold beyond a wall this long


@dataclass
class DynamicsWindow:
    """Rolling market-state history -> per-tick derivative signals."""
    path: Optional[str] = None
    cfg: DynamicsConfig = field(default_factory=DynamicsConfig)
    _hist: list = field(default_factory=list)   # [{t, spot, flip, cw, pw, gex, be}]

    def __post_init__(self):
        self._load()

    # -- public -----------------------------------------------------------------
    def update(self, now_epoch: float, *, spot: float, gamma_flip: float,
               call_wall: float, put_wall: float, net_gex: float,
               straddle_be: float, session_open: Optional[float] = None) -> dict:
        """Record this tick's state and return the derivative signals.

        Values that can't be computed yet (not enough history) are simply
        absent from the dict — missing data drops out downstream, honestly.
        """
        rec = {"t": float(now_epoch), "spot": spot, "flip": gamma_flip,
               "cw": call_wall, "pw": put_wall, "gex": net_gex, "be": straddle_be}
        self._hist.append(rec)
        cutoff = now_epoch - self.cfg.max_age_min * 60.0
        self._hist = [h for h in self._hist if h["t"] >= cutoff]
        self._save()

        out: dict = {}
        past = self._at_least_ago(now_epoch, self.cfg.horizon_min * 60.0)
        if past is not None:
            dt_min = max((now_epoch - past["t"]) / 60.0, 1e-9)
            per5 = 5.0 / dt_min
            if _fin(gamma_flip, past["flip"]):
                out["flip_velocity"] = (gamma_flip - past["flip"]) * per5
                # is the flip chasing spot? +1 = converging on spot, -1 = diverging
                d_now = abs(spot - gamma_flip)
                d_then = abs(past["spot"] - past["flip"])
                out["flip_chase"] = d_then - d_now
            if _fin(call_wall, past["cw"]):
                out["call_wall_velocity"] = (call_wall - past["cw"]) * per5
            if _fin(put_wall, past["pw"]):
                out["put_wall_velocity"] = (put_wall - past["pw"]) * per5
            if _fin(net_gex, past["gex"]):
                out["gex_velocity_bn"] = (net_gex - past["gex"]) / 1e9 * per5
            if _fin(straddle_be, past["be"]) and past["be"] > 0:
                # premium ramp/crush: % change of the straddle over 5 minutes
                out["straddle_ramp"] = (straddle_be / past["be"] - 1.0) * per5

        rupture = self._wall_rupture(now_epoch)
        if rupture is not None:
            out["wall_rupture"] = rupture

        if session_open is not None and session_open > 0 and straddle_be > 0 \
                and math.isfinite(straddle_be):
            # THE cheap 0DTE number: how much of the day's implied move is
            # already spent. > ~0.7 late morning = continuation needs help.
            out["expected_move_consumed"] = abs(spot - session_open) / straddle_be

        return out

    # -- internals ----------------------------------------------------------------
    def _at_least_ago(self, now_epoch: float, seconds: float) -> Optional[dict]:
        """Newest record at least `seconds` old (None until history exists)."""
        target = now_epoch - seconds
        best = None
        for h in self._hist:
            if h["t"] <= target and (best is None or h["t"] > best["t"]):
                best = h
        return best

    def _wall_rupture(self, now_epoch: float) -> Optional[float]:
        """Signed acceptance beyond a wall: +1 = spot has held ABOVE the call
        wall for the rupture window (resistance failed), -1 = held below the
        put wall, 0 = walls holding. None until a full window of history."""
        w0 = now_epoch - self.cfg.rupture_window_min * 60.0
        window = [h for h in self._hist if h["t"] >= w0]
        if len(window) < 2 or window[0]["t"] > w0 + 60.0:
            return None
        if all(_fin(h["spot"], h["cw"]) and h["spot"] > h["cw"] for h in window):
            return 1.0
        if all(_fin(h["spot"], h["pw"]) and h["spot"] < h["pw"] for h in window):
            return -1.0
        return 0.0

    def _load(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._hist = json.load(f).get("hist", [])
        except Exception:
            self._hist = []

    def _save(self) -> None:
        if not self.path:
            return
        try:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".dyn_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"hist": self._hist}, f)
            os.replace(tmp, self.path)
        except Exception:
            pass


def _fin(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def session_open_from_bars(bars, now) -> Optional[float]:
    """First bar open of the CURRENT session (>= 09:30 ET of `now`'s date).

    Bar timestamps arrive in two naive conventions: UTC epochs (live feeds)
    and ET wall time (synthetic/replay). Rather than guess, try both cutoffs
    and keep the one whose bar count best matches the minutes actually
    elapsed since the open.
    """
    try:
        import datetime as _dt
        import numpy as np
        from zoneinfo import ZoneInfo
        et = now.astimezone(ZoneInfo("America/New_York"))
        start_et = et.replace(hour=9, minute=30, second=0, microsecond=0)
        if et < start_et:
            return None
        expected = max((et - start_et).total_seconds() / 60.0, 1.0)

        ts = np.asarray(bars.ts, dtype="datetime64[ns]")
        candidates = [
            np.datetime64(start_et.replace(tzinfo=None)),                       # ET-naive
            np.datetime64(start_et.astimezone(_dt.timezone.utc).replace(tzinfo=None)),  # UTC-naive
        ]
        best = None
        for c in candidates:
            idx = np.nonzero(ts >= c)[0]
            if idx.size == 0:
                continue
            err = abs(idx.size - expected)
            if best is None or err < best[0]:
                best = (err, idx[0])
        if best is None:
            return None
        return float(bars.open[best[1]])
    except Exception:
        return None
