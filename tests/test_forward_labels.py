"""
tests/test_forward_labels.py
============================
PR 3 acceptance — multi-horizon forward-return, direction, excursion, and
volatility labels:
  * log-return convention against known deterministic paths;
  * horizons extending past the session close become None (never truncated);
  * terminal price tolerance is one base bar;
  * MFE/MAE and remaining-realized-move match hand computations.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from prediction.labels import (DEFAULT_MIN_RETURN, SessionLabeler,
                               direction_label)

UTC = dt.timezone.utc
SESSION_START = dt.datetime(2026, 7, 6, 13, 30)      # naive UTC bar clock


def _labeler(closes, highs=None, lows=None, start=SESSION_START,
             step_min=1) -> SessionLabeler:
    closes = np.asarray(closes, dtype=float)
    ts = (np.datetime64(start) +
          np.arange(len(closes)) * np.timedelta64(step_min, "m"))
    return SessionLabeler(
        ts=ts.astype("datetime64[ns]"),
        high=np.asarray(highs if highs is not None else closes, dtype=float),
        low=np.asarray(lows if lows is not None else closes, dtype=float),
        close=closes,
    )


def _obs(minute: int) -> dt.datetime:
    return (SESSION_START + dt.timedelta(minutes=minute)).replace(tzinfo=UTC)


class TestForwardReturns:
    def test_log_return_at_each_horizon(self):
        closes = [600.0 + 0.1 * i for i in range(390)]
        lab = _labeler(closes)
        spot = closes[10]                              # observe at bar 10
        out = lab.label_observation(_obs(10), spot)
        for h, mins in (("5m", 5), ("15m", 15), ("30m", 30), ("60m", 60)):
            expected = math.log(closes[10 + mins] / spot)
            assert out[f"fwd_return_{h}"] == pytest.approx(expected), h
        assert out["fwd_return_close"] == pytest.approx(
            math.log(closes[-1] / spot))

    def test_horizon_past_close_is_none(self):
        closes = [600.0] * 60                          # one-hour session
        lab = _labeler(closes)
        out = lab.label_observation(_obs(40), 600.0)
        assert out["fwd_return_5m"] is not None
        assert out["fwd_return_15m"] is not None
        assert out["fwd_return_30m"] is None           # 40+30 > 59
        assert out["fwd_return_60m"] is None
        assert out["up_30m"] is None
        assert out["direction_30m"] is None
        assert out["up_mfe_30m"] is None
        assert out["realized_variance_30m"] is None

    def test_close_horizon_uses_last_bar(self):
        closes = [600.0] * 100
        closes[-1] = 606.0
        lab = _labeler(closes)
        out = lab.label_observation(_obs(50), 600.0)
        assert out["fwd_return_close"] == pytest.approx(math.log(606.0 / 600.0))

    def test_gap_beyond_tolerance_is_none(self):
        # bars every 5 minutes: the 5m boundary lands on a bar, but the
        # 15m boundary's nearest following bar may exceed the 1-bar tolerance
        closes = [600.0] * 30
        lab = _labeler(closes, step_min=5)             # bars at 0,5,10,...
        obs = (SESSION_START + dt.timedelta(minutes=3)).replace(tzinfo=UTC)
        out = lab.label_observation(obs, 600.0)
        # boundary 3+5=8 -> next bar at 10, gap 2min > 1min tolerance
        assert out["fwd_return_5m"] is None

    def test_observation_at_last_bar_everything_none(self):
        closes = [600.0] * 30
        lab = _labeler(closes)
        out = lab.label_observation(_obs(29), 600.0)
        assert out["fwd_return_close"] is None
        assert out["remaining_realized_move"] is None


class TestDirectionLabels:
    def test_up_binary(self):
        closes = [600.0] * 60
        closes[40] = 600.5                             # up at 30m from obs 10
        lab = _labeler(closes)
        out = lab.label_observation(_obs(10), 600.0)
        assert out["up_30m"] == 1
        assert out["up_5m"] == 0                       # flat = not up

    def test_actionable_threshold(self):
        # a +1bp move is above the 2bp default? No: 0.0001 < 0.0002 -> 0
        assert direction_label(0.0001) == 0
        assert direction_label(0.0025) == 1
        assert direction_label(-0.0025) == -1
        # implied-move fraction dominates when larger
        assert direction_label(0.0004,
                               implied_remaining_move=0.02,
                               move_fraction=0.05) == 0     # thr = 0.001
        assert direction_label(0.0015,
                               implied_remaining_move=0.02,
                               move_fraction=0.05) == 1
        assert direction_label(None) is None
        assert direction_label(0.0001,
                               min_return=DEFAULT_MIN_RETURN) == 0


class TestExcursions:
    def test_mfe_mae_from_shaped_path(self):
        # rally to 603 at bar 20, dip to 598 at bar 40, close 601
        closes = [600.0] * 120
        highs = [600.0] * 120
        lows = [600.0] * 120
        highs[20] = 603.0
        lows[40] = 598.0
        closes[-1] = 601.0
        highs[-1] = lows[-1] = 601.0
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0)
        assert out["up_mfe_close"] == pytest.approx(math.log(603.0 / 600.0))
        assert out["up_mae_close"] == pytest.approx(math.log(598.0 / 600.0))
        assert out["down_mfe_close"] == pytest.approx(-math.log(598.0 / 600.0))
        assert out["down_mae_close"] == pytest.approx(-math.log(603.0 / 600.0))

    def test_window_respects_horizon(self):
        # the dip at bar 40 must NOT appear in the 30m window from bar 0
        closes = [600.0] * 120
        lows = [600.0] * 120
        lows[40] = 595.0
        lab = _labeler(closes, None, lows)
        out = lab.label_observation(_obs(0), 600.0)
        assert out["up_mae_30m"] == pytest.approx(0.0)
        assert out["up_mae_60m"] == pytest.approx(math.log(595.0 / 600.0))


class TestVolatilityLabels:
    def test_realized_variance_matches_hand_sum(self):
        closes = [600.0, 600.6, 600.0, 600.9, 600.3, 600.3, 600.3]
        lab = _labeler(closes)
        out = lab.label_observation(_obs(0), closes[0])
        path = np.log(np.array(closes))
        var = float(np.sum(np.diff(path) ** 2))
        assert out["realized_variance_close"] == pytest.approx(var)
        assert out["realized_volatility_close"] == pytest.approx(math.sqrt(var))
        assert out["abs_return_close"] == pytest.approx(
            abs(math.log(closes[-1] / closes[0])))

    def test_remaining_realized_move(self):
        closes = [600.0] * 50
        highs = [600.0] * 50
        lows = [600.0] * 50
        highs[30] = 604.5                              # +0.75%
        lows[45] = 597.0                               # -0.50%
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0)
        assert out["remaining_realized_move"] == pytest.approx(0.0075)

    def test_high_low_range(self):
        closes = [600.0] * 40
        highs = [601.0] * 40
        lows = [599.0] * 40
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0)
        assert out["high_low_range_30m"] == pytest.approx(math.log(601.0 / 599.0))
        assert out["max_intrahorizon_move_30m"] == pytest.approx(
            max(abs(math.log(601.0 / 600.0)), abs(math.log(599.0 / 600.0))))


class TestNoFutureLeak:
    def test_bar_at_observation_excluded_from_future_window(self):
        # the bar ENDING at the observation is history, not future path
        closes = [600.0] * 60
        highs = [600.0] * 60
        highs[10] = 605.0                              # spike ON the obs bar
        lab = _labeler(closes, highs)
        out = lab.label_observation(_obs(10), 600.0)
        assert out["up_mfe_30m"] == pytest.approx(0.0)
