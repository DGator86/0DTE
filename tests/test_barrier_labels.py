"""
tests/test_barrier_labels.py
============================
PR 3 acceptance — wall/flip touch, first-passage, and range-survival labels:
  * structural levels are FROZEN at observation time (passed in, never
    looked up later);
  * ambiguous same-bar target/stop events are marked and conservatively
    resolved to the ADVERSE event;
  * first-passage ordering and time-to-event match hand-built paths.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from prediction.labels import SessionLabeler, first_passage, range_survival

UTC = dt.timezone.utc
SESSION_START = dt.datetime(2026, 7, 6, 13, 30)


def _labeler(closes, highs=None, lows=None) -> SessionLabeler:
    closes = np.asarray(closes, dtype=float)
    ts = (np.datetime64(SESSION_START) +
          np.arange(len(closes)) * np.timedelta64(1, "m"))
    return SessionLabeler(
        ts=ts.astype("datetime64[ns]"),
        high=np.asarray(highs if highs is not None else closes, dtype=float),
        low=np.asarray(lows if lows is not None else closes, dtype=float),
        close=closes,
    )


def _obs(minute: int) -> dt.datetime:
    return (SESSION_START + dt.timedelta(minutes=minute)).replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# first_passage                                                                #
# --------------------------------------------------------------------------- #
class TestFirstPassage:
    def test_target_first_up(self):
        out = first_passage(highs=[601, 605, 601], lows=[599, 600, 594],
                            minutes=[1, 2, 3], target=604, stop=595,
                            direction="up")
        assert out["first_event"] == "target"
        assert out["time_to_first_event"] == 2.0
        assert out["ambiguous_same_bar"] == 0

    def test_stop_first_up(self):
        out = first_passage(highs=[601, 601, 606], lows=[599, 594, 600],
                            minutes=[1, 2, 3], target=604, stop=595,
                            direction="up")
        assert out["first_event"] == "stop"
        assert out["time_to_first_event"] == 2.0

    def test_same_bar_ambiguous_resolves_conservative(self):
        # one wide bar contains BOTH levels: never assume the favorable order
        out = first_passage(highs=[606], lows=[594], minutes=[1],
                            target=604, stop=595, direction="up")
        assert out["first_event"] == "ambiguous"
        assert out["first_event_conservative"] == "stop"
        assert out["ambiguous_same_bar"] == 1

    def test_neither(self):
        out = first_passage(highs=[601, 602], lows=[599, 598],
                            minutes=[1, 2], target=610, stop=590,
                            direction="up")
        assert out["first_event"] == "neither"
        assert out["time_to_first_event"] is None

    def test_down_direction(self):
        out = first_passage(highs=[601, 601], lows=[599, 594],
                            minutes=[1, 2], target=595, stop=603,
                            direction="down")
        assert out["first_event"] == "target"

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError):
            first_passage([601], [599], [1], 604, 595, direction="sideways")


# --------------------------------------------------------------------------- #
# range_survival                                                               #
# --------------------------------------------------------------------------- #
class TestRangeSurvival:
    def test_survives_strictly_inside(self):
        assert range_survival(highs=[601, 602], lows=[599, 598],
                              lower=595, upper=605) == 1

    def test_touching_boundary_kills(self):
        # STRICTLY within: equality is a breach
        assert range_survival(highs=[605], lows=[599],
                              lower=595, upper=605) == 0
        assert range_survival(highs=[601], lows=[595],
                              lower=595, upper=605) == 0


# --------------------------------------------------------------------------- #
# Frozen-level wall / flip labels                                              #
# --------------------------------------------------------------------------- #
class TestWallFlipLabels:
    def test_call_wall_touch_and_first(self):
        closes = [600.0] * 60
        highs = [600.0] * 60
        highs[20] = 606.0                              # touch call wall at bar 20
        lab = _labeler(closes, highs)
        out = lab.label_observation(_obs(0), 600.0,
                                    call_wall=605.0, put_wall=595.0,
                                    gamma_flip=598.0)
        assert out["touch_call_wall_30m"] == 1
        assert out["touch_call_wall_15m"] == 0         # before the touch
        assert out["touch_put_wall_close"] == 0
        assert out["call_wall_first"] == 1
        assert out["put_wall_first"] == 0
        assert out["neither_wall"] == 0
        assert out["time_to_call_wall"] == 20.0
        assert out["time_to_put_wall"] is None

    def test_put_wall_first(self):
        closes = [600.0] * 60
        lows = [600.0] * 60
        highs = [600.0] * 60
        lows[10] = 594.5
        highs[30] = 605.5
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0,
                                    call_wall=605.0, put_wall=595.0)
        assert out["put_wall_first"] == 1
        assert out["call_wall_first"] == 0
        assert out["time_to_put_wall"] == 10.0
        assert out["time_to_call_wall"] == 30.0

    def test_same_bar_both_walls_ambiguous(self):
        closes = [600.0] * 30
        highs = [600.0] * 30
        lows = [600.0] * 30
        highs[5] = 606.0
        lows[5] = 594.0                                 # one bar spans both walls
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0,
                                    call_wall=605.0, put_wall=595.0)
        assert out["wall_first_ambiguous"] == 1
        assert out["call_wall_first"] is None
        assert out["put_wall_first"] is None
        assert out["neither_wall"] == 0

    def test_neither_wall(self):
        closes = [600.0] * 30
        lab = _labeler(closes)
        out = lab.label_observation(_obs(0), 600.0,
                                    call_wall=605.0, put_wall=595.0)
        assert out["neither_wall"] == 1
        assert out["call_wall_first"] == 0
        assert out["put_wall_first"] == 0

    def test_gamma_flip_touch_and_cross(self):
        closes = [600.0] * 60
        lows = [600.0] * 60
        # bar 15 dips through the flip but closes back above (touch, no cross)
        lows[15] = 597.5
        lab = _labeler(closes, None, lows)
        out = lab.label_observation(_obs(0), 600.0, gamma_flip=598.0)
        assert out["touch_gamma_flip_30m"] == 1
        assert out["cross_gamma_flip_30m"] == 0
        assert out["time_to_flip"] == 15.0

        # now close below the flip -> crossed
        closes2 = list(closes)
        closes2[20] = 597.0
        lows2 = list(lows)
        lows2[20] = 597.0
        lab2 = _labeler(closes2, None, lows2)
        out2 = lab2.label_observation(_obs(0), 600.0, gamma_flip=598.0)
        assert out2["cross_gamma_flip_30m"] == 1

    def test_levels_are_frozen_at_observation(self):
        # two observations with DIFFERENT frozen walls over the same path
        # get different labels — proof the label uses the passed-in level,
        # not any later value
        closes = [600.0] * 60
        highs = [600.0] * 60
        highs[20] = 603.0
        lab = _labeler(closes, highs)
        tight = lab.label_observation(_obs(0), 600.0,
                                      call_wall=602.0, put_wall=595.0)
        wide = lab.label_observation(_obs(0), 600.0,
                                     call_wall=610.0, put_wall=595.0)
        assert tight["touch_call_wall_close"] == 1
        assert wide["touch_call_wall_close"] == 0

    def test_missing_levels_yield_none(self):
        closes = [600.0] * 30
        lab = _labeler(closes)
        out = lab.label_observation(_obs(0), 600.0)     # no levels passed
        assert out["touch_call_wall_15m"] is None
        assert out["touch_gamma_flip_15m"] is None
        assert out["range_survive_15m"] is None
        assert out["call_wall_first"] is None

    def test_wall_channel_survival(self):
        closes = [600.0] * 60
        highs = [601.0] * 60
        lows = [599.0] * 60
        lab = _labeler(closes, highs, lows)
        out = lab.label_observation(_obs(0), 600.0,
                                    call_wall=605.0, put_wall=595.0)
        assert out["range_survive_close"] == 1
        # breach late in the session: 30m survives, close does not
        highs2 = list(highs)
        highs2[50] = 605.0
        lab2 = _labeler(closes, highs2, lows)
        out2 = lab2.label_observation(_obs(0), 600.0,
                                      call_wall=605.0, put_wall=595.0)
        assert out2["range_survive_30m"] == 1
        assert out2["range_survive_close"] == 0
