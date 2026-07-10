"""
tests/test_asof_builder.py
==========================
PR 3 acceptance — as-of source rules and stable observation identity:
  * future bars and quotes are REJECTED (AsOfViolation), never silently used;
  * missing values stay missing (no neutral imputation);
  * snapshot ids are stable and deterministic;
  * session metadata distinguishes regular / early-close / non-sessions.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from prediction.asof import (AsOfFeatureBuilder, AsOfViolation, bars_asof,
                             ensure_asof)
from prediction.dataset import (FEATURE_VERSION, make_snapshot_id,
                                normalize_ts, session_metadata)
from resample import RawBars

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def _bars(start: dt.datetime, n: int, price0: float = 600.0) -> RawBars:
    ts = (np.datetime64(start.replace(tzinfo=None)) +
          np.arange(n) * np.timedelta64(1, "m"))
    close = price0 + 0.01 * np.arange(n)
    return RawBars(ts=ts.astype("datetime64[ns]"), open=close, high=close + 0.05,
                   low=close - 0.05, close=close, volume=np.ones(n))


# --------------------------------------------------------------------------- #
# ensure_asof                                                                  #
# --------------------------------------------------------------------------- #
class TestEnsureAsof:
    def test_past_source_returns_age(self):
        obs = dt.datetime(2026, 7, 6, 14, 0, tzinfo=UTC)
        src = dt.datetime(2026, 7, 6, 13, 59, 30, tzinfo=UTC)
        assert ensure_asof("bar", src, obs) == pytest.approx(30.0)

    def test_equal_timestamp_allowed(self):
        obs = dt.datetime(2026, 7, 6, 14, 1, tzinfo=UTC)
        assert ensure_asof("bar", obs, obs) == 0.0

    def test_future_source_rejected(self):
        obs = dt.datetime(2026, 7, 6, 14, 1, 0, tzinfo=UTC)
        quote = dt.datetime(2026, 7, 6, 14, 1, 7, tzinfo=UTC)
        with pytest.raises(AsOfViolation):
            ensure_asof("option_quote", quote, obs)

    def test_timezone_mix_normalized(self):
        # 10:01 ET == 14:01 UTC — the same instant must not be "future"
        obs = dt.datetime(2026, 7, 6, 14, 1, tzinfo=UTC)
        src = dt.datetime(2026, 7, 6, 10, 1, tzinfo=ET)
        assert ensure_asof("x", src, obs) == 0.0


# --------------------------------------------------------------------------- #
# bars_asof                                                                    #
# --------------------------------------------------------------------------- #
class TestBarsAsOf:
    def test_future_bars_truncated(self):
        start = dt.datetime(2026, 7, 6, 13, 30)
        bars = _bars(start, 60)
        obs = dt.datetime(2026, 7, 6, 13, 59, tzinfo=UTC)   # bar 29 ends 13:59
        out = bars_asof(bars, obs)
        assert len(out.ts) == 30                            # 13:30..13:59 inclusive
        assert out.ts[-1] == np.datetime64("2026-07-06T13:59:00", "ns")

    def test_bar_ending_at_observation_is_usable(self):
        # spec: a one-minute bar ending at 10:01 may be used at 10:01
        start = dt.datetime(2026, 7, 6, 13, 30)
        bars = _bars(start, 60)
        obs = dt.datetime(2026, 7, 6, 13, 30, tzinfo=UTC)
        out = bars_asof(bars, obs)
        assert len(out.ts) == 1

    def test_all_future_yields_empty(self):
        start = dt.datetime(2026, 7, 6, 13, 30)
        bars = _bars(start, 10)
        obs = dt.datetime(2026, 7, 6, 13, 0, tzinfo=UTC)
        assert len(bars_asof(bars, obs).ts) == 0

    def test_inserting_future_bar_does_not_change_snapshot(self):
        # leakage test: appending a future bar to the recording must not
        # change what an earlier observation sees
        start = dt.datetime(2026, 7, 6, 13, 30)
        obs = dt.datetime(2026, 7, 6, 13, 45, tzinfo=UTC)
        a = bars_asof(_bars(start, 16), obs)
        b = bars_asof(_bars(start, 300), obs)
        assert np.array_equal(a.ts, b.ts)
        assert np.array_equal(a.close, b.close)


# --------------------------------------------------------------------------- #
# AsOfFeatureBuilder                                                           #
# --------------------------------------------------------------------------- #
class TestFeatureBuilder:
    def setup_method(self):
        self.obs = dt.datetime(2026, 7, 6, 14, 1, tzinfo=UTC)

    def test_observed_value_recorded_with_age(self):
        b = AsOfFeatureBuilder(observation_ts=self.obs)
        b.add("adx", 22.5, source_ts=self.obs - dt.timedelta(seconds=12))
        out = b.build()
        assert out["features"]["adx"] == 22.5
        assert out["missingness"]["adx"] == 0
        assert out["source_ages"]["adx"] == pytest.approx(12.0)

    def test_future_quote_rejected(self):
        b = AsOfFeatureBuilder(observation_ts=self.obs)
        with pytest.raises(AsOfViolation):
            b.add("chain_mid", 3.1,
                  source_ts=self.obs + dt.timedelta(seconds=7))

    def test_missing_values_stay_missing(self):
        b = AsOfFeatureBuilder(observation_ts=self.obs)
        b.add("vix", None)
        b.add("cvd_slope", float("nan"))
        b.add_missing("rsp_spy_div")
        out = b.build()
        for k in ("vix", "cvd_slope", "rsp_spy_div"):
            assert out["features"][k] is None
            assert out["missingness"][k] == 1
            assert out["source_ages"][k] is None

    def test_coverage_fraction(self):
        b = AsOfFeatureBuilder(observation_ts=self.obs)
        b.add("a", 1.0)
        b.add("b", None)
        b.add("c", 2.0)
        b.add("d", None)
        assert b.build()["coverage"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Stable snapshot ids                                                          #
# --------------------------------------------------------------------------- #
class TestSnapshotId:
    TS = dt.datetime(2026, 7, 6, 10, 1, tzinfo=ET)

    def test_deterministic(self):
        a = make_snapshot_id("SPY", self.TS, FEATURE_VERSION, 7)
        b = make_snapshot_id("SPY", self.TS, FEATURE_VERSION, 7)
        assert a == b and len(a) == 64

    def test_distinct_by_component(self):
        base = make_snapshot_id("SPY", self.TS, FEATURE_VERSION, 7)
        assert make_snapshot_id("QQQ", self.TS, FEATURE_VERSION, 7) != base
        assert make_snapshot_id("SPY", self.TS + dt.timedelta(minutes=1),
                                FEATURE_VERSION, 7) != base
        assert make_snapshot_id("SPY", self.TS, "v9.9.9", 7) != base
        assert make_snapshot_id("SPY", self.TS, FEATURE_VERSION, 8) != base

    def test_timezone_representation_irrelevant(self):
        # the same instant in UTC and ET must hash identically
        utc = self.TS.astimezone(UTC)
        assert (make_snapshot_id("SPY", self.TS, FEATURE_VERSION, 0)
                == make_snapshot_id("SPY", utc, FEATURE_VERSION, 0))

    def test_normalize_ts_second_precision(self):
        a = self.TS.replace(microsecond=123456)
        assert normalize_ts(a) == normalize_ts(self.TS)


# --------------------------------------------------------------------------- #
# Session identity                                                             #
# --------------------------------------------------------------------------- #
class TestSessionMetadata:
    def test_regular_session(self):
        ts = dt.datetime(2026, 7, 10, 10, 30, tzinfo=ET)     # Friday
        m = session_metadata(ts)
        assert m["is_session"] is True
        assert m["is_early_close"] is False
        assert m["minutes_since_open"] == pytest.approx(60.0)
        assert m["minutes_to_close"] == pytest.approx(330.0)
        assert m["day_of_week"] == 4

    def test_early_close_session(self):
        ts = dt.datetime(2026, 11, 27, 10, 30, tzinfo=ET)    # day after Thanksgiving
        m = session_metadata(ts)
        assert m["is_session"] is True
        assert m["is_early_close"] is True
        assert m["minutes_to_close"] < 240.0                 # 13:00 ET close

    def test_weekend_is_not_a_session(self):
        ts = dt.datetime(2026, 7, 11, 10, 30, tzinfo=ET)     # Saturday
        m = session_metadata(ts)
        assert m["is_session"] is False
        assert m["minutes_since_open"] is None
        assert m["minutes_to_close"] is None
        assert m["session_date"] == "2026-07-11"
