"""
tests/test_turn_lag_study.py
============================
Turn detection + signal-lag measurement over synthetic journal data.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from journal import COLUMNS, Journal  # noqa: E402

ET = ZoneInfo("America/New_York")

_spec = importlib.util.spec_from_file_location(
    "turn_lag_study", os.path.join(ROOT, "scripts", "turn_lag_study.py"))
tls = importlib.util.module_from_spec(_spec)
sys.modules["turn_lag_study"] = tls   # dataclasses need the module registered
_spec.loader.exec_module(tls)


def _tick(ts, spot, bias_fast=None, blend=None, cross=None, dirword=None):
    sig = {}
    if bias_fast is not None:
        sig["bias_fast"] = bias_fast
    if blend is not None:
        sig["regime_bias_value"] = blend
    if cross is not None:
        sig["bias_cross"] = cross
    return tls.Tick(ts=ts, spot=spot, regime_direction=dirword, signals=sig)


def _series(spots, start=None):
    start = start or dt.datetime(2026, 7, 8, 10, 0, tzinfo=ET)
    return [_tick(start + dt.timedelta(minutes=i), s) for i, s in enumerate(spots)]


# --------------------------------------------------------------------------- #
# Turn detection                                                               #
# --------------------------------------------------------------------------- #
def test_detect_turns_finds_v_bottom():
    # 750 -> 745 over 20 min, then 745 -> 749.5 over 25 min (0.60% reversal)
    spots = [750 - 0.25 * i for i in range(21)] + \
            [745 + 0.18 * i for i in range(1, 26)]
    turns = tls.detect_turns(_series(spots), min_move_pct=0.3, hold_min=10)
    assert len(turns) == 1
    t = turns[0]
    assert t.direction == "up"
    assert t.spot == 745.0
    assert t.idx == 20                       # the session low is the pivot


def test_detect_turns_rejects_unheld_bounce():
    # bounce clears the move threshold but a NEW low prints 5 min later
    spots = ([750 - 0.25 * i for i in range(21)]          # decline to 745
             + [745 + 0.6 * i for i in range(1, 6)]       # sharp 0.40% bounce
             + [747 - 0.7 * i for i in range(1, 6)])      # new low inside hold
    turns = tls.detect_turns(_series(spots), min_move_pct=0.3, hold_min=10)
    assert all(t.direction != "up" or t.spot < 745 for t in turns)


def test_detect_turns_short_series_safe():
    assert tls.detect_turns(_series([750.0, 750.1]), 0.3, 10) == []


# --------------------------------------------------------------------------- #
# Lag measurement                                                              #
# --------------------------------------------------------------------------- #
def test_measure_lags_fast_leads_blend():
    """Construct a V-bottom where the fast composite flips bull 5 minutes
    after the low and the blend flips 20 minutes after: the study must
    report exactly those lags, and 'never' for a signal that never agrees."""
    start = dt.datetime(2026, 7, 8, 10, 0, tzinfo=ET)
    ticks = []
    for i in range(60):
        ts = start + dt.timedelta(minutes=i)
        spot = 750 - 0.25 * i if i <= 20 else 745 + 0.2 * (i - 20)
        mins_after_low = i - 20
        ticks.append(_tick(
            ts, spot,
            bias_fast=(62.0 if mins_after_low >= 5 else 35.0),
            blend=(58.0 if mins_after_low >= 20 else 40.0),
            cross=(1.0 if mins_after_low == 6 else None),
            dirword="put",                    # never flips: reports 'never'
        ))

    turns = tls.detect_turns(ticks, min_move_pct=0.3, hold_min=10)
    assert len(turns) == 1
    lags = tls.measure_lags("2026-07-08", ticks, turns)[0].lag_min
    assert lags["fast"] == 5.0
    assert lags["blend"] == 20.0
    assert lags["cross"] == 6.0
    assert math.isinf(lags["dirword"])       # measured but never flipped


def test_lags_report_na_when_signal_absent():
    """Sessions journaled before bias_fast existed: fast/blend report None
    (n/a), not a fake lag."""
    spots = [750 - 0.25 * i for i in range(21)] + \
            [745 + 0.2 * i for i in range(1, 30)]
    ticks = _series(spots)                   # no signals at all
    turns = tls.detect_turns(ticks, min_move_pct=0.3, hold_min=10)
    assert len(turns) == 1
    lags = tls.measure_lags("2026-07-07", ticks, turns)[0].lag_min
    assert lags["fast"] is None
    assert lags["blend"] is None


# --------------------------------------------------------------------------- #
# End to end against a real journal DB                                         #
# --------------------------------------------------------------------------- #
def test_run_study_reads_journal_db(tmp_path):
    db = str(tmp_path / "shadow.db")
    jrn = Journal(db)
    start = dt.datetime(2026, 7, 8, 10, 0, tzinfo=ET)
    for i in range(60):
        spot = 750 - 0.25 * i if i <= 20 else 745 + 0.2 * (i - 20)
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": "2026-07-08",
            "ts": (start + dt.timedelta(minutes=i)).isoformat(),
            "spot": spot,
            "decision": "NO_TRADE",
            "regime_direction": "put",
            "signals_json": json.dumps({
                "bias_fast": 62.0 if i - 20 >= 5 else 35.0,
                "regime_bias_value": 58.0 if i - 20 >= 20 else 40.0,
            }),
        })
        jrn.log(row)
    jrn.close()

    results = tls.run_study(db, session=None, min_move_pct=0.3, hold_min=10)
    assert len(results) == 1
    assert results[0].lag_min["fast"] == 5.0
    assert results[0].lag_min["blend"] == 20.0
    # the report must render without raising
    tls.print_report(results)
