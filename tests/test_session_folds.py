"""
Session-unit fold construction (validation/session_folds.py, PR 1 of
Prediction Engine V2): folds are built from complete session dates, an
embargo of whole sessions separates warm-up from test, no session is ever
split, and the holdout carve reserves complete sessions.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from validation.session_folds import (
    SessionSpan, make_session_folds, session_date, session_spans,
    split_holdout_by_sessions,
)

ET = ZoneInfo("America/New_York")


def _ticks(n_sessions: int, ticks_per_session: int = 10,
           start: dt.date = dt.date(2026, 6, 1)) -> list[dt.datetime]:
    """Consecutive weekday sessions, `ticks_per_session` minutes from 09:30."""
    out = []
    day = start
    for _ in range(n_sessions):
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        for m in range(ticks_per_session):
            out.append(dt.datetime(day.year, day.month, day.day, 9, 30 + m,
                                   tzinfo=ET))
        day += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# session grouping                                                             #
# --------------------------------------------------------------------------- #
def test_session_spans_groups_contiguously():
    ticks = _ticks(3, ticks_per_session=5)
    spans = session_spans(ticks)
    assert len(spans) == 3
    assert [s.n_ticks for s in spans] == [5, 5, 5]
    # spans tile the timeline exactly
    assert spans[0].start == 0
    assert spans[-1].end == len(ticks)
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start
    # every tick inside a span belongs to that session date
    for s in spans:
        assert all(session_date(t) == s.date for t in ticks[s.start:s.end])


def test_session_spans_rejects_unsorted():
    ticks = _ticks(2)
    ticks[0], ticks[-1] = ticks[-1], ticks[0]
    with pytest.raises(ValueError, match="not sorted"):
        session_spans(ticks)


def test_session_spans_sorted_input_yields_unique_sessions():
    a, b = _ticks(1), _ticks(1, start=dt.date(2026, 6, 2))
    # day 1, day 2, then day 1 again is unsorted — the guard fires before any
    # session could be split into two runs
    with pytest.raises(ValueError, match="not sorted"):
        session_spans(a[:5] + b + a[5:])
    dates = [s.date for s in session_spans(a + b)]
    assert dates == sorted(set(dates))


# --------------------------------------------------------------------------- #
# fold construction invariants                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_no_session_split_or_overlap(mode):
    ticks = _ticks(10)
    spans = session_spans(ticks)
    boundaries = {s.start for s in spans} | {s.end for s in spans}
    folds = make_session_folds(ticks, mode=mode, n_folds=3, train_frac=0.6,
                               embargo_sessions=1)
    assert folds
    for f in folds:
        # tick ranges land exactly on session boundaries — no session split
        assert {f.warm_start, f.warm_end, f.test_start, f.test_end} <= boundaries
        # no session on both sides of the fold
        assert not (set(f.warm_sessions) & set(f.test_sessions))
        assert not (set(f.embargoed_sessions) & set(f.test_sessions))
        assert not (set(f.warm_sessions) & set(f.embargoed_sessions))
        assert f.test_sessions  # every fold has at least one test session


def test_embargo_separates_warm_and_test():
    ticks = _ticks(12)
    for embargo in (0, 1, 2):
        folds = make_session_folds(ticks, mode="expanding", n_folds=2,
                                   train_frac=0.5, embargo_sessions=embargo)
        assert folds
        for f in folds:
            # warm ends exactly `embargo` complete sessions before test starts
            assert len(f.embargoed_sessions) == embargo
            ordered = f.warm_sessions + f.embargoed_sessions + f.test_sessions
            assert list(ordered) == sorted(ordered)


def test_expanding_test_windows_tile_forward():
    ticks = _ticks(10)
    folds = make_session_folds(ticks, mode="expanding", n_folds=3,
                               train_frac=0.6, embargo_sessions=1)
    # test windows are consecutive and non-overlapping, ending at the last tick
    seen = []
    for f in folds:
        seen.extend(f.test_sessions)
    assert len(seen) == len(set(seen))
    assert folds[-1].test_end == len(ticks)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start
        # expanding: warm-up grows monotonically
        assert b.warm_end >= a.warm_end


def test_initial_warm_sessions_pins_first_test_session():
    ticks = _ticks(8)
    spans = session_spans(ticks)
    folds = make_session_folds(ticks, mode="expanding", n_folds=1,
                               embargo_sessions=1, initial_warm_sessions=6)
    assert len(folds) == 1
    f = folds[0]
    assert f.test_sessions == (spans[6].date, spans[7].date)
    assert f.test_start == spans[6].start
    # embargo eats the last warm session
    assert f.embargoed_sessions == (spans[5].date,)
    assert f.warm_sessions == tuple(s.date for s in spans[:5])


def test_zero_embargo_keeps_full_warm_window():
    ticks = _ticks(6)
    folds = make_session_folds(ticks, mode="expanding", n_folds=1,
                               train_frac=0.5, embargo_sessions=0)
    f = folds[0]
    assert f.embargoed_sessions == ()
    assert f.warm_end == f.test_start


# --------------------------------------------------------------------------- #
# holdout carve                                                                #
# --------------------------------------------------------------------------- #
def test_holdout_reserves_complete_final_sessions():
    ticks = _ticks(8, ticks_per_session=7)
    search, hold = split_holdout_by_sessions(ticks, 0.25)
    assert search + hold == ticks
    # 25% of 8 sessions = 2 complete sessions
    assert len(session_spans(hold)) == 2
    assert len(session_spans(search)) == 6
    # boundary is a session boundary: no shared session date
    assert not ({session_date(t) for t in search}
                & {session_date(t) for t in hold})


def test_holdout_never_partial_session():
    # awkward fraction: 30% of 7 sessions -> 2 complete sessions, not 2.1
    ticks = _ticks(7, ticks_per_session=9)
    search, hold = split_holdout_by_sessions(ticks, 0.30)
    assert len(session_spans(hold)) == 2
    assert len(hold) == 2 * 9


def test_holdout_degenerate_inputs():
    ticks = _ticks(1)
    search, hold = split_holdout_by_sessions(ticks, 0.5)
    assert hold == [] and search == ticks
    search, hold = split_holdout_by_sessions(ticks, 0.0)
    assert hold == [] and search == ticks
