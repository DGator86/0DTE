"""
validation/session_folds.py
===========================
Build walk-forward folds from COMPLETE trading sessions.

Why this exists (docs/PREDICTION_ENGINE_V2_HANDOFF.md §3.1–3.2, §18)
--------------------------------------------------------------------
The legacy fold builder divided the tick timeline by observation index, which
allowed the morning of a session to land in warm-up while the afternoon of
the SAME session landed in the test window — model state and the shared
terminal settlement leaked across the boundary. Here folds are built from
whole session dates (America/New_York), so:

  * no session is ever split between warm-up and test,
  * no session appears on both sides of a fold,
  * a configurable EMBARGO of complete sessions separates the end of warm-up
    from the start of test (default one session).

The embargo is taken out of the warm side: warm-up covers sessions strictly
before ``test_start_session - embargo``. Because every label in this system
settles at its own session close, dropping the embargo sessions also purges
any label whose evaluation window could brush the test boundary.

All functions operate on a chronologically sorted list of aware datetimes and
return TICK INDEX ranges aligned to session boundaries, so callers can keep
slicing their existing timestamp lists.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def session_date(ts: dt.datetime) -> str:
    """Exchange-local (America/New_York) session date for a tick."""
    return ts.astimezone(ET).date().isoformat()


@dataclass(frozen=True)
class SessionSpan:
    """One complete session's contiguous tick range: [start, end)."""
    date: str
    start: int
    end: int

    @property
    def n_ticks(self) -> int:
        return self.end - self.start


def session_spans(timestamps: list[dt.datetime]) -> list[SessionSpan]:
    """
    Group a sorted tick sequence into per-session contiguous spans.
    Raises ValueError when timestamps are not chronologically sorted — that
    would make "complete session" folds meaningless, so fail loudly instead
    of building corrupt folds. (Sorted timestamps guarantee each session's
    ticks are contiguous, since the ET session date is monotone in time.)
    """
    spans: list[SessionSpan] = []
    cur_date: str | None = None
    cur_start = 0
    prev: dt.datetime | None = None

    for i, t in enumerate(timestamps):
        if prev is not None and t < prev:
            raise ValueError(
                f"timestamps not sorted at index {i}: {t} < {prev}")
        prev = t
        d = session_date(t)
        if d != cur_date:
            if cur_date is not None:
                spans.append(SessionSpan(cur_date, cur_start, i))
            cur_date, cur_start = d, i
    if cur_date is not None:
        spans.append(SessionSpan(cur_date, cur_start, len(timestamps)))
    return spans


@dataclass(frozen=True)
class SessionFold:
    """
    One walk-forward fold expressed in both session and tick-index units.
    Tick ranges are half-open [start, end) into the original timestamp list.
    An empty warm window is represented by warm_start == warm_end.
    """
    warm_start: int
    warm_end: int
    test_start: int
    test_end: int
    warm_sessions: tuple[str, ...]
    embargoed_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def make_session_folds(
    timestamps: list[dt.datetime],
    mode: str = "expanding",
    n_folds: int = 5,
    train_frac: float = 0.6,
    embargo_sessions: int = 1,
    initial_warm_sessions: int | None = None,
) -> list[SessionFold]:
    """
    Session-unit analogue of the legacy tick-index fold builder.

    expanding — warm-up always starts at the first session and grows; the
                test window slides forward one fold at a time.
    rolling   — warm-up and test windows are (roughly) fixed session counts
                that slide forward together.

    initial_warm_sessions (expanding only) pins the first test session
    exactly — used by the session-based holdout evaluation, where the test
    window must be precisely the held-out sessions.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown mode: {mode!r}")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be >= 0")

    spans = session_spans(timestamps)
    n_sess = len(spans)
    if n_sess == 0:
        return []

    # (warm_start_s, test_start_s, test_end_s) in SESSION units
    session_triples: list[tuple[int, int, int]] = []
    if mode == "expanding":
        warm0 = (initial_warm_sessions if initial_warm_sessions is not None
                 else int(n_sess * train_frac))
        warm0 = max(0, min(warm0, n_sess))
        remaining = n_sess - warm0
        fold_size = max(1, remaining // max(1, n_folds))
        for i in range(n_folds):
            test_start_s = warm0 + i * fold_size
            test_end_s = (test_start_s + fold_size) if i < n_folds - 1 else n_sess
            if test_start_s >= n_sess:
                break
            session_triples.append((0, test_start_s, min(test_end_s, n_sess)))
    else:  # rolling
        fold_size = max(1, n_sess // max(1, n_folds))
        warm_size = max(1, round(fold_size * train_frac / max(1e-9, 1.0 - train_frac)))
        for i in range(n_folds):
            test_start_s = i * fold_size
            test_end_s = (test_start_s + fold_size) if i < n_folds - 1 else n_sess
            if test_start_s >= n_sess:
                break
            warm_start_s = max(0, test_start_s - embargo_sessions - warm_size)
            session_triples.append((warm_start_s, test_start_s, min(test_end_s, n_sess)))

    folds: list[SessionFold] = []
    for warm_start_s, test_start_s, test_end_s in session_triples:
        warm_end_s = max(warm_start_s, test_start_s - embargo_sessions)
        warm = spans[warm_start_s:warm_end_s]
        embargoed = spans[warm_end_s:test_start_s]
        test = spans[test_start_s:test_end_s]
        if not test:
            continue
        folds.append(SessionFold(
            warm_start=warm[0].start if warm else test[0].start,
            warm_end=warm[-1].end if warm else test[0].start,
            test_start=test[0].start,
            test_end=test[-1].end,
            warm_sessions=tuple(s.date for s in warm),
            embargoed_sessions=tuple(s.date for s in embargoed),
            test_sessions=tuple(s.date for s in test),
        ))
    return folds


def split_holdout_by_sessions(
    timestamps: list[dt.datetime], holdout_frac: float
) -> tuple[list[dt.datetime], list[dt.datetime]]:
    """
    Reserve the FINAL fraction of complete sessions (never a partial session)
    as an untouched holdout. Returns (search_ts, holdout_ts).

    At least one session is always kept on each side; with fewer than two
    sessions there is nothing meaningful to hold out, so the holdout is empty.
    """
    if holdout_frac <= 0.0:
        return list(timestamps), []
    spans = session_spans(timestamps)
    n_sess = len(spans)
    if n_sess < 2:
        return list(timestamps), []
    n_hold = max(1, min(n_sess - 1, round(n_sess * holdout_frac)))
    cut = spans[n_sess - n_hold].start
    return list(timestamps[:cut]), list(timestamps[cut:])
