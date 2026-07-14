"""
tests/test_nested_session_folds.py
==================================
Leakage and structure tests for prediction.crossfit.build_nested_session_folds
(Prediction Engine V3 Part 1 §4 / §12.1).
"""
from __future__ import annotations

import pytest

from prediction.crossfit import NestedCrossFitConfig, build_nested_session_folds


def _sessions(n: int, start: str = "2026-01-") -> list[str]:
    # Produce n weekday-like ISO dates as simple zero-padded day strings.
    return [f"{start}{i + 1:02d}" for i in range(n)]


def test_fold_definitions_are_deterministic():
    sessions = _sessions(60)
    cfg = NestedCrossFitConfig(
        outer_folds=4, inner_folds=3, embargo_sessions=1,
        min_train_sessions=10, min_validation_sessions=3, random_state=42,
    )
    a = build_nested_session_folds(sessions, cfg)
    b = build_nested_session_folds(list(reversed(sessions)), cfg)
    assert [f.fold_id for f in a] == [f.fold_id for f in b]
    for fa, fb in zip(a, b):
        assert fa.train_sessions == fb.train_sessions
        assert fa.validation_sessions == fb.validation_sessions
        assert fa.calibration_sessions == fb.calibration_sessions
        assert fa.embargoed_sessions == fb.embargoed_sessions


def test_session_never_on_both_train_and_validation():
    sessions = _sessions(50)
    cfg = NestedCrossFitConfig(
        outer_folds=4, embargo_sessions=1,
        min_train_sessions=8, min_validation_sessions=3,
    )
    for fd in build_nested_session_folds(sessions, cfg):
        assert not (set(fd.train_sessions) & set(fd.validation_sessions))
        assert not (set(fd.calibration_sessions) & set(fd.validation_sessions))
        assert not (set(fd.train_sessions) & set(fd.calibration_sessions))


def test_calibration_sessions_are_not_test_sessions():
    sessions = _sessions(50)
    cfg = NestedCrossFitConfig(
        outer_folds=3, embargo_sessions=1,
        min_train_sessions=8, min_validation_sessions=3,
    )
    for fd in build_nested_session_folds(sessions, cfg):
        assert not (set(fd.calibration_sessions) & set(fd.validation_sessions))


def test_embargo_absent_from_fit_and_test():
    sessions = _sessions(50)
    cfg = NestedCrossFitConfig(
        outer_folds=3, embargo_sessions=2,
        min_train_sessions=8, min_validation_sessions=3,
    )
    for fd in build_nested_session_folds(sessions, cfg):
        emb = set(fd.embargoed_sessions)
        assert not (emb & set(fd.train_sessions))
        assert not (emb & set(fd.validation_sessions))


def test_expanding_time_ordered_outer_folds():
    sessions = _sessions(40)
    cfg = NestedCrossFitConfig(
        outer_folds=3, embargo_sessions=1,
        min_train_sessions=6, min_validation_sessions=2,
    )
    folds = build_nested_session_folds(sessions, cfg)
    assert len(folds) == 3
    # Train window grows (or stays) across expanding folds
    for i in range(1, len(folds)):
        assert folds[i].train_sessions[-1] >= folds[i - 1].train_sessions[-1]
        # validation blocks move forward in time
        assert folds[i].validation_sessions[0] > folds[i - 1].validation_sessions[0]


def test_unsorted_input_is_sorted():
    sessions = _sessions(30)
    cfg = NestedCrossFitConfig(
        outer_folds=2, embargo_sessions=1,
        min_train_sessions=6, min_validation_sessions=2,
    )
    folds = build_nested_session_folds(sessions[::-1], cfg)
    for fd in folds:
        assert list(fd.train_sessions) == sorted(fd.train_sessions)
        assert list(fd.validation_sessions) == sorted(fd.validation_sessions)


def test_insufficient_sessions_raises():
    cfg = NestedCrossFitConfig(
        outer_folds=4, embargo_sessions=1,
        min_train_sessions=20, min_validation_sessions=5,
    )
    with pytest.raises(ValueError):
        build_nested_session_folds(_sessions(10), cfg)
