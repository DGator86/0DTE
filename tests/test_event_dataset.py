"""
tests/test_event_dataset.py
===========================
V3 Part 2 PR13 — competing-risk event dataset (§45).
"""
from __future__ import annotations

import pytest

from prediction.event_dataset import (
    EventDatasetConfig,
    expand_observation_to_event_rows,
)


def test_expand_target_first():
    prices = [100.0, 100.5, 101.0, 101.5, 102.0]
    minutes = [0, 1, 2, 3, 4]
    rows = expand_observation_to_event_rows(
        snapshot_id="s1", session_date="2026-07-14", origin_ts="t0",
        prices=prices, minutes=minutes,
        target=101.2, stop=99.0, direction="up",
        expected_remaining_move=2.0,
        frozen_features={"dist_flip": 0.1},
        cfg=EventDatasetConfig(horizon_minutes=10),
    )
    assert rows
    assert any(r.event_target for r in rows)
    assert not any(r.event_stop for r in rows)
    assert all(
        r.features["target_distance"] == rows[0].features["target_distance"]
        for r in rows)


def test_same_bar_adverse_first():
    # Wide bar crosses both target (above) and stop (below)
    rows = expand_observation_to_event_rows(
        snapshot_id="s", session_date="d", origin_ts="t",
        prices=[100.0, 100.0],
        highs=[100.0, 101.5],
        lows=[100.0, 98.5],
        minutes=[0, 1],
        target=101.0, stop=99.0, direction="up",
        cfg=EventDatasetConfig(same_bar_policy="adverse_first", horizon_minutes=5),
    )
    amb = [r for r in rows if r.ambiguous_same_bar]
    assert amb
    assert amb[0].event_stop == 1
    assert amb[0].event_target == 0


def test_censored_rows_retained():
    prices = [100.0, 100.1, 100.05, 100.08]
    minutes = [0, 1, 2, 3]
    rows = expand_observation_to_event_rows(
        snapshot_id="s", session_date="d", origin_ts="t",
        prices=prices, minutes=minutes,
        target=110.0, stop=90.0, direction="up",
        cfg=EventDatasetConfig(horizon_minutes=3),
    )
    assert any(r.censored for r in rows)
    assert all(r.event_target == 0 and r.event_stop == 0 for r in rows)


def test_no_future_feature_mutation():
    frozen = {"gamma_flip": 99.0, "call_wall": 105.0}
    rows = expand_observation_to_event_rows(
        snapshot_id="s", session_date="d", origin_ts="t",
        prices=[100, 101, 102], minutes=[0, 1, 2],
        target=103, stop=98, direction="up",
        frozen_features=frozen,
        cfg=EventDatasetConfig(horizon_minutes=5),
    )
    assert all(r.features["gamma_flip"] == 99.0 for r in rows)
    assert all(r.features["call_wall"] == 105.0 for r in rows)
