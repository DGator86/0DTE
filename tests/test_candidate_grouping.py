"""
tests/test_candidate_grouping.py
================================
PR 8 acceptance — candidates from one snapshot stay in one fold:
  * grouped_snapshot_folds never splits a snapshot_id across train/test;
  * assert_snapshots_not_split catches deliberate leakage;
  * training-frame load preserves snapshot identity.
"""
from __future__ import annotations

import pytest

from prediction.candidate_dataset import (
    CandidateTrainingFrame, append_settled_candidate,
    assert_snapshots_not_split, grouped_snapshot_folds,
    load_candidate_training_frame,
)
from prediction.storage import PredictionStore, make_candidate_id


def _frame(n_sessions=10, cands_per_snap=4):
    """Synthetic frame: 2 snapshots per session, several candidates each."""
    frame = CandidateTrainingFrame()
    for s in range(n_sessions):
        session = f"2026-07-{s + 1:02d}"
        for snap_i in range(2):
            snap = f"{session}|t{snap_i}"
            for c in range(cands_per_snap):
                cid = f"{snap}|c{c}"
                append_settled_candidate(
                    frame,
                    candidate_id=cid,
                    snapshot_id=snap,
                    session_date=session,
                    feature_row={"legacy_candidate_score": 0.1 * c,
                                 "ev": 0.01 * c, "max_loss": 1.0},
                    outcome={"pnl_expected_fill": 0.05 - 0.02 * c,
                             "pnl_mid": 0.05 - 0.02 * c, "settled": 1},
                    fill_uncertainty=0.1 * c,
                    capital=1.0,
                )
    return frame


class TestSnapshotFolds:
    def test_no_snapshot_split(self):
        frame = _frame()
        folds = grouped_snapshot_folds(
            frame.snapshot_ids, frame.session_dates,
            n_folds=3, embargo_sessions=1, min_train_sessions=2)
        assert len(folds) == 3
        for fold in folds:
            assert_snapshots_not_split(
                frame.snapshot_ids, fold["train_indices"], fold["test_indices"])
            train_snaps = set(fold["train_snapshots"])
            test_snaps = set(fold["test_snapshots"])
            assert not (train_snaps & test_snaps)
            # every test index's snapshot is wholly in test
            for i in fold["test_indices"]:
                assert frame.snapshot_ids[i] in test_snaps
                assert frame.snapshot_ids[i] not in train_snaps

    def test_assert_catches_leak(self):
        snaps = ["a", "a", "b", "b"]
        with pytest.raises(AssertionError, match="split"):
            assert_snapshots_not_split(snaps, [0, 2], [1, 3])

    def test_all_candidates_of_snapshot_move_together(self):
        frame = _frame(n_sessions=8, cands_per_snap=5)
        folds = grouped_snapshot_folds(
            frame.snapshot_ids, frame.session_dates, n_folds=2,
            embargo_sessions=1, min_train_sessions=2)
        for fold in folds:
            by_snap = {}
            for i in fold["train_indices"] + fold["test_indices"]:
                by_snap.setdefault(frame.snapshot_ids[i], set()).add(
                    "train" if i in fold["train_indices"] else "test")
            for snap, sides in by_snap.items():
                assert len(sides) == 1, f"{snap} on both sides: {sides}"


class TestStoreRoundTrip:
    def test_load_frame_keeps_snapshot_ids(self, tmp_path):
        store = PredictionStore(db_path=str(tmp_path / "p.sqlite"))
        legs = [{"strike": 599.0, "kind": "P", "qty": -1},
                {"strike": 598.0, "kind": "P", "qty": 1}]
        for s in range(4):
            session = f"2026-07-{s + 1:02d}"
            snap = f"{session}|0900"
            for c in range(3):
                cid = make_candidate_id(snap, "put_credit",
                                        [{**legs[0], "strike": 599.0 - c},
                                         legs[1]])
                store.log_candidate_snapshot(
                    cid, snap, "put_credit",
                    [{**legs[0], "strike": 599.0 - c}, legs[1]],
                    legacy_metrics={"score": 1.0, "ev": 0.05, "credit": 0.3,
                                    "max_loss": 0.7},
                    geometry={"ev": 0.05, "legacy_candidate_score": 1.0},
                )
                store.log_candidate_outcome(cid, {
                    "settled": 1, "settlement_price": 601.0,
                    "pnl_mid": 0.30, "pnl_expected_fill": 0.25,
                    "pnl_conservative": 0.20, "pnl_policy": None,
                    "mfe": 0.01, "mae": -0.01,
                    "target_hit": 0, "stop_hit": 0, "first_event": "neither",
                })
        frame = load_candidate_training_frame(store)
        assert len(frame) == 12
        folds = grouped_snapshot_folds(
            frame.snapshot_ids, frame.session_dates, n_folds=1,
            embargo_sessions=1, min_train_sessions=2)
        assert_snapshots_not_split(
            frame.snapshot_ids, folds[0]["train_indices"],
            folds[0]["test_indices"])
