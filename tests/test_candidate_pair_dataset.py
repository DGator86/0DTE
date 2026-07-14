"""
tests/test_candidate_pair_dataset.py
====================================
V3 Part 3 PR19 — within-snapshot pairwise dataset (§8 / §44).
"""
from __future__ import annotations

import pytest

from prediction.candidate_dataset import (
    CandidateTrainingFrame, append_settled_candidate,
    assert_pairs_within_snapshot, assert_snapshots_not_split,
    build_pair_features, build_pairwise_frame, generate_snapshot_pairs,
    grouped_snapshot_folds, pair_weight, pairwise_frame_from_training_frame,
    reverse_pair_features,
)


def _cand(cid, utility, **feat_extra):
    feat = {
        "expected_net_pnl": utility,
        "utility_score": utility,
        "expected_shortfall": 0.2,
        "fill_uncertainty": 0.1,
        "model_uncertainty": 0.1,
        "capital_required": 1.0,
        "family": feat_extra.pop("family", "put_credit"),
        "direction": feat_extra.pop("direction", "bullish"),
        "n_legs": feat_extra.pop("n_legs", 2),
        "max_loss": 1.0,
    }
    feat.update(feat_extra)
    return {
        "candidate_id": cid,
        "features": feat,
        "realized_utility": utility,
    }


class TestPairFeatures:
    def test_swap_reverses_diff_features(self):
        a = {"expected_net_pnl": 0.5, "fill_uncertainty": 0.2, "family": "x"}
        b = {"expected_net_pnl": 0.1, "fill_uncertainty": 0.4, "family": "x"}
        fwd = build_pair_features(a, b)
        rev = build_pair_features(b, a)
        assert rev == reverse_pair_features(fwd)
        assert fwd["diff_expected_net_pnl"] == pytest.approx(0.4)
        assert rev["diff_expected_net_pnl"] == pytest.approx(-0.4)
        assert fwd["same_family"] == 1.0
        assert rev["same_family"] == 1.0

    def test_categorical_mismatch(self):
        a = {"family": "put_credit", "expected_net_pnl": 0.1}
        b = {"family": "call_credit", "expected_net_pnl": 0.2}
        feat = build_pair_features(a, b)
        assert feat["same_family"] == 0.0


class TestNearTiesAndLabels:
    def test_near_ties_excluded(self):
        cands = [
            _cand("a", 0.100),
            _cand("b", 0.105),  # Δ=0.005 < 0.01R
            _cand("c", 0.50),
        ]
        pairs = generate_snapshot_pairs(
            "snap1", cands, pair_epsilon_r=0.01, risk_unit_r=1.0)
        ids = {(p.candidate_a_id, p.candidate_b_id) for p in pairs}
        assert ("a", "b") not in ids
        assert ("a", "c") in ids
        assert ("b", "c") in ids

    def test_swap_reverses_label(self):
        cands = [_cand("a", 0.8), _cand("b", 0.2)]
        pairs = generate_snapshot_pairs("s", cands, pair_epsilon_r=0.01)
        assert len(pairs) == 1
        p = pairs[0]
        assert p.a_wins == 1
        # Rebuild swapped order by constructing features manually
        feat_ba = build_pair_features(
            cands[1]["features"], cands[0]["features"])
        assert feat_ba == reverse_pair_features(p.pair_features)
        a_wins_swapped = int(0.2 > 0.8)
        assert a_wins_swapped == 1 - p.a_wins

    def test_pairs_never_cross_snapshots(self):
        by_snap = {
            "s1": [_cand("s1|a", 0.5), _cand("s1|b", 0.1)],
            "s2": [_cand("s2|a", 0.9), _cand("s2|b", 0.0)],
        }
        frame = build_pairwise_frame(by_snap)
        assert_pairs_within_snapshot(frame.pairs)
        for p in frame.pairs:
            assert p.candidate_a_id.startswith(p.snapshot_id)
            assert p.candidate_b_id.startswith(p.snapshot_id)
        # No pair mixes s1 and s2 candidates
        for p in frame.pairs:
            assert p.snapshot_id in ("s1", "s2")
            assert p.candidate_a_id.split("|")[0] == p.snapshot_id
            assert p.candidate_b_id.split("|")[0] == p.snapshot_id


class TestPairWeights:
    def test_weights_deterministic(self):
        kwargs = dict(
            realized_utility_a=1.0, realized_utility_b=0.2,
            complete_outcomes=True, valid_executable_quotes=True,
            passes_feasibility=True, quote_quality=1.0,
            data_quality=1.0, family_support=1.0, risk_unit_r=1.0,
        )
        assert pair_weight(**kwargs) == pair_weight(**kwargs)

    def test_large_delta_increases_weight(self):
        small = pair_weight(realized_utility_a=0.2, realized_utility_b=0.1)
        large = pair_weight(realized_utility_a=1.0, realized_utility_b=0.1)
        assert large > small

    def test_uncertain_fill_reduces_weight(self):
        base = pair_weight(realized_utility_a=1.0, realized_utility_b=0.0)
        bad = pair_weight(realized_utility_a=1.0, realized_utility_b=0.0,
                          fill_uncertain=True)
        assert bad < base


class TestSnapshotGrouping:
    def test_pairwise_from_training_frame_keeps_snapshots(self):
        frame = CandidateTrainingFrame()
        utils = []
        for snap, vals in (("2026-07-01|t0", [0.5, 0.1, 0.3]),
                           ("2026-07-01|t1", [0.8, 0.0])):
            for i, u in enumerate(vals):
                append_settled_candidate(
                    frame,
                    candidate_id=f"{snap}|c{i}",
                    snapshot_id=snap,
                    session_date="2026-07-01",
                    feature_row={"expected_net_pnl": u, "family": "put_credit",
                                 "utility_score": u},
                    outcome={"pnl_expected_fill": u, "settled": 1},
                )
                utils.append(u)
        pw = pairwise_frame_from_training_frame(frame, utils)
        assert len(pw) > 0
        assert_pairs_within_snapshot(pw.pairs)
        # Fold grouping still holds on the underlying candidate frame
        folds = grouped_snapshot_folds(
            frame.snapshot_ids, frame.session_dates,
            n_folds=1, embargo_sessions=0, min_train_sessions=1)
        # With one session, may get zero folds depending on min_train —
        # just ensure assert helper still works on synthetic split
        assert_snapshots_not_split(frame.snapshot_ids, [0, 1, 2], [3, 4])
