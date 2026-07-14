"""
tests/test_candidate_pair_ranker.py
===================================
V3 Part 3 PR20 — pairwise ranker fit/infer (§8 / §44).
"""
from __future__ import annotations

import pytest

from prediction.candidate_dataset import generate_snapshot_pairs
from prediction.models.candidate_rank import (
    CandidateRankConfig, PairwiseCandidateRanker, ranking_regret,
)


def _cands():
    # Clear utility ordering a > b > c
    rows = []
    for cid, u, fam in (
        ("a", 1.0, "put_credit"),
        ("b", 0.4, "put_credit"),
        ("c", 0.0, "call_credit"),
    ):
        rows.append({
            "candidate_id": cid,
            "features": {
                "expected_net_pnl": u,
                "utility_score": u,
                "expected_shortfall": 0.2,
                "fill_uncertainty": 0.1,
                "model_uncertainty": 0.1,
                "capital_required": 1.0 if cid != "c" else 2.0,
                "family": fam,
                "n_legs": 2,
            },
            "realized_utility": u,
            "absolute_utility": u,
        })
    return rows


def test_fit_and_rank_deterministic():
    cands = _cands()
    pairs = generate_snapshot_pairs("snap", cands, pair_epsilon_r=0.01)
    r1 = PairwiseCandidateRanker(CandidateRankConfig(estimator="logistic"))
    r1.fit(pairs)
    out1 = r1.rank_snapshot("snap", cands)
    r2 = PairwiseCandidateRanker(CandidateRankConfig(estimator="logistic"))
    r2.fit(pairs)
    out2 = r2.rank_snapshot("snap", cands)
    assert out1.to_dict() == out2.to_dict()
    assert out1.top_candidate_id == "a"


def test_swap_pair_proba_consistent():
    cands = _cands()
    pairs = generate_snapshot_pairs("snap", cands)
    ranker = PairwiseCandidateRanker().fit(pairs)
    from prediction.candidate_dataset import build_pair_features, reverse_pair_features
    feat_ab = build_pair_features(cands[0]["features"], cands[1]["features"])
    feat_ba = reverse_pair_features(feat_ab)
    p_ab = ranker.predict_pair_proba(feat_ab)
    p_ba = ranker.predict_pair_proba(feat_ba)
    # Not required to sum to 1 after independent scoring, but both in (0,1)
    assert 0.0 < p_ab < 1.0
    assert 0.0 < p_ba < 1.0


def test_vetoed_cannot_be_top():
    cands = _cands()
    pairs = generate_snapshot_pairs("snap", cands)
    ranker = PairwiseCandidateRanker().fit(pairs)
    out = ranker.rank_snapshot("snap", cands, vetoed_ids={"a"})
    assert out.top_candidate_id != "a"
    assert out.top_candidate_id == "b"


def test_single_candidate_ranking():
    cands = [_cands()[0]]
    ranker = PairwiseCandidateRanker()
    out = ranker.rank_snapshot("snap", cands)
    assert out.top_candidate_id == "a"
    assert out.ordered_candidate_ids == ("a",)
    assert out.pairwise_scores["a"] == 0.5


def test_tied_scores_deterministic_tiebreak():
    cands = [
        {
            "candidate_id": "z",
            "features": {"utility_score": 0.5, "capital_required": 2.0,
                         "family": "x", "expected_net_pnl": 0.5},
            "absolute_utility": 0.5, "uncertainty": 0.1, "capital": 2.0,
        },
        {
            "candidate_id": "a",
            "features": {"utility_score": 0.5, "capital_required": 1.0,
                         "family": "x", "expected_net_pnl": 0.5},
            "absolute_utility": 0.5, "uncertainty": 0.1, "capital": 1.0,
        },
    ]
    ranker = PairwiseCandidateRanker()
    out = ranker.rank_snapshot("snap", cands)
    # Same combined score → lower capital wins, then stable id
    assert out.top_candidate_id == "a"


def test_ranking_regret():
    realized = {"a": 0.2, "b": 1.0, "c": 0.5}
    assert ranking_regret("a", realized) == pytest.approx(0.8)
    assert ranking_regret("b", realized) == pytest.approx(0.0)


def test_hgb_challenger_fits():
    cands = _cands()
    pairs = generate_snapshot_pairs("snap", cands)
    ranker = PairwiseCandidateRanker(
        CandidateRankConfig(estimator="hgb")).fit(pairs)
    out = ranker.rank_snapshot("snap", cands)
    assert out.top_candidate_id in {"a", "b", "c"}
    assert out.diagnostics["estimator"] == "hgb"
