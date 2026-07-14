"""
tests/test_candidate_ranking_v3.py
==================================
V3 Part 3 PR20 — CandidateRanking contract + storage persistence.
"""
from __future__ import annotations

from prediction.models.candidate_rank import (
    CandidateRanking, PairwiseCandidateRanker,
)
from prediction.storage import PredictionStore


def test_ranking_roundtrip_dict():
    r = CandidateRanking(
        snapshot_id="s1",
        ordered_candidate_ids=("a", "b"),
        combined_scores={"a": 0.9, "b": 0.1},
        absolute_utilities={"a": 1.0, "b": 0.0},
        pairwise_scores={"a": 0.8, "b": 0.2},
        expected_regret={"a": 0.0, "b": 1.0},
        selection_uncertainty={"a": 0.1, "b": 0.5},
        top_candidate_id="a",
        second_candidate_id="b",
        top_score_margin=0.8,
    )
    assert CandidateRanking.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_persist_ranking(tmp_path):
    store = PredictionStore(str(tmp_path / "p.sqlite"))
    ranking = PairwiseCandidateRanker().rank_snapshot(
        "snap",
        [{"candidate_id": "c1", "features": {"utility_score": 1.0},
          "absolute_utility": 1.0}],
    )
    rid = store.log_candidate_ranking(
        "snap", ranking.model_version, ranking.to_dict(),
        generated_at="2026-07-14T12:00:00Z", mode="shadow",
    )
    assert rid >= 1
    rows = store.fetch_candidate_rankings("snap")
    assert len(rows) == 1
    assert rows[0]["ranking"]["top_candidate_id"] == "c1"
    store.close()
