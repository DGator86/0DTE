"""
tests/test_part3_end_to_end_replay.py
====================================
V3 Part 3 PR33 — deterministic offline replay (§54).
"""
from __future__ import annotations

from execution.estimate_v3 import build_execution_estimate_v3
from prediction.deployment import write_deployment_pointer, load_deployment_pointer
from prediction.models.candidate_rank import PairwiseCandidateRanker
from prediction.part3_shadow import run_part3_shadow_decision
from prediction.storage import PredictionStore


def test_replay_identical_outputs(tmp_path):
    snapshot_id = "2026-07-01|t0"
    cands = [
        {"candidate_id": "a",
         "features": {"utility_score": 0.8, "expected_net_pnl": 0.8,
                      "family": "put_credit"},
         "absolute_utility": 0.8},
        {"candidate_id": "b",
         "features": {"utility_score": 0.1, "expected_net_pnl": 0.1,
                      "family": "put_credit"},
         "absolute_utility": 0.1},
    ]
    utils = {"a": 0.8, "b": 0.1}
    kwargs = dict(
        snapshot_id=snapshot_id,
        ts="2026-07-01T15:00:00Z",
        symbol="SPY",
        candidates=cands,
        absolute_utilities=utils,
        mid_credit=0.50,
        natural_credit=0.30,
        family="put_credit",
        n_legs=2,
        configuration_hash="cfghash",
        p_positive_utility=0.70,
        composite_uncertainty=0.15,
        ood_score=0.05,
        data_quality=0.95,
    )
    r1 = run_part3_shadow_decision(**kwargs)
    r2 = run_part3_shadow_decision(**kwargs)
    assert r1.decision.to_dict() == r2.decision.to_dict()
    assert r1.ranking == r2.ranking
    assert r1.execution == r2.execution

    store = PredictionStore(str(tmp_path / "replay.sqlite"))
    store.log_candidate_ranking(
        snapshot_id, "v3.0.0", r1.ranking,
        generated_at=kwargs["ts"], mode="shadow",
    )
    store.log_meta_decision(
        snapshot_id, "v3.0.0", r1.meta,
        generated_at=kwargs["ts"], mode="shadow",
        candidate_id=r1.decision.selected_candidate_id,
    )
    rows = store.fetch_candidate_rankings(snapshot_id)
    assert rows[0]["ranking"]["top_candidate_id"] == \
        r1.decision.selected_candidate_id
    store.close()

    ptr_path = str(tmp_path / "deployment.json")
    write_deployment_pointer(ptr_path, {
        "mode": "shadow",
        "prediction_model_group": "g",
        "candidate_value_model": "cv",
        "candidate_rank_model": "cr",
        "fill_probability_model": "fp",
        "fill_concession_model": "fc",
        "meta_model": "mm",
    })
    assert load_deployment_pointer(ptr_path)["mode"] == "shadow"

    # Execution estimate alone is deterministic
    e1 = build_execution_estimate_v3(
        mid_credit=0.5, natural_credit=0.3, family="put_credit", n_legs=2)
    e2 = build_execution_estimate_v3(
        mid_credit=0.5, natural_credit=0.3, family="put_credit", n_legs=2)
    assert e1.to_dict() == e2.to_dict()

    # Ranker unfitted path deterministic
    assert PairwiseCandidateRanker().rank_snapshot(
        snapshot_id, cands, absolute_utilities=utils
    ).to_dict() == PairwiseCandidateRanker().rank_snapshot(
        snapshot_id, cands, absolute_utilities=utils
    ).to_dict()
