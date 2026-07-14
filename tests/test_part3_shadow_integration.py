"""
tests/test_part3_shadow_integration.py
======================================
V3 Part 3 PR32/33 — shadow path does not alter legacy (§32 / §53).
"""
from __future__ import annotations

from dashboard.state import serialize_part3_decision
from prediction.part3_shadow import run_part3_shadow_decision


def test_shadow_decision_and_legacy_flag():
    cands = [
        {"candidate_id": "a", "features": {"utility_score": 1.0},
         "absolute_utility": 1.0},
        {"candidate_id": "b", "features": {"utility_score": 0.2},
         "absolute_utility": 0.2},
    ]
    result = run_part3_shadow_decision(
        snapshot_id="2026-07-01|t0",
        ts="2026-07-01T15:00:00Z",
        symbol="SPY",
        candidates=cands,
        absolute_utilities={"a": 1.0, "b": 0.2},
        mid_credit=0.5,
        natural_credit=0.3,
        family="put_credit",
        n_legs=2,
        mode="shadow",
    )
    assert result.legacy_unchanged is True
    assert result.decision.mode == "shadow"
    assert result.decision.source == "v3_shadow"
    payload = serialize_part3_decision(result.decision)
    assert "SHADOW" in payload["shadow_label"]
    assert payload["generated_at"]
    assert payload["model_versions"]


def test_hard_veto_in_shadow():
    result = run_part3_shadow_decision(
        snapshot_id="s",
        ts="t",
        symbol="SPY",
        candidates=[{"candidate_id": "a", "features": {},
                     "absolute_utility": 1.0}],
        absolute_utilities={"a": 1.0},
        mid_credit=0.5,
        natural_credit=0.3,
        family="put_credit",
        n_legs=2,
        hard_vetoes=("daily_loss_limit",),
        p_positive_utility=0.9,
    )
    assert result.decision.action == "HARD_VETO"
    assert result.decision.statistical_action != "HARD_VETO"
