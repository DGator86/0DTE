"""
tests/test_authority_router.py
"""
from __future__ import annotations

from decision_stack.authority import resolve_authority


def test_hard_veto_always_wins():
    r = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "l1"},
        v3_decision={"action": "TRADE", "candidate_id": "v1"},
        hard_vetoes=("stale_chain",),
    )
    assert r.final_action == "HARD_VETO"


def test_shadow_keeps_legacy():
    r = resolve_authority(
        mode="shadow",
        legacy_decision={"action": "TRADE", "candidate_id": "l1",
                         "structure": "pcs"},
        v3_decision={"action": "ABSTAIN", "candidate_id": "v1"},
    )
    assert r.authority_source == "legacy"
    assert r.final_action == "TRADE"
    assert r.selected_candidate_id == "l1"


def test_advisory_keeps_legacy_with_advisory():
    r = resolve_authority(
        mode="advisory",
        legacy_decision={"action": "NO_EDGE"},
        v3_decision={"action": "TRADE", "candidate_id": "v1"},
    )
    assert r.authority_source == "legacy"
    assert r.advisory_action == "TRADE"


def test_candidate_dual_accounts():
    r = resolve_authority(
        mode="candidate",
        legacy_decision={"action": "TRADE", "candidate_id": "l1"},
        v3_decision={"action": "TRADE", "candidate_id": "v1"},
    )
    assert r.reference_account == "legacy"
    assert r.candidate_account == "v3"
    assert r.selected_candidate_id == "l1"
    assert r.advisory_candidate_id == "v1"


def test_champion_uses_v3():
    r = resolve_authority(
        mode="champion",
        legacy_decision={"action": "NO_EDGE"},
        v3_decision={"action": "TRADE", "candidate_id": "v1",
                     "structure": "cds"},
        fallback_policy="abstain",
    )
    assert r.authority_source == "v3"
    assert r.final_action == "TRADE"
    assert r.selected_candidate_id == "v1"


def test_champion_fallback_abstain():
    r = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "l1"},
        v3_decision={"action": "UNAVAILABLE"},
        fallback_policy="abstain",
    )
    assert r.fallback_used is True
    assert r.final_action == "ABSTAIN"


def test_champion_fallback_legacy():
    r = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "l1"},
        v3_decision=None,
        fallback_policy="legacy",
    )
    assert r.fallback_used is True
    assert r.authority_source == "legacy"
    assert r.final_action == "TRADE"
