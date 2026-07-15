"""tests/test_dual_paper_accounts.py"""
from decision_stack.authority import resolve_authority


def test_dual_paper_authority_split():
    r = resolve_authority(
        mode="candidate",
        legacy_decision={"action": "TRADE", "candidate_id": "L"},
        v3_decision={"action": "NO_EDGE", "candidate_id": "V"},
    )
    assert r.reference_account == "legacy"
    assert r.candidate_account == "v3"
