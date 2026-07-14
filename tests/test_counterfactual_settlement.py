"""tests/test_counterfactual_settlement.py"""
from learning.settlement import settle_session_counterfactuals


def test_no_trades_and_nonselected_settle():
    out = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1", "action": "NO_EDGE"}],
        candidate_evaluations=[
            {"candidate_id": "c1", "snapshot_id": "s1"},
            {"candidate_id": "c2", "snapshot_id": "s1"},
        ],
    )
    assert out["complete"]
    assert len(out["candidate_outcomes"]) == 2


def test_idempotent_settlement():
    kwargs = dict(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1"}],
        candidate_evaluations=[{"candidate_id": "c1", "snapshot_id": "s1"}],
    )
    a = settle_session_counterfactuals(**kwargs)
    b = settle_session_counterfactuals(**kwargs)
    assert len(a["candidate_outcomes"]) == len(b["candidate_outcomes"])
