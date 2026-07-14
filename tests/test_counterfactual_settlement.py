"""tests/test_counterfactual_settlement.py"""
from learning.settlement import settle_session_counterfactuals


def test_no_settlement_fn_is_incomplete():
    out = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1", "action": "NO_EDGE"}],
        candidate_evaluations=[
            {"candidate_id": "c1", "snapshot_id": "s1"},
        ],
    )
    assert out["complete"] is False
    assert out["settled_journal"] == []


def test_allow_incomplete_never_invents_zero_pnl():
    out = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1"}],
        candidate_evaluations=[
            {"candidate_id": "c1", "snapshot_id": "s1"},
        ],
        allow_incomplete=True,
    )
    # Fixture path may mark rows unresolved — never coerce PnL to 0.0
    for row in out["settled_journal"]:
        assert row.get("net_pnl") != 0.0 or "net_pnl" in row and row.get(
            "settlement_status") == "settled"
    for co in out["candidate_outcomes"]:
        assert co.get("net_pnl") is None
        assert co.get("settlement_status") == "unresolved"


def test_real_settlement_fn_with_pnl():
    def _fn(session_date, rows):
        return [{"snapshot_id": "s1", "net_pnl": 1.5, "settled_at": "t"}]

    out = settle_session_counterfactuals(
        session_date="2026-07-14",
        journal_rows=[{"snapshot_id": "s1"}],
        settlement_fn=_fn,
    )
    assert out["complete"] is True
    assert out["settled_journal"][0]["net_pnl"] == 1.5
