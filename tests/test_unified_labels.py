"""tests/test_unified_labels.py"""
from learning.labels import meta_decision_labels


def test_meta_decision_labels():
    labels = meta_decision_labels([
        {"snapshot_id": "s1", "final_action": "TRADE",
         "realized_executable_pnl": 12.0},
        {"snapshot_id": "s2", "final_action": "ABSTAIN",
         "realized_executable_pnl": None},
    ])
    assert labels[0]["positive_executable_value"] is True
    assert labels[1]["positive_executable_value"] is None
