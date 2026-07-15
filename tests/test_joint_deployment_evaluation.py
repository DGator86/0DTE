"""tests/test_joint_deployment_evaluation.py"""
from learning.deployment_evaluation import evaluate_deployment_bundle


def test_evaluate_complete_stack():
    ev = evaluate_deployment_bundle(
        deployment_id="d1",
        comparison_deployment_id="legacy",
        sessions=["a", "b", "c"],
        metrics={"net_pnl": 1.2, "drawdown": -0.1},
    )
    assert ev["sessions_count"] == 3
    assert ev["promoted"] is False
