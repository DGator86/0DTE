"""tests/test_learning_orchestrator.py"""
import pytest

from learning.orchestrator import LearningOrchestrator


def test_holdout_mandatory():
    orch = LearningOrchestrator()
    with pytest.raises(ValueError, match="holdout"):
        orch.run_weekly(sessions=[])


def test_no_independent_champion():
    orch = LearningOrchestrator()
    out = orch.run_weekly(sessions=["2026-07-01"])
    assert out["promoted"] is False
    daily = orch.run_daily(session_date="2026-07-01")
    assert daily["promoted"] is False
