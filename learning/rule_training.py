"""
learning/rule_training.py
=========================
V1 rule-config search that produces versioned RuleConfigArtifact candidates.
Never independently promotes the live system.

NOT financial advice.
"""
from __future__ import annotations

from typing import Any, Optional

from adaptive_learning.config_store import RuleConfigArtifact, new_rule_config_artifact


def produce_rule_config_candidate(
    *,
    overrides: dict,
    regime_overrides: Optional[dict] = None,
    parent_id: Optional[str] = None,
    metrics: Optional[dict] = None,
    author: str = "learning_orchestrator",
) -> RuleConfigArtifact:
    return new_rule_config_artifact(
        overrides=overrides,
        regime_overrides=regime_overrides or {},
        parent_id=parent_id,
        metrics=metrics or {},
        author=author,
        status="candidate",
    )
