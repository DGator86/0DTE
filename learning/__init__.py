"""
learning/
========
Unified learning orchestrator
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §12–§15).

Coordinates settlement labels, model-family retraining, rule-config
candidates, drift evaluation, and complete deployment evaluation.
Never independently promotes a champion.

NOT financial advice.
"""
from learning.orchestrator import LearningOrchestrator
from learning.settlement import settle_session_counterfactuals
from learning.promotion_packet import build_joint_promotion_packet

__all__ = [
    "LearningOrchestrator",
    "settle_session_counterfactuals",
    "build_joint_promotion_packet",
]
