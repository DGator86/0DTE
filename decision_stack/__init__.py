"""
decision_stack/
==============
Unified V1 + V2 + V3 decision stack
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §7.5, §9).

NOT financial advice.
"""
from decision_stack.contracts import (
    FINAL_ACTIONS,
    CandidateEvaluation,
    UnifiedDecisionRecord,
)
from decision_stack.authority import AuthorityResult, resolve_authority
from decision_stack.stack import UnifiedDecisionStack

__all__ = [
    "FINAL_ACTIONS",
    "CandidateEvaluation",
    "UnifiedDecisionRecord",
    "AuthorityResult",
    "resolve_authority",
    "UnifiedDecisionStack",
]
