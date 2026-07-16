"""Versioned, dependency-light contracts shared across the runtime."""

from .candidates import CandidateEconomics, CandidateLeg, CandidateSummary
from .decisions import (
    AgentAction,
    AgentDecision,
    AgentDecisionPacket,
    PolicyDecisionView,
)
from .market import (
    CanonicalMarketSnapshot,
    DataQuality,
    FeedObservation,
    FeedStatus,
)
from .risk import OperationalState, PortfolioState, RiskEnvelope

__all__ = [
    "AgentAction",
    "AgentDecision",
    "AgentDecisionPacket",
    "CandidateEconomics",
    "CandidateLeg",
    "CandidateSummary",
    "CanonicalMarketSnapshot",
    "DataQuality",
    "FeedObservation",
    "FeedStatus",
    "OperationalState",
    "PolicyDecisionView",
    "PortfolioState",
    "RiskEnvelope",
]
