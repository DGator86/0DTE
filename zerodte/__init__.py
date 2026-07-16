"""Canonical package boundary for the 0DTE decision platform.

The legacy top-level modules remain operational during the migration. New code
should depend on the contracts and service interfaces exposed from this package
instead of importing live feeds, dashboard serializers, or the monolithic
``unified_loop`` directly.
"""

from .contracts.decisions import AgentDecision, AgentDecisionPacket
from .contracts.market import CanonicalMarketSnapshot

__all__ = [
    "AgentDecision",
    "AgentDecisionPacket",
    "CanonicalMarketSnapshot",
]

__version__ = "0.1.0"
