"""Thin SPY-DER integration surface owned by 0DTE.

0DTE publishes MarketPacket / OutcomePacket, calls the local HTTP decision
service, and reads DashboardPacket files. No imports of SPY-DER internals.
"""

from integrations.spy_der.contracts import (
    DASHBOARD_SCHEMA,
    DECISION_REQUEST_SCHEMA,
    DECISION_RESPONSE_SCHEMA,
    MARKET_PACKET_SCHEMA,
    OUTCOME_PACKET_SCHEMA,
    PARALLEL_TRACK_ID,
    PARALLEL_TRACK_LABEL,
)
from integrations.spy_der.decision_client import (
    DEFAULT_DECISION_URL,
    DecisionClient,
    DecisionResult,
)

__all__ = [
    "DASHBOARD_SCHEMA",
    "DECISION_REQUEST_SCHEMA",
    "DECISION_RESPONSE_SCHEMA",
    "DEFAULT_DECISION_URL",
    "MARKET_PACKET_SCHEMA",
    "OUTCOME_PACKET_SCHEMA",
    "PARALLEL_TRACK_ID",
    "PARALLEL_TRACK_LABEL",
    "DecisionClient",
    "DecisionResult",
]
