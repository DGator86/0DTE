"""Canonical policy and AI-agent decision contracts."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .candidates import CandidateSummary
from .risk import OperationalState, PortfolioState, RiskEnvelope


DECISION_PACKET_SCHEMA = "decision.packet.v1"
AGENT_DECISION_SCHEMA = "agent.decision.v1"


class AgentAction(str, Enum):
    SELECT_CANDIDATE = "SELECT_CANDIDATE"
    ABSTAIN = "ABSTAIN"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"


@dataclass(frozen=True)
class PolicyDecisionView:
    source: str
    action: str
    candidate_id: str | None = None
    structure: str = ""
    direction: str = "none"
    confidence: float = 0.0
    uncertainty: float = 1.0
    size_cap: float = 0.0
    hard_vetoes: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()
    version: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("policy source is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("policy confidence must be within [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("policy uncertainty must be within [0, 1]")
        if not 0.0 <= self.size_cap <= 1.0:
            raise ValueError("policy size_cap must be within [0, 1]")


@dataclass(frozen=True)
class AgentDecisionPacket:
    packet_id: str
    snapshot_id: str
    timestamp: dt.datetime
    symbol: str
    candidates: tuple[CandidateSummary, ...]
    risk_envelope: RiskEnvelope
    operational_state: OperationalState
    portfolio_state: PortfolioState
    forecasts: Mapping[str, Any] = field(default_factory=dict)
    structural_state: Mapping[str, Any] = field(default_factory=dict)
    policy_decisions: tuple[PolicyDecisionView, ...] = ()
    disagreements: tuple[str, ...] = ()
    drift_state: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DECISION_PACKET_SCHEMA
    deployment_id: str = ""
    configuration_hash: str = ""

    def __post_init__(self) -> None:
        if not self.packet_id:
            raise ValueError("packet_id is required")
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.symbol:
            raise ValueError("symbol is required")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a packet")
        if any(candidate.snapshot_id != self.snapshot_id for candidate in self.candidates):
            raise ValueError("all candidates must belong to the packet snapshot")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "policy_decisions", tuple(self.policy_decisions))
        object.__setattr__(self, "forecasts", MappingProxyType(dict(self.forecasts)))
        object.__setattr__(self, "structural_state", MappingProxyType(dict(self.structural_state)))
        object.__setattr__(self, "drift_state", MappingProxyType(dict(self.drift_state)))

    def candidate(self, candidate_id: str) -> CandidateSummary | None:
        return next(
            (candidate for candidate in self.candidates if candidate.candidate_id == candidate_id),
            None,
        )


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction
    candidate_id: str | None = None
    size_scalar: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 1.0
    exit_policy_id: str | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    contradictory_evidence_ids: tuple[str, ...] = ()
    rationale: str = ""
    schema_version: str = AGENT_DECISION_SCHEMA
    model_id: str = ""
    prompt_version: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.size_scalar <= 1.0:
            raise ValueError("size_scalar must be within [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be within [0, 1]")
        if self.action == AgentAction.SELECT_CANDIDATE and not self.candidate_id:
            raise ValueError("SELECT_CANDIDATE requires candidate_id")
        if self.action != AgentAction.SELECT_CANDIDATE and self.candidate_id is not None:
            raise ValueError("candidate_id is only valid for SELECT_CANDIDATE")
        if self.action != AgentAction.SELECT_CANDIDATE and self.size_scalar != 0.0:
            raise ValueError("non-entry actions must use size_scalar=0")


def validate_agent_decision(
    packet: AgentDecisionPacket,
    decision: AgentDecision,
) -> CandidateSummary | None:
    """Validate model output against deterministic packet constraints.

    The agent may select only an existing, non-vetoed candidate and may only
    reduce the deterministic size cap. It can never create option geometry.
    """
    if decision.action != AgentAction.SELECT_CANDIDATE:
        return None
    assert decision.candidate_id is not None
    candidate = packet.candidate(decision.candidate_id)
    if candidate is None:
        raise ValueError("agent selected a candidate outside the packet whitelist")
    if not candidate.selectable:
        raise ValueError("agent selected a hard-vetoed candidate")
    if packet.operational_state.hard_vetoes:
        raise ValueError("operational hard veto blocks candidate selection")
    if decision.size_scalar > packet.risk_envelope.max_size_scalar:
        raise ValueError("agent size exceeds deterministic risk cap")
    return candidate
