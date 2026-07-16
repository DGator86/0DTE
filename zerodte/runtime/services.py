"""Service protocols for the staged decision runtime.

Interfaces are intentionally narrow. Implementations may wrap the current
legacy modules during migration or point at the eventual package-native code.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Protocol, Sequence

from zerodte.contracts.candidates import CandidateSummary
from zerodte.contracts.decisions import (
    AgentDecision,
    AgentDecisionPacket,
    PolicyDecisionView,
)
from zerodte.contracts.market import CanonicalMarketSnapshot
from zerodte.contracts.risk import OperationalState, PortfolioState, RiskEnvelope


class SnapshotAssembler(Protocol):
    def snapshot(self, now: dt.datetime) -> CanonicalMarketSnapshot | None: ...


class FeatureService(Protocol):
    def compute(
        self, snapshot: CanonicalMarketSnapshot
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Return (features, structural_state)."""
        ...


class ForecastService(Protocol):
    def predict(
        self,
        snapshot: CanonicalMarketSnapshot,
        features: Mapping[str, Any],
        structural_state: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class CandidateService(Protocol):
    def generate(
        self,
        snapshot: CanonicalMarketSnapshot,
        features: Mapping[str, Any],
        structural_state: Mapping[str, Any],
        forecasts: Mapping[str, Any],
    ) -> Sequence[CandidateSummary]: ...


class PolicyService(Protocol):
    def evaluate(
        self,
        snapshot: CanonicalMarketSnapshot,
        features: Mapping[str, Any],
        structural_state: Mapping[str, Any],
        forecasts: Mapping[str, Any],
        candidates: Sequence[CandidateSummary],
    ) -> Sequence[PolicyDecisionView]: ...


class RiskService(Protocol):
    def operational_state(
        self, snapshot: CanonicalMarketSnapshot
    ) -> OperationalState: ...

    def portfolio_state(self, account_id: str) -> PortfolioState: ...

    def envelope(
        self,
        snapshot: CanonicalMarketSnapshot,
        portfolio: PortfolioState,
        candidates: Sequence[CandidateSummary],
    ) -> RiskEnvelope: ...


class DecisionAgent(Protocol):
    def decide(self, packet: AgentDecisionPacket) -> AgentDecision: ...


class JournalSink(Protocol):
    def record(self, event_type: str, payload: Mapping[str, Any]) -> None: ...
