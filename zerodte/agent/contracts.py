"""AI provider protocol kept separate from the financial runtime."""
from __future__ import annotations

from typing import Protocol

from zerodte.contracts.decisions import AgentDecision, AgentDecisionPacket


class AgentProviderError(RuntimeError):
    """Raised when a provider cannot produce a valid structured decision."""


class AgentProvider(Protocol):
    provider_id: str
    model_id: str
    prompt_version: str

    def decide(self, packet: AgentDecisionPacket) -> AgentDecision: ...
