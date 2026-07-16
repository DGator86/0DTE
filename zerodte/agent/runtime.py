"""Fail-closed runtime wrapper for any AI decision provider."""
from __future__ import annotations

import dataclasses
import logging

from zerodte.contracts.decisions import (
    AgentAction,
    AgentDecision,
    AgentDecisionPacket,
    validate_agent_decision,
)

from .contracts import AgentProvider

log = logging.getLogger("zerodte.agent")


class FailClosedAgentRuntime:
    """Validate provider output and convert every failure to ABSTAIN.

    The provider has no authority to construct candidates, increase the risk
    cap, or override deterministic vetoes. Those rules are checked here before
    the pipeline can expose a selected candidate to execution.
    """

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def decide(self, packet: AgentDecisionPacket) -> AgentDecision:
        try:
            decision = self.provider.decide(packet)
            decision = dataclasses.replace(
                decision,
                model_id=decision.model_id or self.provider.model_id,
                prompt_version=(
                    decision.prompt_version or self.provider.prompt_version
                ),
            )
            validate_agent_decision(packet, decision)
            return decision
        except Exception as exc:  # model failure must never become trade authority
            log.warning("agent decision rejected: %s", exc)
            return AgentDecision(
                action=AgentAction.ABSTAIN,
                confidence=0.0,
                uncertainty=1.0,
                rationale=f"agent_failure:{type(exc).__name__}:{exc}",
                model_id=getattr(self.provider, "model_id", ""),
                prompt_version=getattr(self.provider, "prompt_version", ""),
            )
