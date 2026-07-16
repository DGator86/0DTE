"""Staged canonical decision pipeline.

This module contains orchestration only. Financial calculations remain in the
services supplied to the pipeline. The AI agent is the final statistical choice
layer, while deterministic validation and risk remain authoritative.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from zerodte.contracts.candidates import CandidateSummary
from zerodte.contracts.decisions import (
    AgentAction,
    AgentDecision,
    AgentDecisionPacket,
    validate_agent_decision,
)
from zerodte.contracts.market import CanonicalMarketSnapshot

from .services import (
    CandidateService,
    DecisionAgent,
    FeatureService,
    ForecastService,
    JournalSink,
    PolicyService,
    RiskService,
    SnapshotAssembler,
)


@dataclass(frozen=True)
class PipelineResult:
    snapshot: CanonicalMarketSnapshot
    packet: AgentDecisionPacket
    decision: AgentDecision
    selected_candidate: CandidateSummary | None
    error: str = ""


class CanonicalDecisionPipeline:
    """Compose independently testable processing stages.

    Failure policy is deliberately conservative: missing snapshots return None;
    any downstream exception produces an auditable ABSTAIN decision.
    """

    def __init__(
        self,
        *,
        snapshots: SnapshotAssembler,
        features: FeatureService,
        forecasts: ForecastService,
        candidates: CandidateService,
        policies: PolicyService,
        risk: RiskService,
        agent: DecisionAgent,
        account_id: str = "agent",
        deployment_id: str = "",
        configuration_hash: str = "",
        journal: JournalSink | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.features = features
        self.forecasts = forecasts
        self.candidates = candidates
        self.policies = policies
        self.risk = risk
        self.agent = agent
        self.account_id = account_id
        self.deployment_id = deployment_id
        self.configuration_hash = configuration_hash
        self.journal = journal

    def run(self, now: dt.datetime) -> PipelineResult | None:
        snapshot = self.snapshots.snapshot(now)
        if snapshot is None:
            return None

        packet: AgentDecisionPacket | None = None
        try:
            features, structural_state = self.features.compute(snapshot)
            forecasts = self.forecasts.predict(
                snapshot, features, structural_state
            )
            candidates = tuple(
                self.candidates.generate(
                    snapshot, features, structural_state, forecasts
                )
            )
            policy_decisions = tuple(
                self.policies.evaluate(
                    snapshot,
                    features,
                    structural_state,
                    forecasts,
                    candidates,
                )
            )
            operational = self.risk.operational_state(snapshot)
            portfolio = self.risk.portfolio_state(self.account_id)
            envelope = self.risk.envelope(snapshot, portfolio, candidates)
            packet = AgentDecisionPacket(
                packet_id=_packet_id(
                    snapshot.snapshot_id,
                    self.deployment_id,
                    self.configuration_hash,
                ),
                snapshot_id=snapshot.snapshot_id,
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                candidates=candidates,
                risk_envelope=envelope,
                operational_state=operational,
                portfolio_state=portfolio,
                forecasts=forecasts,
                structural_state=structural_state,
                policy_decisions=policy_decisions,
                disagreements=_policy_disagreements(policy_decisions),
                deployment_id=self.deployment_id,
                configuration_hash=self.configuration_hash,
            )

            if not operational.entries_allowed or not envelope.approved:
                reason = ",".join(
                    (*operational.hard_vetoes, *envelope.hard_vetoes)
                ) or "deterministic_risk_rejected"
                decision = _abstain(reason)
            elif not candidates:
                decision = _abstain("no_candidates")
            else:
                decision = self.agent.decide(packet)
                selected = validate_agent_decision(packet, decision)
                result = PipelineResult(
                    snapshot=snapshot,
                    packet=packet,
                    decision=decision,
                    selected_candidate=selected,
                )
                self._record(result)
                return result
        except Exception as exc:  # fail closed and keep the failure observable
            decision = _abstain(f"pipeline_error:{type(exc).__name__}:{exc}")
            if packet is None:
                packet = _minimal_packet(snapshot, self.account_id)
            result = PipelineResult(
                snapshot=snapshot,
                packet=packet,
                decision=decision,
                selected_candidate=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._record(result)
            return result

        result = PipelineResult(
            snapshot=snapshot,
            packet=packet,
            decision=decision,
            selected_candidate=None,
        )
        self._record(result)
        return result

    def _record(self, result: PipelineResult) -> None:
        if self.journal is None:
            return
        self.journal.record(
            "agent_decision",
            {
                "snapshot_id": result.snapshot.snapshot_id,
                "packet_id": result.packet.packet_id,
                "deployment_id": result.packet.deployment_id,
                "configuration_hash": result.packet.configuration_hash,
                "decision": dataclasses.asdict(result.decision),
                "selected_candidate_id": (
                    result.selected_candidate.candidate_id
                    if result.selected_candidate is not None
                    else None
                ),
                "error": result.error,
            },
        )


def _packet_id(snapshot_id: str, deployment_id: str, config_hash: str) -> str:
    payload = {
        "snapshot_id": snapshot_id,
        "deployment_id": deployment_id,
        "configuration_hash": config_hash,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"packet_{digest[:20]}"


def _policy_disagreements(decisions) -> tuple[str, ...]:
    if len(decisions) < 2:
        return ()
    first = decisions[0]
    disagreements: list[str] = []
    for other in decisions[1:]:
        if first.action != other.action:
            disagreements.append(
                f"{first.source}:{first.action}!={other.source}:{other.action}"
            )
        elif first.candidate_id != other.candidate_id:
            disagreements.append(
                f"{first.source}:{first.candidate_id}!={other.source}:{other.candidate_id}"
            )
    return tuple(disagreements)


def _abstain(reason: str) -> AgentDecision:
    return AgentDecision(
        action=AgentAction.ABSTAIN,
        confidence=0.0,
        uncertainty=1.0,
        rationale=reason,
    )


def _minimal_packet(
    snapshot: CanonicalMarketSnapshot,
    account_id: str,
) -> AgentDecisionPacket:
    from zerodte.contracts.risk import OperationalState, PortfolioState, RiskEnvelope

    return AgentDecisionPacket(
        packet_id=_packet_id(snapshot.snapshot_id, "", ""),
        snapshot_id=snapshot.snapshot_id,
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        candidates=(),
        risk_envelope=RiskEnvelope.rejected("pipeline_failed_before_risk"),
        operational_state=OperationalState(
            market_open=False,
            data_valid=False,
            broker_available=False,
            hard_vetoes=("pipeline_failed",),
        ),
        portfolio_state=PortfolioState(
            account_id=account_id,
            equity=0.0,
            cash=0.0,
        ),
    )
