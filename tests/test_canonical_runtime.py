from __future__ import annotations

import datetime as dt

from zerodte.agent.runtime import FailClosedAgentRuntime
from zerodte.contracts.candidates import (
    CandidateEconomics,
    CandidateLeg,
    CandidateSummary,
    make_candidate_id,
)
from zerodte.contracts.decisions import AgentAction, AgentDecision, PolicyDecisionView
from zerodte.contracts.market import CanonicalMarketSnapshot, DataQuality
from zerodte.contracts.risk import OperationalState, PortfolioState, RiskEnvelope
from zerodte.runtime.pipeline import CanonicalDecisionPipeline

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 7, 16, 14, 0, tzinfo=UTC)


def _candidate(snapshot_id: str = "snap-1") -> CandidateSummary:
    legs = (
        CandidateLeg("P", 620.0, -1, "2026-07-16"),
        CandidateLeg("P", 618.0, 1, "2026-07-16"),
    )
    return CandidateSummary(
        candidate_id=make_candidate_id(snapshot_id, "put_credit", legs),
        snapshot_id=snapshot_id,
        family="put_credit",
        direction="bullish",
        legs=legs,
        economics=CandidateEconomics(
            entry_price=0.42,
            expected_fill_price=0.40,
            fill_probability=0.72,
            max_profit=40.0,
            max_loss=160.0,
            expected_value=8.0,
            expected_utility=0.15,
            probability_profit=0.68,
            probability_touch=0.25,
            cvar_95=-120.0,
            liquidity_score=0.85,
            data_quality=0.9,
        ),
    )


def _snapshot() -> CanonicalMarketSnapshot:
    return CanonicalMarketSnapshot(
        snapshot_id="snap-1",
        timestamp=NOW,
        symbol="SPY",
        spot=622.0,
        data_quality=DataQuality(score=0.95, coverage=0.95),
    )


class _Provider:
    provider_id = "test"
    model_id = "test-model"
    prompt_version = "prompt-v1"

    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.calls = 0

    def decide(self, packet):
        self.calls += 1
        return AgentDecision(
            action=AgentAction.SELECT_CANDIDATE,
            candidate_id=self.candidate_id,
            size_scalar=0.5,
            confidence=0.7,
            uncertainty=0.3,
            rationale="best executable utility after model disagreement",
        )


class _Snapshots:
    def snapshot(self, now):
        return _snapshot()


class _Features:
    def compute(self, snapshot):
        return ({"bias": 0.61}, {"gamma_regime": "long"})


class _Forecasts:
    def predict(self, snapshot, features, structural_state):
        return {"p_up_30m": 0.62, "p_range_survive_30m": 0.66}


class _Candidates:
    def generate(self, snapshot, features, structural_state, forecasts):
        return (_candidate(snapshot.snapshot_id),)


class _Policies:
    def evaluate(self, snapshot, features, structural_state, forecasts, candidates):
        return (
            PolicyDecisionView(
                source="legacy",
                action="TRADE",
                candidate_id=candidates[0].candidate_id,
                structure="PCS",
                direction="put",
                confidence=0.65,
                uncertainty=0.35,
                size_cap=0.8,
            ),
            PolicyDecisionView(
                source="v3",
                action="TRADE",
                candidate_id=candidates[0].candidate_id,
                structure="PCS",
                direction="put",
                confidence=0.7,
                uncertainty=0.3,
                size_cap=0.6,
            ),
        )


class _Risk:
    def __init__(self, veto: bool = False) -> None:
        self.veto = veto

    def operational_state(self, snapshot):
        return OperationalState(
            market_open=True,
            data_valid=not self.veto,
            broker_available=True,
            hard_vetoes=("stale_chain",) if self.veto else (),
        )

    def portfolio_state(self, account_id):
        return PortfolioState(account_id=account_id, equity=10_000, cash=10_000)

    def envelope(self, snapshot, portfolio, candidates):
        if self.veto:
            return RiskEnvelope.rejected("stale_chain")
        return RiskEnvelope(
            approved=True,
            max_risk_dollars=200.0,
            max_size_scalar=0.6,
            remaining_daily_loss_budget=500.0,
            remaining_position_slots=1,
        )


class _Journal:
    def __init__(self) -> None:
        self.events = []

    def record(self, event_type, payload):
        self.events.append((event_type, payload))


def _pipeline(provider, *, veto: bool = False, journal=None):
    return CanonicalDecisionPipeline(
        snapshots=_Snapshots(),
        features=_Features(),
        forecasts=_Forecasts(),
        candidates=_Candidates(),
        policies=_Policies(),
        risk=_Risk(veto=veto),
        agent=FailClosedAgentRuntime(provider),
        journal=journal,
        deployment_id="test-shadow",
        configuration_hash="abc123",
    )


def test_candidate_id_is_stable():
    first = _candidate().candidate_id
    second = _candidate().candidate_id
    assert first == second
    assert first.startswith("cand_")


def test_agent_cannot_select_candidate_outside_packet():
    provider = _Provider("cand_not_in_packet")
    result = _pipeline(provider).run(NOW)
    assert result is not None
    assert result.decision.action == AgentAction.ABSTAIN
    assert result.selected_candidate is None
    assert "agent_failure" in result.decision.rationale


def test_pipeline_selects_only_whitelisted_candidate_and_journals():
    candidate = _candidate()
    provider = _Provider(candidate.candidate_id)
    journal = _Journal()
    result = _pipeline(provider, journal=journal).run(NOW)
    assert result is not None
    assert result.decision.action == AgentAction.SELECT_CANDIDATE
    assert result.selected_candidate is not None
    assert result.selected_candidate.candidate_id == candidate.candidate_id
    assert result.decision.size_scalar <= result.packet.risk_envelope.max_size_scalar
    assert journal.events[0][0] == "agent_decision"


def test_deterministic_veto_prevents_agent_call():
    provider = _Provider(_candidate().candidate_id)
    result = _pipeline(provider, veto=True).run(NOW)
    assert result is not None
    assert result.decision.action == AgentAction.ABSTAIN
    assert "stale_chain" in result.decision.rationale
    assert provider.calls == 0
