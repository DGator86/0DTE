"""
decision_stack/contracts.py
===========================
UnifiedDecisionRecord and CandidateEvaluation contracts
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §7.4–§7.5).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FINAL_ACTIONS = (
    "TRADE",
    "NO_EDGE",
    "ABSTAIN",
    "HARD_VETO",
    "NO_CANDIDATE",
    "UNAVAILABLE",
)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    legacy_score: Optional[float] = None
    legacy_ev: Optional[float] = None
    legacy_prob_profit: Optional[float] = None
    expected_net_pnl: Optional[float] = None
    p_positive_pnl: Optional[float] = None
    pnl_quantiles: dict = field(default_factory=dict)
    expected_shortfall: Optional[float] = None
    absolute_utility: Optional[float] = None
    pairwise_rank_score: Optional[float] = None
    final_rank: Optional[int] = None
    ranking_uncertainty: Optional[float] = None
    fill_probability: Optional[float] = None
    expected_fill_price: Optional[float] = None
    conservative_fill_price: Optional[float] = None
    expected_concession: Optional[float] = None
    fees: Optional[float] = None
    expected_exit_cost: Optional[float] = None
    expected_order_value: Optional[float] = None
    model_versions: dict = field(default_factory=dict)
    vetoes: tuple = ()
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "legacy_score": self.legacy_score,
            "legacy_ev": self.legacy_ev,
            "legacy_prob_profit": self.legacy_prob_profit,
            "expected_net_pnl": self.expected_net_pnl,
            "p_positive_pnl": self.p_positive_pnl,
            "pnl_quantiles": dict(self.pnl_quantiles),
            "expected_shortfall": self.expected_shortfall,
            "absolute_utility": self.absolute_utility,
            "pairwise_rank_score": self.pairwise_rank_score,
            "final_rank": self.final_rank,
            "ranking_uncertainty": self.ranking_uncertainty,
            "fill_probability": self.fill_probability,
            "expected_fill_price": self.expected_fill_price,
            "conservative_fill_price": self.conservative_fill_price,
            "expected_concession": self.expected_concession,
            "fees": self.fees,
            "expected_exit_cost": self.expected_exit_cost,
            "expected_order_value": self.expected_order_value,
            "model_versions": dict(self.model_versions),
            "vetoes": list(self.vetoes),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class UnifiedDecisionRecord:
    snapshot_id: str
    ts: str
    session_date: str
    symbol: str
    deployment_id: str
    deployment_mode: str
    authority_source: str
    legacy_action: str
    legacy_candidate_id: Optional[str] = None
    legacy_structure: Optional[str] = None
    legacy_direction: Optional[str] = None
    legacy_size_mult: float = 1.0
    v3_statistical_action: str = "UNAVAILABLE"
    v3_final_action: str = "UNAVAILABLE"
    v3_candidate_id: Optional[str] = None
    v3_structure: Optional[str] = None
    v3_direction: Optional[str] = None
    selected_candidate_id: Optional[str] = None
    final_action: str = "UNAVAILABLE"
    final_structure: Optional[str] = None
    final_direction: Optional[str] = None
    final_size_mult: float = 1.0
    hard_vetoes: tuple = ()
    reasons: tuple = ()
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    forecast_summary: dict = field(default_factory=dict)
    selected_candidate_evaluation: Optional[dict] = None
    legacy_v3_disagreement: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)
    configuration_hash: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "ts": self.ts,
            "session_date": self.session_date,
            "symbol": self.symbol,
            "deployment_id": self.deployment_id,
            "deployment_mode": self.deployment_mode,
            "authority_source": self.authority_source,
            "legacy_action": self.legacy_action,
            "legacy_candidate_id": self.legacy_candidate_id,
            "legacy_structure": self.legacy_structure,
            "legacy_direction": self.legacy_direction,
            "legacy_size_mult": self.legacy_size_mult,
            "v3_statistical_action": self.v3_statistical_action,
            "v3_final_action": self.v3_final_action,
            "v3_candidate_id": self.v3_candidate_id,
            "v3_structure": self.v3_structure,
            "v3_direction": self.v3_direction,
            "selected_candidate_id": self.selected_candidate_id,
            "final_action": self.final_action,
            "final_structure": self.final_structure,
            "final_direction": self.final_direction,
            "final_size_mult": self.final_size_mult,
            "hard_vetoes": list(self.hard_vetoes),
            "reasons": list(self.reasons),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "forecast_summary": dict(self.forecast_summary),
            "selected_candidate_evaluation": self.selected_candidate_evaluation,
            "legacy_v3_disagreement": dict(self.legacy_v3_disagreement),
            "model_versions": dict(self.model_versions),
            "configuration_hash": self.configuration_hash,
            "diagnostics": dict(self.diagnostics),
        }
