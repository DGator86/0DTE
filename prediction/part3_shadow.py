"""
prediction/part3_shadow.py
=========================
Shadow-only Part 3 decision sequence helper (§32).

Never alters legacy tickets. Component failures are recorded and excluded;
missing required components prevent V3 advisory action.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from execution.estimate_v3 import build_execution_estimate_v3, expected_order_value
from policy.contracts import TradeDecisionV3
from prediction.models.candidate_rank import PairwiseCandidateRanker
from prediction.models.trade_meta import (
    MetaThresholdConfig, apply_hard_vetoes, decide_meta_action,
)


@dataclass
class Part3ShadowResult:
    decision: TradeDecisionV3
    ranking: Optional[dict] = None
    execution: Optional[dict] = None
    meta: Optional[dict] = None
    component_errors: dict = field(default_factory=dict)
    legacy_unchanged: bool = True


def run_part3_shadow_decision(
    *,
    snapshot_id: str,
    ts: str,
    symbol: str,
    candidates: list,
    absolute_utilities: dict,
    mid_credit: float,
    natural_credit: float,
    family: str,
    n_legs: int,
    hard_vetoes: tuple = (),
    composite_uncertainty: float = 0.2,
    ood_score: float = 0.1,
    data_quality: float = 0.9,
    p_positive_utility: float = 0.65,
    mode: str = "shadow",
    configuration_hash: str = "",
    ranker: Optional[PairwiseCandidateRanker] = None,
) -> Part3ShadowResult:
    """Deterministic shadow path using provided utilities (no network)."""
    errors: dict = {}
    ranking_dict = None
    execution_dict = None
    top_id = None
    util = None
    p_fill = None
    eov = None

    try:
        ranker = ranker or PairwiseCandidateRanker()
        ranking = ranker.rank_snapshot(
            snapshot_id, candidates,
            absolute_utilities=absolute_utilities,
            vetoed_ids=set(),
        )
        ranking_dict = ranking.to_dict()
        top_id = ranking.top_candidate_id
        util = absolute_utilities.get(top_id) if top_id else None
    except Exception as exc:  # noqa: BLE001 — isolate component
        errors["candidate_rank"] = f"{type(exc).__name__}: {exc}"

    try:
        est = build_execution_estimate_v3(
            mid_credit=mid_credit,
            natural_credit=natural_credit,
            family=family,
            n_legs=n_legs,
        )
        execution_dict = est.to_dict()
        p_fill = est.p_fill
        eov = expected_order_value(
            est.p_fill, float(util or 0.0), opportunity_cost_unfilled=0.0)
    except Exception as exc:  # noqa: BLE001
        errors["execution"] = f"{type(exc).__name__}: {exc}"

    statistical, reasons = decide_meta_action(
        p_positive_utility=p_positive_utility,
        expected_order_value=float(eov if eov is not None else -1.0),
        selected_candidate_id=top_id,
        selected_candidate_utility=util,
        composite_uncertainty=composite_uncertainty,
        ood_score=ood_score,
        data_quality=data_quality,
        cfg=MetaThresholdConfig(),
    )
    if errors:
        statistical = "ABSTAIN"
        reasons = tuple(reasons) + ("required_component_missing",)

    final, vetoes = apply_hard_vetoes(statistical, hard_vetoes)
    out_reasons = tuple(reasons)
    if final == "HARD_VETO":
        out_reasons = out_reasons + tuple(f"hard_veto:{v}" for v in vetoes)

    decision = TradeDecisionV3(
        snapshot_id=snapshot_id,
        ts=ts,
        symbol=symbol,
        action=final,
        statistical_action=statistical,
        selected_candidate_id=top_id,
        direction="unknown",
        family=family if top_id else None,
        p_positive_utility=p_positive_utility,
        expected_order_value=eov,
        candidate_utility=util,
        confidence=max(0.0, 1.0 - composite_uncertainty),
        uncertainty=composite_uncertainty,
        ood_score=ood_score,
        fill_probability=p_fill,
        hard_vetoes=tuple(hard_vetoes),
        reasons=out_reasons,
        policy_version="v3.0.0-part3",
        model_versions={"part3": "v3.0.0"},
        source="v3_shadow",
        mode=mode,
        diagnostics={
            "configuration_hash": configuration_hash,
            "component_errors": errors,
            "legacy_unchanged": True,
        },
    )
    return Part3ShadowResult(
        decision=decision,
        ranking=ranking_dict,
        execution=execution_dict,
        meta={"action": decision.action, "reasons": list(decision.reasons)},
        component_errors=errors,
        legacy_unchanged=True,
    )
