"""
prediction/part3_decision.py
============================
Complete V3 candidate / execution / meta decision path
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §11.4 / PR5).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from decision_stack.contracts import CandidateEvaluation


@dataclass
class V3DecisionResult:
    statistical_action: str
    final_action: str
    candidate_id: Optional[str] = None
    structure: Optional[str] = None
    direction: Optional[str] = None
    size_mult: float = 1.0
    reasons: tuple = ()
    evaluations: tuple = ()
    ranking: Optional[dict] = None
    execution: Optional[dict] = None
    meta: Optional[dict] = None
    selected_candidate_evaluation: Optional[dict] = None
    component_errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "statistical_action": self.statistical_action,
            "final_action": self.final_action,
            "candidate_id": self.candidate_id,
            "structure": self.structure,
            "direction": self.direction,
            "size_mult": self.size_mult,
            "reasons": list(self.reasons),
            "evaluations": [
                e.to_dict() if hasattr(e, "to_dict") else e
                for e in self.evaluations
            ],
            "ranking": self.ranking,
            "execution": self.execution,
            "meta": self.meta,
            "selected_candidate_evaluation": self.selected_candidate_evaluation,
            "component_errors": dict(self.component_errors),
        }


def _cand_field(c: Any, name: str, default=None):
    if isinstance(c, dict):
        return c.get(name, default)
    return getattr(c, name, default)


def _as_rank_dict(c: Any) -> dict:
    if isinstance(c, dict):
        out = dict(c)
        out.setdefault("candidate_id", str(c.get("candidate_id") or ""))
        return out
    return {
        "candidate_id": str(getattr(c, "candidate_id", "") or ""),
        "family": getattr(c, "family", None),
        "direction": getattr(c, "direction", None),
        "ev": getattr(c, "ev", None),
        "prob_profit": getattr(c, "prob_profit", None),
        "absolute_utility": getattr(c, "absolute_utility", None),
    }


def build_v3_candidate_evaluations(
    *,
    snapshot: Any,
    forecast: Any,
    universe: Any,
    runtime: Any = None,
    mode: str = "shadow",
) -> tuple[CandidateEvaluation, ...]:
    """Score every candidate in the shared universe."""
    candidates = tuple(getattr(universe, "candidates", ()) or ())
    if not candidates:
        return ()

    value_model = None
    rank_model = None
    if runtime is not None and getattr(runtime, "artifacts", None):
        art = runtime.artifacts
        value_model = art.candidate_value
        rank_model = art.candidate_rank

    utilities: dict[str, float] = {}
    evaluations: list[CandidateEvaluation] = []
    for c in candidates:
        cid = str(_cand_field(c, "candidate_id") or "")
        legacy_ev = _cand_field(c, "ev")
        legacy_pop = _cand_field(c, "prob_profit")
        legacy_score = _cand_field(c, "score")
        expected_net = None
        p_pos = None
        util = None
        model_versions: dict = {}

        if value_model is not None:
            try:
                pred = (value_model.predict([c])
                        if hasattr(value_model, "predict") else None)
                if pred is not None:
                    first = pred[0] if isinstance(pred, (list, tuple)) else pred
                    expected_net = float(
                        getattr(first, "expected_net_pnl", None)
                        or getattr(first, "expected_pnl", 0)
                        or 0)
                    p_pos = float(
                        getattr(first, "p_positive_pnl", None)
                        or getattr(first, "p_positive", 0.5)
                        or 0.5)
                    util = float(
                        getattr(first, "utility", None) or expected_net)
                    model_versions["candidate_value"] = "trained"
            except Exception:
                model_versions["candidate_value"] = "failed"

        if util is None:
            util = (float(legacy_ev)
                    if isinstance(legacy_ev, (int, float)) else 0.0)
            expected_net = util
            p_pos = (float(legacy_pop)
                     if isinstance(legacy_pop, (int, float)) else 0.5)
            model_versions["candidate_value"] = "legacy_ev_baseline"
            if mode in ("candidate", "champion") and value_model is None:
                evaluations.append(CandidateEvaluation(
                    candidate_id=cid,
                    legacy_score=(
                        float(legacy_score) if legacy_score is not None
                        else None),
                    legacy_ev=(
                        float(legacy_ev) if legacy_ev is not None else None),
                    legacy_prob_profit=(
                        float(legacy_pop) if legacy_pop is not None else None),
                    vetoes=("required_component_missing",),
                    model_versions=model_versions,
                    diagnostics={"mode": mode},
                ))
                continue

        utilities[cid] = float(util)
        evaluations.append(CandidateEvaluation(
            candidate_id=cid,
            legacy_score=(
                float(legacy_score) if legacy_score is not None else None),
            legacy_ev=float(legacy_ev) if legacy_ev is not None else None,
            legacy_prob_profit=(
                float(legacy_pop) if legacy_pop is not None else None),
            expected_net_pnl=expected_net,
            p_positive_pnl=p_pos,
            absolute_utility=float(util),
            model_versions=model_versions,
        ))

    ranked_ids: list[str] = []
    ranking_uncertainty = 0.25
    try:
        from prediction.models.candidate_rank import PairwiseCandidateRanker
        ranker = (rank_model if rank_model is not None
                  else PairwiseCandidateRanker())
        rank_inputs = [_as_rank_dict(c) for c in candidates]
        ranking = ranker.rank_snapshot(
            getattr(snapshot, "snapshot_id", ""),
            rank_inputs,
            absolute_utilities=utilities,
            vetoed_ids=set(),
        )
        ranked_ids = list(
            getattr(ranking, "ordered_candidate_ids", None)
            or getattr(ranking, "ordered_ids", None)
            or [])
        if not ranked_ids:
            # Fall back to ranking rows if present
            rows = getattr(ranking, "rows", None) or getattr(
                ranking, "ranked", None) or []
            for row in rows:
                if isinstance(row, dict) and row.get("candidate_id"):
                    ranked_ids.append(str(row["candidate_id"]))
                elif hasattr(row, "candidate_id"):
                    ranked_ids.append(str(row.candidate_id))
        if not ranked_ids and getattr(ranking, "top_candidate_id", None):
            ranked_ids = [str(ranking.top_candidate_id)]
        ranking_uncertainty = float(
            getattr(ranking, "ranking_uncertainty", None) or 0.25)
    except Exception:
        ranked_ids = sorted(utilities, key=utilities.get, reverse=True)

    id_to_rank = {cid: i + 1 for i, cid in enumerate(ranked_ids)}
    out: list[CandidateEvaluation] = []
    for ev in evaluations:
        rank = id_to_rank.get(ev.candidate_id)
        fill_p = None
        concession = None
        fees = 1.0
        eov = None
        fill_price = None
        if (rank is not None and rank <= 5
                and ev.absolute_utility is not None):
            mid = float(ev.legacy_ev or 0.0)
            natural = mid * 0.85  # never treat mid as filled
            try:
                from execution.estimate_v3 import (
                    build_execution_estimate_v3, expected_order_value,
                )
                fam = "unknown"
                n_legs = 2
                for c in candidates:
                    if str(_cand_field(c, "candidate_id")) == ev.candidate_id:
                        fam = str(_cand_field(c, "family") or "unknown")
                        legs = _cand_field(c, "legs") or ()
                        n_legs = max(1, len(list(legs)) if legs else 2)
                        break
                est = build_execution_estimate_v3(
                    mid_credit=mid,
                    natural_credit=natural,
                    family=fam,
                    n_legs=n_legs,
                )
                fill_p = float(getattr(est, "p_fill", None)
                               or getattr(est, "fill_probability", 0.5)
                               or 0.5)
                concession = float(
                    getattr(est, "expected_concession", 0) or 0)
                fill_price = float(
                    getattr(est, "expected_fill_credit", None)
                    or getattr(est, "expected_fill_price", natural)
                    or natural)
                fees = float(getattr(est, "fees", 1.0) or 1.0)
                eov = float(expected_order_value(
                    p_fill=fill_p,
                    expected_net_pnl_given_fill=float(ev.absolute_utility),
                ))
            except Exception:
                fill_p = 0.5
                fill_price = natural
                eov = float(ev.absolute_utility) * fill_p - fees

        out.append(CandidateEvaluation(
            candidate_id=ev.candidate_id,
            legacy_score=ev.legacy_score,
            legacy_ev=ev.legacy_ev,
            legacy_prob_profit=ev.legacy_prob_profit,
            expected_net_pnl=ev.expected_net_pnl,
            p_positive_pnl=ev.p_positive_pnl,
            pnl_quantiles=dict(ev.pnl_quantiles),
            expected_shortfall=ev.expected_shortfall,
            absolute_utility=ev.absolute_utility,
            pairwise_rank_score=(
                float(ev.absolute_utility)
                if ev.absolute_utility is not None else None),
            final_rank=rank,
            ranking_uncertainty=ranking_uncertainty if rank else None,
            fill_probability=fill_p,
            expected_fill_price=fill_price,
            conservative_fill_price=(
                (fill_price * 0.95) if fill_price is not None else None),
            expected_concession=concession,
            fees=fees if fill_p is not None else None,
            expected_exit_cost=None,
            expected_order_value=eov,
            model_versions=dict(ev.model_versions),
            vetoes=tuple(ev.vetoes),
            diagnostics=dict(ev.diagnostics),
        ))
    return tuple(out)


def build_v3_decision(
    *,
    snapshot: Any,
    forecast: Any,
    universe: Any,
    runtime: Any = None,
    model_set: Any = None,
    hard_vetoes: tuple[str, ...] = (),
    mode: str = "shadow",
) -> V3DecisionResult:
    """
    Full Part 3 path: value → utility → rank → fill → EOV → meta → hard veto.
    """
    errors: dict = {}
    if mode in ("candidate", "champion"):
        if runtime is not None and getattr(runtime, "artifacts", None):
            art = runtime.artifacts
            required = (
                ("candidate_value", art.candidate_value),
                ("candidate_rank", art.candidate_rank),
                ("fill_probability", art.fill_probability),
                ("fill_concession", art.fill_concession),
                ("meta_model", art.meta_model),
            )
            missing = [n for n, m in required if m is None]
            if len(missing) == 5:
                return V3DecisionResult(
                    statistical_action="ABSTAIN",
                    final_action="ABSTAIN",
                    reasons=("required_component_missing",),
                    component_errors={"required": ",".join(missing)},
                )

    evaluations = build_v3_candidate_evaluations(
        snapshot=snapshot,
        forecast=forecast,
        universe=universe,
        runtime=runtime,
        mode=mode,
    )

    if not evaluations:
        return V3DecisionResult(
            statistical_action="NO_CANDIDATE",
            final_action=("HARD_VETO" if hard_vetoes else "NO_CANDIDATE"),
            reasons=(("hard_veto",) + tuple(hard_vetoes)
                     if hard_vetoes else ("no_candidate",)),
            evaluations=evaluations,
        )

    ranked = sorted(
        [e for e in evaluations if e.final_rank is not None],
        key=lambda e: e.final_rank or 999,
    )
    top = ranked[0] if ranked else evaluations[0]

    if "required_component_missing" in (top.vetoes or ()):
        action = "ABSTAIN"
        final_action, veto_reasons = _apply_vetoes(action, hard_vetoes)
        return V3DecisionResult(
            statistical_action=action,
            final_action=final_action,
            candidate_id=top.candidate_id,
            reasons=("required_component_missing",) + veto_reasons,
            evaluations=evaluations,
            selected_candidate_evaluation=top.to_dict(),
        )

    uncertainty = float(getattr(forecast, "uncertainty", None) or 0.3)
    ood = float(getattr(forecast, "ood_score", None) or 0.1)
    dq = float(getattr(forecast, "data_quality", None) or 0.8)
    p_pos = float(top.p_positive_pnl or 0.5)
    eov = float(top.expected_order_value
                if top.expected_order_value is not None else 0.0)
    util = float(top.absolute_utility
                 if top.absolute_utility is not None else 0.0)

    try:
        from prediction.models.trade_meta import decide_meta_action
        action, reasons = decide_meta_action(
            p_positive_utility=p_pos,
            expected_order_value=eov,
            selected_candidate_id=top.candidate_id,
            selected_candidate_utility=util,
            composite_uncertainty=uncertainty,
            ood_score=ood,
            data_quality=dq,
        )
        meta_dict = {
            "action": action,
            "reasons": list(reasons),
            "expected_order_value": eov,
            "p_positive_utility": p_pos,
        }
    except Exception as exc:
        errors["meta"] = f"{type(exc).__name__}: {exc}"
        action, reasons = "ABSTAIN", ("meta_failed",)
        meta_dict = {"action": action, "reasons": list(reasons)}

    final_action, veto_reasons = _apply_vetoes(action, hard_vetoes)

    fam = None
    direction = None
    for c in getattr(universe, "candidates", ()) or ():
        if str(_cand_field(c, "candidate_id")) == top.candidate_id:
            fam = _cand_field(c, "family")
            direction = _cand_field(c, "direction")
            break

    return V3DecisionResult(
        statistical_action=str(action),
        final_action=str(final_action),
        candidate_id=top.candidate_id,
        structure=str(fam) if fam else None,
        direction=str(direction) if direction else None,
        reasons=tuple(reasons) + veto_reasons,
        evaluations=evaluations,
        ranking={
            "top_candidate_id": top.candidate_id,
            "final_rank": top.final_rank,
        },
        execution={
            "fill_probability": top.fill_probability,
            "expected_fill_price": top.expected_fill_price,
            "expected_order_value": top.expected_order_value,
            "fees": top.fees,
            "note": "midpoint_diagnostic_only",
        },
        meta=meta_dict,
        selected_candidate_evaluation=top.to_dict(),
        component_errors=errors,
    )


def _apply_vetoes(action: str, hard_vetoes: tuple[str, ...]) -> tuple[str, tuple]:
    from prediction.models.trade_meta import apply_hard_vetoes
    final, reasons = apply_hard_vetoes(action, hard_vetoes)
    return str(final), tuple(reasons or ())
