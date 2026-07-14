"""
prediction/part3_decision.py
============================
Complete V3 candidate / execution / meta decision path
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §11.4 / PR5).

Fail-closed when any required Part 3 artifact is missing in candidate/champion.
Never treats midpoint as filled. Never uses EV as a market price.
Never invents fill_p=0.5 on failure.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from decision_stack.contracts import CandidateEvaluation


class Part3DecisionError(RuntimeError):
    """Required Part 3 component missing or unusable."""


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
        "candidate_id": str(getattr(c, "candidate_id", "")
                            or getattr(c, "v2_candidate_id", "") or ""),
        "family": getattr(c, "family", None),
        "direction": getattr(c, "direction", None),
        "ev": getattr(c, "ev", None),
        "credit": getattr(c, "credit", None),
        "prob_profit": getattr(c, "prob_profit", None),
        "absolute_utility": getattr(c, "absolute_utility", None),
        "execution": getattr(c, "execution", None),
    }


def _mid_and_natural_credit(c: Any) -> tuple[Optional[float], Optional[float], dict]:
    """
    Extract market credit levels. NEVER use EV as a price.

    Prefer execution panel (natural / mid). Fall back to candidate.credit
    (mid) with natural derived only from the execution panel or an explicit
    natural field — not from EV.
    """
    diag: dict = {}
    exec_panel = _cand_field(c, "execution")
    mid = None
    natural = None
    if isinstance(exec_panel, dict):
        mid = exec_panel.get("mid_credit", exec_panel.get("mid"))
        natural = exec_panel.get(
            "natural_credit", exec_panel.get("natural"))
        diag["source"] = "execution_panel"
    credit = _cand_field(c, "credit")
    if mid is None and isinstance(credit, (int, float)):
        mid = float(credit)
        diag["source"] = diag.get("source") or "candidate.credit"
    if natural is None:
        # Without an executable natural quote, leave None — do not invent
        # natural = mid * 0.85 from EV or from mid alone in decision path.
        nat_field = _cand_field(c, "natural_credit")
        if isinstance(nat_field, (int, float)):
            natural = float(nat_field)
            diag["natural_source"] = "candidate.natural_credit"
        elif isinstance(exec_panel, dict) and exec_panel.get("natural") is not None:
            natural = float(exec_panel["natural"])
            diag["natural_source"] = "execution_panel.natural"
        else:
            diag["natural_source"] = "unavailable"
    if mid is not None:
        mid = float(mid)
    if natural is not None:
        natural = float(natural)
        if mid is not None and natural > mid + 1e-9:
            natural = mid
    return mid, natural, diag


def _required_part3_missing(runtime: Any) -> list[str]:
    if runtime is None or not getattr(runtime, "artifacts", None):
        return [
            "candidate_value", "candidate_rank", "fill_probability",
            "fill_concession", "meta_model",
        ]
    art = runtime.artifacts
    required = (
        ("candidate_value", art.candidate_value),
        ("candidate_rank", art.candidate_rank),
        ("fill_probability", art.fill_probability),
        ("fill_concession", art.fill_concession),
        ("meta_model", art.meta_model),
    )
    return [n for n, m in required if m is None]


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

    missing = _required_part3_missing(runtime) if mode in (
        "candidate", "champion") else []
    if missing and mode in ("candidate", "champion"):
        # Fail closed: emit vetoed rows and stop economic invention.
        return tuple(
            CandidateEvaluation(
                candidate_id=str(_cand_field(c, "candidate_id") or ""),
                vetoes=("required_component_missing",),
                diagnostics={"missing": list(missing), "mode": mode},
            )
            for c in candidates
        )

    value_model = None
    rank_model = None
    fill_p_model = None
    fill_c_model = None
    if runtime is not None and getattr(runtime, "artifacts", None):
        art = runtime.artifacts
        value_model = art.candidate_value
        rank_model = art.candidate_rank
        fill_p_model = art.fill_probability
        fill_c_model = art.fill_concession

    utilities: dict[str, float] = {}
    evaluations: list[CandidateEvaluation] = []
    cand_by_id: dict[str, Any] = {}
    adapted_by_id: dict[str, dict] = {}

    # Batch typed candidate-value inference when a trained model is present.
    if value_model is not None and hasattr(value_model, "predict_v3"):
        try:
            from prediction.adapters import (
                AdapterError, adapt_candidate_forecast_v3, candidate_value_rows,
                verify_candidate_feature_schema,
            )
            mkt = getattr(snapshot, "market", None)
            raw = getattr(snapshot, "raw_features", None) or {}
            spot = None
            if mkt is not None:
                spot = getattr(mkt, "spot", None)
            if spot is None:
                spot = raw.get("spot")
            rows, ids = candidate_value_rows(
                candidates,
                snapshot_id=str(getattr(snapshot, "snapshot_id", "") or ""),
                spot=float(spot) if spot is not None else None,
                call_wall=(getattr(mkt, "call_wall", None)
                           if mkt is not None else raw.get("call_wall")),
                put_wall=(getattr(mkt, "put_wall", None)
                          if mkt is not None else raw.get("put_wall")),
                gamma_flip=(getattr(mkt, "gamma_flip", None)
                            if mkt is not None else raw.get("gamma_flip")),
                minutes_to_close=raw.get("minutes_to_close"),
                net_gex=(getattr(mkt, "net_gex", None)
                         if mkt is not None else raw.get("net_gex")),
                data_quality=(
                    float(getattr(forecast, "data_quality"))
                    if forecast is not None
                    and getattr(forecast, "data_quality", None) is not None
                    else None),
            )
            # Schema parity: refuse silent median-imputation of trained cols.
            meta = getattr(value_model, "metadata", None) or {}
            trained_names = None
            vec = getattr(value_model, "vectorizer", None)
            if vec is not None and getattr(vec, "feature_names", None):
                trained_names = list(vec.feature_names)
            missing_feats = verify_candidate_feature_schema(
                rows,
                expected_hash=meta.get("candidate_feature_schema_hash"),
                trained_feature_names=trained_names,
            )
            if missing_feats and mode in ("candidate", "champion"):
                raise AdapterError(
                    f"candidate feature schema mismatch; missing={missing_feats[:8]}")
            preds = value_model.predict_v3(rows, candidate_ids=ids)
            for pred in preds:
                adapted = adapt_candidate_forecast_v3(pred)
                adapted_by_id[adapted["candidate_id"]] = adapted
        except Exception as exc:
            if mode in ("candidate", "champion"):
                return tuple(
                    CandidateEvaluation(
                        candidate_id=str(
                            _cand_field(c, "candidate_id")
                            or _cand_field(c, "v2_candidate_id") or ""),
                        vetoes=("candidate_value_unusable",),
                        diagnostics={"error": f"{type(exc).__name__}: {exc}"},
                    )
                    for c in candidates
                )

    for c in candidates:
        cid = str(
            _cand_field(c, "candidate_id")
            or _cand_field(c, "v2_candidate_id")
            or "")
        cand_by_id[cid] = c
        legacy_ev = _cand_field(c, "ev")
        legacy_pop = _cand_field(c, "prob_profit")
        legacy_score = _cand_field(c, "score")
        expected_net = None
        p_pos = None
        util = None
        pnl_quantiles: dict = {}
        expected_shortfall = None
        model_versions: dict = {}

        adapted = adapted_by_id.get(cid)
        if adapted is not None:
            expected_net = adapted["expected_net_pnl"]
            p_pos = adapted["p_positive_pnl"]
            util = adapted["absolute_utility"]
            pnl_quantiles = dict(adapted["pnl_quantiles"])
            expected_shortfall = adapted["expected_shortfall"]
            model_versions.update(adapted["model_versions"])

        if util is None:
            # Labeled baseline from legacy EV — research/shadow only.
            if mode in ("candidate", "champion"):
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
                    model_versions={"candidate_value": "absent"},
                    diagnostics={"mode": mode},
                ))
                continue
            util = (float(legacy_ev)
                    if isinstance(legacy_ev, (int, float)) else 0.0)
            expected_net = util
            p_pos = (float(legacy_pop)
                     if isinstance(legacy_pop, (int, float)) else 0.5)
            model_versions["candidate_value"] = "legacy_ev_baseline"

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
            pnl_quantiles=pnl_quantiles,
            expected_shortfall=expected_shortfall,
            absolute_utility=float(util),
            model_versions=model_versions,
        ))

    # Pairwise ranking
    ranked_ids: list[str] = []
    ranking_uncertainty = 0.25
    try:
        from prediction.models.candidate_rank import PairwiseCandidateRanker
        ranker = (rank_model if rank_model is not None
                  else PairwiseCandidateRanker())
        if mode in ("candidate", "champion") and rank_model is None:
            raise Part3DecisionError("candidate_rank required")
        rank_inputs = [_as_rank_dict(c) for c in candidates
                       if str(_cand_field(c, "candidate_id")
                              or _cand_field(c, "v2_candidate_id") or "")
                       in utilities]
        ranking = ranker.rank_snapshot(
            getattr(snapshot, "snapshot_id", ""),
            rank_inputs,
            absolute_utilities=utilities,
            vetoed_ids=set(),
        )
        ranked_ids = list(getattr(ranking, "ordered_candidate_ids", ()) or [])
        if not ranked_ids and getattr(ranking, "top_candidate_id", None):
            ranked_ids = [str(ranking.top_candidate_id)]
        ranking_uncertainty = float(
            getattr(ranking, "ranking_uncertainty", None) or 0.25)
    except Exception as exc:
        if mode in ("candidate", "champion"):
            return tuple(
                CandidateEvaluation(
                    candidate_id=e.candidate_id,
                    vetoes=tuple(e.vetoes) + ("ranking_failed",),
                    diagnostics={"rank_error": str(exc)},
                    model_versions=dict(e.model_versions),
                )
                for e in evaluations
            )
        ranked_ids = sorted(utilities, key=utilities.get, reverse=True)

    id_to_rank = {cid: i + 1 for i, cid in enumerate(ranked_ids)}
    out: list[CandidateEvaluation] = []
    for ev in evaluations:
        rank = id_to_rank.get(ev.candidate_id)
        fill_p = None
        concession = None
        fees = None
        eov = None
        fill_price = None
        cons_fill = None
        exec_diag: dict = {"note": "midpoint_diagnostic_only"}
        model_versions = dict(ev.model_versions)

        if (rank is not None and rank <= 5
                and ev.absolute_utility is not None
                and "required_component_missing" not in (ev.vetoes or ())):
            c = cand_by_id.get(ev.candidate_id)
            mid, natural, credit_diag = _mid_and_natural_credit(c)
            exec_diag["credit"] = credit_diag
            fam = str(_cand_field(c, "family") or "unknown")
            legs = _cand_field(c, "legs") or ()
            n_legs = max(1, len(list(legs)) if legs else 2)

            if mid is None or natural is None:
                # Cannot price without real mid+natural credits.
                exec_diag["execution_status"] = "credit_unavailable"
                if mode in ("candidate", "champion"):
                    out.append(CandidateEvaluation(
                        candidate_id=ev.candidate_id,
                        legacy_score=ev.legacy_score,
                        legacy_ev=ev.legacy_ev,
                        legacy_prob_profit=ev.legacy_prob_profit,
                        expected_net_pnl=ev.expected_net_pnl,
                        p_positive_pnl=ev.p_positive_pnl,
                        absolute_utility=ev.absolute_utility,
                        final_rank=rank,
                        ranking_uncertainty=ranking_uncertainty,
                        vetoes=tuple(ev.vetoes) + ("credit_unavailable",),
                        model_versions=model_versions,
                        diagnostics=exec_diag,
                    ))
                    continue
            else:
                try:
                    from execution.estimate_v3 import (
                        build_execution_estimate_v3, expected_order_value,
                    )
                    p_fill_arg = None
                    exp_frac = None
                    cons_frac = None
                    fill_unc = None
                    versions: dict = {}

                    # Use loaded fill-probability model when fitted.
                    if fill_p_model is not None and getattr(
                            fill_p_model, "fitted", False):
                        from prediction.adapters import (
                            fill_attempt_features_from_candidate,
                        )
                        quote_age = (
                            getattr(snapshot, "source_ages_seconds", {})
                            or {}).get("chain")
                        if (mode in ("candidate", "champion")
                                and quote_age is None):
                            raise Part3DecisionError(
                                "quote_age_seconds unknown; refuse optimistic "
                                "fill inference in candidate/champion mode")
                        feats = fill_attempt_features_from_candidate(
                            candidate=c,
                            mid_credit=mid,
                            natural_credit=natural,
                            family=fam,
                            n_legs=n_legs,
                            quote_age_seconds=quote_age,
                            minutes_to_close=(
                                getattr(snapshot, "raw_features", {})
                                or {}).get("minutes_to_close"),
                            data_quality=float(
                                getattr(forecast, "data_quality", None) or 0.0)
                            if forecast is not None else None,
                        )
                        fp = fill_p_model.predict(feats, family=fam)
                        p_fill_arg = float(fp.p_fill_60s)
                        fill_unc = float(fp.uncertainty)
                        versions["fill_probability"] = getattr(
                            fp, "model_version", "trained")
                        model_versions["fill_probability"] = "trained"
                    elif mode in ("candidate", "champion"):
                        raise Part3DecisionError(
                            "fill_probability model missing or unfitted")

                    # Use loaded fill-concession model when fitted.
                    if fill_c_model is not None and getattr(
                            fill_c_model, "fitted", False):
                        from prediction.adapters import (
                            fill_attempt_features_from_candidate,
                        )
                        feats = fill_attempt_features_from_candidate(
                            candidate=c,
                            mid_credit=mid,
                            natural_credit=natural,
                            family=fam,
                            n_legs=n_legs,
                        )
                        fc = fill_c_model.predict(
                            feats, family=fam, n_legs=n_legs)
                        exp_frac = float(fc.expected_fill_fraction)
                        cons_frac = float(fc.conservative_fill_fraction)
                        versions["fill_concession"] = getattr(
                            fc, "model_version", "trained")
                        model_versions["fill_concession"] = "trained"
                    elif mode in ("candidate", "champion"):
                        raise Part3DecisionError(
                            "fill_concession model missing or unfitted")

                    est = build_execution_estimate_v3(
                        mid_credit=mid,
                        natural_credit=natural,
                        family=fam,
                        n_legs=n_legs,
                        p_fill=p_fill_arg,
                        expected_fill_fraction=exp_frac,
                        conservative_fill_fraction=cons_frac,
                        fill_uncertainty=fill_unc,
                        model_versions=versions,
                        require_empirical=(mode in ("candidate", "champion")),
                    )
                    fill_p = float(est.p_fill)
                    fill_price = float(est.expected_credit)
                    cons_fill = float(est.conservative_credit)
                    concession = float(
                        abs(mid - fill_price) if mid is not None else 0.0)
                    fees = float(est.entry_fees) + float(est.expected_exit_fees)
                    if ev.expected_net_pnl is None:
                        if mode in ("candidate", "champion"):
                            raise Part3DecisionError(
                                "expected_net_pnl required for EOV")
                        net_for_eov = 0.0
                    else:
                        net_for_eov = float(ev.expected_net_pnl)
                    eov = float(expected_order_value(
                        p_fill=fill_p,
                        expected_net_pnl_given_fill=net_for_eov,
                    ))
                    exec_diag.update({
                        "fallback_level": est.fallback_level,
                        "model_versions": dict(est.model_versions),
                        "execution_status": "ok",
                        "mid_credit": mid,
                        "natural_credit": natural,
                        "expected_credit": fill_price,
                        "conservative_credit": cons_fill,
                    })
                except Exception as exc:
                    exec_diag["execution_status"] = "failed"
                    exec_diag["error"] = f"{type(exc).__name__}: {exc}"
                    if mode in ("candidate", "champion"):
                        out.append(CandidateEvaluation(
                            candidate_id=ev.candidate_id,
                            legacy_score=ev.legacy_score,
                            legacy_ev=ev.legacy_ev,
                            legacy_prob_profit=ev.legacy_prob_profit,
                            expected_net_pnl=ev.expected_net_pnl,
                            p_positive_pnl=ev.p_positive_pnl,
                            absolute_utility=ev.absolute_utility,
                            final_rank=rank,
                            ranking_uncertainty=ranking_uncertainty,
                            vetoes=tuple(ev.vetoes) + ("execution_failed",),
                            model_versions=model_versions,
                            diagnostics=exec_diag,
                        ))
                        continue
                    # Research/shadow: leave fill fields None — do NOT invent
                    # fill_p=0.5. Ranking/utility still available.
                    fill_p = None
                    fill_price = None
                    eov = None

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
            conservative_fill_price=cons_fill,
            expected_concession=concession,
            fees=fees,
            expected_exit_cost=None,
            expected_order_value=eov,
            model_versions=model_versions,
            vetoes=tuple(ev.vetoes),
            diagnostics=exec_diag,
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
    if mode in ("candidate", "champion") and forecast is None:
        return V3DecisionResult(
            statistical_action="UNAVAILABLE",
            final_action="UNAVAILABLE",
            reasons=("forecast_unavailable",),
            component_errors={"forecast": "None"},
        )
    if mode in ("candidate", "champion"):
        # Require uncertainty, OOD, and data_quality — all must be present
        # (None is missing; 0.0 is a valid observed value).
        for field_name in ("uncertainty", "ood_score", "data_quality"):
            if getattr(forecast, field_name, None) is None:
                return V3DecisionResult(
                    statistical_action="UNAVAILABLE",
                    final_action="UNAVAILABLE",
                    reasons=(f"forecast_{field_name}_unavailable",),
                )

    missing = _required_part3_missing(runtime)
    if mode in ("candidate", "champion") and missing:
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

    # Prefer ranked, non-vetoed candidates
    actionable = [
        e for e in evaluations
        if e.final_rank is not None
        and not e.vetoes
    ]
    ranked = sorted(actionable, key=lambda e: e.final_rank or 999)
    if not ranked:
        # All vetoed / unranked
        top = sorted(
            evaluations,
            key=lambda e: (e.final_rank is None, e.final_rank or 999),
        )[0]
        action = "ABSTAIN"
        final_action, veto_reasons = _apply_vetoes(action, hard_vetoes)
        return V3DecisionResult(
            statistical_action=action,
            final_action=final_action,
            candidate_id=top.candidate_id,
            reasons=("no_actionable_candidate",) + veto_reasons,
            evaluations=evaluations,
            selected_candidate_evaluation=top.to_dict(),
            component_errors=errors,
        )

    top = ranked[0]
    # Explicit None checks — never use truthiness defaults that turn 0.0 into
    # optimistic values (data_quality=0 → 0.8, p_pos=0 → 0.5, etc.).
    unc_raw = getattr(forecast, "uncertainty", None)
    ood_raw = getattr(forecast, "ood_score", None)
    dq_raw = getattr(forecast, "data_quality", None)
    p_pos_raw = top.p_positive_pnl
    if mode in ("candidate", "champion"):
        required = {
            "uncertainty": unc_raw,
            "ood_score": ood_raw,
            "data_quality": dq_raw,
            "p_positive_pnl": p_pos_raw,
        }
        missing_fields = [n for n, v in required.items() if v is None]
        if missing_fields:
            return V3DecisionResult(
                statistical_action="UNAVAILABLE",
                final_action="UNAVAILABLE",
                candidate_id=top.candidate_id,
                reasons=("forecast_fields_unavailable",) + tuple(
                    f"missing:{n}" for n in missing_fields),
                evaluations=evaluations,
                selected_candidate_evaluation=top.to_dict(),
                component_errors={"missing_fields": ",".join(missing_fields)},
            )
        if top.expected_net_pnl is None:
            return V3DecisionResult(
                statistical_action="UNAVAILABLE",
                final_action="UNAVAILABLE",
                candidate_id=top.candidate_id,
                reasons=("expected_net_pnl_unavailable",),
                evaluations=evaluations,
                selected_candidate_evaluation=top.to_dict(),
            )
        uncertainty = float(unc_raw)
        ood = float(ood_raw)
        dq = float(dq_raw)
        p_pos = float(p_pos_raw)
    else:
        # Research/shadow: labeled defaults only when truly absent (None).
        uncertainty = float(unc_raw) if unc_raw is not None else 0.3
        ood = float(ood_raw) if ood_raw is not None else 0.1
        dq = float(dq_raw) if dq_raw is not None else 0.8
        p_pos = float(p_pos_raw) if p_pos_raw is not None else 0.5
    eov = float(top.expected_order_value
                if top.expected_order_value is not None else 0.0)
    util = float(top.absolute_utility
                 if top.absolute_utility is not None else 0.0)

    meta_dict: dict = {}
    action = "ABSTAIN"
    reasons: tuple = ()

    try:
        from prediction.models.trade_meta import (
            decide_meta_action, meta_features_from_inputs,
        )
        meta_model = None
        if runtime is not None and runtime.artifacts.meta_model is not None:
            meta_model = runtime.artifacts.meta_model

        if meta_model is not None and getattr(meta_model, "fitted", False):
            feats = meta_features_from_inputs(
                forecast={
                    "uncertainty": uncertainty,
                    "ood_score": ood,
                    "data_quality": dq,
                },
                candidate={
                    "absolute_utility": util,
                    "p_positive_pnl": p_pos,
                },
                execution={
                    "expected_order_value": eov,
                    "fill_probability": top.fill_probability,
                },
            )
            meta = meta_model.decide(
                feats,
                expected_order_value=eov,
                selected_candidate_id=top.candidate_id,
                selected_candidate_utility=util,
                composite_uncertainty=uncertainty,
                ood_score=ood,
                data_quality=dq,
                hard_vetoes=(),  # applied below once
            )
            # Statistical action before hard vetoes
            action = str(meta.diagnostics.get("statistical_action")
                         or meta.action)
            if action == "HARD_VETO":
                action = "ABSTAIN"  # hard veto applied separately
            reasons = tuple(r for r in meta.reasons
                            if not str(r).startswith("hard_veto"))
            meta_dict = meta.to_dict()
            meta_dict["model_source"] = "trained_meta_model"
        elif mode in ("candidate", "champion"):
            return V3DecisionResult(
                statistical_action="ABSTAIN",
                final_action="ABSTAIN",
                candidate_id=top.candidate_id,
                reasons=("meta_model_unusable",),
                evaluations=evaluations,
                selected_candidate_evaluation=top.to_dict(),
                component_errors={"meta_model": "missing_or_unfitted"},
            )
        else:
            # Research/shadow threshold baseline (explicitly labeled).
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
                "model_source": "threshold_baseline",
            }
    except Exception as exc:
        errors["meta"] = f"{type(exc).__name__}: {exc}"
        if mode in ("candidate", "champion"):
            return V3DecisionResult(
                statistical_action="ABSTAIN",
                final_action="ABSTAIN",
                candidate_id=top.candidate_id,
                reasons=("meta_failed",),
                evaluations=evaluations,
                selected_candidate_evaluation=top.to_dict(),
                component_errors=errors,
            )
        action, reasons = "ABSTAIN", ("meta_failed",)
        meta_dict = {"action": action, "reasons": list(reasons)}

    final_action, veto_reasons = _apply_vetoes(action, hard_vetoes)

    fam = None
    direction = None
    for c in getattr(universe, "candidates", ()) or ():
        cid = str(_cand_field(c, "candidate_id")
                  or _cand_field(c, "v2_candidate_id") or "")
        if cid == top.candidate_id:
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
            "credit_diagnostics": (top.diagnostics or {}).get("credit"),
        },
        meta=meta_dict,
        selected_candidate_evaluation=top.to_dict(),
        component_errors=errors,
    )


def _apply_vetoes(action: str, hard_vetoes: tuple[str, ...]) -> tuple[str, tuple]:
    from prediction.models.trade_meta import apply_hard_vetoes
    final, reasons = apply_hard_vetoes(action, hard_vetoes)
    return str(final), tuple(reasons or ())
