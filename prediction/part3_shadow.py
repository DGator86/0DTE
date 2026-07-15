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


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def build_part3_live_payload(
    *,
    snapshot_id: str,
    ts: str,
    symbol: str,
    candidates: list | None = None,
    forecasts: dict | None = None,
    signals: dict | None = None,
    hard_vetoes: list | tuple = (),
    mode: str = "shadow",
    store=None,
    direction: str = "unknown",
) -> dict:
    """
    Build the dashboard `live_state.part3` object from a live shadow tick.

    Observation-only: never mutates legacy decisions. Failures become ABSTAIN
    with explicit component_errors — never silent zeros.
    """
    from dashboard.state import serialize_part3_decision

    signals = dict(signals or {})
    forecasts = dict(forecasts or {})
    candidates = list(candidates or [])

    # ---- assemble ranker inputs from V2 forecasts / candidates --------------
    cand_rows = []
    abs_utils = {}
    for c in candidates:
        cid = (
            getattr(c, "candidate_id", None)
            or getattr(c, "v2_candidate_id", None)
            or getattr(c, "_v2_candidate_id", None)
        )
        if not cid:
            continue
        fc = forecasts.get(cid)
        util = float(fc.utility_score) if fc is not None else float(
            getattr(c, "v2_utility_score", None)
            or getattr(c, "score", 0.0)
            or 0.0)
        feat = {
            "utility_score": util,
            "expected_net_pnl": float(fc.expected_net_pnl) if fc else util,
            "family": getattr(c, "family", None),
            "capital_required": float(
                getattr(c, "max_loss", 0.0) or getattr(c, "capital", 0.0) or 0.0),
            "n_legs": len(getattr(c, "legs", ()) or ()),
        }
        if fc is not None:
            feat["expected_shortfall"] = float(fc.expected_shortfall)
            feat["fill_uncertainty"] = float(fc.fill_uncertainty)
            feat["model_uncertainty"] = float(fc.model_uncertainty)
            feat["p_profit"] = float(fc.p_profit)
        cand_rows.append({
            "candidate_id": str(cid),
            "features": feat,
            "absolute_utility": util,
            "uncertainty": float(getattr(fc, "model_uncertainty", 0.0) or 0.0)
            if fc else 0.0,
            "capital": feat["capital_required"],
        })
        abs_utils[str(cid)] = util

    # Fallback: synthesize from flat signals when candidate objects absent
    if not cand_rows:
        v2_top = signals.get("v2_top_candidate_id")
        leg_top = signals.get("legacy_top_candidate_id")
        v2_u = float(signals.get("v2_utility_score") or 0.0)
        if v2_top:
            cand_rows.append({
                "candidate_id": str(v2_top),
                "features": {"utility_score": v2_u, "family": signals.get("v2_top_family")},
                "absolute_utility": v2_u,
            })
            abs_utils[str(v2_top)] = v2_u
        if leg_top and str(leg_top) != str(v2_top):
            cand_rows.append({
                "candidate_id": str(leg_top),
                "features": {
                    "utility_score": float(signals.get("legacy_top_score") or 0.0),
                    "family": signals.get("legacy_top_family"),
                },
                "absolute_utility": float(signals.get("legacy_top_score") or 0.0),
            })
            abs_utils[str(leg_top)] = float(signals.get("legacy_top_score") or 0.0)

    # Credit / natural from top candidate execution when available
    mid_credit = 0.50
    natural_credit = 0.30
    family = str(signals.get("v2_top_family") or signals.get("legacy_top_family")
                 or "unknown")
    n_legs = 2
    if candidates:
        top_c = None
        v2_top = signals.get("v2_top_candidate_id")
        for c in candidates:
            cid = (
                getattr(c, "candidate_id", None)
                or getattr(c, "v2_candidate_id", None)
                or getattr(c, "_v2_candidate_id", None)
            )
            if v2_top and cid == v2_top:
                top_c = c
                break
        if top_c is None:
            top_c = candidates[0]
        family = getattr(top_c, "family", family) or family
        n_legs = max(len(getattr(top_c, "legs", ()) or ()), 1)
        mid_credit = float(getattr(top_c, "credit", 0.0) or 0.0)
        ex = getattr(top_c, "execution", None) or {}
        if isinstance(ex, dict) and ex.get("natural_credit") is not None:
            natural_credit = float(ex["natural_credit"])
        elif isinstance(ex, dict) and ex.get("mid_credit") is not None:
            mid_credit = float(ex.get("mid_credit", mid_credit))
            natural_credit = float(ex.get("natural_credit", mid_credit * 0.6))
        else:
            # debit structures: credit < 0
            if mid_credit >= 0:
                natural_credit = mid_credit * 0.6
            else:
                natural_credit = mid_credit * 1.2

    unc = signals.get("v2_policy_uncertainty")
    if unc is None:
        unc = signals.get("policy_uncertainty")
    if unc is None:
        unc = signals.get("v2_fc_uncertainty", 0.35)
    ood = float(signals.get("ood_score") or signals.get("v2_fc_ood") or 0.1)
    dq = float(signals.get("data_quality") or signals.get("v2_fc_data_quality")
               or 0.85)
    # Map forecast confidence → meta probability prior when no trained meta model
    conf = signals.get("v2_policy_confidence")
    if conf is None:
        conf = signals.get("policy_confidence")
    if conf is None:
        conf = signals.get("v2_fc_p_up_close")
    p_pos = float(conf) if conf is not None else 0.55
    # Soft blend with utility sign
    top_util = max(abs_utils.values()) if abs_utils else 0.0
    if top_util > 0:
        p_pos = _clip01(0.5 * p_pos + 0.5 * min(0.9, 0.55 + top_util))
    elif top_util < 0:
        p_pos = _clip01(0.5 * p_pos + 0.5 * max(0.2, 0.45 + top_util))

    vetoes = tuple(str(v) for v in (hard_vetoes or ()) if v)

    result = run_part3_shadow_decision(
        snapshot_id=snapshot_id,
        ts=ts,
        symbol=symbol,
        candidates=cand_rows,
        absolute_utilities=abs_utils,
        mid_credit=mid_credit,
        natural_credit=natural_credit,
        family=family,
        n_legs=n_legs,
        hard_vetoes=vetoes,
        composite_uncertainty=_clip01(float(unc)),
        ood_score=_clip01(ood),
        data_quality=_clip01(dq),
        p_positive_utility=_clip01(p_pos),
        mode=mode,
    )
    # Patch direction on decision via to_dict roundtrip
    d = result.decision.to_dict()
    d["direction"] = direction or d.get("direction") or "unknown"
    from policy.contracts import TradeDecisionV3
    decision = TradeDecisionV3.from_dict(d)

    payload = serialize_part3_decision(decision, generated_at=ts)
    ranking = dict(result.ranking or {})
    ranking.setdefault("legacy_top_candidate_id",
                       signals.get("legacy_top_candidate_id"))
    ranking.setdefault("top_family", signals.get("v2_top_family") or family)
    if ranking.get("top_candidate_id") and abs_utils:
        tid = ranking["top_candidate_id"]
        ranking["absolute_utility"] = abs_utils.get(tid)
        ranking["combined_score"] = (ranking.get("combined_scores") or {}).get(tid)
        ranking["pairwise_score"] = (ranking.get("pairwise_scores") or {}).get(tid)
        ranking["expected_regret"] = (ranking.get("expected_regret") or {}).get(tid)
        ranking["top_score_margin"] = ranking.get("top_score_margin")
    payload["ranking"] = ranking
    payload["execution"] = dict(result.execution or {})
    payload["model_state"] = {
        "load_status": "ok" if not result.component_errors else "degraded",
        "data_quality": dq,
        "drift_severity": signals.get("drift_severity") or "NORMAL",
        "registry_statuses": ["shadow"],
    }
    payload["meta"] = dict(result.meta or {})
    payload["component_errors"] = dict(result.component_errors or {})
    if "note" in payload and result.decision.action:
        payload.pop("note", None)

    # Persist observation outputs (never raise into the tick)
    if store is not None:
        try:
            if result.ranking:
                store.log_candidate_ranking(
                    snapshot_id, decision.model_versions.get("part3", "v3.0.0"),
                    result.ranking, generated_at=ts, mode=mode)
            store.log_meta_decision(
                snapshot_id,
                decision.model_versions.get("part3", "v3.0.0"),
                result.meta or decision.to_dict(),
                generated_at=ts, mode=mode,
                candidate_id=decision.selected_candidate_id,
            )
        except Exception:
            pass
    return payload
