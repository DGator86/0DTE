"""
decision_stack/stack.py
=======================
UnifiedDecisionStack — single evaluate() entry point
(docs/UNIFIED_V1_V2_V3_HANDOFF.md §9.2, §10).

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from decision_stack.authority import coerce_size_mult, resolve_authority
from decision_stack.contracts import UnifiedDecisionRecord


@dataclass
class UnifiedDecisionStack:
    """
    Required evaluation order (§9.2):
      snapshot → V1 baseline → V2/V3 forecast → physical dist →
      candidate universe once → legacy score → V3 eval → rank →
      execution → meta → hard vetoes → authority → persist.
    """

    deployment: Any
    prediction_runtime: Any = None
    legacy_config: Any = None
    stores: dict = field(default_factory=dict)
    # Optional callables for V1 integration without circular imports.
    legacy_baseline_fn: Optional[Callable] = None
    candidate_universe_fn: Optional[Callable] = None
    legacy_score_fn: Optional[Callable] = None
    hard_veto_fn: Optional[Callable] = None
    persist_fn: Optional[Callable] = None
    # (candidate_id, session_date) -> tuple[str, ...] risk veto reasons.
    # Called after V3 selection; rejections enter hard_vetoes before final auth.
    portfolio_risk_fn: Optional[Callable] = None

    def evaluate(
        self,
        snapshot: Any,
        *,
        position_contexts: list | None = None,
        legacy_decision: Any = None,
        hard_vetoes: tuple[str, ...] | None = None,
    ) -> UnifiedDecisionRecord:
        from prediction.forecast_assembly import build_v3_forecast
        from prediction.part3_decision import build_v3_decision

        deployment = self.deployment
        mode = getattr(deployment, "mode", "shadow")
        dep_id = getattr(deployment, "deployment_id", "")
        cfg_hash = getattr(deployment, "configuration_hash", "")
        fallback_policy = getattr(deployment, "fallback_policy", "abstain")

        diagnostics: dict = {"stages": []}
        forecast = None
        universe = None
        v3_result = None
        forecast_summary: dict = {}
        model_versions: dict = {}

        # 1. Validate snapshot
        snapshot_id = getattr(snapshot, "snapshot_id", None) or ""
        ts = getattr(snapshot, "ts", "") or ""
        session_date = getattr(snapshot, "session_date", "") or ""
        symbol = getattr(snapshot, "symbol", "") or ""
        diagnostics["stages"].append("snapshot_validated")

        # 2. V1 baseline (caller may precompute and pass legacy_decision)
        if legacy_decision is None and self.legacy_baseline_fn is not None:
            legacy_decision = self.legacy_baseline_fn(
                snapshot, position_contexts=position_contexts)
        diagnostics["stages"].append("legacy_baseline")

        # 3–4. V2/V3 forecast + physical distribution inputs
        try:
            if self.prediction_runtime is not None:
                forecast = self.prediction_runtime.forecast(snapshot)
            else:
                forecast = build_v3_forecast(
                    snapshot=snapshot, runtime=None, mode=mode)
            if forecast is not None:
                if hasattr(forecast, "to_dict"):
                    forecast_summary = {
                        k: forecast.to_dict().get(k)
                        for k in (
                            "p_up_30m", "expected_return_30m",
                            "uncertainty", "ood_score", "data_quality",
                            "regime_probabilities", "model_versions",
                        )
                        if k in forecast.to_dict()
                    }
                    model_versions = dict(
                        getattr(forecast, "model_versions", {}) or {})
                elif isinstance(forecast, dict):
                    forecast_summary = dict(forecast.get("summary") or forecast)
                    model_versions = dict(forecast.get("model_versions") or {})
            diagnostics["stages"].append("forecast")
        except Exception as exc:
            diagnostics["forecast_error"] = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            forecast = None

        if mode in ("candidate", "champion") and forecast is None:
            # Fail closed — do not continue into candidate economics with
            # invented uncertainty / probability defaults.
            v3_result = {
                "statistical_action": "UNAVAILABLE",
                "final_action": "UNAVAILABLE",
                "candidate_id": None,
                "reasons": ("forecast_unavailable",),
            }
            # Still need legacy action for disagreement / fallback routing
            legacy_action = "NO_EDGE"
            legacy_cand = None
            legacy_struct = None
            legacy_dir = None
            legacy_size = 1.0
            if legacy_decision is not None:
                if isinstance(legacy_decision, dict):
                    legacy_action = str(
                        legacy_decision.get("action")
                        or legacy_decision.get("final_action")
                        or "NO_EDGE")
                    legacy_cand = legacy_decision.get("candidate_id")
                    legacy_struct = legacy_decision.get("structure")
                    legacy_dir = legacy_decision.get("direction")
                    legacy_size = coerce_size_mult(
                        legacy_decision.get("size_mult"), default=1.0)
            hard = tuple(hard_vetoes or ())
            auth = resolve_authority(
                mode=mode,
                legacy_decision={
                    "action": legacy_action,
                    "candidate_id": legacy_cand,
                    "structure": legacy_struct,
                    "direction": legacy_dir,
                    "size_mult": legacy_size,
                },
                v3_decision=v3_result,
                hard_vetoes=hard,
                fallback_policy=fallback_policy,
                legacy_size_mult=legacy_size,
                v3_size_mult=0.0,
            )
            record = UnifiedDecisionRecord(
                snapshot_id=str(snapshot_id),
                ts=str(ts),
                session_date=str(session_date),
                symbol=str(symbol),
                deployment_id=str(dep_id),
                deployment_mode=str(mode),
                authority_source=auth.authority_source,
                legacy_action=legacy_action,
                legacy_candidate_id=legacy_cand,
                legacy_structure=legacy_struct,
                legacy_direction=legacy_dir,
                legacy_size_mult=legacy_size,
                v3_statistical_action="UNAVAILABLE",
                v3_final_action="UNAVAILABLE",
                selected_candidate_id=auth.selected_candidate_id,
                final_action=auth.final_action,
                final_structure=auth.final_structure,
                final_direction=auth.final_direction,
                final_size_mult=auth.final_size_mult,
                hard_vetoes=hard,
                reasons=tuple(auth.reasons) + ("forecast_unavailable",),
                fallback_used=auth.fallback_used,
                fallback_reason=auth.fallback_reason or "forecast_unavailable",
                forecast_summary={},
                configuration_hash=str(cfg_hash),
                diagnostics=diagnostics,
            )
            if self.persist_fn is not None:
                try:
                    self.persist_fn(record, snapshot=snapshot)
                except Exception as exc:
                    diagnostics["persist_error"] = str(exc)
            return record

        # 5. Candidate universe once
        if self.candidate_universe_fn is not None:
            universe = self.candidate_universe_fn(snapshot, forecast=forecast)
        else:
            from prediction.candidate_universe import build_candidate_universe
            universe = build_candidate_universe(
                snapshot_id=snapshot_id,
                generated_at=ts,
                candidates=(),
                diagnostics={"note": "empty_universe_no_generator"},
            )
        diagnostics["stages"].append("candidate_universe")
        diagnostics["candidate_count"] = len(
            getattr(universe, "candidates", ()) or ())

        # 6. Legacy scoring (optional hook)
        if self.legacy_score_fn is not None and universe is not None:
            legacy_decision = self.legacy_score_fn(
                snapshot, universe, legacy_decision) or legacy_decision
        diagnostics["stages"].append("legacy_score")

        # 7–10. V3 candidate eval, rank, execution, meta
        hard = tuple(hard_vetoes or ())
        if hard_vetoes is None and self.hard_veto_fn is not None:
            hard = tuple(self.hard_veto_fn(snapshot) or ())

        try:
            v3_result = build_v3_decision(
                snapshot=snapshot,
                forecast=forecast,
                universe=universe,
                runtime=self.prediction_runtime,
                hard_vetoes=hard,
                mode=mode,
            )
            diagnostics["stages"].append("v3_decision")
        except Exception as exc:
            diagnostics["v3_error"] = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            v3_result = {
                "statistical_action": "UNAVAILABLE",
                "final_action": "UNAVAILABLE",
                "candidate_id": None,
                "reasons": ("v3_decision_failed",),
            }

        # Normalize legacy action
        legacy_action = "NO_EDGE"
        legacy_cand = None
        legacy_struct = None
        legacy_dir = None
        legacy_size = 1.0
        if legacy_decision is not None:
            if isinstance(legacy_decision, dict):
                legacy_action = str(
                    legacy_decision.get("action")
                    or legacy_decision.get("final_action")
                    or "NO_EDGE")
                legacy_cand = legacy_decision.get("candidate_id")
                legacy_struct = legacy_decision.get("structure")
                legacy_dir = legacy_decision.get("direction")
                legacy_size = coerce_size_mult(
                    legacy_decision.get("size_mult"), default=1.0)
            else:
                take = getattr(legacy_decision, "take", None)
                if take is True:
                    legacy_action = "TRADE"
                elif take is False:
                    legacy_action = "NO_EDGE"
                else:
                    legacy_action = str(
                        getattr(legacy_decision, "action", None)
                        or getattr(legacy_decision, "final_action", None)
                        or "NO_EDGE")
                cand = getattr(legacy_decision, "candidate", None)
                legacy_cand = (
                    getattr(cand, "candidate_id", None) if cand is not None
                    else getattr(legacy_decision, "candidate_id", None))
                legacy_struct = (
                    getattr(cand, "family", None) if cand is not None
                    else getattr(legacy_decision, "structure", None))
                legacy_dir = getattr(legacy_decision, "direction", None)
                legacy_size = coerce_size_mult(
                    getattr(legacy_decision, "size_mult", None), default=1.0)

        if isinstance(v3_result, dict):
            v3_stat = str(v3_result.get("statistical_action")
                          or v3_result.get("final_action")
                          or "UNAVAILABLE")
            v3_final = str(v3_result.get("final_action") or v3_stat)
            v3_cand = v3_result.get("candidate_id")
            v3_struct = v3_result.get("structure")
            v3_dir = v3_result.get("direction")
            v3_reasons = tuple(v3_result.get("reasons") or ())
            selected_eval = v3_result.get("selected_candidate_evaluation")
            v3_size = coerce_size_mult(
                v3_result.get("size_mult"), default=1.0)
        else:
            v3_stat = str(getattr(v3_result, "statistical_action",
                                  "UNAVAILABLE"))
            v3_final = str(getattr(v3_result, "final_action", v3_stat))
            v3_cand = getattr(v3_result, "candidate_id", None)
            v3_struct = getattr(v3_result, "structure", None)
            v3_dir = getattr(v3_result, "direction", None)
            v3_reasons = tuple(getattr(v3_result, "reasons", ()) or ())
            selected_eval = getattr(
                v3_result, "selected_candidate_evaluation", None)
            v3_size = coerce_size_mult(
                getattr(v3_result, "size_mult", None), default=1.0)

        # Portfolio risk against the V3-selected candidate enters hard vetoes
        # before final authority (champion must not bypass RiskManager).
        if (self.portfolio_risk_fn is not None
                and str(v3_final).upper() == "TRADE"
                and v3_cand):
            try:
                risk_extra = self.portfolio_risk_fn(
                    str(v3_cand), str(session_date)) or ()
                if risk_extra:
                    hard = hard + tuple(str(x) for x in risk_extra)
                    diagnostics["portfolio_risk_vetoes"] = list(risk_extra)
                    # Reflect risk into V3 final action so paper intents and
                    # dual-account candidate mode also see the veto.
                    from prediction.part3_decision import _apply_vetoes
                    v3_final, risk_reasons = _apply_vetoes(v3_final, hard)
                    v3_reasons = v3_reasons + tuple(risk_reasons)
            except Exception as exc:
                diagnostics["portfolio_risk_error"] = str(exc)
                hard = hard + ("risk:check_failed",)
                v3_final = "HARD_VETO"
                v3_reasons = v3_reasons + ("risk:check_failed",)

        # 11–12. Hard vetoes already applied inside build_v3_decision;
        # authority router re-checks (including portfolio risk).
        auth = resolve_authority(
            mode=mode,
            legacy_decision={
                "action": legacy_action,
                "candidate_id": legacy_cand,
                "structure": legacy_struct,
                "direction": legacy_dir,
                "size_mult": legacy_size,
            },
            v3_decision={
                "statistical_action": v3_stat,
                "final_action": v3_final,
                "candidate_id": v3_cand,
                "structure": v3_struct,
                "direction": v3_dir,
                "size_mult": v3_size,
            },
            hard_vetoes=hard,
            fallback_policy=fallback_policy,
            legacy_size_mult=legacy_size,
            v3_size_mult=v3_size,
        )
        diagnostics["stages"].append("authority")

        # If authority still wants TRADE on a non-V3 candidate (legacy
        # shadow/reference), also portfolio-check that candidate.
        if (self.portfolio_risk_fn is not None
                and auth.final_action == "TRADE"
                and auth.selected_candidate_id
                and str(auth.selected_candidate_id) != str(v3_cand or "")):
            try:
                risk_extra = self.portfolio_risk_fn(
                    str(auth.selected_candidate_id), str(session_date)) or ()
                if risk_extra:
                    hard = hard + tuple(str(x) for x in risk_extra)
                    diagnostics["portfolio_risk_vetoes_auth"] = list(risk_extra)
                    auth = resolve_authority(
                        mode=mode,
                        legacy_decision={
                            "action": legacy_action,
                            "candidate_id": legacy_cand,
                            "structure": legacy_struct,
                            "direction": legacy_dir,
                            "size_mult": legacy_size,
                        },
                        v3_decision={
                            "statistical_action": v3_stat,
                            "final_action": v3_final,
                            "candidate_id": v3_cand,
                            "structure": v3_struct,
                            "direction": v3_dir,
                            "size_mult": v3_size,
                        },
                        hard_vetoes=hard,
                        fallback_policy=fallback_policy,
                        legacy_size_mult=legacy_size,
                        v3_size_mult=v3_size,
                    )
            except Exception as exc:
                diagnostics["portfolio_risk_error_auth"] = str(exc)

        disagreement = {
            "action": legacy_action != v3_final,
            "legacy_action": legacy_action,
            "v3_action": v3_final,
            "candidate": legacy_cand != v3_cand,
            "legacy_candidate_id": legacy_cand,
            "v3_candidate_id": v3_cand,
            "structure": legacy_struct != v3_struct,
            "direction": legacy_dir != v3_dir,
        }

        record = UnifiedDecisionRecord(
            snapshot_id=str(snapshot_id),
            ts=str(ts),
            session_date=str(session_date),
            symbol=str(symbol),
            deployment_id=str(dep_id),
            deployment_mode=str(mode),
            authority_source=auth.authority_source,
            legacy_action=legacy_action,
            legacy_candidate_id=legacy_cand,
            legacy_structure=legacy_struct,
            legacy_direction=legacy_dir,
            legacy_size_mult=legacy_size,
            v3_statistical_action=v3_stat,
            v3_final_action=v3_final,
            v3_candidate_id=v3_cand,
            v3_structure=v3_struct,
            v3_direction=v3_dir,
            selected_candidate_id=auth.selected_candidate_id,
            final_action=auth.final_action,
            final_structure=auth.final_structure,
            final_direction=auth.final_direction,
            final_size_mult=auth.final_size_mult,
            hard_vetoes=hard,
            reasons=tuple(auth.reasons) + v3_reasons,
            fallback_used=auth.fallback_used,
            fallback_reason=auth.fallback_reason,
            forecast_summary=forecast_summary,
            selected_candidate_evaluation=(
                selected_eval if isinstance(selected_eval, dict)
                else (selected_eval.to_dict()
                      if selected_eval is not None
                      and hasattr(selected_eval, "to_dict")
                      else selected_eval)),
            legacy_v3_disagreement=disagreement,
            model_versions=model_versions,
            configuration_hash=str(cfg_hash),
            diagnostics=diagnostics,
        )

        # 13. Persist
        if self.persist_fn is not None:
            try:
                self.persist_fn(record, snapshot=snapshot, universe=universe,
                                forecast=forecast, v3_result=v3_result)
                diagnostics["stages"].append("persisted")
            except Exception as exc:
                diagnostics["persist_error"] = str(exc)

        return record
