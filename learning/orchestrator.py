"""
learning/orchestrator.py
========================
LearningOrchestrator — daily/evening/weekly/manual coordination.

Never writes a standalone champion disconnected from the V2/V3 stack.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from learning.deployment_evaluation import evaluate_deployment_bundle
from learning.drift_evaluation import evaluate_drift
from learning.model_training import retrain_eligible_families
from learning.rule_training import produce_rule_config_candidate
from learning.settlement import settle_session_counterfactuals
from learning.weight_updates import update_dynamic_weights


@dataclass
class LearningOrchestrator:
    registry: Any = None
    store: Any = None
    configs_dir: str = "configs"
    state: dict = field(default_factory=dict)

    def run_daily(self, *, session_date: str, journal_rows: list | None = None,
                  candidate_evaluations: list | None = None,
                  settled_sessions: list | None = None) -> dict:
        settlement = settle_session_counterfactuals(
            session_date=session_date,
            journal_rows=journal_rows,
            candidate_evaluations=candidate_evaluations,
        )
        drift = evaluate_drift(metrics=self.state.get("drift_metrics"))
        # Only update weights from caller-supplied settled sessions with
        # real component losses — never invent empty loss records.
        sessions = list(settled_sessions or self.state.get("settled_sessions") or [])
        real_sessions = [
            s for s in sessions
            if isinstance(s, dict) and s.get("component_losses")
        ]
        weights = dict(self.state.get("weights") or {})
        if real_sessions and settlement.get("complete"):
            weights = update_dynamic_weights(
                settled_sessions=real_sessions,
                current_weights=weights,
            )
            self.state["weights"] = weights
        self.state["last_daily"] = session_date
        return {
            "session_date": session_date,
            "settlement": settlement,
            "drift": drift,
            "weights": weights,
            "weight_update_applied": bool(real_sessions and settlement.get("complete")),
            "promoted": False,
        }

    def run_evening(self, *, session_date: str, **kwargs) -> dict:
        return self.run_daily(session_date=session_date, **kwargs)

    def run_weekly(
        self,
        *,
        sessions: list | None = None,
        rule_overrides: dict | None = None,
    ) -> dict:
        sessions = list(sessions or [])
        # Outer-test sessions must remain untouched by selection — caller
        # is responsible for partitioning; we only refuse empty holdout.
        if not sessions:
            raise ValueError("holdout mandatory: sessions list is empty")
        train = retrain_eligible_families(
            sessions=sessions, registry=self.registry)
        rule_candidate = None
        if rule_overrides:
            rule_candidate = produce_rule_config_candidate(
                overrides=rule_overrides).to_dict()
        return {
            "training": train,
            "rule_candidate": rule_candidate,
            "promoted": False,
            "sessions_count": len(sessions),
        }

    def run_manual(
        self,
        *,
        deployment_id: str,
        comparison_deployment_id: Optional[str] = None,
        sessions: list | None = None,
        metrics: dict | None = None,
    ) -> dict:
        evaluation = evaluate_deployment_bundle(
            deployment_id=deployment_id,
            comparison_deployment_id=comparison_deployment_id,
            sessions=sessions,
            metrics=metrics,
        )
        return {"evaluation": evaluation, "promoted": False}
