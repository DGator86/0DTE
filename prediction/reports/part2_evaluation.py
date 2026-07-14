"""
prediction/reports/part2_evaluation.py
======================================
Part 2 evaluation report scaffold (V3 Part 2 §49, PR 16).

Assembles structural / regime / expert / return / competing-risk / path /
ensemble sections. Headline CIs must be session-bootstrapped by callers
using prediction.session_bootstrap when metrics arrays are provided.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Part2EvaluationReport:
    structural_data_quality: dict = field(default_factory=dict)
    regime_model: dict = field(default_factory=dict)
    experts: dict = field(default_factory=dict)
    return_distribution: dict = field(default_factory=dict)
    competing_risk: dict = field(default_factory=dict)
    path_model: dict = field(default_factory=dict)
    ensemble: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_part2_evaluation_report(
    *,
    structural_summary: Optional[dict] = None,
    regime_metrics: Optional[dict] = None,
    expert_metrics: Optional[dict] = None,
    return_metrics: Optional[dict] = None,
    competing_risk_metrics: Optional[dict] = None,
    path_metrics: Optional[dict] = None,
    ensemble_metrics: Optional[dict] = None,
) -> Part2EvaluationReport:
    """Assemble a Part 2 evaluation report from precomputed section dicts."""
    return Part2EvaluationReport(
        structural_data_quality=dict(structural_summary or {}),
        regime_model=dict(regime_metrics or {}),
        experts=dict(expert_metrics or {}),
        return_distribution=dict(return_metrics or {}),
        competing_risk=dict(competing_risk_metrics or {}),
        path_model=dict(path_metrics or {}),
        ensemble=dict(ensemble_metrics or {}),
        diagnostics={"report_version": "v3.part2"},
    )


def summarize_structural_quality(states: list[dict]) -> dict[str, Any]:
    """Lightweight structural quality summary from StructuralState.to_dict rows."""
    if not states:
        return {"n": 0}
    n = len(states)
    oi = sum(1 for s in states if s.get("net_gex_oi") is not None)
    vol = sum(1 for s in states if s.get("net_gex_volume") is not None)
    hyb = sum(1 for s in states if s.get("net_gex_hybrid") is not None)
    qs = [float(s["quality_score"]) for s in states
          if s.get("quality_score") is not None]
    disag = [float(s["gex_disagreement"]) for s in states
             if s.get("gex_disagreement") is not None]
    return {
        "n": n,
        "source_availability": {
            "oi": oi / n, "volume": vol / n, "hybrid": hyb / n,
        },
        "mean_quality_score": (sum(qs) / len(qs)) if qs else None,
        "mean_gex_disagreement": (sum(disag) / len(disag)) if disag else None,
        "missingness": {
            "oi": 1.0 - oi / n,
            "volume": 1.0 - vol / n,
            "hybrid": 1.0 - hyb / n,
        },
    }
