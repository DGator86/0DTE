"""
prediction/reports/part3_evaluation.py
======================================
Part 3 evaluation report sections (§41).
"""
from __future__ import annotations

from typing import Optional


def build_part3_evaluation_report(
    *,
    candidate_value: Optional[dict] = None,
    candidate_ranking: Optional[dict] = None,
    execution: Optional[dict] = None,
    meta_decision: Optional[dict] = None,
    dynamic_weights: Optional[dict] = None,
    drift: Optional[dict] = None,
    deployment: Optional[dict] = None,
) -> dict:
    """Machine-readable Part 3 evaluation bundle with required sections."""
    return {
        "candidate_value": dict(candidate_value or {}),
        "candidate_ranking": dict(candidate_ranking or {}),
        "execution": dict(execution or {}),
        "meta_decision": dict(meta_decision or {}),
        "dynamic_weights": dict(dynamic_weights or {}),
        "drift": dict(drift or {}),
        "deployment": dict(deployment or {}),
        "sections": [
            "candidate_value",
            "candidate_ranking",
            "execution",
            "meta_decision",
            "dynamic_weights",
            "drift",
            "deployment",
        ],
    }


def render_part3_evaluation_markdown(report: dict) -> str:
    lines = ["# Part 3 Evaluation Report", ""]
    for section in report.get("sections") or []:
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append("```")
        lines.append(str(report.get(section) or {}))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
