"""
adaptive_learning/reports.py
============================
Promotion-candidate reports. Every learning cycle that evaluates a challenger
persists the outcome two ways:

  * a validation_reports row with report_type="promotion_candidate" so the
    dashboard's Learning tab (and the existing Validation tab plumbing) can
    render it, and
  * a JSON + Markdown pair under reports/promotion/YYYY_MM_DD.json|.md for
    humans reviewing the pending candidate offline.

The headline metrics block follows the spec:

    {
      "current_config_score": -0.32,
      "candidate_config_score": 0.18,
      "holdout_score": 0.11,
      "changed_params": {"gate.max_adx": [20, 24], ...},   # [old, new]
      "reason": "gate_effectiveness_reversed",
      "promote": true
    }

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Optional

from journal import Journal


def changed_params_map(base_overrides: dict, best_params: dict,
                       base_engine_cfg=None) -> dict:
    """{param: [old, new]} for every searched parameter whose value moved.
    `old` comes from the champion's overrides when present, else from the
    dataclass default on base_engine_cfg."""
    from decision_engine import EngineConfig

    base = base_engine_cfg or EngineConfig()
    out: dict = {}
    for path, new in (best_params or {}).items():
        if path in (base_overrides or {}):
            old = base_overrides[path]
        else:
            prefix, _, key = path.partition(".")
            section = getattr(base, prefix, None)
            old = getattr(section, key, None) if section is not None else None
        if old != new:
            out[path] = [old, new]
    return out


def build_promotion_report(
    reason: str,
    champion_eval: dict,
    challenger_eval: dict,
    decision: dict,
    changed_params: dict,
    candidate: Optional[dict] = None,
    diagnostics: Optional[list] = None,
    stability: Optional[dict] = None,
    feature_report: Optional[dict] = None,
    walk_forward: Optional[dict] = None,
    holdout: Optional[dict] = None,
    optimizer_summary: Optional[dict] = None,
) -> dict:
    """Assemble the full promotion report (metrics for validation_reports)."""
    return {
        # -- spec headline block --
        "current_config_score": champion_eval.get("score"),
        "candidate_config_score": challenger_eval.get("score"),
        "holdout_score": challenger_eval.get("holdout_score"),
        "changed_params": changed_params,
        "reason": reason,
        "promote": bool(decision.get("promote")),
        # -- full detail --
        "champion_eval": champion_eval,
        "challenger_eval": challenger_eval,
        "decision": decision,
        "candidate": candidate or {},
        "diagnostics": diagnostics or [],
        "stability": stability or {},
        "feature_report": feature_report or {},
        "walk_forward": walk_forward or {},
        "holdout": holdout or {},
        "optimizer": optimizer_summary or {},
    }


def _summary_line(report: dict) -> str:
    verdict = "PROMOTE (pending human review)" if report["promote"] else "REJECT"
    n = len(report.get("changed_params") or {})
    return (f"Promotion candidate — {verdict}; reason={report.get('reason')}; "
            f"{n} param(s) changed; candidate score "
            f"{_num(report.get('candidate_config_score'))} vs champion "
            f"{_num(report.get('current_config_score'))}, holdout "
            f"{_num(report.get('holdout_score'))}")


def _num(v, nd: int = 4) -> str:
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def render_markdown(report: dict, report_date: str) -> str:
    """Human-readable promotion report."""
    lines = [
        "# Promotion Candidate Report",
        "",
        f"*Date:* {report_date}  ",
        f"*Reason:* `{report.get('reason')}`  ",
        f"*Recommendation:* **{'PROMOTE (pending human review)' if report['promote'] else 'REJECT'}**",
        "",
        "## Scores",
        "",
        "| metric | champion | challenger |",
        "|---|---|---|",
        f"| search score | {_num(report.get('current_config_score'))} "
        f"| {_num(report.get('candidate_config_score'))} |",
        f"| holdout score | {_num((report.get('champion_eval') or {}).get('holdout_score'))} "
        f"| {_num(report.get('holdout_score'))} |",
        f"| gate edge | {_num((report.get('champion_eval') or {}).get('gate_edge'))} "
        f"| {_num((report.get('challenger_eval') or {}).get('gate_edge'))} |",
        f"| Brier skill | {_num((report.get('champion_eval') or {}).get('brier_skill'))} "
        f"| {_num((report.get('challenger_eval') or {}).get('brier_skill'))} |",
        f"| trades | {(report.get('champion_eval') or {}).get('trade_count', 'n/a')} "
        f"| {(report.get('challenger_eval') or {}).get('trade_count', 'n/a')} |",
        f"| max drawdown | {_num((report.get('champion_eval') or {}).get('max_drawdown'))} "
        f"| {_num((report.get('challenger_eval') or {}).get('max_drawdown'))} |",
        f"| profitable folds | "
        f"{(report.get('champion_eval') or {}).get('n_profitable', '?')}/"
        f"{(report.get('champion_eval') or {}).get('n_folds', '?')} | "
        f"{(report.get('challenger_eval') or {}).get('n_profitable', '?')}/"
        f"{(report.get('challenger_eval') or {}).get('n_folds', '?')} |",
        "",
        "## Parameter changes",
        "",
    ]
    changed = report.get("changed_params") or {}
    if changed:
        lines += ["| parameter | old | new |", "|---|---|---|"]
        lines += [f"| `{k}` | {old} | {new} |" for k, (old, new) in changed.items()]
    else:
        lines.append("_none_")

    lines += ["", "## Promotion rules", ""]
    for r in (report.get("decision") or {}).get("rules", []):
        mark = "PASS" if r.get("passed") else "FAIL"
        lines.append(f"- **{mark}** `{r.get('name')}` — {r.get('detail')}")

    diags = report.get("diagnostics") or []
    if diags:
        lines += ["", "## Diagnostics", ""]
        for d in diags:
            d = d if isinstance(d, dict) else d.to_dict()
            lines.append(f"- [{d.get('severity', '?').upper()}] "
                         f"`{d.get('issue')}` (confidence "
                         f"{d.get('confidence', 0):.0%}) — {d.get('likely_cause')}")

    stab = report.get("stability") or {}
    if stab:
        lines += ["", "## Parameter stability", "",
                  "| parameter | verdict | sign consistency | sensitivity |",
                  "|---|---|---|---|"]
        for k, v in stab.items():
            lines.append(f"| `{k}` | {v.get('verdict')} "
                         f"| {v.get('fold_sign_consistency')} "
                         f"| {v.get('sensitivity')} |")

    lines += ["", "## Risk assessment", ""]
    if report["promote"]:
        lines.append(
            "All promotion rules passed on out-of-sample data. Residual risks: "
            "the holdout window is finite and regime-conditional; promote via "
            "`python3 -m adaptive_learning.promoter --approve <config_id>` only "
            "after reviewing the parameter changes above, and expect to revert "
            "if the next validation cycle degrades.")
    else:
        failing = (report.get("decision") or {}).get("failing_rules") or []
        lines.append("Rejected: " + (", ".join(f"`{f}`" for f in failing)
                                     if failing else "no failing rule recorded")
                     + ". The candidate config remains archived under "
                       "configs/candidates/ for reference; nothing changes live.")
    lines.append("")
    return "\n".join(lines)


def persist_promotion_report(jrn: Journal, report: dict,
                             reports_dir: str = os.path.join("reports", "promotion"),
                             report_date: Optional[str] = None) -> dict:
    """validation_reports row + reports/promotion/YYYY_MM_DD.{json,md}."""
    report_date = report_date or dt.date.today().isoformat()
    summary = _summary_line(report)
    flags = []
    if not report["promote"]:
        failing = (report.get("decision") or {}).get("failing_rules") or []
        flags = [{"flag": f"promotion_rule_failed:{name}", "severity": "info",
                  "detail": name} for name in failing]

    report_id = jrn.log_validation_report(
        report_date, "promotion_candidate", report, summary, flags)

    os.makedirs(reports_dir, exist_ok=True)
    stem = report_date.replace("-", "_")
    json_path = os.path.join(reports_dir, f"{stem}.json")
    md_path = os.path.join(reports_dir, f"{stem}.md")
    # Several cycles on one date: suffix rather than overwrite the audit trail.
    n = 1
    while os.path.exists(json_path):
        n += 1
        json_path = os.path.join(reports_dir, f"{stem}_{n}.json")
        md_path = os.path.join(reports_dir, f"{stem}_{n}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"report_date": report_date, "summary": summary,
                   "metrics": report}, f, indent=2, default=str)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report, report_date))

    return {"report_id": report_id, "json_path": json_path,
            "md_path": md_path, "summary": summary}


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    champion = {"score": -0.32, "holdout_score": -0.20, "gate_edge": -0.4,
                "brier_skill": 0.02, "trade_count": 40, "max_drawdown": 0.9,
                "n_folds": 4, "n_profitable": 1}
    challenger = {"score": 0.18, "holdout_score": 0.11, "gate_edge": 0.15,
                  "brier_skill": 0.04, "trade_count": 38, "max_drawdown": 0.6,
                  "n_folds": 4, "n_profitable": 3}
    from adaptive_learning.promoter import check_promotion
    decision = check_promotion(champion, challenger)
    report = build_promotion_report(
        reason="gate_effectiveness_reversed",
        champion_eval=champion, challenger_eval=challenger,
        decision=decision.to_dict(),
        changed_params={"gate.max_adx": [20.0, 24.0],
                        "gate.min_gex_pct_rank": [0.60, 0.50]},
    )
    print("=" * 70)
    print("  reports demo")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as d:
        jrn = Journal(":memory:")
        out = persist_promotion_report(jrn, report, reports_dir=d,
                                       report_date="2026-07-09")
        print(f"  {out['summary']}")
        print(f"  json -> {os.path.basename(out['json_path'])}  "
              f"md -> {os.path.basename(out['md_path'])}")
        rows = jrn.fetch_validation_reports(report_type="promotion_candidate")
        print(f"  validation_reports rows: {len(rows)}  "
              f"promote={rows[0]['metrics']['promote']}")
        jrn.close()
        print("-" * 70)
        with open(out["md_path"], encoding="utf-8") as f:
            print("\n".join("  " + line for line in f.read().splitlines()[:24]))
    print("=" * 70)
