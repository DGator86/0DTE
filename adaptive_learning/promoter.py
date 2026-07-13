"""
adaptive_learning/promoter.py
=============================
Champion / challenger promotion. Two halves:

1. check_promotion() — the rule engine. A challenger is recommended for
   promotion only when EVERY rule passes:
     * holdout score beats the champion's
     * >= 3/4 walk-forward folds profitable
     * gate edge does not regress
     * Brier skill does not regress below zero
     * trade count >= 90% of the champion's (no trade-frequency collapse)
     * max drawdown not worse
     * no changed parameter judged 'unstable' by the stability engine
     * no active alert-severity drift diagnosis
   Any failure -> reject, with the failing rule named.

2. The human CLI — the ONLY code path that writes configs/champion.json:

     python3 -m adaptive_learning.promoter --list
     python3 -m adaptive_learning.promoter --approve <config_id> [--author NAME]
     python3 -m adaptive_learning.promoter --reject  <config_id>

   Approval archives the previous champion to configs/archive/ and installs
   the pending candidate; both actions update the journal's promotions and
   candidate_configs tables when a --db is given.

The learner never calls approve(). It stops at configs/promoted/
pending_review.json plus a promotions row with status 'pending_review'.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Optional

from adaptive_learning.config_store import (
    ConfigRecord, archive_dir, champion_path, load_config,
    pending_review_path, save_config,
)
from adaptive_learning.stability import stability_acceptable

PROMOTION_THRESHOLDS = {
    "min_profitable_fold_frac": 0.75,   # "3/4 or better folds profitable"
    "min_trade_count_ratio": 0.90,      # trade count must not collapse
    "min_brier_skill": 0.0,
    "max_drawdown_tolerance": 0.0,      # absolute $/share slack (0 = strict)
}


# --------------------------------------------------------------------------- #
# Evaluation extraction                                                         #
# --------------------------------------------------------------------------- #
def evaluation_from_wf(wf_result, score: Optional[float] = None,
                       holdout_score: Optional[float] = None) -> dict:
    """Flatten a WalkForwardResult (+ scores) into the comparable evaluation
    dict the rule engine consumes."""
    folds = list(getattr(wf_result, "folds", []) or [])
    edges, skills, dds = [], [], []
    trade_count = 0
    for f in folds:
        ts = f.tearsheet
        trade_count += ts.trade_ticks
        dds.append(ts.max_drawdown)
        eff = ts.gate_effectiveness or {}
        t = (eff.get("trades_taken") or {}).get("mean")
        b = (eff.get("blocked_by_gate") or {}).get("mean")
        if t is not None and b is not None:
            edges.append(t - b)
        if ts.brier_skill is not None:
            skills.append(ts.brier_skill)
    return {
        "score": score,
        "holdout_score": holdout_score,
        "n_folds": len(folds),
        "n_profitable": (wf_result.n_profitable() if folds else 0),
        "gate_edge": round(sum(edges) / len(edges), 4) if edges else None,
        "brier_skill": round(sum(skills) / len(skills), 4) if skills else None,
        "trade_count": trade_count,
        "max_drawdown": round(sum(dds) / len(dds), 6) if dds else 0.0,
    }


# --------------------------------------------------------------------------- #
# Rule engine                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class PromotionRule:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class PromotionDecision:
    promote: bool
    rules: list[PromotionRule] = field(default_factory=list)

    @property
    def failing(self) -> list[str]:
        return [r.name for r in self.rules if not r.passed]

    def to_dict(self) -> dict:
        return {"promote": self.promote,
                "failing_rules": self.failing,
                "rules": [r.to_dict() for r in self.rules]}


def _fmt(v, nd=4):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def check_promotion(champion_eval: dict, challenger_eval: dict,
                    stability: Optional[dict] = None,
                    changed_params: Optional[list[str]] = None,
                    diagnoses: Optional[list] = None,
                    thresholds: Optional[dict] = None) -> PromotionDecision:
    """Apply every promotion rule; ALL must pass. Metrics that are
    unmeasurable on BOTH sides (tiny windows) pass with a note — no regression
    is detectable — but a metric the champion has and the challenger lost is a
    failure."""
    cfg = {**PROMOTION_THRESHOLDS, **(thresholds or {})}
    rules: list[PromotionRule] = []

    def rule(name: str, passed: bool, detail: str) -> None:
        rules.append(PromotionRule(name, bool(passed), detail))

    # 1. holdout must improve vs champion holdout (no search-score fallback
    #    unless BOTH sides lack holdout — first cycle only).
    champ_hold = champion_eval.get("holdout_score")
    chall_hold = challenger_eval.get("holdout_score")
    if champ_hold is None and chall_hold is None:
        # First cycle with no holdout on either side: compare search scores.
        champ_ref = champion_eval.get("score")
        chall_ref = challenger_eval.get("score")
        rule("holdout_improves",
             chall_ref is not None and champ_ref is not None and chall_ref > champ_ref,
             f"no holdout on either side; challenger search {_fmt(chall_ref)} "
             f"vs champion search {_fmt(champ_ref)}")
    elif champ_hold is None:
        rule("holdout_improves",
             chall_hold is not None,
             f"challenger holdout {_fmt(chall_hold)}; champion has no holdout "
             f"(require challenger holdout present)")
    else:
        rule("holdout_improves",
             chall_hold is not None and chall_hold > champ_hold,
             f"challenger holdout {_fmt(chall_hold)} vs champion holdout "
             f"{_fmt(champ_hold)}")

    # 2. walk-forward consistency
    n_folds = challenger_eval.get("n_folds") or 0
    n_prof = challenger_eval.get("n_profitable") or 0
    frac = (n_prof / n_folds) if n_folds else 0.0
    rule("walk_forward_consistency",
         n_folds > 0 and frac >= cfg["min_profitable_fold_frac"],
         f"{n_prof}/{n_folds} folds profitable "
         f"(need >= {cfg['min_profitable_fold_frac']:.0%})")

    # 3. gate edge must not regress
    ce, che = champion_eval.get("gate_edge"), challenger_eval.get("gate_edge")
    if ce is None and che is None:
        rule("gate_edge_improves", True,
             "gate edge unmeasurable on both sides (no blocked candidates)")
    elif che is None:
        rule("gate_edge_improves", False,
             f"champion gate edge {_fmt(ce)} but challenger's is unmeasurable")
    elif ce is None:
        rule("gate_edge_improves", che > 0,
             f"challenger gate edge {_fmt(che)} (champion unmeasurable; "
             f"require > 0)")
    else:
        rule("gate_edge_improves", che > ce,
             f"challenger gate edge {_fmt(che)} vs champion {_fmt(ce)}")

    # 4. Brier skill must remain positive AND improve when champion measurable
    cs, chs = champion_eval.get("brier_skill"), challenger_eval.get("brier_skill")
    if cs is None and chs is None:
        rule("brier_skill_positive", True,
             "Brier skill unmeasurable on both sides")
    elif chs is None:
        rule("brier_skill_positive", False,
             f"champion Brier skill {_fmt(cs)} but challenger's is unmeasurable")
    elif cs is None:
        rule("brier_skill_positive", chs > cfg["min_brier_skill"],
             f"challenger Brier skill {_fmt(chs)} "
             f"(need > {cfg['min_brier_skill']})")
    else:
        rule("brier_skill_positive",
             chs > cfg["min_brier_skill"] and chs >= cs,
             f"challenger Brier skill {_fmt(chs)} vs champion {_fmt(cs)} "
             f"(need > {cfg['min_brier_skill']} and not regress)")

    # 5. trade count must not collapse (absolute floor when champion traded)
    ct = champion_eval.get("trade_count") or 0
    cht = challenger_eval.get("trade_count") or 0
    need = ct * cfg["min_trade_count_ratio"]
    min_abs = 5 if ct > 0 else 0
    rule("trade_count_maintained",
         (ct == 0 and cht >= 0) or (cht >= need and cht >= min_abs),
         f"challenger {cht} trades vs champion {ct} "
         f"(need >= {need:.0f}"
         + (f" and >= {min_abs}" if min_abs else "") + ")")

    # 6. drawdown must not get worse
    cd = champion_eval.get("max_drawdown")
    chd = challenger_eval.get("max_drawdown")
    if cd is None or chd is None:
        rule("drawdown_not_worse", True, "drawdown unmeasurable")
    else:
        rule("drawdown_not_worse",
             chd <= cd + cfg["max_drawdown_tolerance"],
             f"challenger max drawdown {chd:.4f} vs champion {cd:.4f}")

    # 7. changed parameters must be fold-stable
    if stability is not None and changed_params:
        ok, why = stability_acceptable(stability, changed_params)
        rule("parameters_stable", ok, why)
    else:
        rule("parameters_stable", True, "no stability analysis supplied")

    # 8. no active alert-severity drift
    severe = [d for d in (diagnoses or [])
              if getattr(d, "issue", str(d)).endswith("_DRIFT")
              and getattr(d, "severity", "") == "alert"]
    rule("no_severe_drift", not severe,
         ("active drift: " + ", ".join(d.issue for d in severe)) if severe
         else "no alert-severity drift diagnosis")

    return PromotionDecision(promote=all(r.passed for r in rules), rules=rules)


# --------------------------------------------------------------------------- #
# Pending-review handoff (called by the learner on a passing decision)          #
# --------------------------------------------------------------------------- #
def write_pending_review(record: ConfigRecord, decision: PromotionDecision,
                         configs_dir: str = "configs",
                         jrn=None) -> str:
    """Stage a passing challenger for HUMAN review. Never touches
    champion.json."""
    import dataclasses
    staged = dataclasses.replace(record, status="pending_review")
    path = save_config(staged, pending_review_path(configs_dir))
    if jrn is not None:
        jrn.log_promotion(record.config_id, decision.to_dict(),
                          status="pending_review")
        jrn.update_candidate_status(record.config_id, "pending_review")
    return path


# --------------------------------------------------------------------------- #
# Human CLI: the only writer of champion.json                                   #
# --------------------------------------------------------------------------- #
def _load_pending(configs_dir: str) -> ConfigRecord:
    path = pending_review_path(configs_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no pending candidate at {path}")
    return load_config(path)


def approve(config_id: str, configs_dir: str = "configs",
            db_path: Optional[str] = None,
            author: str = "") -> str:
    """Install the pending candidate as champion. Archives the previous
    champion first; refuses on a config_id mismatch."""
    import dataclasses

    pending = _load_pending(configs_dir)
    if not pending.config_id.startswith(config_id):
        raise ValueError(f"pending candidate is {pending.config_id}, "
                         f"not {config_id!r}")

    champ_file = champion_path(configs_dir)
    if os.path.isfile(champ_file):
        old = load_config(champ_file)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = dataclasses.replace(old, status="archived")
        save_config(archived, os.path.join(
            archive_dir(configs_dir),
            f"champion_{old.config_id[:8]}_{stamp}.json"))

    promoted = dataclasses.replace(
        pending, status="promoted",
        author=author or pending.author)
    save_config(promoted, champ_file)
    os.remove(pending_review_path(configs_dir))

    if db_path and os.path.isfile(db_path):
        from journal import Journal
        jrn = Journal(db_path)
        try:
            jrn.update_promotion(pending.config_id, "approved",
                                 approved_by=author or None)
            jrn.update_candidate_status(pending.config_id, "promoted")
        finally:
            jrn.close()
    return champ_file


def reject(config_id: str, configs_dir: str = "configs",
           db_path: Optional[str] = None,
           author: str = "") -> str:
    """Reject the pending candidate: archived (not deleted) for the audit
    trail, journal rows updated."""
    import dataclasses

    pending = _load_pending(configs_dir)
    if not pending.config_id.startswith(config_id):
        raise ValueError(f"pending candidate is {pending.config_id}, "
                         f"not {config_id!r}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rejected = dataclasses.replace(pending, status="rejected")
    path = save_config(rejected, os.path.join(
        archive_dir(configs_dir),
        f"rejected_{pending.config_id[:8]}_{stamp}.json"))
    os.remove(pending_review_path(configs_dir))

    if db_path and os.path.isfile(db_path):
        from journal import Journal
        jrn = Journal(db_path)
        try:
            jrn.update_promotion(pending.config_id, "rejected",
                                 approved_by=author or None)
            jrn.update_candidate_status(pending.config_id, "rejected")
        finally:
            jrn.close()
    return path


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Human promotion CLI — the only writer of champion.json.")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--db", default="", help="journal SQLite (audit rows)")
    ap.add_argument("--author", default=os.environ.get("USER", ""))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="show the pending candidate (if any) and the champion")
    g.add_argument("--approve", metavar="CONFIG_ID",
                   help="install the pending candidate as champion")
    g.add_argument("--reject", metavar="CONFIG_ID",
                   help="reject the pending candidate (archived, not deleted)")
    args = ap.parse_args()

    if args.list:
        champ_file = champion_path(args.configs_dir)
        if os.path.isfile(champ_file):
            c = load_config(champ_file)
            print(f"champion: {c.config_id}  label={c.label!r}  "
                  f"created={c.created_at[:19]}")
            print(f"  overrides: {json.dumps(c.overrides)}")
        else:
            print("champion: <none> (dataclass defaults)")
        pend_file = pending_review_path(args.configs_dir)
        if os.path.isfile(pend_file):
            p = load_config(pend_file)
            print(f"pending:  {p.config_id}  label={p.label!r}  "
                  f"reason={p.promotion_reason!r}")
            print(f"  overrides: {json.dumps(p.overrides)}")
            print(f"  approve with: python3 -m adaptive_learning.promoter "
                  f"--approve {p.config_id[:8]}")
        else:
            print("pending:  <none>")
        return

    if args.approve:
        path = approve(args.approve, args.configs_dir,
                       db_path=args.db or None, author=args.author)
        print(f"PROMOTED -> {path}")
    else:
        path = reject(args.reject, args.configs_dir,
                      db_path=args.db or None, author=args.author)
        print(f"REJECTED -> {path}")


if __name__ == "__main__":
    main()
