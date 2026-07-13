"""
adaptive_learning/learner.py
============================
Orchestrates one complete learning cycle:

    load journal -> diagnose failures -> generate hypotheses ->
    optimize (holdout MANDATORY) -> stability analysis -> feature lab ->
    promotion recommendation -> persist report + candidate config

Hard guarantees, enforced here:
  * holdout_frac must be > 0 — the learner refuses to run a search whose
    winner was never scored out-of-search-sample;
  * the learner NEVER writes configs/champion.json. On a passing decision it
    stops at configs/promoted/pending_review.json plus a promotions row with
    status 'pending_review'; a human runs the promoter CLI.

Entry points
------------
  run_daily(...)     cheap: diagnostics + drift + calibration only
  run_evening(...)   weekday post-close: full optimize cycle (lighter trials)
  run_weekly(...)    weekend deep cycle: full optimize + more trials/folds
  run_manual(...)    the full cycle, triggered by hand
  run_learning_cycle(...)  the underlying engine both of the above call

Data sources: pass an explicit feed_factory/timestamps (tests, demos), or a
record_dir of shadow recordings (chain_store.RecordedFeed) for real data.
Scheduling stays external (cron / systemd the daily/evening/weekly CLIs);
no daemon here.

CLI
---
    python3 -m adaptive_learning.learner --mode daily  --db shadow.db
    python3 -m adaptive_learning.learner --mode evening --db shadow.db \
        --record-dir /var/lib/zerodte/ticks
    python3 -m adaptive_learning.learner --mode weekly --db shadow.db \
        --record-dir /var/lib/zerodte/ticks

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from journal import Journal
from decision_engine import EngineConfig
from walk_forward import WalkForwardConfig, run_walk_forward
from optimizer import OptimizerConfig, run_optimizer, _score

from adaptive_learning import config_store
from adaptive_learning.diagnostics import (
    compute_drift, diagnose, drift_diagnoses, log_drift_report,
)
from adaptive_learning.feature_lab import run_feature_lab
from adaptive_learning.hypothesis import combined_param_space, generate
from adaptive_learning.promoter import (
    check_promotion, evaluation_from_wf, write_pending_review,
)
from adaptive_learning.reports import (
    build_promotion_report, changed_params_map, persist_promotion_report,
)
from adaptive_learning.stability import feature_stability, parameter_stability


@dataclass
class LearnerConfig:
    db_path: str = "shadow.db"
    record_dir: str = ""                      # shadow recordings (RecordedFeed)
    configs_dir: str = "configs"
    reports_dir: str = os.path.join("reports", "promotion")
    # search
    search: str = "tpe"                       # "tpe" | "random" ("grid" allowed)
    n_trials: int = 30
    metric: str = "composite"
    seed: int = 42
    holdout_frac: float = 0.25                # MANDATORY > 0
    # walk-forward
    wf_mode: str = "expanding"
    wf_folds: int = 4
    wf_train_frac: float = 0.6
    # search-space cap (dimensionality guard)
    max_params: int = 8
    # data floors
    min_ticks: int = 100
    min_sessions: int = 3


# --------------------------------------------------------------------------- #
# Data resolution                                                               #
# --------------------------------------------------------------------------- #
def _resolve_feed(cfg: LearnerConfig,
                  feed_factory: Optional[Callable],
                  timestamps: Optional[list]) -> tuple[Callable, list]:
    if feed_factory is not None and timestamps is not None:
        return feed_factory, list(timestamps)
    if cfg.record_dir and os.path.isdir(cfg.record_dir):
        from chain_store import RecordedFeed
        probe = RecordedFeed(cfg.record_dir)
        ticks = probe.timestamps()
        sessions = {t.date() for t in ticks}
        if len(ticks) >= cfg.min_ticks and len(sessions) >= cfg.min_sessions:
            return (lambda: RecordedFeed(cfg.record_dir)), ticks
        raise ValueError(
            f"record_dir {cfg.record_dir!r} has only {len(ticks)} ticks over "
            f"{len(sessions)} sessions (need >= {cfg.min_ticks} / "
            f">= {cfg.min_sessions}); let shadow mode record longer")
    raise ValueError("no data source: pass feed_factory+timestamps or a "
                     "record_dir with enough recorded history")


def _evaluate_cfg(feed_factory: Callable, search_ts: list, holdout_ts: list,
                  wf_cfg: WalkForwardConfig, engine_cfg: EngineConfig,
                  metric: str) -> tuple[dict, object, object]:
    """Score one config on the search window (walk-forward) AND the holdout
    (one expanding fold warmed on the whole search window) — the same protocol
    run_optimizer uses for its winner, so champion and challenger numbers are
    comparable. Returns (evaluation dict, wf_result, holdout_result)."""
    wf_result = run_walk_forward(
        feed_factory=feed_factory, timestamps=search_ts,
        wf_cfg=wf_cfg, engine_cfg=engine_cfg)
    score = _score(wf_result, metric)
    holdout_result = None
    holdout_score = None
    if holdout_ts:
        if wf_cfg.fold_unit == "session":
            from validation.session_folds import session_spans
            hold_cfg = WalkForwardConfig(
                mode="expanding", n_folds=1, fold_unit="session",
                embargo_sessions=wf_cfg.embargo_sessions,
                max_failed_tick_frac=wf_cfg.max_failed_tick_frac,
                # Pin the test window to exactly the held-out sessions.
                initial_warm_sessions=len(session_spans(search_ts)),
            )
        else:
            hold_cfg = WalkForwardConfig(
                mode="expanding", n_folds=1, fold_unit="tick",
                train_frac=len(search_ts) / max(1, len(search_ts) + len(holdout_ts)),
            )
        holdout_result = run_walk_forward(
            feed_factory=feed_factory,
            timestamps=search_ts + holdout_ts,
            wf_cfg=hold_cfg,
            engine_cfg=engine_cfg)
        holdout_score = _score(holdout_result, metric)
    ev = evaluation_from_wf(wf_result, score=score, holdout_score=holdout_score)
    return ev, wf_result, holdout_result


# --------------------------------------------------------------------------- #
# Daily: diagnostics only (cheap)                                               #
# --------------------------------------------------------------------------- #
def run_daily(cfg: Optional[LearnerConfig] = None,
              report_date: Optional[str] = None) -> dict:
    """Diagnostics + drift + calibration snapshot. No optimization, no
    candidates — the observability half of the loop."""
    cfg = cfg or LearnerConfig()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = uuid.uuid4().hex
    jrn = Journal(cfg.db_path)
    try:
        drift = compute_drift(jrn)
        log_drift_report(jrn, drift, report_date=report_date)
        diagnoses = diagnose(jrn, include_drift=False) + drift_diagnoses(drift)
        calibration = jrn.calibration()
        jrn.log_learning_run(
            run_id, "daily", started,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            diagnostics=[d.to_dict() for d in diagnoses],
            outcome="diagnostics_only",
        )
        return {"run_id": run_id, "mode": "daily",
                "diagnoses": [d.to_dict() for d in diagnoses],
                "drift": drift, "calibration": calibration,
                "outcome": "diagnostics_only"}
    finally:
        jrn.close()


# --------------------------------------------------------------------------- #
# The full cycle                                                                #
# --------------------------------------------------------------------------- #
def run_learning_cycle(cfg: Optional[LearnerConfig] = None,
                       mode: str = "manual",
                       feed_factory: Optional[Callable] = None,
                       timestamps: Optional[list] = None,
                       report_date: Optional[str] = None) -> dict:
    """
    One complete learning cycle. Returns a summary dict; every artifact
    (learning_runs row, candidate file, promotion report, pending_review) is
    persisted along the way. Raises if holdout_frac <= 0 — no exceptions to
    the anti-selection-bias rule.

    Insufficient recorded history returns outcome='insufficient_data' instead
    of raising, so scheduled evening/weekly timers can no-op cleanly until
    shadow has enough sessions.
    """
    cfg = cfg or LearnerConfig()
    if cfg.holdout_frac <= 0.0:
        raise ValueError(
            "holdout_frac must be > 0: a search winner that was never scored "
            "on an untouched window is in-sample by construction")

    run_id = uuid.uuid4().hex
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    jrn = Journal(cfg.db_path)
    summary: dict = {"run_id": run_id, "mode": mode}
    try:
        # -- 1. diagnose ----------------------------------------------------
        drift = compute_drift(jrn)
        log_drift_report(jrn, drift, report_date=report_date)
        diagnoses = diagnose(jrn, include_drift=False) + drift_diagnoses(drift)
        summary["diagnoses"] = [d.to_dict() for d in diagnoses]

        # -- 2. hypothesize ---------------------------------------------------
        hypotheses = generate(diagnoses)
        summary["hypotheses"] = [h.to_dict() for h in hypotheses]
        if not hypotheses:
            jrn.log_learning_run(
                run_id, mode, started,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                diagnostics=summary["diagnoses"], outcome="no_action",
                notes="no diagnosable failure maps to a parameter search")
            summary["outcome"] = "no_action"
            return summary
        param_space = combined_param_space(hypotheses, max_params=cfg.max_params)
        reason = hypotheses[0].issue
        summary["param_space"] = param_space
        summary["reason"] = reason

        # -- 3. data + champion baseline -------------------------------------
        try:
            feed_factory, ticks = _resolve_feed(cfg, feed_factory, timestamps)
        except ValueError as exc:
            note = str(exc)
            jrn.log_learning_run(
                run_id, mode, started,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                diagnostics=summary["diagnoses"],
                param_space=param_space,
                outcome="insufficient_data",
                notes=note)
            summary["outcome"] = "insufficient_data"
            summary["notes"] = note
            return summary
        # Session-based carve: the holdout is complete sessions, never a
        # partial trading day (see validation/session_folds.py).
        from validation.session_folds import split_holdout_by_sessions
        search_ts, holdout_ts = split_holdout_by_sessions(ticks, cfg.holdout_frac)
        wf_cfg = WalkForwardConfig(mode=cfg.wf_mode, n_folds=cfg.wf_folds,
                                   train_frac=cfg.wf_train_frac)

        champion = config_store.load_champion(cfg.configs_dir)
        champ_overrides = champion.record.overrides if champion else {}
        champ_engine = champion.engine_cfg if champion else EngineConfig()
        champ_id = champion.record.config_id if champion else None

        print(f"  [learner] champion baseline "
              f"({champ_id[:8] if champ_id else 'defaults'}) …", flush=True)
        champ_eval, champ_wf, _champ_hold = _evaluate_cfg(
            feed_factory, search_ts, holdout_ts, wf_cfg, champ_engine, cfg.metric)
        summary["champion_eval"] = champ_eval

        # -- 4. optimize (holdout enforced above) ------------------------------
        result = run_optimizer(
            feed_factory=feed_factory,
            timestamps=ticks,
            param_space=param_space,
            opt_cfg=OptimizerConfig(search=cfg.search, n_trials=cfg.n_trials,
                                    metric=cfg.metric, seed=cfg.seed,
                                    holdout_frac=cfg.holdout_frac),
            wf_cfg=wf_cfg,
            base_engine_cfg=champ_engine,
        )
        best = result.best_trial
        chall_eval = evaluation_from_wf(
            best.wf_result, score=best.score, holdout_score=result.holdout_score)
        summary["challenger_eval"] = chall_eval

        # -- 5. stability + feature lab ---------------------------------------
        stability = parameter_stability(result.trials)
        summary["stability"] = stability
        feat_stab = feature_stability(jrn, champ_wf)
        feature_report = run_feature_lab(jrn, stability=feat_stab,
                                         as_of=report_date, seed=cfg.seed)

        # -- 6. candidate record ----------------------------------------------
        merged_overrides = {**champ_overrides, **best.params}
        candidate = config_store.new_candidate(
            merged_overrides,
            label=f"{reason}",
            parent_id=champ_id,
            regime_overrides=(champion.record.regime_overrides if champion else {}),
            optimizer={**result.to_dict(),
                       "trials": result.to_dict()["trials"][:20]},
            metrics={"search_score": best.score,
                     "holdout_score": result.holdout_score,
                     "champion_score": champ_eval.get("score"),
                     "champion_holdout": champ_eval.get("holdout_score")},
            promotion_reason=reason,
        )
        candidate_path = config_store.save_candidate(candidate, cfg.configs_dir)
        jrn.log_candidate_config(
            candidate.config_id, candidate.created_at, candidate.overrides,
            parent_id=champ_id, label=candidate.label,
            metrics=candidate.metrics)
        summary["candidate"] = {"config_id": candidate.config_id,
                                "path": candidate_path,
                                "overrides": candidate.overrides}

        # -- 7. promotion decision --------------------------------------------
        changed = changed_params_map(champ_overrides, best.params, champ_engine)
        decision = check_promotion(
            champ_eval, chall_eval,
            stability=stability, changed_params=list(best.params),
            diagnoses=diagnoses)
        summary["decision"] = decision.to_dict()
        summary["changed_params"] = changed

        # -- 8. report ---------------------------------------------------------
        report = build_promotion_report(
            reason=reason,
            champion_eval=champ_eval, challenger_eval=chall_eval,
            decision=decision.to_dict(), changed_params=changed,
            candidate={"config_id": candidate.config_id,
                       "label": candidate.label,
                       "overrides": candidate.overrides},
            diagnostics=summary["diagnoses"],
            stability=stability,
            feature_report=feature_report,
            walk_forward=best.wf_result.to_dict(),
            holdout=(result.holdout_result.to_dict()
                     if result.holdout_result else {}),
            optimizer_summary={k: v for k, v in result.to_dict().items()
                               if k != "trials"},
        )
        persisted = persist_promotion_report(
            jrn, report, reports_dir=cfg.reports_dir, report_date=report_date)
        summary["report"] = persisted

        # -- 9. stage for human review (never auto-promote) --------------------
        if decision.promote:
            pending = write_pending_review(candidate, decision,
                                           configs_dir=cfg.configs_dir, jrn=jrn)
            summary["pending_review"] = pending
            outcome = "promotion_recommended"
        else:
            jrn.log_promotion(candidate.config_id, decision.to_dict(),
                              status="rejected",
                              notes=", ".join(decision.failing))
            jrn.update_candidate_status(candidate.config_id, "rejected")
            outcome = "rejected"
        summary["outcome"] = outcome

        jrn.log_learning_run(
            run_id, mode, started,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            diagnostics=summary["diagnoses"],
            param_space=param_space,
            n_trials=len(result.trials),
            best_score=best.score,
            holdout_score=result.holdout_score,
            trials=result.to_dict()["trials"],
            outcome=outcome,
            notes=f"candidate={candidate.config_id} reason={reason}")
        return summary
    finally:
        jrn.close()


def run_evening(cfg: Optional[LearnerConfig] = None,
                feed_factory: Optional[Callable] = None,
                timestamps: Optional[list] = None,
                report_date: Optional[str] = None) -> dict:
    """
    Weekday post-close optimize cycle: find candidate settings from recorded
    ticks + journal failure modes. Scheduled timers pass lighter --trials /
    --folds; still uses mandatory holdout. Never auto-writes champion.json.
    """
    return run_learning_cycle(cfg or LearnerConfig(), mode="evening",
                              feed_factory=feed_factory,
                              timestamps=timestamps, report_date=report_date)


def run_weekly(cfg: Optional[LearnerConfig] = None,
               feed_factory: Optional[Callable] = None,
               timestamps: Optional[list] = None,
               report_date: Optional[str] = None) -> dict:
    """The scheduled deep cycle: full optimization + candidate generation."""
    return run_learning_cycle(cfg, mode="weekly", feed_factory=feed_factory,
                              timestamps=timestamps, report_date=report_date)


def run_manual(cfg: Optional[LearnerConfig] = None,
               feed_factory: Optional[Callable] = None,
               timestamps: Optional[list] = None,
               report_date: Optional[str] = None) -> dict:
    """Hand-triggered full cycle (identical engine, labeled 'manual')."""
    return run_learning_cycle(cfg, mode="manual", feed_factory=feed_factory,
                              timestamps=timestamps, report_date=report_date)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Adaptive Learning Engine — diagnose, hypothesize, "
                    "optimize with mandatory holdout, recommend promotions. "
                    "Never touches champion.json.")
    ap.add_argument("--mode", choices=["daily", "evening", "weekly", "manual"],
                    default="manual")
    ap.add_argument("--db", default="shadow.db")
    ap.add_argument("--record-dir", default="")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--reports-dir", default="")
    ap.add_argument("--search", choices=["tpe", "random", "grid"], default="tpe")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    reports_dir = args.reports_dir or os.path.join("reports", "promotion")
    cfg = LearnerConfig(db_path=args.db, record_dir=args.record_dir,
                        configs_dir=args.configs_dir, reports_dir=reports_dir,
                        search=args.search,
                        n_trials=args.trials, holdout_frac=args.holdout,
                        wf_folds=args.folds)
    if args.mode == "daily":
        out = run_daily(cfg)
    elif args.mode == "evening":
        out = run_evening(cfg)
    elif args.mode == "weekly":
        out = run_weekly(cfg)
    else:
        out = run_manual(cfg)

    print(f"\n  learning run {out['run_id'][:8]} ({out['mode']}): "
          f"{out['outcome']}")
    for d in out.get("diagnoses", []):
        print(f"    [{d['severity'].upper():5}] {d['issue']} "
              f"conf={d['confidence']:.0%}")
    if out.get("decision"):
        for r in out["decision"]["rules"]:
            print(f"    {'PASS' if r['passed'] else 'FAIL'} {r['name']}: "
                  f"{r['detail']}")
    if args.json:
        print(_json.dumps(out, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Demo: full gate-inversion cycle on the coupled synthetic world               #
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import json
    import tempfile
    from journal import COLUMNS
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    print("=" * 76)
    print("  Adaptive Learning Engine demo — gate-inversion cycle, coupled world")
    print("=" * 76)

    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "shadow.db")
        # Seed a journal exhibiting the failure mode: taken trades lose while
        # gate-blocked candidates would have won (gate inversion).
        jrn = Journal(db)
        session = "2026-07-08"
        for i in range(40):
            traded = i % 2 == 0
            row = {c: None for c in COLUMNS}
            row.update({
                "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
                "spot": 600.0, "gex_regime": "long" if i % 3 else "short",
                "was_traded": 1 if traded else 0, "candidate_present": 1,
                "gate_pass": 1 if traded else 0,
                "decision": "TRADE" if traded else "NO_TRADE",
                "credit": 0.4 if traded else 1.2,
                "ev": 0.1, "prob_profit": 0.6,
                "legs_json": json.dumps([
                    {"qty": -1, "strike": 599.0 if traded else 604.0, "kind": "C"},
                    {"qty": 1, "strike": 601.0 if traded else 606.0, "kind": "C"},
                ]),
                "regime_direction": "call",
            })
            jrn.log(row)
        jrn.settle_session(session, 601.0)
        jrn.close()

        DAYS, STRIDE = 8, 15
        def make_feed():
            return CoupledSyntheticFeed(WorldConfig(days=DAYS, seed=11,
                                                    tick_stride=STRIDE))
        ticks = make_feed().timestamps()

        cfg = LearnerConfig(
            db_path=db,
            configs_dir=os.path.join(d, "configs"),
            reports_dir=os.path.join(d, "reports"),
            search="random", n_trials=3, holdout_frac=0.25,
            wf_folds=2, max_params=4)
        out = run_learning_cycle(cfg, mode="manual",
                                 feed_factory=make_feed, timestamps=ticks,
                                 report_date="2026-07-09")

        print(f"\n  outcome: {out['outcome']}  (reason={out.get('reason')})")
        print(f"  diagnoses: {[d['issue'] for d in out['diagnoses']]}")
        print(f"  candidate: {os.path.basename(out['candidate']['path'])}")
        print(f"  report:    {os.path.basename(out['report']['md_path'])}")
        for r in out["decision"]["rules"]:
            print(f"    {'PASS' if r['passed'] else 'FAIL'} {r['name']}")
        jrn = Journal(db)
        runs = jrn.fetch_learning_runs()
        print(f"  learning_runs rows: {len(runs)}  "
              f"(best={runs[0]['best_score']}, holdout={runs[0]['holdout_score']})")
        jrn.close()
    print("=" * 76)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        _demo()
