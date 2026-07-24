"""
sequential_dojo.py
==================
Anchored sequential (chronological curriculum) evaluation — the measurement
spine that keeps accelerated training from becoming backtest overfitting.

The one rule
------------
Score Day t *before* learning from it. The only number that represents genuine
learning is the PREQUENTIAL score — the system's first-pass result on a blind
day, using only information available through the prior sessions. Re-running a
day until it "wins" just memorizes that day's answer key; the dojo docs already
say replaying adds no new information.

What this module measures
-------------------------
For each session t in chronological order it computes, leak-free (each session
is scored warmed only on the sessions before it, via walk_forward's
``initial_warm_sessions`` — the same session-fold + embargo machinery the rest
of the stack uses):

  * prequential J(champion, t)  — the carried-in (learned) state on blind day t
  * prequential J(baseline, t)  — the untrained baseline on the same blind day
  * forward transfer  FT_t = J(champion, t) − J(baseline, t)

FT_t answers the boxed success criterion of the design: *does what was learned
through the prior sessions improve the first-pass result on this blind day?*
A curriculum that teaches generalized behavior has FT_t > 0 on average; one
that overfit has FT_t ≤ 0.

`retention_forgetting` scores a fixed panel of earlier sessions under two
configs (champion-before vs candidate-after a learning step) and returns the
anti-forgetting penalty F — the scaffold a learner-curriculum gate uses so a
Day-t update cannot silently degrade Day-1 behavior.

Sealed holdout
--------------
Sessions listed in the sealed set are removed from the curriculum entirely, so
they are never scored or learned against here — a benchmark that stays sealed
because nothing in this loop ever touches it.

Scope (Stage 1)
--------------
This is the prequential MEASUREMENT + governance spine. It reuses the existing
learner for the optional per-day update but does NOT itself implement the
replay-mixture retraining, the three-memory agent-lesson store, the intraday
forecast-vintage grading, or selection-regret — those are named follow-on
stages (see docs/sequential_dojo.md). Run with the current champion to verify
it generalizes forward; wire the learner in a later stage to close the loop.

NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from decision_engine import EngineConfig
from journal import Journal
from optimizer import _score
from validation.session_folds import session_spans
from walk_forward import WalkForwardConfig, run_walk_forward

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class SequentialDojoConfig:
    db_path: str = "shadow.db"
    record_dir: str = ""
    configs_dir: str = "configs"
    reports_dir: str = os.path.join("reports", "sequential")
    report_date: Optional[str] = None
    metric: str = "composite"          # optimizer._score objective
    start_session: Optional[str] = None   # ISO date; default: first learnable
    end_session: Optional[str] = None
    min_warm_sessions: int = 2         # need >= this many prior sessions to score
    sealed_sessions: tuple[str, ...] = ()   # never scored/learned here
    # 0DTE sessions settle same-day, so D_{t-1}'s label can't leak into D_t —
    # the honest prequential setup tests D_t warmed on D_0..D_{t-1} with no gap.
    embargo_sessions: int = 0


# --------------------------------------------------------------------------- #
# Feed resolution                                                             #
# --------------------------------------------------------------------------- #
def _resolve_feed(cfg: SequentialDojoConfig,
                  feed_factory: Optional[Callable],
                  timestamps: Optional[list]) -> tuple[Callable, list]:
    if feed_factory is not None and timestamps is not None:
        return feed_factory, list(timestamps)
    if cfg.record_dir and os.path.isdir(cfg.record_dir):
        from chain_store import RecordedFeed
        ticks = RecordedFeed(cfg.record_dir).timestamps()
        return (lambda: RecordedFeed(cfg.record_dir)), ticks
    raise ValueError("no data source: pass feed_factory+timestamps or a "
                     "record_dir with recorded sessions")


def _champion_engine(cfg: SequentialDojoConfig) -> tuple[EngineConfig, Optional[str]]:
    """The carried-in learned state: the promoted champion if present, else
    the baseline defaults (in which case forward transfer is 0 by
    construction and the run is a pure baseline sanity check)."""
    try:
        from adaptive_learning import config_store
        champ = config_store.load_champion(cfg.configs_dir)
        if champ is not None:
            return champ.engine_cfg, champ.record.config_id
    except Exception:
        pass
    return EngineConfig(), None


# --------------------------------------------------------------------------- #
# Prequential single-session scoring (leak-free)                              #
# --------------------------------------------------------------------------- #
def _session_score(feed_factory: Callable, ts_upto: list, warm_sessions: int,
                   engine_cfg: EngineConfig, metric: str,
                   embargo: int) -> Optional[float]:
    """Score exactly the LAST session in ``ts_upto`` (index == warm_sessions),
    warmed only on the ``warm_sessions`` sessions before it. Uses one pinned
    expanding session-fold, so it is out-of-sample and leak-free by
    construction. Returns None when the fold is invalid/absent."""
    wf = run_walk_forward(
        feed_factory=feed_factory,
        timestamps=ts_upto,
        wf_cfg=WalkForwardConfig(
            mode="expanding", n_folds=1, fold_unit="session",
            embargo_sessions=embargo, initial_warm_sessions=warm_sessions),
        engine_cfg=engine_cfg,
    )
    if not wf.valid_folds:
        return None
    try:
        return _score(wf, metric)
    except Exception:
        return None


def retention_forgetting(feed_factory: Callable, all_ts: list,
                         panel_session_idx: list[int],
                         cfg_before: EngineConfig, cfg_after: EngineConfig,
                         metric: str = "composite",
                         embargo: int = 1) -> dict:
    """Anti-forgetting: re-score a fixed panel of EARLIER sessions under the
    config before vs after a learning step. F = mean over the panel of
    max(0, J_before − J_after) — the amount of prior-session skill a candidate
    gives up. The gate a learner-curriculum uses so Day-t learning can't
    silently degrade Day-1 behavior."""
    spans = session_spans(all_ts)
    losses, per = [], {}
    for i in panel_session_idx:
        if i <= 0 or i >= len(spans):
            continue
        ts_upto = all_ts[: spans[i].end]
        jb = _session_score(feed_factory, ts_upto, i, cfg_before, metric, embargo)
        ja = _session_score(feed_factory, ts_upto, i, cfg_after, metric, embargo)
        if jb is None or ja is None:
            continue
        loss = max(0.0, jb - ja)
        losses.append(loss)
        per[spans[i].date] = {"before": round(jb, 6), "after": round(ja, 6),
                              "forgetting": round(loss, 6)}
    return {"forgetting": round(sum(losses) / len(losses), 6) if losses else 0.0,
            "n": len(losses), "per_session": per}


# --------------------------------------------------------------------------- #
# The chronological loop                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class SequentialResult:
    days: list[dict] = field(default_factory=list)
    champion_id: Optional[str] = None
    n_sessions: int = 0
    n_sealed: int = 0

    def mean_forward_transfer(self) -> Optional[float]:
        fts = [d["forward_transfer"] for d in self.days
               if d.get("forward_transfer") is not None]
        return round(sum(fts) / len(fts), 6) if fts else None

    def cumulative_prequential(self) -> Optional[float]:
        js = [d["prequential_J"] for d in self.days
              if d.get("prequential_J") is not None]
        return round(sum(js), 6) if js else None

    def n_forward_positive(self) -> int:
        return sum(1 for d in self.days
                   if (d.get("forward_transfer") or 0) > 0)

    def to_metrics(self) -> dict:
        return {
            "champion_id": self.champion_id,
            "n_sessions": self.n_sessions,
            "n_sealed": self.n_sealed,
            "n_scored": len(self.days),
            "mean_forward_transfer": self.mean_forward_transfer(),
            "cumulative_prequential_J": self.cumulative_prequential(),
            "n_forward_positive": self.n_forward_positive(),
            "days": self.days,
        }


def run_sequential_dojo(cfg: Optional[SequentialDojoConfig] = None,
                        feed_factory: Optional[Callable] = None,
                        timestamps: Optional[list] = None) -> dict:
    cfg = cfg or SequentialDojoConfig()
    report_date = cfg.report_date or dt.datetime.now(ET).date().isoformat()

    feed_factory, all_ts = _resolve_feed(cfg, feed_factory, timestamps)
    spans = session_spans(all_ts)
    sealed = set(cfg.sealed_sessions)
    champion_cfg, champ_id = _champion_engine(cfg)
    baseline_cfg = EngineConfig()

    result = SequentialResult(champion_id=champ_id, n_sessions=len(spans),
                              n_sealed=sum(1 for s in spans if s.date in sealed))

    for i, span in enumerate(spans):
        if span.date in sealed:
            continue
        if i < cfg.min_warm_sessions:
            continue                        # not enough prior sessions to warm
        if cfg.start_session and span.date < cfg.start_session:
            continue
        if cfg.end_session and span.date > cfg.end_session:
            break

        ts_upto = all_ts[: span.end]
        preq = _session_score(feed_factory, ts_upto, i, champion_cfg,
                              cfg.metric, cfg.embargo_sessions)
        base = _session_score(feed_factory, ts_upto, i, baseline_cfg,
                              cfg.metric, cfg.embargo_sessions)
        ft = (preq - base) if (preq is not None and base is not None) else None
        result.days.append({
            "session": span.date,
            "warm_sessions": i,
            "prequential_J": round(preq, 6) if preq is not None else None,
            "baseline_J": round(base, 6) if base is not None else None,
            "forward_transfer": round(ft, 6) if ft is not None else None,
        })

    metrics = result.to_metrics()
    metrics["config"] = {
        "metric": cfg.metric, "record_dir": cfg.record_dir,
        "min_warm_sessions": cfg.min_warm_sessions,
        "embargo_sessions": cfg.embargo_sessions,
        "sealed_sessions": list(cfg.sealed_sessions),
    }
    mft = metrics["mean_forward_transfer"]
    verdict = ("no scored sessions" if not result.days
               else "curriculum generalizes forward (mean FT > 0)"
               if (mft or 0) > 0
               else "no forward transfer yet (champion ~ baseline or overfit)")
    summary = (f"sequential dojo: {len(result.days)} blind sessions scored, "
               f"mean forward transfer {mft}, "
               f"{result.n_forward_positive()}/{len(result.days)} days FT>0 — {verdict}")

    flags = []
    if mft is not None and mft <= 0 and champ_id is not None:
        flags.append({"severity": "warn", "flag": "no_forward_transfer",
                      "detail": "the champion does not beat baseline on blind "
                                "sessions — possible overfit; do not size up"})

    jrn = Journal(cfg.db_path)
    report_id = jrn.log_validation_report(
        report_date, "sequential_dojo", metrics, summary, flags=flags)
    jrn.close()

    os.makedirs(cfg.reports_dir, exist_ok=True)
    stamp = dt.datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(cfg.reports_dir, f"sequential_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"report_date": report_date, "summary": summary,
                   "flags": flags, "metrics": metrics}, f, indent=2, default=str)

    return {"report_id": report_id, "report_date": report_date,
            "summary": summary, "flags": flags, "json_path": json_path,
            "metrics": metrics}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Anchored sequential dojo: prequential (predict-then-learn) "
                    "chronological scoring with forward-transfer — proves "
                    "learning generalizes forward instead of memorizing days.")
    ap.add_argument("--db", default="shadow.db")
    ap.add_argument("--record-dir", default="")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--reports-dir", default=os.path.join("reports", "sequential"))
    ap.add_argument("--metric", default="composite")
    ap.add_argument("--start-session", default=None)
    ap.add_argument("--end-session", default=None)
    ap.add_argument("--min-warm", type=int, default=2)
    ap.add_argument("--sealed", default="",
                    help="comma-separated ISO session dates to seal out")
    args = ap.parse_args()

    sealed = tuple(s for s in (args.sealed.split(",") if args.sealed else []) if s)
    cfg = SequentialDojoConfig(
        db_path=args.db, record_dir=args.record_dir,
        configs_dir=args.configs_dir, reports_dir=args.reports_dir,
        metric=args.metric, start_session=args.start_session,
        end_session=args.end_session, min_warm_sessions=args.min_warm,
        sealed_sessions=sealed)
    if not cfg.record_dir:
        ap.error("--record-dir is required (recorded sessions to walk)")

    out = run_sequential_dojo(cfg)
    print(f"\n  sequential dojo report #{out['report_id']} ({out['report_date']})")
    print(f"  {out['summary']}")
    print(f"\n  {'session':<12} {'warm':>4} {'preq_J':>9} {'base_J':>9} {'fwd_xfer':>9}")
    for d in out["metrics"]["days"]:
        def f(x): return f"{x:+.4f}" if isinstance(x, (int, float)) else "—"
        print(f"  {d['session']:<12} {d['warm_sessions']:>4} "
              f"{f(d['prequential_J']):>9} {f(d['baseline_J']):>9} "
              f"{f(d['forward_transfer']):>9}")
    for fl in out["flags"]:
        print(f"    [{fl['severity'].upper()}] {fl['flag']}: {fl['detail']}")
    print(f"\n  JSON: {out['json_path']}")


if __name__ == "__main__":
    main()
