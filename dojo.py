"""
dojo.py
=======
Matrix-style accelerated training for the 0DTE / SPY-DER stack: replay every
experience the VPS has recorded, run the adaptive learner on it, then spar
the pipeline against a combinatoric catalog of Markov-generated universes
(matrix_universe.py) so its behavior is measured in situations the live tape
has never shown it. One command, one persisted report.

Phases
------
  1. recorded   Walk-forward over the shadow recordings (chain_store ticks)
                plus full-window calibration — the real-tape baseline.
  2. learner    One adaptive_learning cycle (mode='dojo') on the same
                recordings: diagnose → hypothesize → optimize (holdout
                mandatory) → stage pending_review. Never touches
                champion.json — promotion stays human.
  3. universe   Backtest the pipeline across N universes per generation from
                the UniverseCatalog; aggregate P&L / win rate / directional
                hit per market archetype; evolve the catalog toward the
                weakest archetypes and repeat. Produces the robustness
                matrix and the (archetype × regime) coverage map.

The report is persisted to journal.validation_reports with
report_type='dojo' (rendered by the dashboard's Dojo tab, which the Vercel
proxy serves) and written as JSON under reports/dojo/.

Phases degrade honestly: with no recorded data yet, phases 1–2 report
'insufficient_data' and phase 3 still runs — the universe matrix never
requires the tape.

Usage
-----
    python3 dojo.py --db shadow.db --record-dir /var/lib/zerodte/ticks
    python3 dojo.py --universes 8 --generations 2 --days 8 --stride 5
    python3 dojo.py --full-lattice          # every archetype × tilt × vol cell

NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from backtest import run_backtest
from journal import Journal, economic_pnl
from matrix_universe import (
    ARCHETYPES, REGIMES, MarkovWorldFeed, UniverseCatalog, UniverseSpec,
    merge_coverage,
)
from walk_forward import WalkForwardConfig, run_walk_forward

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class DojoConfig:
    db_path: str = "shadow.db"
    record_dir: str = ""
    configs_dir: str = "configs"
    reports_dir: str = os.path.join("reports", "dojo")
    report_date: Optional[str] = None      # default: today ET
    # phase toggles
    skip_recorded: bool = False
    skip_learner: bool = False
    skip_universe: bool = False
    # recorded-phase walk-forward
    wf_folds: int = 3
    wf_train_frac: float = 0.6
    min_ticks: int = 100
    min_sessions: int = 3
    # learner phase
    learn_trials: int = 15
    learn_holdout: float = 0.25
    # universe phase
    universes_per_gen: int = 6
    generations: int = 2
    full_lattice: bool = False
    universe_days: int = 8
    tick_stride: int = 5
    catalog_seed: int = 20260723


# --------------------------------------------------------------------------- #
# Phase 1 — recorded tape                                                     #
# --------------------------------------------------------------------------- #
def _phase_recorded(cfg: DojoConfig) -> dict:
    if cfg.skip_recorded:
        return {"status": "skipped", "note": "--skip-recorded"}
    if not cfg.record_dir or not os.path.isdir(cfg.record_dir):
        return {"status": "insufficient_data",
                "note": f"no recording directory at {cfg.record_dir!r} — "
                        "let shadow mode record sessions first"}

    from chain_store import RecordedFeed
    probe = RecordedFeed(cfg.record_dir)
    ticks = probe.timestamps()
    sessions = sorted({t.date().isoformat() for t in ticks})
    if len(ticks) < cfg.min_ticks or len(sessions) < cfg.min_sessions:
        return {"status": "insufficient_data",
                "n_ticks": len(ticks), "n_sessions": len(sessions),
                "note": f"{len(ticks)} ticks / {len(sessions)} sessions "
                        f"recorded (need >= {cfg.min_ticks} / "
                        f">= {cfg.min_sessions})"}

    print(f"  [dojo] phase 1 — recorded tape: {len(ticks):,} ticks / "
          f"{len(sessions)} sessions", flush=True)
    wf = run_walk_forward(
        feed_factory=lambda: RecordedFeed(cfg.record_dir),
        timestamps=ticks,
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=cfg.wf_folds,
                                 train_frac=cfg.wf_train_frac),
    )
    jrn = Journal(":memory:")
    run_backtest(RecordedFeed(cfg.record_dir), ticks, journal=jrn)
    calibration = jrn.calibration()
    jrn.close()
    return {"status": "ok", "n_ticks": len(ticks),
            "n_sessions": len(sessions), "sessions": sessions,
            "walk_forward": wf.to_dict(), "calibration": calibration}


# --------------------------------------------------------------------------- #
# Phase 2 — adaptive learner                                                  #
# --------------------------------------------------------------------------- #
_LEARNER_KEYS = ("run_id", "mode", "outcome", "reason", "notes", "diagnoses",
                 "decision", "changed_params", "champion_eval",
                 "challenger_eval", "stability")


def _phase_learner(cfg: DojoConfig) -> dict:
    if cfg.skip_learner:
        return {"status": "skipped", "note": "--skip-learner"}
    from adaptive_learning.learner import LearnerConfig, run_learning_cycle
    print("  [dojo] phase 2 — adaptive learning cycle", flush=True)
    lcfg = LearnerConfig(
        db_path=cfg.db_path, record_dir=cfg.record_dir,
        configs_dir=cfg.configs_dir,
        reports_dir=os.path.join(os.path.dirname(cfg.reports_dir) or ".",
                                 "promotion"),
        n_trials=cfg.learn_trials, holdout_frac=cfg.learn_holdout,
        min_ticks=cfg.min_ticks, min_sessions=cfg.min_sessions,
    )
    try:
        out = run_learning_cycle(lcfg, mode="dojo",
                                 report_date=cfg.report_date)
    except Exception as exc:  # a broken learner must not sink the report
        return {"status": "error", "note": f"{type(exc).__name__}: {exc}"}
    trimmed = {k: out[k] for k in _LEARNER_KEYS if k in out}
    if "candidate" in out:
        trimmed["candidate"] = {"config_id": out["candidate"]["config_id"],
                                "overrides": out["candidate"]["overrides"]}
    trimmed["status"] = "ok"
    return trimmed


# --------------------------------------------------------------------------- #
# Phase 3 — universe sparring                                                 #
# --------------------------------------------------------------------------- #
def _run_universe(spec: UniverseSpec) -> tuple[dict, MarkovWorldFeed]:
    feed = MarkovWorldFeed(spec)
    ticks = feed.timestamps()
    jrn = Journal(":memory:")
    tearsheet = run_backtest(MarkovWorldFeed(spec), ticks, journal=jrn)
    cal = jrn.calibration()
    # per-session trade stats so archetype attribution matches the P&L's
    # session-level granularity (a universe can span several archetypes)
    session_stats: dict[str, dict] = {}
    for r in jrn.fetch(settled_only=True):
        if r["was_traded"] != 1:
            continue
        pnl = economic_pnl(r)
        if pnl is None:
            continue
        s = session_stats.setdefault(
            r["session_date"], {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["pnl"] += pnl
        s["wins"] += 1 if pnl > 0 else 0
    jrn.close()

    d = cal["directional"]["overall"]
    row = {
        **spec.to_dict(),
        "day_archetypes": feed.day_archetype,
        "total_pnl": tearsheet.total_pnl,
        "trades": tearsheet.trade_ticks,
        "win_rate": tearsheet.win_rate,
        "sharpe": tearsheet.sharpe,
        "max_drawdown": tearsheet.max_drawdown,
        "gate_pass_rate": tearsheet.gate_pass_rate,
        "daily_pnl": tearsheet.daily_pnl or {},
        "session_stats": session_stats,
        "dir_hit": d.get("hit_rate"), "dir_n": d.get("n"),
        "brier_skill": cal["prob_profit"].get("brier_skill"),
    }
    return row, feed


def _archetype_matrix(universe_rows: list[dict],
                      feeds: list[MarkovWorldFeed]) -> dict:
    """Aggregate universe results into the per-archetype robustness matrix.
    Attribution is at session level: each session's P&L is charged to that
    day's archetype; sessions without trades count as 0 (standing aside is an
    observation too)."""
    agg: dict[str, dict] = {
        a: {"n_universes": 0, "n_sessions": 0, "total_pnl": 0.0,
            "session_pnls": [], "trades": 0, "trade_wins": 0,
            "dir_hits": [], "dir_ns": []}
        for a in ARCHETYPES}
    for row, feed in zip(universe_rows, feeds):
        touched = set()
        for session, arch in feed.day_archetype.items():
            a = agg[arch]
            stats = (row.get("session_stats") or {}).get(session) or {}
            pnl = stats.get("pnl", 0.0)
            a["n_sessions"] += 1
            a["total_pnl"] += pnl
            a["session_pnls"].append(pnl)
            a["trades"] += stats.get("trades", 0)
            a["trade_wins"] += stats.get("wins", 0)
            touched.add(arch)
        for arch in touched:
            agg[arch]["n_universes"] += 1
        # directional stats have no per-session decomposition; charge them to
        # the START archetype (dominant by construction)
        if row["dir_hit"] is not None and row["dir_n"]:
            agg[row["start_archetype"]]["dir_hits"].append(
                row["dir_hit"] * row["dir_n"])
            agg[row["start_archetype"]]["dir_ns"].append(row["dir_n"])

    out = {}
    for arch, a in agg.items():
        pnls = a.pop("session_pnls")
        hits, ns = a.pop("dir_hits"), a.pop("dir_ns")
        wins = a.pop("trade_wins")
        mean = (sum(pnls) / len(pnls)) if pnls else None
        s_wins = sum(1 for p in pnls if p > 0)
        s_losses = sum(1 for p in pnls if p < 0)
        out[arch] = {
            **a,
            "total_pnl": round(a["total_pnl"], 4),
            "mean_session_pnl": round(mean, 4) if mean is not None else None,
            "win_rate": round(wins / a["trades"], 4) if a["trades"] else None,
            "session_win_rate": (round(s_wins / (s_wins + s_losses), 4)
                                 if (s_wins + s_losses) else None),
            "dir_hit": (round(sum(hits) / sum(ns), 4) if ns else None),
        }
    return out


def _phase_universe(cfg: DojoConfig) -> dict:
    if cfg.skip_universe:
        return {"status": "skipped", "note": "--skip-universe"}
    catalog = UniverseCatalog(seed=cfg.catalog_seed, days=cfg.universe_days,
                              tick_stride=cfg.tick_stride)
    lattice_size = len(catalog.lattice())
    generations: list[dict] = []
    all_rows: list[dict] = []
    all_feeds: list[MarkovWorldFeed] = []

    for gen in range(cfg.generations):
        specs = (catalog.lattice() if cfg.full_lattice
                 else catalog.sample(cfg.universes_per_gen))
        print(f"  [dojo] phase 3 — generation {gen + 1}/{cfg.generations}: "
              f"{len(specs)} universes "
              f"(weights={ {k: round(v, 2) for k, v in catalog.weights.items()} })",
              flush=True)
        rows, feeds = [], []
        for spec in specs:
            t0 = time.time()
            row, feed = _run_universe(spec)
            rows.append(row)
            feeds.append(feed)
            print(f"    {spec.universe_id} {spec.start_archetype:<15} "
                  f"pnl={row['total_pnl']:+8.4f} trades={row['trades']:>3} "
                  f"dir_hit={row['dir_hit'] if row['dir_hit'] is not None else '—'} "
                  f"({time.time() - t0:.1f}s)", flush=True)

        gen_matrix = _archetype_matrix(rows, feeds)
        generations.append({
            "generation": gen,
            "weights": dict(catalog.weights),
            "universes": rows,
            "archetype_matrix": gen_matrix,
        })
        all_rows.extend(rows)
        all_feeds.extend(feeds)

        # evolve toward weakness: score = mean session P&L per archetype
        scores = {a: m["mean_session_pnl"]
                  for a, m in gen_matrix.items()
                  if m["mean_session_pnl"] is not None}
        catalog = catalog.evolve(scores)

    matrix = _archetype_matrix(all_rows, all_feeds)
    # two coverage matrices: generated minutes describe the ENVIRONMENT;
    # evaluated ticks are what the pipeline actually sparred (tick_stride
    # subset) — the honest coverage claim, and the one the flags use.
    coverage = merge_coverage(all_feeds)
    coverage_eval = merge_coverage(all_feeds, evaluated=True)
    visited = sum(1 for a in coverage for r in coverage[a] if coverage[a][r] > 0)
    visited_eval = sum(1 for a in coverage_eval for r in coverage_eval[a]
                       if coverage_eval[a][r] > 0)
    return {
        "status": "ok",
        "lattice_size": lattice_size,
        "n_universes": len(all_rows),
        "generations": generations,
        "archetype_matrix": matrix,
        "coverage": coverage,
        "coverage_evaluated": coverage_eval,
        "coverage_cells_visited": visited,
        "coverage_cells_visited_evaluated": visited_eval,
        "coverage_cells_total": len(ARCHETYPES) * len(REGIMES),
    }


# --------------------------------------------------------------------------- #
# Flags + summary                                                             #
# --------------------------------------------------------------------------- #
def _build_flags(recorded: dict, learner: dict, universe: dict) -> list[dict]:
    flags: list[dict] = []
    if recorded.get("status") == "insufficient_data":
        flags.append({"severity": "info", "flag": "no_recorded_tape",
                      "detail": recorded.get("note", "")})
    if learner.get("outcome") == "promotion_recommended":
        flags.append({"severity": "info", "flag": "promotion_pending_review",
                      "detail": "learner staged a candidate — run the "
                                "promoter CLI to review"})
    if universe.get("status") == "ok":
        for arch, m in universe["archetype_matrix"].items():
            mean = m.get("mean_session_pnl")
            if m["n_sessions"] >= 3 and mean is not None and mean < 0:
                flags.append({
                    "severity": "warn", "flag": f"weak_archetype:{arch}",
                    "detail": f"mean session P&L {mean:+.4f} over "
                              f"{m['n_sessions']} sessions — the pipeline "
                              f"loses in this market type"})
        cov = universe.get("coverage_evaluated") or universe["coverage"]
        missing = [f"{a}×{r}" for a in cov for r in cov[a] if cov[a][r] == 0]
        if missing:
            flags.append({"severity": "info", "flag": "uncovered_situations",
                          "detail": f"{len(missing)} (archetype × regime) "
                                    f"cells not yet evaluated by the "
                                    f"pipeline: {', '.join(missing[:6])}"
                                    + ("…" if len(missing) > 6 else "")})
    return flags


def _summary_text(recorded: dict, learner: dict, universe: dict,
                  flags: list[dict]) -> str:
    parts = []
    if recorded.get("status") == "ok":
        wf = recorded["walk_forward"]
        parts.append(f"recorded tape: {recorded['n_sessions']} sessions, "
                     f"mean fold P&L {wf.get('mean_pnl')}")
    else:
        parts.append(f"recorded tape: {recorded.get('status')}")
    parts.append(f"learner: {learner.get('outcome', learner.get('status'))}")
    if universe.get("status") == "ok":
        weak = sum(1 for f in flags if f["flag"].startswith("weak_archetype"))
        visited = universe.get("coverage_cells_visited_evaluated",
                               universe["coverage_cells_visited"])
        parts.append(
            f"universe sparring: {universe['n_universes']} universes, "
            f"{visited}/{universe['coverage_cells_total']} situation cells "
            f"evaluated, {weak} weak archetype(s)")
    else:
        parts.append(f"universe sparring: {universe.get('status')}")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run_dojo(cfg: Optional[DojoConfig] = None) -> dict:
    cfg = cfg or DojoConfig()
    report_date = cfg.report_date or dt.datetime.now(ET).date().isoformat()
    cfg.report_date = report_date
    started = time.time()

    recorded = _phase_recorded(cfg)
    learner = _phase_learner(cfg)
    universe = _phase_universe(cfg)

    flags = _build_flags(recorded, learner, universe)
    summary = _summary_text(recorded, learner, universe, flags)
    metrics = {
        "phases": {"recorded": recorded, "learner": learner,
                   "universe": universe},
        "elapsed_s": round(time.time() - started, 1),
        "config": {
            "record_dir": cfg.record_dir, "wf_folds": cfg.wf_folds,
            "learn_trials": cfg.learn_trials,
            "universes_per_gen": cfg.universes_per_gen,
            "generations": cfg.generations,
            "full_lattice": cfg.full_lattice,
            "universe_days": cfg.universe_days,
            "tick_stride": cfg.tick_stride,
            "catalog_seed": cfg.catalog_seed,
        },
    }

    jrn = Journal(cfg.db_path)
    report_id = jrn.log_validation_report(
        report_date, "dojo", metrics, summary, flags=flags)
    jrn.close()

    os.makedirs(cfg.reports_dir, exist_ok=True)
    stamp = dt.datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(cfg.reports_dir, f"dojo_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"report_date": report_date, "summary": summary,
                   "flags": flags, "metrics": metrics}, f, indent=2,
                  default=str)

    return {"report_id": report_id, "report_date": report_date,
            "summary": summary, "flags": flags, "json_path": json_path,
            "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Matrix-style training: recorded-tape walk-forward + "
                    "adaptive learning + Markov universe sparring, one "
                    "persisted dojo report (dashboard Dojo tab).")
    ap.add_argument("--db", default="shadow.db")
    ap.add_argument("--record-dir", default="")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--reports-dir", default=os.path.join("reports", "dojo"))
    ap.add_argument("--report-date", default=None)
    ap.add_argument("--skip-recorded", action="store_true")
    ap.add_argument("--skip-learner", action="store_true")
    ap.add_argument("--skip-universe", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--universes", type=int, default=6,
                    help="universes per generation (weighted sample)")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--full-lattice", action="store_true",
                    help="run every archetype × tilt × vol cell each "
                         "generation (72 universes — slow, exhaustive)")
    ap.add_argument("--days", type=int, default=8,
                    help="sessions per generated universe")
    ap.add_argument("--stride", type=int, default=5,
                    help="serve every Nth minute of each universe")
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = DojoConfig(
        db_path=args.db, record_dir=args.record_dir,
        configs_dir=args.configs_dir, reports_dir=args.reports_dir,
        report_date=args.report_date,
        skip_recorded=args.skip_recorded, skip_learner=args.skip_learner,
        skip_universe=args.skip_universe,
        wf_folds=args.folds, learn_trials=args.trials,
        universes_per_gen=args.universes, generations=args.generations,
        full_lattice=args.full_lattice, universe_days=args.days,
        tick_stride=args.stride, catalog_seed=args.seed,
    )
    out = run_dojo(cfg)

    print(f"\n  dojo report #{out['report_id']} ({out['report_date']})")
    print(f"  {out['summary']}")
    for fl in out["flags"]:
        print(f"    [{fl['severity'].upper():4}] {fl['flag']}: {fl['detail']}")
    uni = out["metrics"]["phases"]["universe"]
    if uni.get("status") == "ok":
        print("\n  Robustness matrix (per archetype):")
        # dir(start): directional hit is charged to each universe's START
        # archetype (it has no per-session decomposition)
        print(f"    {'archetype':<16} {'univ':>4} {'sess':>4} {'trades':>6} "
              f"{'total_pnl':>10} {'mean/sess':>10} {'win%':>5} {'dir(start)':>10}")
        for arch, m in uni["archetype_matrix"].items():
            wr = (f"{m['session_win_rate'] * 100:.0f}%"
                  if m["session_win_rate"] is not None else "—")
            dh = (f"{m['dir_hit'] * 100:.0f}%"
                  if m["dir_hit"] is not None else "—")
            mean = (f"{m['mean_session_pnl']:+.4f}"
                    if m["mean_session_pnl"] is not None else "—")
            print(f"    {arch:<16} {m['n_universes']:>4} {m['n_sessions']:>4} "
                  f"{m['trades']:>6} {m['total_pnl']:>+10.4f} {mean:>10} "
                  f"{wr:>5} {dh:>10}")
    print(f"\n  JSON: {out['json_path']}")
    if args.json:
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
