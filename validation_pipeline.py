"""
validation_pipeline.py
======================
Scheduled validation for the 0DTE pipeline: a lightweight DAILY health check
and a deeper WEEKLY review. Each run produces a structured report that is
persisted to the journal's validation_reports table (the dashboard's
"Validation" tab reads from there) and can push an alert through notifier.py
when key metrics degrade beyond thresholds.

Daily (after market close)
--------------------------
  * short walk-forward over the most recent recorded sessions (3-5 folds)
  * core health metrics from the shadow journal: win rate, mean P&L, Sharpe,
    gate effectiveness, Brier skill / EV bias, RAS action frequency, regime
    diversity
  * deltas vs the previous daily report + degradation flags

Weekly (weekend)
----------------
  * full walk-forward across more folds on the whole recorded window
  * everything the daily has, plus per-regime P&L breakdown, feature
    contribution analysis (component_correlations incl. signals_json),
    gate-effectiveness trend across prior reports, and high-level
    recommendations

CLI
---
    python validation_pipeline.py --mode daily  --db shadow.db --record-dir ticks
    python validation_pipeline.py --mode weekly --db shadow.db --record-dir ticks --notify

Degradation thresholds are conservative on purpose (alert fatigue is a real
failure mode); tune FLAG_THRESHOLDS as the report history accumulates.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional
from zoneinfo import ZoneInfo

from journal import Journal

ET = ZoneInfo("America/New_York")

# Conservative starting thresholds -- a flag fires only on clear degradation.
FLAG_THRESHOLDS = {
    "sharpe_drop_frac": 0.20,      # Sharpe drop > 20% vs trailing average
    "win_rate_drop_frac": 0.15,    # win rate drop > 15% (relative) vs trailing avg
    "min_brier_skill": 0.0,        # prob_profit must beat the base rate
    "min_wf_sessions": 3,          # recorded sessions needed for a walk-forward
    "min_wf_ticks": 100,           # recorded ticks needed for a walk-forward
}


# --------------------------------------------------------------------------- #
# Journal-derived health metrics                                              #
# --------------------------------------------------------------------------- #
def _ras_summary(jrn: Journal) -> dict:
    """RAS score distribution + action frequency across all evaluations."""
    try:
        rows = jrn.fetch_ras()
    except Exception:
        return {"n": 0}
    if not rows:
        return {"n": 0}
    scores = [r["score"] for r in rows if r.get("score") is not None]
    actions: dict[str, int] = {}
    for r in rows:
        a = r.get("action") or "unknown"
        actions[a] = actions.get(a, 0) + 1
    out: dict = {"n": len(rows), "actions": actions}
    if scores:
        s = sorted(scores)
        out["score_mean"] = round(sum(scores) / len(scores), 2)
        out["score_min"] = round(s[0], 2)
        out["score_p50"] = round(s[len(s) // 2], 2)
    return out


def journal_health_metrics(jrn: Journal) -> dict:
    """The core health panel, computed from what the journal already stores."""
    gate = jrn.gate_effectiveness()
    cal = jrn.calibration()
    diversity = jrn.regime_diversity()

    taken = gate["trades_taken"]
    pp = cal["prob_profit"]
    ev = cal["ev"]
    direction = cal["directional"]["overall"]

    return {
        "n_settled_trades": taken["n"],
        "win_rate": taken["win_rate"],
        "mean_pnl_per_trade": taken["mean"],
        "gate_effectiveness": gate,
        "brier": pp.get("brier"),
        "brier_skill": pp.get("brier_skill"),
        "ev_bias": ev.get("mean_ev_error"),
        "ev_mae": ev.get("mae_ev_error"),
        "directional_hit_rate": direction.get("hit_rate"),
        "directional_n": direction.get("n"),
        "regime_diversity": diversity,
        "ras": _ras_summary(jrn),
    }


def per_regime_breakdown(jrn: Journal) -> dict:
    """Settled-candidate P&L grouped by gex_regime, split taken vs blocked."""
    rows = [r for r in jrn.fetch(settled_only=True)
            if r["realized_pnl"] is not None]
    out: dict[str, dict] = {}
    for r in rows:
        regime = r["gex_regime"] or "unknown"
        bucket = out.setdefault(regime, {"taken": [], "blocked": []})
        if r["was_traded"] == 1:
            bucket["taken"].append(r["realized_pnl"])
        elif r["candidate_present"] == 1 and r["gate_pass"] == 0:
            bucket["blocked"].append(r["realized_pnl"])

    def stats(xs: list) -> dict:
        if not xs:
            return {"n": 0, "mean_pnl": None, "win_rate": None}
        wins = sum(1 for x in xs if x > 0)
        return {"n": len(xs), "mean_pnl": round(sum(xs) / len(xs), 4),
                "win_rate": round(wins / len(xs), 3)}

    return {regime: {"taken": stats(b["taken"]), "blocked": stats(b["blocked"])}
            for regime, b in out.items()}


# --------------------------------------------------------------------------- #
# Walk-forward on recorded ticks                                              #
# --------------------------------------------------------------------------- #
def _recorded_walk_forward(record_dir: str, n_folds: int,
                           lookback_sessions: Optional[int] = None) -> Optional[dict]:
    """
    Run a walk-forward over the shadow recordings in record_dir. When
    lookback_sessions is given (daily mode), the warm-up fraction is chosen so
    the test region covers roughly the most recent N sessions. Returns
    WalkForwardResult.to_dict() or None when there isn't enough history.
    """
    if not record_dir or not os.path.isdir(record_dir):
        return None
    from chain_store import RecordedFeed
    from walk_forward import WalkForwardConfig, run_walk_forward

    probe = RecordedFeed(record_dir)
    ticks = probe.timestamps()
    sessions = sorted({t.date() for t in ticks})
    if (len(ticks) < FLAG_THRESHOLDS["min_wf_ticks"]
            or len(sessions) < FLAG_THRESHOLDS["min_wf_sessions"]):
        return None

    train_frac = 0.6
    if lookback_sessions and len(sessions) > lookback_sessions:
        window = set(sessions[-lookback_sessions:])
        start_idx = next(i for i, t in enumerate(ticks) if t.date() in window)
        train_frac = max(0.5, min(0.9, start_idx / len(ticks)))

    wf = run_walk_forward(
        feed_factory=lambda: RecordedFeed(record_dir),
        timestamps=ticks,
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=n_folds,
                                 train_frac=train_frac),
    )
    d = wf.to_dict()
    d["n_recorded_ticks"] = len(ticks)
    d["n_recorded_sessions"] = len(sessions)
    return d


# --------------------------------------------------------------------------- #
# Flags + deltas                                                              #
# --------------------------------------------------------------------------- #
def _flag(name: str, severity: str, detail: str) -> dict:
    return {"flag": name, "severity": severity, "detail": detail}


def _trailing_mean(prior_reports: list[dict], *path) -> Optional[float]:
    vals = []
    for rep in prior_reports:
        v = rep.get("metrics") or {}
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
            if v is None:
                break
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def compute_flags(metrics: dict, prior_reports: list[dict],
                  thresholds: Optional[dict] = None) -> list[dict]:
    """Degradation flags for one report, judged against absolute floors and
    the trailing average of prior reports of the same type."""
    cfg = {**FLAG_THRESHOLDS, **(thresholds or {})}
    flags: list[dict] = []
    jm = metrics.get("journal") or {}
    wf = metrics.get("walk_forward")

    if wf is None:
        flags.append(_flag(
            "insufficient_data", "info",
            "not enough recorded sessions for a walk-forward; journal metrics only"))

    # Gate effectiveness reversal (absolute)
    gate = (jm.get("gate_effectiveness") or {})
    taken_mean = (gate.get("trades_taken") or {}).get("mean")
    blocked_mean = (gate.get("blocked_by_gate") or {}).get("mean")
    if taken_mean is not None and blocked_mean is not None and blocked_mean > taken_mean:
        flags.append(_flag(
            "gate_effectiveness_reversed", "alert",
            f"blocked-trade mean P&L ({blocked_mean:+.4f}) is better than taken "
            f"({taken_mean:+.4f}) — the gate may be costing edge"))

    # Brier skill below floor (absolute)
    skill = jm.get("brier_skill")
    if skill is not None and skill < cfg["min_brier_skill"]:
        flags.append(_flag(
            "brier_skill_negative", "alert",
            f"Brier skill {skill:+.4f} < {cfg['min_brier_skill']} — prob_profit no "
            f"longer beats quoting the base rate"))

    # Sharpe degradation vs trailing average of prior reports (relative)
    sharpe = (wf or {}).get("mean_sharpe")
    trail_sharpe = _trailing_mean(prior_reports, "walk_forward", "mean_sharpe")
    if (sharpe is not None and trail_sharpe is not None and trail_sharpe > 0
            and sharpe < trail_sharpe * (1.0 - cfg["sharpe_drop_frac"])):
        flags.append(_flag(
            "sharpe_degraded", "alert",
            f"walk-forward Sharpe {sharpe:+.3f} is >{cfg['sharpe_drop_frac']:.0%} "
            f"below the trailing average {trail_sharpe:+.3f}"))

    # Win-rate degradation vs trailing average (relative)
    wr = jm.get("win_rate")
    trail_wr = _trailing_mean(prior_reports, "journal", "win_rate")
    if (wr is not None and trail_wr is not None and trail_wr > 0
            and wr < trail_wr * (1.0 - cfg["win_rate_drop_frac"])):
        flags.append(_flag(
            "win_rate_degraded", "warn",
            f"win rate {wr:.1%} is >{cfg['win_rate_drop_frac']:.0%} below the "
            f"trailing average {trail_wr:.1%}"))

    return flags


def _deltas_vs_previous(metrics: dict, prior_reports: list[dict]) -> dict:
    """Signed change of the headline numbers vs the most recent prior report."""
    if not prior_reports:
        return {}
    prev = prior_reports[0].get("metrics") or {}
    prev_jm, jm = prev.get("journal") or {}, metrics.get("journal") or {}
    prev_wf, wf = prev.get("walk_forward") or {}, metrics.get("walk_forward") or {}
    out = {}
    for label, cur, old in (
        ("win_rate", jm.get("win_rate"), prev_jm.get("win_rate")),
        ("mean_pnl_per_trade", jm.get("mean_pnl_per_trade"), prev_jm.get("mean_pnl_per_trade")),
        ("brier_skill", jm.get("brier_skill"), prev_jm.get("brier_skill")),
        ("ev_bias", jm.get("ev_bias"), prev_jm.get("ev_bias")),
        ("mean_sharpe", wf.get("mean_sharpe"), prev_wf.get("mean_sharpe")),
    ):
        if isinstance(cur, (int, float)) and isinstance(old, (int, float)):
            out[label] = round(cur - old, 6)
    if out:
        out["vs_report_date"] = prior_reports[0].get("report_date")
    return out


# --------------------------------------------------------------------------- #
# Weekly extras                                                               #
# --------------------------------------------------------------------------- #
def _gate_trend(prior_reports: list[dict], current_metrics: dict,
                report_date: str) -> list[dict]:
    """Gate edge (taken mean - blocked mean) over time, oldest first."""
    points = []

    def point(date: str, metrics: dict) -> Optional[dict]:
        gate = ((metrics.get("journal") or {}).get("gate_effectiveness") or {})
        t = (gate.get("trades_taken") or {}).get("mean")
        b = (gate.get("blocked_by_gate") or {}).get("mean")
        if t is None or b is None:
            return None
        return {"report_date": date, "taken_mean": t, "blocked_mean": b,
                "gate_edge": round(t - b, 4)}

    for rep in reversed(prior_reports):          # oldest first
        p = point(rep.get("report_date", "?"), rep.get("metrics") or {})
        if p:
            points.append(p)
    p = point(report_date, current_metrics)
    if p:
        points.append(p)
    return points


def _recommendations(metrics: dict, flags: list[dict]) -> list[str]:
    recs: list[str] = []
    flag_names = {f["flag"] for f in flags}
    jm = metrics.get("journal") or {}

    if "gate_effectiveness_reversed" in flag_names:
        recs.append("Review gate thresholds: blocked candidates are outperforming "
                    "taken trades. Consider a light optimizer pass on gate.* params.")
    if "brier_skill_negative" in flag_names:
        recs.append("prob_profit calibration has degraded — inspect "
                    "prob_calibration() reliability bins before trusting EV-based sizing.")
    if "sharpe_degraded" in flag_names:
        recs.append("Out-of-sample Sharpe is degrading — re-run the optimizer on "
                    "recent recordings or reduce size until it stabilizes.")

    diversity = (jm.get("regime_diversity") or {})
    if diversity.get("n", 0) >= 10 and diversity.get("distinct", 0) < 2:
        recs.append("Track record is concentrated in a single GEX regime; treat "
                    "aggregate stats as untested in other regimes.")

    corr = metrics.get("feature_contributions") or {}
    weak = [k for k, v in corr.items()
            if k.startswith("sig:") and isinstance(v, (int, float)) and abs(v) < 0.05]
    if len(weak) >= 3:
        recs.append(f"{len(weak)} observation-only signals show |r| < 0.05 vs realized "
                    "P&L — candidates for removal via the feature-impact workflow.")

    if not recs:
        recs.append("No concerning trends detected; keep accumulating history.")
    return recs


# --------------------------------------------------------------------------- #
# Report builders                                                             #
# --------------------------------------------------------------------------- #
def _summarize(report_type: str, metrics: dict, flags: list[dict]) -> str:
    jm = metrics.get("journal") or {}
    wf = metrics.get("walk_forward")
    parts = []
    if wf:
        sh = wf.get("mean_sharpe")
        parts.append(f"walk-forward: {wf.get('n_profitable', '?')}/{wf.get('n_folds', '?')} "
                     f"folds profitable"
                     + (f", mean Sharpe {sh:+.2f}" if isinstance(sh, (int, float)) else ""))
    n = jm.get("n_settled_trades") or 0
    if n:
        wr = jm.get("win_rate")
        mp = jm.get("mean_pnl_per_trade")
        parts.append(f"{n} settled trades"
                     + (f", win rate {wr:.0%}" if isinstance(wr, (int, float)) else "")
                     + (f", mean P&L {mp:+.4f}" if isinstance(mp, (int, float)) else ""))
    else:
        parts.append("no settled trades yet")
    alerts = [f for f in flags if f.get("severity") == "alert"]
    if alerts:
        parts.append(f"{len(alerts)} ALERT flag(s): "
                     + ", ".join(f["flag"] for f in alerts))
    else:
        parts.append("no degradation alerts")
    return f"{report_type.capitalize()} validation — " + "; ".join(parts)


def run_daily_validation(db_path: str, record_dir: str = "",
                         lookback_days: int = 20, n_folds: int = 4,
                         report_date: Optional[str] = None,
                         log_to_journal: bool = True) -> dict:
    """
    Lightweight daily health check. Returns the report dict:
      {report_date, report_type, metrics, summary, flags}
    and (by default) persists it to validation_reports.
    """
    report_date = report_date or dt.datetime.now(ET).date().isoformat()
    jrn = Journal(db_path)
    try:
        metrics: dict = {
            "journal": journal_health_metrics(jrn),
            "walk_forward": _recorded_walk_forward(record_dir, n_folds,
                                                   lookback_sessions=lookback_days),
        }
        prior = jrn.fetch_validation_reports(report_type="daily", limit=10)
        metrics["deltas"] = _deltas_vs_previous(metrics, prior)
        flags = compute_flags(metrics, prior)
        summary = _summarize("daily", metrics, flags)
        report = {"report_date": report_date, "report_type": "daily",
                  "metrics": metrics, "summary": summary, "flags": flags}
        if log_to_journal:
            report["id"] = jrn.log_validation_report(
                report_date, "daily", metrics, summary, flags)
        return report
    finally:
        jrn.close()


def run_weekly_validation(db_path: str, record_dir: str = "",
                          n_folds: int = 6,
                          report_date: Optional[str] = None,
                          log_to_journal: bool = True) -> dict:
    """
    Deeper weekly review: full-window walk-forward plus regime-level and
    feature-level analysis, trend across prior reports, and recommendations.
    """
    report_date = report_date or dt.datetime.now(ET).date().isoformat()
    jrn = Journal(db_path)
    try:
        metrics: dict = {
            "journal": journal_health_metrics(jrn),
            "walk_forward": _recorded_walk_forward(record_dir, n_folds),
            "per_regime": per_regime_breakdown(jrn),
            "feature_contributions": jrn.component_correlations(),
        }

        # Aggregate the past week's daily reports so the weekly is the roll-up.
        week_ago = (dt.date.fromisoformat(report_date) - dt.timedelta(days=7)).isoformat()
        dailies = jrn.fetch_validation_reports(report_type="daily", since=week_ago)
        metrics["daily_aggregate"] = {
            "n_daily_reports": len(dailies),
            "mean_win_rate": _trailing_mean(dailies, "journal", "win_rate"),
            "mean_sharpe": _trailing_mean(dailies, "walk_forward", "mean_sharpe"),
            "total_alert_flags": sum(
                1 for rep in dailies for f in (rep.get("flags") or [])
                if isinstance(f, dict) and f.get("severity") == "alert"),
        }

        prior = jrn.fetch_validation_reports(report_type="weekly", limit=10)
        metrics["deltas"] = _deltas_vs_previous(metrics, prior)
        metrics["gate_trend"] = _gate_trend(prior, metrics, report_date)
        flags = compute_flags(metrics, prior)
        metrics["recommendations"] = _recommendations(metrics, flags)
        summary = _summarize("weekly", metrics, flags)
        report = {"report_date": report_date, "report_type": "weekly",
                  "metrics": metrics, "summary": summary, "flags": flags}
        if log_to_journal:
            report["id"] = jrn.log_validation_report(
                report_date, "weekly", metrics, summary, flags)
        return report
    finally:
        jrn.close()


# --------------------------------------------------------------------------- #
# Alerting                                                                    #
# --------------------------------------------------------------------------- #
def send_degradation_alert(report: dict) -> bool:
    """Push alert-severity flags through notifier.py (stdout always; ntfy/email
    when configured). Returns True when an alert was sent."""
    alerts = [f for f in report.get("flags", [])
              if isinstance(f, dict) and f.get("severity") == "alert"]
    if not alerts:
        return False
    from notifier import Notifier
    title = (f"0DTE {report['report_type']} validation: "
             f"{len(alerts)} degradation alert(s)")
    body = "\n".join(f"[{f['flag']}] {f['detail']}" for f in alerts)
    body += f"\n\n{report.get('summary', '')}"
    Notifier().send_text(title, body, tags="warning")
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Scheduled validation pipeline: daily health check or "
                    "weekly deep review, persisted to validation_reports.")
    ap.add_argument("--mode", choices=["daily", "weekly"], required=True)
    ap.add_argument("--db", default="shadow.db", help="journal SQLite path")
    ap.add_argument("--record-dir", default="",
                    help="directory of ticks_*.jsonl.gz shadow recordings "
                         "(VPS default: /var/lib/zerodte/ticks)")
    ap.add_argument("--lookback-days", type=int, default=20,
                    help="daily mode: recent sessions the test window should cover")
    ap.add_argument("--folds", type=int, default=0,
                    help="walk-forward folds (default: 4 daily, 6 weekly)")
    ap.add_argument("--date", default=None, help="report date override (YYYY-MM-DD)")
    ap.add_argument("--notify", action="store_true",
                    help="send an alert via notifier.py when alert flags fire")
    ap.add_argument("--json", action="store_true", help="print the full report JSON")
    args = ap.parse_args()

    if args.mode == "daily":
        report = run_daily_validation(
            args.db, args.record_dir,
            lookback_days=args.lookback_days,
            n_folds=args.folds or 4, report_date=args.date)
    else:
        report = run_weekly_validation(
            args.db, args.record_dir,
            n_folds=args.folds or 6, report_date=args.date)

    print(f"\n{report['summary']}")
    for f in report["flags"]:
        print(f"  [{f['severity'].upper()}] {f['flag']}: {f['detail']}")
    if args.json:
        print(_json.dumps(report, indent=2, default=str))

    if args.notify:
        sent = send_degradation_alert(report)
        print("  alert dispatched" if sent else "  no alert-severity flags; nothing sent")


if __name__ == "__main__":
    main()
