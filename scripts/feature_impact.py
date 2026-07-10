"""
scripts/feature_impact.py
=========================
Standardized feature-impact evaluation for mtf_matrix.py features: run the
FULL pipeline (backtest + walk-forward) under a baseline config and a variant
config, diff the results, and emit a Markdown report plus an optional
validation_reports journal entry (report_type='feature_impact', which the
dashboard's Validation tab renders).

This is step 2-6 of the standard workflow (docs/feature_impact_workflow.md):
every new or modified matrix variable goes through this comparison BEFORE it
earns a permanent place in the registry or a regime-blend weight.

Usage
-----
    python scripts/feature_impact.py \
      --feature bollinger_keltner_donchian_v1 \
      --baseline-config configs/baseline.yaml \
      --new-config configs/with_channels.yaml \
      --recorded /var/lib/zerodte/ticks \
      --output reports/feature_impact_2026-07-08.md \
      --db /var/lib/zerodte/shadow.db

Without --recorded the comparison runs on the coupled synthetic world
(synthetic_world.py) — useful for plumbing checks, but real recorded ticks are
the evidence that counts.

NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import RunConfig, load_run_config          # noqa: E402
from journal import Journal                                   # noqa: E402
from mtf_matrix import get_disabled_vars, set_disabled_vars   # noqa: E402

EPS = 1e-9

# Recommendation tiers, decided by a simple net-improvement score across the
# headline metrics (see _recommend). Conservative: a mixed picture is Neutral.
TIERS = ["Strong Positive", "Positive", "Neutral", "Negative"]


# --------------------------------------------------------------------------- #
# Evaluation of one config                                                    #
# --------------------------------------------------------------------------- #
def _evaluate(cfg: RunConfig, feed_factory, timestamps, n_folds: int) -> dict:
    """Backtest + walk-forward + per-regime breakdown under one RunConfig.
    The mtf disabled-vars toggle is process-wide, so set it around the run and
    always restore."""
    from backtest import run_backtest
    from validation_pipeline import per_regime_breakdown
    from walk_forward import WalkForwardConfig, run_walk_forward

    prev = get_disabled_vars()
    set_disabled_vars(cfg.disabled_vars)
    try:
        jrn = Journal(":memory:")
        tear = run_backtest(
            feed_factory(), timestamps,
            engine_cfg=cfg.engine_cfg,
            classifier_cfg=cfg.classifier_cfg,
            journal=jrn,
        )
        regimes = per_regime_breakdown(jrn)
        jrn.close()

        wf = run_walk_forward(
            feed_factory=feed_factory,
            timestamps=timestamps,
            wf_cfg=WalkForwardConfig(mode="expanding", n_folds=n_folds),
            engine_cfg=cfg.engine_cfg,
            classifier_cfg=cfg.classifier_cfg,
        )
    finally:
        set_disabled_vars(prev)

    bt = tear.to_dict()
    gate = tear.gate_effectiveness or {}
    taken = (gate.get("trades_taken") or {}).get("mean")
    blocked = (gate.get("blocked_by_gate") or {}).get("mean")
    bt["gate_edge"] = round(taken - blocked, 6) \
        if taken is not None and blocked is not None else None
    return {
        "config_name": cfg.name,
        "disabled_vars": sorted(cfg.disabled_vars),
        "backtest": bt,
        "walk_forward": wf.to_dict(),
        "per_regime": regimes,
    }


# --------------------------------------------------------------------------- #
# Comparison                                                                  #
# --------------------------------------------------------------------------- #
_DELTA_KEYS = [
    # (label, section, key, higher_is_better)
    ("sharpe", "backtest", "sharpe", True),
    ("win_rate", "backtest", "win_rate", True),
    ("mean_pnl_per_trade", "backtest", "mean_pnl_per_trade", True),
    ("total_pnl", "backtest", "total_pnl", True),
    ("max_drawdown", "backtest", "max_drawdown", False),
    ("gate_pass_rate", "backtest", "gate_pass_rate", True),
    ("gate_edge", "backtest", "gate_edge", True),
    ("ev_accuracy", "backtest", "ev_accuracy", True),
    ("wf_mean_sharpe", "walk_forward", "mean_sharpe", True),
    ("wf_mean_win_rate", "walk_forward", "mean_win_rate", True),
    ("wf_n_profitable", "walk_forward", "n_profitable", True),
    ("trade_count", "backtest", "trade_ticks", True),
]

# Metrics that decide the recommendation (trade_count/gate_pass_rate are
# activity indicators — signal-noise context, not quality by themselves).
_SCORED = {"sharpe", "win_rate", "mean_pnl_per_trade", "max_drawdown",
           "wf_mean_sharpe", "wf_n_profitable"}


def _deltas(baseline: dict, variant: dict) -> dict:
    out = {}
    for label, section, key, _ in _DELTA_KEYS:
        b = (baseline.get(section) or {}).get(key)
        v = (variant.get(section) or {}).get(key)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            out[label] = round(v - b, 6)
    return out


def _recommend(deltas: dict) -> tuple[str, list[str]]:
    """Net-improvement vote across the scored metrics -> tier + reasons."""
    score = 0
    reasons = []
    for label, _, _, higher_better in _DELTA_KEYS:
        if label not in _SCORED or label not in deltas:
            continue
        d = deltas[label]
        if abs(d) <= EPS:
            continue
        improved = (d > 0) == higher_better
        score += 1 if improved else -1
        reasons.append(f"{label} {'improved' if improved else 'degraded'} "
                       f"by {d:+.4f}")
    if score >= 3:
        tier = "Strong Positive"
    elif score >= 1:
        tier = "Positive"
    elif score <= -2:
        tier = "Negative"
    else:
        tier = "Neutral"
    if not reasons:
        reasons.append("no measurable metric change between configs")
    return tier, reasons


# --------------------------------------------------------------------------- #
# Markdown report                                                             #
# --------------------------------------------------------------------------- #
def _fmt(v, d=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:+.{d}f}"
    return str(v)


def render_markdown(feature: str, metrics: dict) -> str:
    base, var = metrics["baseline"], metrics["variant"]
    deltas = metrics["deltas"]
    lines = [
        f"# Feature Impact Report — `{feature}`",
        "",
        f"*Generated {metrics['generated_at']} · data source: {metrics['data_source']}*",
        "",
        f"- **Baseline config:** `{base['config_name']}`"
        + (f" (disabled: {', '.join(base['disabled_vars'])})" if base["disabled_vars"] else ""),
        f"- **Variant config:** `{var['config_name']}`"
        + (f" (disabled: {', '.join(var['disabled_vars'])})" if var["disabled_vars"] else ""),
        "",
        f"## Recommendation: **{metrics['recommendation']}**",
        "",
    ]
    lines += [f"- {r}" for r in metrics["recommendation_reasons"]]
    lines += [
        "",
        "## Key metric deltas (variant − baseline)",
        "",
        "| Metric | Baseline | With feature | Δ |",
        "|---|---:|---:|---:|",
    ]
    for label, section, key, _ in _DELTA_KEYS:
        b = (base.get(section) or {}).get(key)
        v = (var.get(section) or {}).get(key)
        d = deltas.get(label)
        lines.append(f"| {label} | {_fmt(b)} | {_fmt(v)} | {_fmt(d)} |")

    lines += [
        "",
        "## Per-regime impact (mean P&L of taken trades)",
        "",
        "| Regime | Baseline n | Baseline P&L | Variant n | Variant P&L |",
        "|---|---:|---:|---:|---:|",
    ]
    all_regimes = sorted(set(base["per_regime"]) | set(var["per_regime"]))
    for regime in all_regimes:
        bt = (base["per_regime"].get(regime) or {}).get("taken") or {}
        vt = (var["per_regime"].get(regime) or {}).get("taken") or {}
        lines.append(f"| {regime} | {bt.get('n', 0)} | {_fmt(bt.get('mean_pnl'))} "
                     f"| {vt.get('n', 0)} | {_fmt(vt.get('mean_pnl'))} |")

    lines += [
        "",
        "## Walk-forward consistency",
        "",
        f"- Baseline: {base['walk_forward'].get('n_profitable')}/"
        f"{base['walk_forward'].get('n_folds')} folds profitable, "
        f"mean Sharpe {_fmt(base['walk_forward'].get('mean_sharpe'), 3)}",
        f"- Variant:  {var['walk_forward'].get('n_profitable')}/"
        f"{var['walk_forward'].get('n_folds')} folds profitable, "
        f"mean Sharpe {_fmt(var['walk_forward'].get('mean_sharpe'), 3)}",
        "",
        "## Signal activity / noise",
        "",
        f"- Trade count: {base['backtest'].get('trade_ticks')} → "
        f"{var['backtest'].get('trade_ticks')}; gate pass rate "
        f"{_fmt(base['backtest'].get('gate_pass_rate'))} → "
        f"{_fmt(var['backtest'].get('gate_pass_rate'))}. A large activity jump "
        "without a quality improvement usually means added noise or "
        "conflicting signals.",
        "",
        "## Next steps",
        "",
        "1. If the recommendation is Positive or better, run the shadow runner "
        "with the feature enabled and watch real-market behavior (incl. RAS "
        "action quality) before making it permanent.",
        "2. Record the Keep / Modify / Remove decision in the journal notes.",
        "",
        "*Backtest/walk-forward on "
        + ("recorded real ticks." if metrics["data_source"].startswith("recorded")
           else "the coupled synthetic world — re-run on recorded ticks before deciding.")
        + " NOT financial advice.*",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Journal integration                                                         #
# --------------------------------------------------------------------------- #
def log_feature_impact(db_path: str, feature: str, metrics: dict,
                       notes: str = "") -> int:
    """Persist the feature-impact result to validation_reports so it appears
    in the dashboard's Validation tab alongside daily/weekly reports."""
    jrn = Journal(db_path)
    try:
        summary = (f"Feature impact — {feature}: {metrics['recommendation']}. "
                   + "; ".join(metrics["recommendation_reasons"][:3]))
        flags = []
        if metrics["recommendation"] == "Negative":
            flags.append({"flag": "feature_negative_impact", "severity": "warn",
                          "detail": f"{feature} degraded headline metrics"})
        return jrn.log_validation_report(
            metrics["report_date"], "feature_impact", metrics, summary,
            flags=flags, notes=notes or None)
    finally:
        jrn.close()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run_feature_impact(feature: str, baseline_cfg: RunConfig,
                       variant_cfg: RunConfig, recorded: str = "",
                       days: int = 10, seed: int = 11, stride: int = 5,
                       n_folds: int = 3) -> dict:
    """Evaluate both configs on the same data and return the metrics dict."""
    if recorded:
        from chain_store import RecordedFeed
        probe = RecordedFeed(recorded)
        timestamps = probe.timestamps()
        sessions = {t.date() for t in timestamps}
        if len(timestamps) < 100 or len(sessions) < 3:
            raise SystemExit(
                f"only {len(timestamps)} recorded ticks across {len(sessions)} "
                f"sessions in {recorded!r} — need >=3 sessions; let shadow "
                "mode record longer or omit --recorded for a synthetic run")
        feed_factory = lambda: RecordedFeed(recorded)   # noqa: E731
        data_source = f"recorded ({len(sessions)} sessions, {len(timestamps):,} ticks)"
    else:
        from synthetic_world import CoupledSyntheticFeed, WorldConfig
        feed_factory = lambda: CoupledSyntheticFeed(    # noqa: E731
            WorldConfig(days=days, seed=seed, tick_stride=stride))
        timestamps = feed_factory().timestamps()
        data_source = f"synthetic ({days} days, seed {seed})"

    print(f"== Feature impact: {feature} ==")
    print(f"   data: {data_source}")
    print(f"\n-- baseline: {baseline_cfg.name} --")
    baseline = _evaluate(baseline_cfg, feed_factory, timestamps, n_folds)
    print(f"\n-- variant: {variant_cfg.name} --")
    variant = _evaluate(variant_cfg, feed_factory, timestamps, n_folds)

    deltas = _deltas(baseline, variant)
    recommendation, reasons = _recommend(deltas)

    return {
        "feature": feature,
        "report_date": dt.date.today().isoformat(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
        "baseline": baseline,
        "variant": variant,
        "deltas": deltas,
        "recommendation": recommendation,
        "recommendation_reasons": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standardized feature-impact evaluation: backtest + "
                    "walk-forward under baseline vs variant configs, Markdown "
                    "report, optional journal logging.")
    ap.add_argument("--feature", required=True, help="feature label for the report")
    ap.add_argument("--baseline-config", required=True, help="baseline YAML overlay")
    ap.add_argument("--new-config", required=True, help="variant YAML overlay")
    ap.add_argument("--recorded", default="",
                    help="directory of ticks_*.jsonl.gz shadow recordings; "
                         "omit to run on the coupled synthetic world")
    ap.add_argument("--days", type=int, default=10, help="synthetic mode: days")
    ap.add_argument("--seed", type=int, default=11, help="synthetic mode: seed")
    ap.add_argument("--stride", type=int, default=5,
                    help="synthetic mode: minutes between ticks")
    ap.add_argument("--folds", type=int, default=3, help="walk-forward folds")
    ap.add_argument("--output", default="",
                    help="write the Markdown report here (default: stdout)")
    ap.add_argument("--db", default="",
                    help="journal SQLite path; when set, the report is logged "
                         "to validation_reports (Validation dashboard tab)")
    ap.add_argument("--notes", default="", help="freeform notes stored with the report")
    args = ap.parse_args()

    baseline_cfg = load_run_config(args.baseline_config)
    variant_cfg = load_run_config(args.new_config)

    metrics = run_feature_impact(
        args.feature, baseline_cfg, variant_cfg,
        recorded=args.recorded, days=args.days, seed=args.seed,
        stride=args.stride, n_folds=args.folds)

    md = render_markdown(args.feature, metrics)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nreport written to {args.output}")
    else:
        print("\n" + md)

    if args.db:
        rid = log_feature_impact(args.db, args.feature, metrics, notes=args.notes)
        print(f"logged to validation_reports (id={rid}) in {args.db}")

    print(f"\nRECOMMENDATION: {metrics['recommendation']}")


if __name__ == "__main__":
    main()
