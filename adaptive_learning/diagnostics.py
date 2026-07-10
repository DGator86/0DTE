"""
adaptive_learning/diagnostics.py
================================
Failure-mode detection for the learning loop. Reads the journal (and the
validation_reports history) and emits structured Diagnosis records — never
parameter changes. Every diagnosis carries the raw numbers in `evidence` so
the downstream hypothesis generator (and a human) can audit the reasoning.

Detected failure modes
----------------------
  gate_effectiveness_reversed  blocked candidates outperform taken trades
  brier_skill_negative         prob_profit no longer beats the base rate
  ev_bias                      physical-density EV is systematically off
  directional_weak             regime direction bias below coin-flip
  sharpe_collapse              walk-forward Sharpe fell vs report history
  trade_frequency_collapse     settled trade count fell vs report history
  regime_concentration         track record concentrated in one gex_regime
  MODEL_DRIFT / REGIME_DRIFT / FEATURE_DRIFT
                               recent window departed from the trailing one

Confidence is a bounded heuristic combining sample size and effect size —
it is a triage signal for the hypothesis generator, not a p-value.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from journal import Journal

# Conservative floors: a diagnosis needs a real sample behind it.
THRESHOLDS = {
    "min_gate_n": 5,             # per side (taken / blocked)
    "min_calibration_n": 20,
    "min_directional_n": 50,
    "max_abs_ev_bias": 0.10,     # $/share
    "min_regime_n": 10,
    "sharpe_drop_frac": 0.20,
    "trade_drop_frac": 0.50,
    "drift_rel": 0.25,           # relative change that counts as drift
    "drift_min_n": 10,           # settled rows per window
    "recent_sessions": 30,
    "baseline_sessions": 90,
}


@dataclass
class Diagnosis:
    issue: str
    severity: str                # "info" | "warn" | "alert"
    confidence: float            # 0..1 heuristic
    affected_module: str
    likely_cause: str
    recommendation: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "issue": self.issue, "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "affected_module": self.affected_module,
            "likely_cause": self.likely_cause,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
        }


def _confidence(n: int, n_ref: int, effect: float, effect_ref: float) -> float:
    """Bounded heuristic: half from sample size (saturating at n_ref), half
    from effect size (saturating at effect_ref). Never claims certainty."""
    size_term = n / (n + max(1, n_ref))
    effect_term = min(1.0, abs(effect) / max(1e-9, effect_ref))
    return round(min(0.95, 0.5 * size_term + 0.5 * effect_term), 3)


# --------------------------------------------------------------------------- #
# Journal-level checks                                                          #
# --------------------------------------------------------------------------- #
def _check_gate_inversion(jrn: Journal, cfg: dict) -> Optional[Diagnosis]:
    gate = jrn.gate_effectiveness()
    t, b = gate["trades_taken"], gate["blocked_by_gate"]
    if (t["n"] < cfg["min_gate_n"] or b["n"] < cfg["min_gate_n"]
            or t["mean"] is None or b["mean"] is None):
        return None
    if b["mean"] <= t["mean"]:
        return None
    gap = b["mean"] - t["mean"]
    return Diagnosis(
        issue="gate_effectiveness_reversed",
        severity="alert",
        confidence=_confidence(min(t["n"], b["n"]), 30, gap, 0.5),
        affected_module="gate_scorer",
        likely_cause=("one or more hard gates (GEX rank / ADX / flip buffer) "
                      "or the selector EV floor are inverted for current "
                      "conditions — the gate blocks better trades than it lets "
                      "through"),
        recommendation=("loosen and reshape the gate: search "
                        "gate.min_gex_pct_rank / gate.max_adx / "
                        "gate.flip_buffer_frac / selector.min_ev through "
                        "walk-forward + holdout; do NOT blindly tighten"),
        evidence={"gate_effectiveness": gate, "gap": round(gap, 4)},
    )


def _check_brier(jrn: Journal, cfg: dict) -> Optional[Diagnosis]:
    pp = jrn.prob_calibration()
    n, skill = pp.get("n", 0), pp.get("brier_skill")
    if n < cfg["min_calibration_n"] or skill is None or skill >= 0:
        return None
    return Diagnosis(
        issue="brier_skill_negative",
        severity="alert",
        confidence=_confidence(n, 60, skill, 0.2),
        affected_module="mc / spread_selector (prob_profit)",
        likely_cause="prob_profit no longer beats always quoting the base rate "
                     "— probability model is miscalibrated for current vol",
        recommendation="inspect prob_calibration() reliability bins; do not "
                       "trust EV-based sizing until skill recovers",
        evidence={"prob_calibration": {k: pp.get(k) for k in
                                       ("n", "base_rate", "brier", "brier_skill")}},
    )


def _check_ev_bias(jrn: Journal, cfg: dict) -> Optional[Diagnosis]:
    ev = jrn.calibration()["ev"]
    n, bias = ev.get("n", 0), ev.get("mean_ev_error")
    if n < cfg["min_calibration_n"] or bias is None:
        return None
    if abs(bias) <= cfg["max_abs_ev_bias"]:
        return None
    direction = "optimistic (EV overstated)" if bias < 0 else "pessimistic (EV understated)"
    return Diagnosis(
        issue="ev_bias",
        severity="warn",
        confidence=_confidence(n, 60, bias, 0.3),
        affected_module="rnd_extractor (physical density)",
        likely_cause=f"physical-density EV is systematically {direction}: "
                     f"mean ev_error {bias:+.4f} $/share",
        recommendation="search rnd.vol_risk_premium and selector.min_ev; "
                       "verify realized-vol squeeze inputs",
        evidence={"ev": ev},
    )


def _check_directional(jrn: Journal, cfg: dict) -> Optional[Diagnosis]:
    d = jrn.directional_accuracy()["overall"]
    n, hit = d.get("n", 0), d.get("hit_rate")
    if n < cfg["min_directional_n"] or hit is None or hit >= 0.5:
        return None
    return Diagnosis(
        issue="directional_weak",
        severity="warn",
        confidence=_confidence(n, 200, 0.5 - hit, 0.10),
        affected_module="decision_matrix / regime_classifier (direction bias)",
        likely_cause=f"direction bias hit rate {hit:.1%} < 50% over {n} "
                     f"resolved-bias ticks — the directional premise is not "
                     f"holding",
        recommendation="search rnd.dir_drift_frac and the directional ADX ramp "
                       "(gate.dir_adx_floor / gate.dir_adx_full)",
        evidence={"directional": d},
    )


def _check_regime_concentration(jrn: Journal, cfg: dict) -> Optional[Diagnosis]:
    div = jrn.regime_diversity()
    if div["n"] < cfg["min_regime_n"] or div["distinct"] >= 2:
        return None
    return Diagnosis(
        issue="regime_concentration",
        severity="info",
        confidence=_confidence(div["n"], 30, 1.0, 1.0),
        affected_module="track record (not a code defect)",
        likely_cause=f"all {div['n']} settled trades sit in one gex_regime "
                     f"({', '.join(div['regimes'])}) — untested elsewhere",
        recommendation="treat aggregate stats as regime-conditional; consider "
                       "regime_overrides rather than global re-tuning",
        evidence={"regime_diversity": div},
    )


# --------------------------------------------------------------------------- #
# Report-history checks (needs prior validation reports)                        #
# --------------------------------------------------------------------------- #
def _history_vals(prior_reports: list[dict], *path) -> list[float]:
    vals = []
    for rep in prior_reports:
        v = rep.get("metrics") or {}
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
            if v is None:
                break
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def _check_sharpe_collapse(prior_reports: list[dict], cfg: dict) -> Optional[Diagnosis]:
    vals = _history_vals(prior_reports, "walk_forward", "mean_sharpe")
    if len(vals) < 3:
        return None
    latest, trail = vals[0], sum(vals[1:]) / len(vals[1:])
    if trail <= 0 or latest >= trail * (1.0 - cfg["sharpe_drop_frac"]):
        return None
    return Diagnosis(
        issue="sharpe_collapse",
        severity="alert",
        confidence=_confidence(len(vals), 8, (trail - latest) / trail, 0.5),
        affected_module="whole pipeline (out-of-sample performance)",
        likely_cause=f"walk-forward Sharpe {latest:+.3f} is "
                     f">{cfg['sharpe_drop_frac']:.0%} below the trailing "
                     f"average {trail:+.3f}",
        recommendation="run a full learning cycle; reduce size until a "
                       "validated challenger lands",
        evidence={"latest": latest, "trailing_mean": round(trail, 4),
                  "history": vals},
    )


def _check_trade_collapse(prior_reports: list[dict], cfg: dict) -> Optional[Diagnosis]:
    vals = _history_vals(prior_reports, "journal", "n_settled_trades")
    if len(vals) < 3:
        return None
    latest, trail = vals[0], sum(vals[1:]) / len(vals[1:])
    if trail <= 0 or latest >= trail * (1.0 - cfg["trade_drop_frac"]):
        return None
    return Diagnosis(
        issue="trade_frequency_collapse",
        severity="warn",
        confidence=_confidence(len(vals), 8, (trail - latest) / trail, 0.8),
        affected_module="gate_scorer / spread_selector (funnel)",
        likely_cause=f"settled trade count {latest:.0f} is "
                     f">{cfg['trade_drop_frac']:.0%} below the trailing "
                     f"average {trail:.1f} — a gate or veto started firing "
                     f"far more often",
        recommendation="inspect decision_funnel(); the gate-inversion search "
                       "space also covers over-tight gates",
        evidence={"latest": latest, "trailing_mean": round(trail, 2),
                  "history": vals},
    )


# --------------------------------------------------------------------------- #
# Drift: current window vs trailing window                                      #
# --------------------------------------------------------------------------- #
def _window_metrics(rows: list[dict]) -> dict:
    """Health metrics over one set of settled journal rows."""
    taken = [r["realized_pnl"] for r in rows
             if r["was_traded"] == 1 and r["realized_pnl"] is not None]
    blocked = [r["realized_pnl"] for r in rows
               if r["was_traded"] == 0 and r["candidate_present"] == 1
               and r["gate_pass"] == 0 and r["realized_pnl"] is not None]
    sessions = {r["session_date"] for r in rows}
    regimes: dict[str, int] = {}
    for r in rows:
        if r["was_traded"] == 1:
            key = r["gex_regime"] or "unknown"
            regimes[key] = regimes.get(key, 0) + 1

    pairs = [(float(r["prob_profit"]), 1.0 if r["realized_pnl"] > 0 else 0.0)
             for r in rows
             if r["prob_profit"] is not None and r["realized_pnl"] is not None]
    brier = None
    if pairs:
        brier = sum((p - w) ** 2 for p, w in pairs) / len(pairs)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    taken_mean, blocked_mean = mean(taken), mean(blocked)
    return {
        "n_settled": len(rows),
        "n_taken": len(taken),
        "sessions": len(sessions),
        "win_rate": (sum(1 for p in taken if p > 0) / len(taken)) if taken else None,
        "gate_edge": (round(taken_mean - blocked_mean, 4)
                      if taken_mean is not None and blocked_mean is not None else None),
        "brier": round(brier, 4) if brier is not None else None,
        "trades_per_session": (round(len(taken) / len(sessions), 3)
                               if sessions else None),
        "regime_mix": regimes,
    }


def compute_drift(jrn: Journal,
                  recent_sessions: Optional[int] = None,
                  baseline_sessions: Optional[int] = None,
                  thresholds: Optional[dict] = None) -> dict:
    """Compare the most recent N sessions against the trailing M sessions
    before them. Returns {recent, baseline, deltas, drifts} where drifts is a
    list of {kind, metric, rel_change, detail}."""
    cfg = {**THRESHOLDS, **(thresholds or {})}
    recent_n = recent_sessions or cfg["recent_sessions"]
    baseline_n = baseline_sessions or cfg["baseline_sessions"]

    rows = [r for r in jrn.fetch(settled_only=True)
            if r["realized_pnl"] is not None]
    sessions = sorted({r["session_date"] for r in rows})
    if len(sessions) < 2:
        return {"note": "not enough sessions for drift analysis",
                "sessions": len(sessions), "drifts": []}

    recent_set = set(sessions[-recent_n:])
    baseline_set = set(sessions[-(recent_n + baseline_n):-recent_n]) or \
        set(sessions[:-recent_n])
    recent = _window_metrics([r for r in rows if r["session_date"] in recent_set])
    baseline = _window_metrics([r for r in rows if r["session_date"] in baseline_set])

    drifts: list[dict] = []
    if (recent["n_settled"] < cfg["drift_min_n"]
            or baseline["n_settled"] < cfg["drift_min_n"]):
        return {"recent": recent, "baseline": baseline, "drifts": drifts,
                "note": "window too small; drift not judged"}

    def rel(cur, old):
        if cur is None or old is None or abs(old) < 1e-12:
            return None
        return (cur - old) / abs(old)

    model_metrics = [("win_rate", recent["win_rate"], baseline["win_rate"]),
                     ("gate_edge", recent["gate_edge"], baseline["gate_edge"]),
                     ("brier", recent["brier"], baseline["brier"])]
    for name, cur, old in model_metrics:
        r = rel(cur, old)
        if r is None:
            continue
        degraded = r < -cfg["drift_rel"] if name != "brier" else r > cfg["drift_rel"]
        if degraded:
            drifts.append({"kind": "MODEL_DRIFT", "metric": name,
                           "rel_change": round(r, 3),
                           "detail": f"{name} moved {r:+.0%} vs trailing window "
                                     f"({old} -> {cur})"})

    r = rel(recent["trades_per_session"], baseline["trades_per_session"])
    if r is not None and abs(r) > cfg["drift_rel"]:
        drifts.append({"kind": "FEATURE_DRIFT", "metric": "trades_per_session",
                       "rel_change": round(r, 3),
                       "detail": f"trade frequency moved {r:+.0%} vs trailing "
                                 f"window — the funnel's inputs shifted"})

    # Regime mix: total-variation distance between the two distributions.
    all_regimes = set(recent["regime_mix"]) | set(baseline["regime_mix"])
    if all_regimes and recent["n_taken"] and baseline["n_taken"]:
        tv = 0.5 * sum(
            abs(recent["regime_mix"].get(k, 0) / recent["n_taken"]
                - baseline["regime_mix"].get(k, 0) / baseline["n_taken"])
            for k in all_regimes)
        if tv > cfg["drift_rel"]:
            drifts.append({"kind": "REGIME_DRIFT", "metric": "regime_mix",
                           "rel_change": round(tv, 3),
                           "detail": f"regime distribution moved (TV distance "
                                     f"{tv:.2f}) vs trailing window"})

    return {"recent": recent, "baseline": baseline, "drifts": drifts}


def drift_diagnoses(drift: dict) -> list[Diagnosis]:
    out = []
    for d in drift.get("drifts", []):
        out.append(Diagnosis(
            issue=d["kind"],
            severity="warn" if abs(d.get("rel_change") or 0) < 0.5 else "alert",
            confidence=_confidence(
                (drift.get("recent") or {}).get("n_settled", 0), 30,
                d.get("rel_change") or 0, 0.5),
            affected_module="market regime / model inputs",
            likely_cause=d["detail"],
            recommendation="re-validate the champion on recent recordings "
                           "before trusting historical scores",
            evidence={"drift": d,
                      "recent": drift.get("recent"),
                      "baseline": drift.get("baseline")},
        ))
    return out


def log_drift_report(jrn: Journal, drift: dict,
                     report_date: Optional[str] = None) -> int:
    """Persist one drift snapshot as validation_reports(report_type='drift')."""
    report_date = report_date or dt.date.today().isoformat()
    drifts = drift.get("drifts", [])
    summary = (f"Drift check — {len(drifts)} drift signal(s): "
               + ", ".join(d["kind"] + ":" + d["metric"] for d in drifts)
               if drifts else "Drift check — no drift beyond thresholds")
    flags = [{"flag": d["kind"], "severity": "warn", "detail": d["detail"]}
             for d in drifts]
    return jrn.log_validation_report(report_date, "drift", drift, summary, flags)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
def diagnose(jrn: Journal,
             prior_reports: Optional[list[dict]] = None,
             thresholds: Optional[dict] = None,
             include_drift: bool = True) -> list[Diagnosis]:
    """Run every check; returns diagnoses ordered alert -> warn -> info,
    highest confidence first within a severity."""
    cfg = {**THRESHOLDS, **(thresholds or {})}
    if prior_reports is None:
        prior_reports = jrn.fetch_validation_reports(limit=10)

    out: list[Diagnosis] = []
    for check in (_check_gate_inversion, _check_brier, _check_ev_bias,
                  _check_directional, _check_regime_concentration):
        d = check(jrn, cfg)
        if d:
            out.append(d)
    for check in (_check_sharpe_collapse, _check_trade_collapse):
        d = check(prior_reports, cfg)
        if d:
            out.append(d)
    if include_drift:
        out.extend(drift_diagnoses(compute_drift(jrn, thresholds=thresholds)))

    sev_rank = {"alert": 0, "warn": 1, "info": 2}
    out.sort(key=lambda d: (sev_rank.get(d.severity, 3), -d.confidence))
    return out


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    from journal import COLUMNS

    # Seed an in-memory journal with an INVERTED gate: taken trades lose,
    # gate-blocked candidates would have won.
    jrn = Journal(":memory:")
    session = "2026-07-08"
    for i in range(40):
        row = {c: None for c in COLUMNS}
        traded = i % 2 == 0
        row.update({
            "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0, "gex_regime": "long",
            "was_traded": 1 if traded else 0, "candidate_present": 1,
            "gate_pass": 1 if traded else 0,
            "decision": "TRADE" if traded else "NO_TRADE",
            # taken rows: short call spread that finishes ITM (loser);
            # blocked rows: winner
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

    print("=" * 70)
    print("  diagnostics demo — inverted gate journal")
    print("=" * 70)
    for d in diagnose(jrn, prior_reports=[]):
        print(f"  [{d.severity.upper():5}] {d.issue}  conf={d.confidence:.0%}")
        print(f"          cause: {d.likely_cause[:100]}")
        print(f"          fix:   {d.recommendation[:100]}")
    jrn.close()
    print("=" * 70)
