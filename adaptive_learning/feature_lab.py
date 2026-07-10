"""
adaptive_learning/feature_lab.py
================================
Feature scoring beyond raw Pearson. journal.component_correlations() stays the
cheap first look; this module adds Spearman (rank), histogram mutual
information (nonlinear dependence), and shuffle-based permutation importance
(does the association survive breaking the pairing?) — all hand-rolled on
numpy/scipy, no sklearn/SHAP dependency.

Feature lifecycle (recommendations only — transitions are written to
feature_scores and reports, never auto-applied to the live engine):

    observation -> experimental -> candidate -> production

A feature earns "candidate" only after the stability engine
(adaptive_learning.stability.feature_stability) confirms sign consistency
across folds, adequate effect size, non-collinearity, and multi-regime
support. "production" is a manual step: it means someone wired the feature
into a matrix weight or veto via the feature-impact workflow.

NOT financial advice.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Optional

from journal import Journal

LIFECYCLE = ("observation", "experimental", "candidate", "production")

# score-component columns treated as features alongside sig:* keys
COMPONENT_FEATURES = ("candidate_score", "ev", "ev_per_risk", "wall_safety",
                      "gamma_safety", "touch_safety", "gate_score",
                      "prob_profit")

# promotion floors (recommendations only)
THRESHOLDS = {
    "min_n": 30,
    "min_abs_pearson": 0.05,
    "min_perm_importance": 0.0,      # must survive shuffling at all
}


# --------------------------------------------------------------------------- #
# Data extraction                                                               #
# --------------------------------------------------------------------------- #
def feature_matrix(jrn: Journal) -> tuple[list[float], dict[str, list]]:
    """Settled rows with realized P&L -> (y, {feature: values}). Values are
    aligned with y; None marks a missing observation for that row."""
    rows = [r for r in jrn.fetch(settled_only=True)
            if r["realized_pnl"] is not None]
    y = [float(r["realized_pnl"]) for r in rows]

    features: dict[str, list] = {c: [] for c in COMPONENT_FEATURES}
    sig_dicts = []
    sig_keys: set[str] = set()
    for r in rows:
        for c in COMPONENT_FEATURES:
            v = r.get(c)
            features[c].append(float(v) if isinstance(v, (int, float)) else None)
        try:
            sig = json.loads(r["signals_json"]) if r.get("signals_json") else {}
        except (json.JSONDecodeError, TypeError):
            sig = {}
        sig = sig if isinstance(sig, dict) else {}
        sig_dicts.append(sig)
        sig_keys.update(k for k, v in sig.items() if isinstance(v, (int, float)))

    for k in sorted(sig_keys):
        features[f"sig:{k}"] = [
            float(s[k]) if isinstance(s.get(k), (int, float)) else None
            for s in sig_dicts]

    # drop features with no observations at all
    features = {k: v for k, v in features.items()
                if any(x is not None for x in v)}
    return y, features


def _paired(xs: list, ys: list) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #
def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def mutual_information(xs: list[float], ys: list[float],
                       n_bins: int = 4) -> Optional[float]:
    """Histogram MI in nats between x (quantile-binned) and the win/loss
    outcome of y. Deliberately coarse: with journal-sized samples fine bins
    only measure noise."""
    n = len(xs)
    if n < 3:
        return None
    # quantile bin edges for x
    sorted_x = sorted(xs)
    edges = [sorted_x[min(n - 1, int(n * q / n_bins))] for q in range(1, n_bins)]

    def xbin(v: float) -> int:
        for i, e in enumerate(edges):
            if v <= e:
                return i
        return len(edges)

    joint: dict[tuple[int, int], int] = {}
    px: dict[int, int] = {}
    py: dict[int, int] = {}
    for x, y in zip(xs, ys):
        bx, by = xbin(x), (1 if y > 0 else 0)
        joint[(bx, by)] = joint.get((bx, by), 0) + 1
        px[bx] = px.get(bx, 0) + 1
        py[by] = py.get(by, 0) + 1

    mi = 0.0
    for (bx, by), c in joint.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((px[bx] / n) * (py[by] / n)))
    return max(0.0, mi)


def permutation_importance(xs: list[float], ys: list[float],
                           n_shuffles: int = 20,
                           seed: int = 0) -> Optional[float]:
    """|Pearson| minus the mean |Pearson| after shuffling x. A feature whose
    association vanishes under shuffling scores near its raw |r|; one whose
    'association' was an artifact of ordering scores near zero or negative."""
    import random as _random

    base = pearson(xs, ys)
    if base is None:
        return None
    rng = _random.Random(seed)
    shuffled_abs = []
    xs_copy = list(xs)
    for _ in range(n_shuffles):
        rng.shuffle(xs_copy)
        r = pearson(xs_copy, ys)
        shuffled_abs.append(abs(r) if r is not None else 0.0)
    return abs(base) - (sum(shuffled_abs) / len(shuffled_abs))


# --------------------------------------------------------------------------- #
# Scoring + lifecycle                                                           #
# --------------------------------------------------------------------------- #
def score_features(jrn: Journal, n_shuffles: int = 20,
                   seed: int = 0) -> dict[str, dict]:
    """All four metrics for every component + sig:* feature with data."""
    y, features = feature_matrix(jrn)
    out: dict[str, dict] = {}
    for name, values in features.items():
        xs, ys = _paired(values, y)
        if len(xs) < 3:
            continue
        out[name] = {
            "n": len(xs),
            "pearson": _round(pearson(xs, ys)),
            "spearman": _round(spearman(xs, ys)),
            "mutual_info": _round(mutual_information(xs, ys)),
            "perm_importance": _round(
                permutation_importance(xs, ys, n_shuffles=n_shuffles, seed=seed)),
        }
    return out


def _round(v: Optional[float], nd: int = 4) -> Optional[float]:
    return round(v, nd) if v is not None else None


def recommend_status(score: dict, stability: Optional[dict] = None,
                     current: str = "observation",
                     thresholds: Optional[dict] = None) -> str:
    """Lifecycle recommendation. Monotone: never suggests skipping a step,
    never suggests 'production' (that is a human decision via the
    feature-impact workflow)."""
    cfg = {**THRESHOLDS, **(thresholds or {})}
    n = score.get("n") or 0
    p = score.get("pearson")
    pi = score.get("perm_importance")

    earns_experimental = (
        n >= cfg["min_n"]
        and p is not None and abs(p) >= cfg["min_abs_pearson"]
        and pi is not None and pi > cfg["min_perm_importance"])
    earns_candidate = earns_experimental and bool(
        stability and stability.get("passes"))

    if current == "production":
        return "production"
    if earns_candidate and current in ("experimental", "candidate"):
        return "candidate"
    if earns_experimental:
        return "experimental" if current == "observation" else current
    return "observation" if current == "observation" else current


def run_feature_lab(jrn: Journal,
                    stability: Optional[dict[str, dict]] = None,
                    as_of: Optional[str] = None,
                    log: bool = True,
                    seed: int = 0) -> dict:
    """Score every feature, attach stability (when provided), recommend a
    lifecycle status relative to the last logged status, and persist one
    feature_scores row per feature."""
    as_of = as_of or dt.date.today().isoformat()
    scores = score_features(jrn, seed=seed)
    prior = {r["feature"]: r for r in jrn.fetch_feature_scores(latest_only=True)}

    report: dict[str, dict] = {}
    for name, sc in scores.items():
        stab = (stability or {}).get(name)
        current = (prior.get(name) or {}).get("status", "observation")
        status = recommend_status(sc, stab, current=current)
        entry = {**sc, "stability": stab, "status": status,
                 "prior_status": current}
        report[name] = entry
        if log:
            jrn.log_feature_score(
                name, as_of, sc["n"],
                pearson=sc["pearson"], spearman=sc["spearman"],
                mutual_info=sc["mutual_info"],
                perm_importance=sc["perm_importance"],
                stability=(stab or {}).get("stability_score"),
                status=status,
                details={"stability": stab} if stab else None)
    return report


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random
    from journal import COLUMNS

    rng = random.Random(3)
    jrn = Journal(":memory:")
    session = "2026-07-08"
    for i in range(60):
        pnl_driver = rng.uniform(-1, 1)
        row = {c: None for c in COLUMNS}
        row.update({
            "session_date": session, "ts": f"{session}T10:{i:02d}:00-04:00",
            "spot": 600.0, "gex_regime": "long",
            "was_traded": 1, "candidate_present": 1, "gate_pass": 1,
            "decision": "TRADE",
            "credit": pnl_driver,                # realized pnl == this driver
            "ev": 0.1, "prob_profit": 0.5 + 0.3 * pnl_driver,
            "candidate_score": 50 + 30 * pnl_driver,
            "legs_json": json.dumps([{"qty": -1, "strike": 610.0, "kind": "C"},
                                     {"qty": 1, "strike": 612.0, "kind": "C"}]),
            "signals_json": json.dumps({
                "predictive": pnl_driver + rng.gauss(0, 0.3),
                "noise": rng.gauss(0, 1.0),
            }),
            "regime_direction": "call",
        })
        jrn.log(row)
    jrn.settle_session(session, 600.0)

    print("=" * 76)
    print("  feature_lab demo — one predictive signal, one noise signal")
    print("=" * 76)
    rep = run_feature_lab(jrn, log=False)
    hdr = f"  {'feature':<22} {'n':>4} {'pearson':>8} {'spearman':>9} {'MI':>7} {'perm':>7}  status"
    print(hdr)
    print("-" * 76)
    for name in sorted(rep, key=lambda k: -abs(rep[k].get("pearson") or 0)):
        s = rep[name]
        print(f"  {name:<22} {s['n']:>4} {s['pearson'] if s['pearson'] is not None else float('nan'):>8.3f} "
              f"{s['spearman'] if s['spearman'] is not None else float('nan'):>9.3f} "
              f"{s['mutual_info'] if s['mutual_info'] is not None else float('nan'):>7.3f} "
              f"{s['perm_importance'] if s['perm_importance'] is not None else float('nan'):>7.3f}  {s['status']}")
    jrn.close()
    print("=" * 76)
