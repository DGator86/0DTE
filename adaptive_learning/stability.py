"""
adaptive_learning/stability.py
==============================
Reject parameters and features that only work by luck.

Two engines:

parameter_stability(trials)
    For every searched parameter, measure how consistently its effect on the
    walk-forward score holds across folds: overall effect direction, per-fold
    sign consistency, sensitivity (score span across values), and variance.
    A parameter whose "best" value flips sign fold-to-fold is curve-fit noise
    and must not drive a promotion.

feature_stability(jrn, wf_result)
    The gatekeeper the spec demands before a feature earns weight:
      * same correlation sign in >= 70% of folds where it is measurable
      * absolute overall effect above a floor
      * not highly collinear with a STRONGER existing feature
      * measurable effect in >= 2 regimes
    Anything failing stays observation-only.

NOT financial advice.
"""
from __future__ import annotations

from typing import Optional

from journal import Journal
from adaptive_learning.feature_lab import feature_matrix, pearson, _paired

THRESHOLDS = {
    "min_trials": 4,
    "sign_consistency": 0.70,
    "min_abs_effect": 0.05,       # |Pearson r| floor for features
    "collinear_r": 0.80,
    "min_regimes": 2,
    "min_folds": 2,
}


# --------------------------------------------------------------------------- #
# Parameter stability (over optimizer trials)                                  #
# --------------------------------------------------------------------------- #
def parameter_stability(trials: list, thresholds: Optional[dict] = None) -> dict:
    """
    trials — optimizer.Trial-like objects (.params, .score, .wf_result with
    .folds[i].tearsheet.total_pnl). Returns per-parameter:

        {param: {n_trials, n_values, overall_r, per_value_mean_score,
                 sensitivity, fold_sign_consistency, fold_r_std, verdict}}

    verdict: "stable" | "unstable" | "insufficient".
    Non-numeric parameter values get a per-value-means-only analysis.
    """
    cfg = {**THRESHOLDS, **(thresholds or {})}
    scored = [t for t in trials if t.score > float("-inf")]
    if not scored:
        return {}

    params = sorted({k for t in scored for k in t.params})
    out: dict[str, dict] = {}
    for key in params:
        values = [t.params.get(key) for t in scored]
        scores = [t.score for t in scored]
        distinct = sorted(set(values), key=lambda v: (str(type(v)), str(v)))

        per_value = {}
        for v in distinct:
            vs = [s for val, s in zip(values, scores) if val == v]
            per_value[str(v)] = round(sum(vs) / len(vs), 4)
        sensitivity = (round(max(per_value.values()) - min(per_value.values()), 4)
                       if len(per_value) > 1 else 0.0)

        entry: dict = {
            "n_trials": len(scored),
            "n_values": len(distinct),
            "per_value_mean_score": per_value,
            "sensitivity": sensitivity,
        }

        numeric = all(isinstance(v, (int, float)) for v in values)
        overall_r = pearson([float(v) for v in values], scores) if numeric else None
        entry["overall_r"] = round(overall_r, 4) if overall_r is not None else None

        # Per-fold effect: correlation between the parameter value and each
        # fold's total P&L across trials. The same fold under different
        # parameter values is the closest thing to a controlled experiment
        # the walk-forward gives us.
        fold_rs: list[float] = []
        if numeric:
            n_folds = min((len(t.wf_result.folds) for t in scored
                           if t.wf_result is not None), default=0)
            for f in range(n_folds):
                xs = [float(t.params[key]) for t in scored]
                ys = [t.wf_result.folds[f].tearsheet.total_pnl for t in scored]
                r = pearson(xs, ys)
                if r is not None:
                    fold_rs.append(r)

        if fold_rs and overall_r is not None and abs(overall_r) > 1e-9:
            same = sum(1 for r in fold_rs if r * overall_r > 0)
            consistency = same / len(fold_rs)
            mu = sum(fold_rs) / len(fold_rs)
            var = sum((r - mu) ** 2 for r in fold_rs) / len(fold_rs)
            entry["fold_sign_consistency"] = round(consistency, 3)
            entry["fold_r_std"] = round(var ** 0.5, 4)
        else:
            entry["fold_sign_consistency"] = None
            entry["fold_r_std"] = None

        if (len(scored) < cfg["min_trials"] or len(distinct) < 2
                or entry["fold_sign_consistency"] is None):
            entry["verdict"] = "insufficient"
        elif entry["fold_sign_consistency"] >= cfg["sign_consistency"]:
            entry["verdict"] = "stable"
        else:
            entry["verdict"] = "unstable"
        out[key] = entry
    return out


def stability_acceptable(stability: dict, changed_params: list[str],
                         thresholds: Optional[dict] = None) -> tuple[bool, str]:
    """A challenger is acceptable when none of its CHANGED parameters carries
    an 'unstable' verdict. 'insufficient' does not block (small searches are
    normal early on) but is reported."""
    unstable = [p for p in changed_params
                if (stability.get(p) or {}).get("verdict") == "unstable"]
    if unstable:
        return False, f"unstable parameters: {', '.join(unstable)}"
    insufficient = [p for p in changed_params
                    if (stability.get(p) or {}).get("verdict") == "insufficient"]
    if insufficient:
        return True, f"insufficient stability evidence for: {', '.join(insufficient)}"
    return True, "all changed parameters stable"


# --------------------------------------------------------------------------- #
# Feature stability (over walk-forward folds + the journal)                    #
# --------------------------------------------------------------------------- #
def feature_stability(jrn: Journal, wf_result,
                      thresholds: Optional[dict] = None) -> dict[str, dict]:
    """
    Per-feature stability report combining the overall journal correlation,
    per-fold correlation signs from the walk-forward tearsheets, pairwise
    collinearity, and regime coverage. `passes` means the feature clears every
    bar in the module docstring; everything else stays observation-only.
    """
    cfg = {**THRESHOLDS, **(thresholds or {})}

    # overall correlations + raw columns from the journal
    y, features = feature_matrix(jrn)
    overall: dict[str, Optional[float]] = {}
    for name, values in features.items():
        xs, ys = _paired(values, y)
        overall[name] = pearson(xs, ys) if len(xs) >= 3 else None

    # per-fold correlations + regime coverage from the walk-forward
    folds = list(getattr(wf_result, "folds", []) or [])
    fold_corrs: list[dict] = []
    fold_regimes: list[set] = []
    for f in folds:
        cc = f.tearsheet.component_correlations or {}
        fold_corrs.append({k: v for k, v in cc.items()
                           if k not in ("n", "note") and isinstance(v, (int, float))})
        fold_regimes.append(set((f.tearsheet.regime_counts or {}).keys()))

    # collinearity: pairwise correlation between feature columns
    names = [n for n in features if overall.get(n) is not None]

    def _pairwise(a: str, b: str) -> Optional[float]:
        pairs = [(x1, x2) for x1, x2 in zip(features[a], features[b])
                 if x1 is not None and x2 is not None]
        if len(pairs) < 3:
            return None
        return pearson([p[0] for p in pairs], [p[1] for p in pairs])

    out: dict[str, dict] = {}
    for name in names:
        r_all = overall[name]
        reasons: list[str] = []

        per_fold = [fc.get(name) for fc in fold_corrs]
        measurable = [(i, r) for i, r in enumerate(per_fold) if r is not None]
        if len(measurable) < cfg["min_folds"]:
            sign_frac = None
            reasons.append(f"measurable in only {len(measurable)} fold(s)")
        else:
            same = sum(1 for _, r in measurable if r * r_all > 0)
            sign_frac = same / len(measurable)
            if sign_frac < cfg["sign_consistency"]:
                reasons.append(f"sign consistent in only {sign_frac:.0%} of folds "
                               f"(need >= {cfg['sign_consistency']:.0%})")

        if abs(r_all) < cfg["min_abs_effect"]:
            reasons.append(f"|r|={abs(r_all):.3f} below effect floor "
                           f"{cfg['min_abs_effect']}")

        collinear_with = None
        for other in names:
            if other == name:
                continue
            if abs(overall[other] or 0) <= abs(r_all):
                continue                      # only a STRONGER feature blocks
            rp = _pairwise(name, other)
            if rp is not None and abs(rp) > cfg["collinear_r"]:
                collinear_with = {"feature": other, "r": round(rp, 3),
                                  "other_effect": round(overall[other], 3)}
                reasons.append(f"collinear with stronger feature {other} "
                               f"(|r|={abs(rp):.2f})")
                break

        # regime breadth: union of regimes over folds where the feature had a
        # measurable effect at or above the floor
        regimes: set = set()
        for i, r in measurable:
            if abs(r) >= cfg["min_abs_effect"]:
                regimes |= fold_regimes[i]
        if len(regimes) < cfg["min_regimes"]:
            reasons.append(f"effect seen in {len(regimes)} regime(s) "
                           f"(need >= {cfg['min_regimes']})")

        passes = not reasons
        # single 0..1 stability score for the feature_scores table: sign
        # consistency scaled by effect size, zeroed by collinearity
        score = 0.0
        if sign_frac is not None and collinear_with is None:
            score = sign_frac * min(1.0, abs(r_all) / 0.3)

        out[name] = {
            "overall_r": round(r_all, 4),
            "n_folds_measurable": len(measurable),
            "sign_consistency": round(sign_frac, 3) if sign_frac is not None else None,
            "regimes_covered": sorted(regimes),
            "collinear_with": collinear_with,
            "stability_score": round(score, 3),
            "passes": passes,
            "reasons": reasons,
        }
    return out


# --------------------------------------------------------------------------- #
# Demo                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class _TS:
        total_pnl: float

    @dataclass
    class _Fold:
        tearsheet: object

    @dataclass
    class _WF:
        folds: list = field(default_factory=list)

    @dataclass
    class _Trial:
        params: dict
        score: float
        wf_result: object = None

    # Stable parameter: higher max_adx -> higher P&L in every fold.
    # Unstable parameter: min_ev helps in fold 1 and hurts in fold 2.
    trials = []
    for adx, mev in [(16, -0.02), (16, 0.02), (24, -0.02),
                     (24, 0.02), (20, 0.0), (26, 0.01)]:
        f1 = _Fold(_TS(total_pnl=adx * 0.10 + mev * 20))
        f2 = _Fold(_TS(total_pnl=adx * 0.10 - mev * 20))
        trials.append(_Trial(params={"gate.max_adx": adx, "selector.min_ev": mev},
                             score=adx * 0.10 + mev * 0.5, wf_result=_WF([f1, f2])))

    print("=" * 70)
    print("  stability demo — parameter fold consistency")
    print("=" * 70)
    stab = parameter_stability(trials)
    for k, v in stab.items():
        print(f"  {k:<24} verdict={v['verdict']:<12} "
              f"sign_consistency={v['fold_sign_consistency']} "
              f"sensitivity={v['sensitivity']}")
    ok, why = stability_acceptable(stab, list(stab))
    print(f"  acceptable={ok}  ({why})")
    print("=" * 70)
