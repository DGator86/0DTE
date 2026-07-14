"""
prediction/uncertainty.py
=========================
Observation-specific uncertainty components (V3 Part 1 §7).

Scores use 0.0 = low uncertainty, 1.0 = maximum uncertainty.
Missing components are NEVER treated as zero — weights are renormalized
across available components and reasons are recorded.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

DEFAULT_WEIGHTS = {
    "ensemble": 0.25,
    "conformal": 0.20,
    "out_of_distribution": 0.25,
    "calibration": 0.15,
    "data_quality": 0.10,
    "model_age": 0.05,
}

ABSTAIN_SHADOW_THRESHOLD = 0.85


@dataclass(frozen=True)
class UncertaintyComponents:
    ensemble: float | None
    conformal: float | None
    out_of_distribution: float | None
    calibration: float | None
    data_quality: float | None
    model_age: float | None
    composite: float
    reasons: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ensemble": self.ensemble,
            "conformal": self.conformal,
            "out_of_distribution": self.out_of_distribution,
            "calibration": self.calibration,
            "data_quality": self.data_quality,
            "model_age": self.model_age,
            "composite": self.composite,
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
        }


def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(np.clip(float(x), 0.0, 1.0))


def clipped_weighted_mean(
    components: dict[str, Optional[float]],
    weights: Optional[dict[str, float]] = None,
) -> tuple[float, tuple[str, ...], dict]:
    """
    Weighted mean over available (non-None) components. Missing components
    are excluded and a reason is appended (never treated as zero).
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    reasons: list[str] = []
    available = {}
    for name, weight in w.items():
        val = components.get(name)
        if val is None:
            reasons.append(f"missing_{name}_component")
            continue
        available[name] = (_clip01(val), float(weight))

    if not available:
        reasons.append("no_uncertainty_components_available")
        return 1.0, tuple(reasons), {"available": {}, "renormalized_weights": {}}

    total_w = sum(wt for _, wt in available.values())
    if total_w <= 0:
        return 1.0, tuple(reasons + ["non_positive_weights"]), {}

    renorm = {k: wt / total_w for k, (_, wt) in available.items()}
    composite = sum(val * renorm[k] for k, (val, _) in available.items())
    composite = float(np.clip(composite, 0.0, 1.0))
    return composite, tuple(reasons), {
        "available": {k: v for k, (v, _) in available.items()},
        "renormalized_weights": renorm,
    }


def ensemble_uncertainty_classification(
    probabilities: Sequence[Sequence[float]],
) -> float:
    """Normalized std of ensemble predicted probabilities → [0, 1]."""
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    # std across estimators, average across observations if matrix is
    # (n_estimators,). Accept 1-d as single-obs ensemble.
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim == 1:
        std = float(np.std(arr))
    else:
        std = float(np.mean(np.std(arr, axis=0)))
    # Bernoulli max std is 0.5
    return float(np.clip(std / 0.5, 0.0, 1.0))


def ensemble_uncertainty_regression(
    values: Sequence[Sequence[float]],
) -> float:
    """Normalized IQR of ensemble predicted values → [0, 1]."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    if arr.ndim == 2 and arr.shape[0] >= 2:
        q75 = np.percentile(arr, 75, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        iqr = float(np.mean(q75 - q25))
        scale = float(np.mean(np.abs(np.median(arr, axis=0))) + 1e-6)
    else:
        flat = arr.ravel()
        iqr = float(np.percentile(flat, 75) - np.percentile(flat, 25))
        scale = float(abs(np.median(flat)) + 1e-6)
    return float(np.clip(iqr / (2.0 * scale + iqr), 0.0, 1.0))


def data_quality_uncertainty(
    *,
    feature_coverage: Optional[float] = None,
    required_field_coverage: Optional[float] = None,
    max_source_age_sec: Optional[float] = None,
    chain_strike_coverage: Optional[float] = None,
    feed_failover: bool = False,
    rnd_quality: Optional[float] = None,
    arbitrage_violations: int = 0,
    missingness_count: int = 0,
    quote_consistency: Optional[float] = None,
    max_age_ref_sec: float = 30.0,
) -> tuple[float, tuple[str, ...]]:
    """
    Deterministic, inspectable data-quality → uncertainty (not learned).
    Higher missingness / age / violations ⇒ higher uncertainty.
    """
    reasons: list[str] = []
    parts: list[float] = []

    def _inv_cov(name: str, v: Optional[float]):
        if v is None:
            return
        u = float(np.clip(1.0 - float(v), 0.0, 1.0))
        parts.append(u)
        if u > 0.25:
            reasons.append(f"low_{name}")

    _inv_cov("feature_coverage", feature_coverage)
    _inv_cov("required_field_coverage", required_field_coverage)
    _inv_cov("chain_strike_coverage", chain_strike_coverage)
    _inv_cov("rnd_quality", rnd_quality)
    _inv_cov("quote_consistency", quote_consistency)

    if max_source_age_sec is not None:
        age_u = float(np.clip(float(max_source_age_sec) / max_age_ref_sec, 0.0, 1.0))
        parts.append(age_u)
        if age_u > 0.5:
            reasons.append("stale_source")

    if feed_failover:
        parts.append(0.6)
        reasons.append("feed_failover")

    if arbitrage_violations > 0:
        parts.append(float(np.clip(0.2 * arbitrage_violations, 0.0, 1.0)))
        reasons.append(f"arbitrage_violations={arbitrage_violations}")

    if missingness_count > 0:
        parts.append(float(np.clip(missingness_count / 20.0, 0.0, 1.0)))
        reasons.append(f"missingness_count={missingness_count}")

    if not parts:
        return 0.0, tuple(reasons)
    return float(np.clip(float(np.mean(parts)), 0.0, 1.0)), tuple(reasons)


def calibration_uncertainty(
    *,
    brier_degradation: Optional[float] = None,
    log_loss_degradation: Optional[float] = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    bin_support: Optional[float] = None,
) -> tuple[float, tuple[str, ...]]:
    """
    From rolling OOS calibration diagnostics. Poorly supported probability
    bins and slope/intercept drift increase uncertainty.
    """
    reasons: list[str] = []
    parts: list[float] = []

    if brier_degradation is not None:
        u = float(np.clip(float(brier_degradation), 0.0, 1.0))
        parts.append(u)
        if u > 0.1:
            reasons.append("brier_degradation")
    if log_loss_degradation is not None:
        u = float(np.clip(float(log_loss_degradation) / 0.5, 0.0, 1.0))
        parts.append(u)
        if u > 0.2:
            reasons.append("log_loss_degradation")
    if slope is not None:
        u = float(np.clip(abs(float(slope) - 1.0), 0.0, 1.0))
        parts.append(u)
        if u > 0.25:
            reasons.append("calibration_slope_deviation")
    if intercept is not None:
        u = float(np.clip(abs(float(intercept)), 0.0, 1.0))
        parts.append(u)
        if u > 0.15:
            reasons.append("calibration_intercept_deviation")
    if bin_support is not None:
        u = float(np.clip(1.0 - float(bin_support), 0.0, 1.0))
        parts.append(u)
        if u > 0.5:
            reasons.append("low_probability_bin_support")

    if not parts:
        return None, ("missing_calibration_diagnostics",)
    return float(np.clip(float(np.mean(parts)), 0.0, 1.0)), tuple(reasons)


def model_age_uncertainty(
    *,
    artifact_age_days: Optional[float] = None,
    days_since_last_eval: Optional[float] = None,
    feature_shift_score: Optional[float] = None,
    missing_recent_sessions: int = 0,
    max_age_days: float = 30.0,
    max_eval_gap_days: float = 5.0,
) -> tuple[float, tuple[str, ...]]:
    """Increase uncertainty for stale / unevaluated / shifted models."""
    reasons: list[str] = []
    parts: list[float] = []
    if artifact_age_days is not None:
        u = float(np.clip(float(artifact_age_days) / max_age_days, 0.0, 1.0))
        parts.append(u)
        if u > 0.7:
            reasons.append("artifact_age")
    if days_since_last_eval is not None:
        u = float(np.clip(
            float(days_since_last_eval) / max_eval_gap_days, 0.0, 1.0))
        parts.append(u)
        if u > 0.7:
            reasons.append("stale_evaluation")
    if feature_shift_score is not None:
        u = float(np.clip(float(feature_shift_score), 0.0, 1.0))
        parts.append(u)
        if u > 0.3:
            reasons.append("feature_distribution_shift")
    if missing_recent_sessions > 0:
        u = float(np.clip(missing_recent_sessions / 5.0, 0.0, 1.0))
        parts.append(u)
        reasons.append(f"missing_recent_sessions={missing_recent_sessions}")
    if not parts:
        return 0.0, tuple(reasons)
    return float(np.clip(float(np.mean(parts)), 0.0, 1.0)), tuple(reasons)


def compose_uncertainty(
    *,
    ensemble: Optional[float] = None,
    conformal: Optional[float] = None,
    out_of_distribution: Optional[float] = None,
    calibration: Optional[float] = None,
    data_quality: Optional[float] = None,
    model_age: Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
    extra_reasons: Sequence[str] = (),
    diagnostics: Optional[dict] = None,
) -> UncertaintyComponents:
    components = {
        "ensemble": _clip01(ensemble),
        "conformal": _clip01(conformal),
        "out_of_distribution": _clip01(out_of_distribution),
        "calibration": _clip01(calibration),
        "data_quality": _clip01(data_quality),
        "model_age": _clip01(model_age),
    }
    composite, reasons, diag = clipped_weighted_mean(components, weights)
    all_reasons = tuple(reasons) + tuple(extra_reasons)
    if composite >= ABSTAIN_SHADOW_THRESHOLD:
        all_reasons = all_reasons + ("ABSTAIN_SHADOW",)
    merged = dict(diagnostics or {})
    merged.update(diag)
    return UncertaintyComponents(
        ensemble=components["ensemble"],
        conformal=components["conformal"],
        out_of_distribution=components["out_of_distribution"],
        calibration=components["calibration"],
        data_quality=components["data_quality"],
        model_age=components["model_age"],
        composite=composite,
        reasons=all_reasons,
        diagnostics=merged,
    )


@dataclass
class SessionBootstrapEnsemble:
    """
    Small session-bootstrap ensemble (5–9 estimators). Each member trains on
    a bootstrap sample of complete sessions.
    """
    n_estimators: int = 7
    seed: int = 42
    estimators: list = field(default_factory=list)
    vectorizers: list = field(default_factory=list)
    fitted: bool = False

    def fit(
        self,
        rows: Sequence[dict],
        y: Sequence,
        sessions: Sequence[str],
        estimator_factory,
        *,
        task: str = "classification",
    ) -> "SessionBootstrapEnsemble":
        from prediction.models.base import FeatureVectorizer
        rows = list(rows)
        y_arr = np.asarray(y)
        sessions = list(sessions)
        uniq = sorted(set(sessions))
        rng = np.random.default_rng(self.seed)
        self.estimators = []
        self.vectorizers = []
        n_est = max(5, min(9, self.n_estimators))
        for _ in range(n_est):
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            keep = set(sampled.tolist())
            idx = [i for i, s in enumerate(sessions) if s in keep]
            if not idx:
                continue
            vec = FeatureVectorizer().fit([rows[i] for i in idx])
            est = estimator_factory()
            y_tr = y_arr[idx]
            if task == "classification" and len(np.unique(y_tr)) < 2:
                continue
            est.fit(vec.transform([rows[i] for i in idx]), y_tr)
            self.estimators.append(est)
            self.vectorizers.append(vec)
        self.fitted = True
        return self

    def predict_proba_members(self, rows: Sequence[dict]) -> np.ndarray:
        """Shape (n_estimators, n_rows) of P(class=1)."""
        if not self.fitted or not self.estimators:
            return np.zeros((0, len(rows)))
        out = []
        for est, vec in zip(self.estimators, self.vectorizers):
            X = vec.transform(list(rows))
            out.append(est.predict_proba(X)[:, 1])
        return np.asarray(out, dtype=float)

    def predict_members(self, rows: Sequence[dict]) -> np.ndarray:
        if not self.fitted or not self.estimators:
            return np.zeros((0, len(rows)))
        out = []
        for est, vec in zip(self.estimators, self.vectorizers):
            out.append(est.predict(vec.transform(list(rows))))
        return np.asarray(out, dtype=float)

    def mean_proba(self, rows: Sequence[dict]) -> np.ndarray:
        members = self.predict_proba_members(rows)
        if members.size == 0:
            return np.full(len(rows), 0.5)
        return np.mean(members, axis=0)

    def uncertainty_classification(self, rows: Sequence[dict]) -> np.ndarray:
        members = self.predict_proba_members(rows)
        if members.shape[0] < 2:
            return np.zeros(len(rows))
        std = np.std(members, axis=0)
        return np.clip(std / 0.5, 0.0, 1.0)
