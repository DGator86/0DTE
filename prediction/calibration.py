"""
prediction/calibration.py
=========================
Probability calibration for the V2/V3 model suite
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.7,
 docs/PREDICTION_ENGINE_V3_PART1_VALIDATION.md §5).

Default is sigmoid/Platt scaling. Isotonic regression is gated behind sample
AND session minimums and is compared to sigmoid on a nested holdout of the
calibration predictions — never by fitting both on the exact same labels and
scoring in-sample.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import brier_score, clip_probability, log_loss_score

# Gates for isotonic selection (§11.7 / V3 §5.4). Research defaults, configurable.
ISOTONIC_MIN_SAMPLES = 2000
ISOTONIC_MIN_SESSIONS = 40


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class SigmoidCalibrator:
    """Platt scaling: logistic regression on the raw score's log-odds."""
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, p_raw, y) -> "SigmoidCalibrator":
        from sklearn.linear_model import LogisticRegression
        x = _logit(p_raw).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self.a, self.b = 1.0, 0.0
            self.fitted = True
            return self
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, p_raw) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator used before fit")
        z = self.a * _logit(p_raw) + self.b
        out = np.where(z >= 0,
                       1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
        return clip_probability(out)

    def to_dict(self) -> dict:
        return {"method": "sigmoid", "a": self.a, "b": self.b}


@dataclass
class IsotonicCalibrator:
    """Isotonic regression p_raw -> p_cal; monotone, clipped to [0, 1]."""
    _iso: object = field(default=None, repr=False)
    fitted: bool = False

    def fit(self, p_raw, y) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")
        self._iso.fit(np.asarray(p_raw, dtype=float),
                      np.asarray(y, dtype=float))
        self.fitted = True
        return self

    def transform(self, p_raw) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator used before fit")
        return clip_probability(self._iso.predict(
            np.asarray(p_raw, dtype=float)))

    def to_dict(self) -> dict:
        return {"method": "isotonic"}


@dataclass
class IdentityCalibrator:
    """No-op fallback (still clips to [0, 1])."""
    fitted: bool = True

    def fit(self, p_raw, y) -> "IdentityCalibrator":
        return self

    def transform(self, p_raw) -> np.ndarray:
        return clip_probability(p_raw)

    def to_dict(self) -> dict:
        return {"method": "identity"}


@dataclass
class CalibrationArtifact:
    """Auditable record of an independent calibrator fit (V3 §5.3)."""
    method: str
    calibrator: object
    training_sessions: tuple[str, ...]
    oof_n: int
    oof_session_n: int
    brier_before: float
    brier_after: float
    log_loss_before: float
    log_loss_after: float
    slope: float | None
    intercept: float | None
    reliability_bins: list[dict]
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "training_sessions": list(self.training_sessions),
            "oof_n": self.oof_n,
            "oof_session_n": self.oof_session_n,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "log_loss_before": self.log_loss_before,
            "log_loss_after": self.log_loss_after,
            "slope": self.slope,
            "intercept": self.intercept,
            "reliability_bins": self.reliability_bins,
            "diagnostics": self.diagnostics,
            "calibrator": (self.calibrator.to_dict()
                           if hasattr(self.calibrator, "to_dict") else {}),
        }


def fit_calibrator(p_raw, y, method: str = "sigmoid"):
    """Fit one named calibrator ('sigmoid' | 'isotonic' | 'identity')."""
    cal = {"sigmoid": SigmoidCalibrator,
           "isotonic": IsotonicCalibrator,
           "identity": IdentityCalibrator}.get(method)
    if cal is None:
        raise ValueError(f"unknown calibration method {method!r}")
    return cal().fit(p_raw, y)


def _session_holdout_masks(
    sessions: Sequence[str],
    *,
    eval_frac: float = 0.25,
    embargo_sessions: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Time-ordered session holdout for nested calibrator comparison."""
    sessions = list(sessions)
    uniq = sorted(set(sessions))
    n_eval = max(1, int(round(len(uniq) * eval_frac)))
    n_fit = len(uniq) - n_eval - embargo_sessions
    if n_fit < 1:
        # too few: use first half / second half without embargo
        mid = max(1, len(uniq) // 2)
        fit_s, eval_s = uniq[:mid], uniq[mid:]
        if not eval_s:
            fit_s, eval_s = uniq[:-1], uniq[-1:]
    else:
        fit_s = uniq[:n_fit]
        eval_s = uniq[n_fit + embargo_sessions:]
    fit_set, eval_set = set(fit_s), set(eval_s)
    fit_mask = np.array([s in fit_set for s in sessions], dtype=bool)
    eval_mask = np.array([s in eval_set for s in sessions], dtype=bool)
    return fit_mask, eval_mask, fit_s, eval_s


def select_calibrator(p_raw, y, n_sessions: int, *,
                      sessions: Optional[Sequence[str]] = None,
                      min_samples: int = ISOTONIC_MIN_SAMPLES,
                      min_sessions: int = ISOTONIC_MIN_SESSIONS,
                      eval_frac: float = 0.25,
                      embargo_sessions: int = 1):
    """
    Choose sigmoid vs isotonic (V3 §5.4).

    Sigmoid is the default. Isotonic is considered only when sample AND
    session gates pass, and only when it beats sigmoid on a nested holdout
    of the calibration predictions (not an in-sample comparison of two
    fits on the same labels). The winning method is then refit on all
    provided calibration rows.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    diag: dict = {
        "chosen": "sigmoid",
        "n": n,
        "n_sessions": int(n_sessions),
        "brier_sigmoid": None,
        "brier_isotonic": None,
        "isotonic_rejected_reason": None,
        "comparison": "nested_holdout",
    }

    if n == 0 or len(np.unique(y)) < 2:
        cal = IdentityCalibrator().fit(p_raw, y)
        diag["chosen"] = "identity"
        diag["isotonic_rejected_reason"] = "degenerate_or_empty_labels"
        return cal, diag

    # Build nested comparison split
    if sessions is not None and len(sessions) == n and len(set(sessions)) >= 2:
        fit_m, eval_m, fit_s, eval_s = _session_holdout_masks(
            sessions, eval_frac=eval_frac, embargo_sessions=embargo_sessions)
        diag["fit_sessions"] = list(fit_s)
        diag["eval_sessions"] = list(eval_s)
    else:
        # Index holdout (still nested — never score on the fit labels)
        cut = max(1, int(round(n * (1.0 - eval_frac))))
        if cut >= n:
            cut = n - 1
        fit_m = np.zeros(n, dtype=bool)
        eval_m = np.zeros(n, dtype=bool)
        fit_m[:cut] = True
        eval_m[cut:] = True
        diag["fit_sessions"] = []
        diag["eval_sessions"] = []
        diag["comparison"] = "index_holdout"

    if not fit_m.any() or not eval_m.any() or len(np.unique(y[fit_m])) < 2:
        # Cannot nest — default sigmoid fit on all, never claim isotonic win
        sig = SigmoidCalibrator().fit(p_raw, y)
        diag["brier_sigmoid"] = brier_score(y, sig.transform(p_raw))
        diag["isotonic_rejected_reason"] = "insufficient_nested_split"
        return sig, diag

    sig_fit = SigmoidCalibrator().fit(p_raw[fit_m], y[fit_m])
    brier_sig = brier_score(y[eval_m], sig_fit.transform(p_raw[eval_m]))
    diag["brier_sigmoid"] = brier_sig

    gates_ok = (n >= min_samples and n_sessions >= min_sessions)
    if not gates_ok:
        reasons = []
        if n < min_samples:
            reasons.append(f"n={n}<{min_samples}")
        if n_sessions < min_sessions:
            reasons.append(f"sessions={n_sessions}<{min_sessions}")
        diag["isotonic_rejected_reason"] = "gates:" + ",".join(reasons)
        # Refit sigmoid on all calibration rows
        return SigmoidCalibrator().fit(p_raw, y), diag

    iso_fit = IsotonicCalibrator().fit(p_raw[fit_m], y[fit_m])
    brier_iso = brier_score(y[eval_m], iso_fit.transform(p_raw[eval_m]))
    diag["brier_isotonic"] = brier_iso

    if brier_iso < brier_sig:
        diag["chosen"] = "isotonic"
        return IsotonicCalibrator().fit(p_raw, y), diag

    diag["isotonic_rejected_reason"] = (
        f"nested_brier_isotonic={brier_iso:.6f}>="
        f"sigmoid={brier_sig:.6f}")
    return SigmoidCalibrator().fit(p_raw, y), diag


def reliability_bins(p, y, n_bins: int = 10) -> list:
    """Reliability table: mean predicted vs realized rate per probability bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                        "mean_predicted": float(p[mask].mean()),
                        "realized_rate": float(y[mask].mean())})
    return out


def calibration_slope_intercept(p, y) -> dict:
    """
    Logistic recalibration slope/intercept (promotion criteria §22.2:
    slope ~1, intercept ~0 for an honest probability).
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None}
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0])}


# --------------------------------------------------------------------------- #
# Slice reporting (V3 §5.5)                                                    #
# --------------------------------------------------------------------------- #

_TOD_BUCKETS = (
    ("open", 0, 60),
    ("morning", 60, 150),
    ("midday", 150, 270),
    ("afternoon", 270, 330),
    ("late", 330, 10_000),
)


def _tod_bucket(minutes_from_open) -> str:
    try:
        m = float(minutes_from_open)
    except (TypeError, ValueError):
        return "unknown"
    for name, lo, hi in _TOD_BUCKETS:
        if lo <= m < hi:
            return name
    return "unknown"


def _gex_sign_bucket(net_gex) -> str:
    try:
        v = float(net_gex)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v) or v == 0.0:
        return "flat" if v == 0.0 else "unknown"
    return "positive" if v > 0 else "negative"


def _confidence_bucket(p: float) -> str:
    c = abs(float(p) - 0.5)
    if c < 0.1:
        return "low"
    if c < 0.25:
        return "mid"
    return "high"


def slice_calibration_report(
    p: np.ndarray,
    y: np.ndarray,
    *,
    sessions: Optional[Sequence[str]] = None,
    rows: Optional[Sequence[dict]] = None,
    severe_brier_delta: float = 0.05,
) -> dict:
    """
    Calibration metrics overall and by session / TOD / GEX / vol / confidence /
    data-quality buckets. Flags slices whose Brier exceeds overall by
    `severe_brier_delta`.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    overall_brier = brier_score(y, p) if len(y) else None
    overall = {
        "n": int(len(y)),
        "brier": overall_brier,
        "log_loss": log_loss_score(y, p) if len(y) else None,
        **calibration_slope_intercept(p, y),
        "reliability": reliability_bins(p, y, n_bins=5),
    }
    slices: dict[str, list] = {
        "session": [],
        "time_of_day": [],
        "gex_sign": [],
        "volatility_quartile": [],
        "confidence": [],
        "data_quality": [],
    }
    flags: list[str] = []

    def _add(kind: str, label: str, mask: np.ndarray):
        if mask.sum() < 5:
            return
        b = brier_score(y[mask], p[mask])
        entry = {"bucket": label, "n": int(mask.sum()), "brier": b,
                 "log_loss": log_loss_score(y[mask], p[mask])}
        slices[kind].append(entry)
        if (overall_brier is not None
                and b - overall_brier >= severe_brier_delta):
            flags.append(f"severe_miscalibration:{kind}:{label}")

    if sessions is not None and len(sessions) == len(y):
        for s in sorted(set(sessions)):
            _add("session", s, np.array([x == s for x in sessions]))

    if rows is not None and len(rows) == len(y):
        tod = np.array([_tod_bucket(r.get("minutes_from_open",
                                          r.get("minutes_to_close")))
                        for r in rows])
        for b in sorted(set(tod.tolist())):
            _add("time_of_day", b, tod == b)

        gex = np.array([_gex_sign_bucket(r.get("net_gex", r.get("gex_net")))
                        for r in rows])
        for b in sorted(set(gex.tolist())):
            _add("gex_sign", b, gex == b)

        vols = []
        for r in rows:
            v = r.get("realized_vol", r.get("iv_rank", r.get("expected_realized_move")))
            try:
                vols.append(float(v) if v is not None else float("nan"))
            except (TypeError, ValueError):
                vols.append(float("nan"))
        vols_a = np.asarray(vols, dtype=float)
        finite = np.isfinite(vols_a)
        if finite.sum() >= 8:
            qs = np.nanquantile(vols_a[finite], [0.25, 0.5, 0.75])
            q_labels = np.full(len(y), "unknown", dtype=object)
            q_labels[finite & (vols_a <= qs[0])] = "q1"
            q_labels[finite & (vols_a > qs[0]) & (vols_a <= qs[1])] = "q2"
            q_labels[finite & (vols_a > qs[1]) & (vols_a <= qs[2])] = "q3"
            q_labels[finite & (vols_a > qs[2])] = "q4"
            for b in ("q1", "q2", "q3", "q4"):
                _add("volatility_quartile", b, q_labels == b)

        conf = np.array([_confidence_bucket(float(pi)) for pi in p])
        for b in ("low", "mid", "high"):
            _add("confidence", b, conf == b)

        dq = []
        for r in rows:
            v = r.get("data_quality", r.get("feature_coverage"))
            try:
                dq.append(float(v) if v is not None else float("nan"))
            except (TypeError, ValueError):
                dq.append(float("nan"))
        dq_a = np.asarray(dq, dtype=float)
        finite_dq = np.isfinite(dq_a)
        if finite_dq.sum() >= 8:
            med = float(np.median(dq_a[finite_dq]))
            labels = np.full(len(y), "unknown", dtype=object)
            labels[finite_dq & (dq_a >= med)] = "high_quality"
            labels[finite_dq & (dq_a < med)] = "low_quality"
            for b in ("high_quality", "low_quality"):
                _add("data_quality", b, labels == b)

    return {"overall": overall, "slices": slices, "flags": flags}


def build_calibration_artifact(
    calibrator,
    p_raw: np.ndarray,
    y: np.ndarray,
    *,
    training_sessions: Sequence[str],
    diagnostics: Optional[dict] = None,
) -> CalibrationArtifact:
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    p_cal = calibrator.transform(p_raw)
    slope_int = calibration_slope_intercept(p_cal, y)
    method = getattr(calibrator, "to_dict", lambda: {})().get("method", "unknown")
    return CalibrationArtifact(
        method=method,
        calibrator=calibrator,
        training_sessions=tuple(sorted(set(training_sessions))),
        oof_n=int(len(y)),
        oof_session_n=len(set(training_sessions)),
        brier_before=brier_score(y, p_raw),
        brier_after=brier_score(y, p_cal),
        log_loss_before=log_loss_score(y, p_raw),
        log_loss_after=log_loss_score(y, p_cal),
        slope=slope_int.get("slope"),
        intercept=slope_int.get("intercept"),
        reliability_bins=reliability_bins(p_cal, y, n_bins=5),
        diagnostics=dict(diagnostics or {}),
    )
