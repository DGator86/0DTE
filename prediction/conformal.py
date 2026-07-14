"""
prediction/conformal.py
=======================
Session-grouped split conformal calibration for return intervals
(V3 Part 2 §17, PR 12).

Corrections are fit on calibration sessions only. Test labels must not
alter the correction. OOD observations are flagged as coverage-limited.

Research / shadow only.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

CONFORMAL_VERSION = "v3.0.0-conformal"


@dataclass(frozen=True)
class ConformalInterval:
    nominal_coverage: float
    lower: float
    upper: float
    correction: float
    support_rows: int
    support_sessions: int
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplitConformalCalibrator:
    """
    Symmetric split-conformal interval widener (§17.4–17.5).

    score = max(predicted_lower - y, y - predicted_upper, 0)
    corrected_lower = predicted_lower - correction
    corrected_upper = predicted_upper + correction
    """

    nominal_coverage: float = 0.90
    correction: float = 0.0
    support_rows: int = 0
    support_sessions: int = 0
    calibration_sessions: tuple[str, ...] = ()
    ood_multiplier: float = 1.5
    ood_threshold: float = 0.8
    fitted: bool = False
    model_version: str = CONFORMAL_VERSION
    diagnostics: dict = field(default_factory=dict)

    def fit(
        self,
        y: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        sessions: Sequence[str],
    ) -> "SplitConformalCalibrator":
        y_arr = np.asarray(y, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if len(y_arr) != len(lo) or len(y_arr) != len(hi):
            raise ValueError("y/lower/upper length mismatch")
        scores = np.maximum(np.maximum(lo - y_arr, y_arr - hi), 0.0)
        n = len(scores)
        if n == 0:
            self.correction = 0.0
            self.fitted = True
            self.diagnostics["empty_calibration"] = True
            return self
        # Finite-sample quantile level for split conformal
        alpha = 1.0 - float(self.nominal_coverage)
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        level = float(np.clip(level, 0.0, 1.0))
        self.correction = float(np.quantile(scores, level))
        self.support_rows = int(n)
        self.support_sessions = int(len(set(sessions)))
        self.calibration_sessions = tuple(sorted(set(sessions)))
        self.fitted = True
        self.diagnostics = {
            "alpha": alpha,
            "quantile_level": level,
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
        }
        return self

    def apply(
        self,
        lower: float,
        upper: float,
        *,
        ood_score: Optional[float] = None,
    ) -> ConformalInterval:
        if not self.fitted:
            raise RuntimeError("SplitConformalCalibrator used before fit")
        corr = float(self.correction)
        coverage_limited = False
        if ood_score is not None and float(ood_score) >= self.ood_threshold:
            corr = corr * float(self.ood_multiplier)
            coverage_limited = True
        lo = float(lower) - corr
        hi = float(upper) + corr
        if lo > hi:
            lo, hi = hi, lo
        return ConformalInterval(
            nominal_coverage=float(self.nominal_coverage),
            lower=lo,
            upper=hi,
            correction=corr,
            support_rows=self.support_rows,
            support_sessions=self.support_sessions,
            model_version=self.model_version,
            diagnostics={
                "coverage_limited": coverage_limited,
                "ood_score": ood_score,
                "calibration_sessions": list(self.calibration_sessions),
            },
        )

    def attach_to_distribution(
        self,
        dist,
        *,
        lower_q: float = 0.05,
        upper_q: float = 0.95,
        ood_score: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Return a new ReturnDistribution with a conformal interval attached.
        Does not mutate the input.
        """
        from prediction.return_distribution import ReturnDistribution

        qs = dist.quantiles
        # Allow str keys from serialization
        def _get(q):
            if q in qs:
                return qs[q]
            return qs[str(q)]

        interval = self.apply(_get(lower_q), _get(upper_q), ood_score=ood_score)
        key = name or f"nominal_{int(round(self.nominal_coverage * 100))}"
        new_intervals = dict(dist.conformal_intervals)
        new_intervals[key] = (interval.lower, interval.upper)
        return ReturnDistribution(
            horizon=dist.horizon,
            quantiles=dict(qs),
            expected_return=dist.expected_return,
            variance=dist.variance,
            conformal_intervals=new_intervals,
            conformal_support_rows=self.support_rows,
            conformal_support_sessions=self.support_sessions,
            uncertainty=dist.uncertainty,
            ood_score=ood_score if ood_score is not None else dist.ood_score,
            model_version=dist.model_version,
            diagnostics={
                **dict(dist.diagnostics),
                "conformal": interval.to_dict(),
            },
        )
