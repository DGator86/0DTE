"""
prediction/models/competing_risk.py
===================================
Discrete-time competing-risk target/stop model (V3 Part 2 §21–§22, PR 13).

Cause-specific hazards via multinomial logistic regression (HGB challenger).
Survival and cumulative incidence identities are enforced.

NOT financial advice.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.models.base import (
    RANDOM_STATE,
    FeatureVectorizer,
    clip_probability,
)

COMPETING_RISK_VERSION = "v3.0.0-cr"
_EVENT_NONE = 0
_EVENT_TARGET = 1
_EVENT_STOP = 2


@dataclass(frozen=True)
class CompetingRiskForecast:
    p_target_first: float
    p_stop_first: float
    p_neither: float
    expected_time_target: Optional[float]
    expected_time_stop: Optional[float]
    target_cumulative_incidence: tuple[float, ...]
    stop_cumulative_incidence: tuple[float, ...]
    survival_curve: tuple[float, ...]
    uncertainty: float
    support_rows: int
    support_sessions: int
    model_version: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        s = self.p_target_first + self.p_stop_first + self.p_neither
        if abs(s - 1.0) > 1e-5:
            raise ValueError(
                f"competing-risk probabilities must sum to 1, got {s}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompetingRiskConfig:
    estimator: str = "multinomial"  # multinomial | hgb
    horizon_minutes: int = 30
    max_iter: int = 1000
    random_state: int = RANDOM_STATE
    epsilon: float = 1e-12


def hazards_to_incidence(
    h_target: Sequence[float],
    h_stop: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> dict:
    """
    Convert per-step hazards to survival + cumulative incidence (§21.4–21.5).
    """
    ht = np.asarray(h_target, dtype=float)
    hs = np.asarray(h_stop, dtype=float)
    if ht.shape != hs.shape:
        raise ValueError("hazard length mismatch")
    # Clip so h_t + h_s <= 1
    total = ht + hs
    over = total > 1.0
    if over.any():
        ht = ht.copy()
        hs = hs.copy()
        ht[over] = ht[over] / total[over]
        hs[over] = hs[over] / total[over]
    n = len(ht)
    survival = np.ones(n + 1, dtype=float)
    ci_t = np.zeros(n, dtype=float)
    ci_s = np.zeros(n, dtype=float)
    for t in range(n):
        survival[t + 1] = survival[t] * (1.0 - ht[t] - hs[t])
        ci_t[t] = survival[t] * ht[t]
        ci_s[t] = survival[t] * hs[t]
    p_t = float(ci_t.sum())
    p_s = float(ci_s.sum())
    p_n = float(survival[n])
    # Renormalize tiny numeric drift
    z = p_t + p_s + p_n
    if z > epsilon:
        p_t, p_s, p_n = p_t / z, p_s / z, p_n / z
    return {
        "h_target": ht,
        "h_stop": hs,
        "survival": survival,
        "ci_target": ci_t,
        "ci_stop": ci_s,
        "p_target_first": p_t,
        "p_stop_first": p_s,
        "p_neither": p_n,
    }


def expected_event_time(
    incidence_at_t: Sequence[float],
    p_event: float,
    *,
    times: Optional[Sequence[float]] = None,
    epsilon: float = 1e-12,
) -> Optional[float]:
    if p_event <= epsilon:
        return None
    inc = np.asarray(incidence_at_t, dtype=float)
    if times is None:
        times = np.arange(1, len(inc) + 1, dtype=float)
    t = np.asarray(times, dtype=float)
    return float(np.sum(t * inc) / max(p_event, epsilon))


def _make_estimator(cfg: CompetingRiskConfig):
    if cfg.estimator == "multinomial":
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        kw = dict(
            solver="lbfgs",
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
        ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if ver < (1, 7):
            kw["multi_class"] = "multinomial"
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(**kw)),
        ])
    if cfg.estimator == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=3, max_leaf_nodes=15, min_samples_leaf=20,
            learning_rate=0.05, random_state=cfg.random_state)
    raise ValueError(f"unknown estimator {cfg.estimator!r}")


@dataclass
class CompetingRiskModel:
    cfg: CompetingRiskConfig = field(default_factory=CompetingRiskConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = field(default=None, repr=False)
    fitted: bool = False
    support_rows: int = 0
    support_sessions: int = 0
    model_version: str = COMPETING_RISK_VERSION
    diagnostics: dict = field(default_factory=dict)

    def fit(
        self,
        feature_rows: Sequence[dict],
        event_labels: Sequence[int],
        sessions: Optional[Sequence[str]] = None,
    ) -> "CompetingRiskModel":
        """
        event_labels: 0=none, 1=target, 2=stop for each discrete-time row.
        """
        y = np.asarray(event_labels, dtype=int)
        if not set(y).issubset({0, 1, 2}):
            raise ValueError("event labels must be in {0,1,2}")
        X = self.vectorizer.fit_transform(list(feature_rows))
        self.estimator = _make_estimator(self.cfg)
        self.estimator.fit(X, y)
        self.support_rows = int(len(y))
        self.support_sessions = (
            int(len(set(sessions))) if sessions is not None else 0)
        self.fitted = True
        self.diagnostics = {
            "class_counts": {
                "none": int((y == 0).sum()),
                "target": int((y == 1).sum()),
                "stop": int((y == 2).sum()),
            }
        }
        return self

    def predict_hazards(
        self,
        feature_rows: Sequence[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-row (h_target, h_stop) with h_t + h_s <= 1."""
        if not self.fitted:
            raise RuntimeError("CompetingRiskModel used before fit")
        X = self.vectorizer.transform(list(feature_rows))
        proba = self.estimator.predict_proba(X)
        # Map columns to none/target/stop
        classes = list(getattr(self.estimator, "classes_", [0, 1, 2]))
        if hasattr(self.estimator, "named_steps"):
            classes = list(self.estimator.named_steps["lr"].classes_)
        col = {int(c): i for i, c in enumerate(classes)}
        p_none = proba[:, col.get(0, 0)] if 0 in col else np.zeros(len(proba))
        p_t = proba[:, col[1]] if 1 in col else np.zeros(len(proba))
        p_s = proba[:, col[2]] if 2 in col else np.zeros(len(proba))
        # Hazards conditional on at-risk ≈ class probs among {none,target,stop}
        # which already sum to 1; h_target = P(target), h_stop = P(stop)
        ht = np.clip(p_t, 0.0, 1.0)
        hs = np.clip(p_s, 0.0, 1.0)
        total = ht + hs
        over = total > 1.0
        ht[over] = ht[over] / total[over]
        hs[over] = hs[over] / total[over]
        return ht, hs

    def forecast_from_path_features(
        self,
        step_feature_rows: Sequence[dict],
        *,
        uncertainty: float = 0.0,
    ) -> CompetingRiskForecast:
        ht, hs = self.predict_hazards(step_feature_rows)
        out = hazards_to_incidence(ht, hs, epsilon=self.cfg.epsilon)
        et_t = expected_event_time(
            out["ci_target"], out["p_target_first"], epsilon=self.cfg.epsilon)
        et_s = expected_event_time(
            out["ci_stop"], out["p_stop_first"], epsilon=self.cfg.epsilon)
        return CompetingRiskForecast(
            p_target_first=float(out["p_target_first"]),
            p_stop_first=float(out["p_stop_first"]),
            p_neither=float(out["p_neither"]),
            expected_time_target=et_t,
            expected_time_stop=et_s,
            target_cumulative_incidence=tuple(
                float(x) for x in np.cumsum(out["ci_target"])),
            stop_cumulative_incidence=tuple(
                float(x) for x in np.cumsum(out["ci_stop"])),
            survival_curve=tuple(float(x) for x in out["survival"]),
            uncertainty=float(clip_probability(uncertainty)),
            support_rows=self.support_rows,
            support_sessions=self.support_sessions,
            model_version=self.model_version,
            diagnostics={"n_steps": len(ht)},
        )
