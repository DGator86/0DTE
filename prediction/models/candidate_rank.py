"""
prediction/models/candidate_rank.py
===================================
Within-snapshot pairwise candidate ranking
(docs/PREDICTION_ENGINE_V3_PART3_DECISION_DEPLOYMENT.md §8–§10).

Baseline: pairwise logistic regression.
Challenger: HistGradientBoostingClassifier.

Absolute utility and pairwise scores are blended for the combined ranking.
Legacy selection remains authoritative until promotion — this module is
research / shadow only.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.candidate_dataset import (
    PAIR_EPSILON_R,
    build_pair_features,
    reverse_pair_features,
)
from prediction.models.base import RANDOM_STATE, FeatureVectorizer, clip_probability

CANDIDATE_RANK_VERSION = "v3.0.0"


@dataclass
class CandidateRankConfig:
    estimator: str = "logistic"  # "logistic" | "hgb"
    absolute_weight: float = 0.60
    pairwise_weight: float = 0.40
    pair_epsilon_r: float = PAIR_EPSILON_R
    risk_unit_r: float = 1.0
    margin_reference: float = 0.25
    C: float = 1.0
    max_iter: int = 2000
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 15
    hgb_max_depth: Optional[int] = 3
    hgb_min_samples_leaf: int = 20


@dataclass(frozen=True)
class CandidateRanking:
    snapshot_id: str
    ordered_candidate_ids: tuple
    combined_scores: dict
    absolute_utilities: dict
    pairwise_scores: dict
    expected_regret: dict
    selection_uncertainty: dict
    top_candidate_id: Optional[str]
    second_candidate_id: Optional[str]
    top_score_margin: Optional[float]
    model_version: str = CANDIDATE_RANK_VERSION
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "combined_scores": dict(self.combined_scores),
            "absolute_utilities": dict(self.absolute_utilities),
            "pairwise_scores": dict(self.pairwise_scores),
            "expected_regret": dict(self.expected_regret),
            "selection_uncertainty": dict(self.selection_uncertainty),
            "top_candidate_id": self.top_candidate_id,
            "second_candidate_id": self.second_candidate_id,
            "top_score_margin": self.top_score_margin,
            "model_version": self.model_version,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateRanking":
        return cls(
            snapshot_id=str(d["snapshot_id"]),
            ordered_candidate_ids=tuple(d.get("ordered_candidate_ids") or ()),
            combined_scores=dict(d.get("combined_scores") or {}),
            absolute_utilities=dict(d.get("absolute_utilities") or {}),
            pairwise_scores=dict(d.get("pairwise_scores") or {}),
            expected_regret=dict(d.get("expected_regret") or {}),
            selection_uncertainty=dict(d.get("selection_uncertainty") or {}),
            top_candidate_id=d.get("top_candidate_id"),
            second_candidate_id=d.get("second_candidate_id"),
            top_score_margin=(
                None if d.get("top_score_margin") is None
                else float(d["top_score_margin"])),
            model_version=str(d.get("model_version", CANDIDATE_RANK_VERSION)),
            diagnostics=dict(d.get("diagnostics") or {}),
        )


def _make_estimator(cfg: CandidateRankConfig):
    if cfg.estimator == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=cfg.hgb_learning_rate,
            max_leaf_nodes=cfg.hgb_max_leaf_nodes,
            max_depth=cfg.hgb_max_depth,
            min_samples_leaf=cfg.hgb_min_samples_leaf,
            random_state=RANDOM_STATE,
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=cfg.C, solver="lbfgs", max_iter=cfg.max_iter,
            random_state=RANDOM_STATE,
        )),
    ])


def _minmax_norm(values: dict) -> dict:
    if not values:
        return {}
    xs = list(values.values())
    lo, hi = min(xs), max(xs)
    span = hi - lo
    if span < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def ranking_regret(
    selected_id: Optional[str],
    realized_utilities: dict,
) -> float:
    """hindsight_best - selected (0 if no selection / empty)."""
    if not realized_utilities:
        return 0.0
    best = max(realized_utilities.values())
    if selected_id is None or selected_id not in realized_utilities:
        return float(best)
    return float(best - realized_utilities[selected_id])


def tie_break_key(
    candidate_id: str,
    *,
    absolute_utility: float,
    uncertainty: float,
    capital: float,
) -> tuple:
    """Deterministic tie-break: higher utility, lower unc, lower capital, id."""
    return (-float(absolute_utility), float(uncertainty), float(capital),
            str(candidate_id))


@dataclass
class PairwiseCandidateRanker:
    """Fit on CandidatePairRecord rows; score within a snapshot at inference."""
    cfg: CandidateRankConfig = field(default_factory=CandidateRankConfig)
    vectorizer: FeatureVectorizer = field(default_factory=FeatureVectorizer)
    estimator: object = None
    fitted: bool = False
    model_version: str = CANDIDATE_RANK_VERSION

    def fit(self, pairs: Sequence) -> "PairwiseCandidateRanker":
        if len(pairs) < 2:
            raise ValueError("need at least 2 pairs to fit pairwise ranker")
        rows = [dict(p.pair_features) for p in pairs]
        y = np.asarray([int(p.a_wins) for p in pairs], dtype=int)
        w = np.asarray([float(p.weight) for p in pairs], dtype=float)
        if y.min() == y.max():
            # Degenerate labels — fit a constant-ish model on zeros
            rows = rows + [reverse_pair_features(rows[0])]
            y = np.concatenate([y, 1 - y[:1]])
            w = np.concatenate([w, w[:1]])
        self.vectorizer.fit(rows)
        X = self.vectorizer.transform(rows)
        est = _make_estimator(self.cfg)
        try:
            est.fit(X, y, lr__sample_weight=w)
        except TypeError:
            try:
                est.fit(X, y, sample_weight=w)
            except TypeError:
                est.fit(X, y)
        self.estimator = est
        self.fitted = True
        return self

    def predict_pair_proba(self, pair_features: dict) -> float:
        """P(A beats B)."""
        if not self.fitted:
            raise RuntimeError("PairwiseCandidateRanker.predict before fit")
        X = self.vectorizer.transform([pair_features])
        if hasattr(self.estimator, "predict_proba"):
            p = float(self.estimator.predict_proba(X)[0, 1])
        else:
            p = float(self.estimator.decision_function(X)[0])
            p = 1.0 / (1.0 + np.exp(-p))
        return clip_probability(p)

    def pairwise_scores_for_snapshot(
        self,
        candidates: Sequence[dict],
    ) -> dict:
        """
        candidates: [{candidate_id, features}, ...]
        Returns mean P(i beats j) over j != i.
        """
        items = list(candidates)
        if not items:
            return {}
        if len(items) == 1:
            return {str(items[0]["candidate_id"]): 0.5}
        scores = {}
        for i, a in enumerate(items):
            probs = []
            for j, b in enumerate(items):
                if i == j:
                    continue
                feat = build_pair_features(
                    a.get("features") or {}, b.get("features") or {})
                probs.append(self.predict_pair_proba(feat))
            scores[str(a["candidate_id"])] = float(np.mean(probs))
        return scores

    def rank_snapshot(
        self,
        snapshot_id: str,
        candidates: Sequence[dict],
        *,
        absolute_utilities: Optional[dict] = None,
        vetoed_ids: Optional[set] = None,
        uncertainties: Optional[dict] = None,
        capitals: Optional[dict] = None,
        realized_utilities: Optional[dict] = None,
    ) -> CandidateRanking:
        """
        Blend absolute utility with pairwise scores.

        candidates: [{candidate_id, features, absolute_utility?, ...}]
        Vetoed candidates are scored for diagnostics but cannot be top.
        """
        vetoed = set(vetoed_ids or ())
        abs_u = dict(absolute_utilities or {})
        unc = dict(uncertainties or {})
        caps = dict(capitals or {})
        for c in candidates:
            cid = str(c["candidate_id"])
            if cid not in abs_u:
                abs_u[cid] = float(
                    c.get("absolute_utility",
                          (c.get("features") or {}).get("utility_score", 0.0)
                          or 0.0))
            if cid not in unc:
                unc[cid] = float(c.get("uncertainty", 0.0) or 0.0)
            if cid not in caps:
                caps[cid] = float(
                    c.get("capital",
                          (c.get("features") or {}).get("capital_required", 0.0)
                          or 0.0))

        if self.fitted and len(candidates) >= 1:
            pair_scores = self.pairwise_scores_for_snapshot(candidates)
        else:
            # Unfitted fallback: rank by absolute utility only
            pair_scores = {str(c["candidate_id"]): 0.5 for c in candidates}

        norm_abs = _minmax_norm(abs_u)
        norm_pair = _minmax_norm(pair_scores)
        aw = float(self.cfg.absolute_weight)
        pw = float(self.cfg.pairwise_weight)
        denom = aw + pw
        if denom <= 0:
            aw, pw, denom = 1.0, 0.0, 1.0
        combined = {
            cid: (aw * norm_abs.get(cid, 0.0) + pw * norm_pair.get(cid, 0.0))
            / denom
            for cid in abs_u
        }

        # Sort with deterministic tie-break; actionable order excludes vetoes
        def sort_key(cid: str) -> tuple:
            return (
                -combined.get(cid, 0.0),
                -abs_u.get(cid, 0.0),
                unc.get(cid, 0.0),
                caps.get(cid, 0.0),
                str(cid),
            )

        ordered = tuple(sorted(combined.keys(), key=sort_key))
        actionable = [cid for cid in ordered if cid not in vetoed]
        top = actionable[0] if actionable else None
        second = actionable[1] if len(actionable) > 1 else None
        margin = None
        if top is not None and second is not None:
            margin = float(combined[top] - combined[second])
        elif top is not None:
            margin = 1.0

        margin_unc = 1.0
        if margin is not None:
            ref = max(float(self.cfg.margin_reference), 1e-9)
            margin_unc = 1.0 - float(np.clip(margin / ref, 0.0, 1.0))

        sel_unc = {}
        for cid in combined:
            sel_unc[cid] = float(np.clip(
                0.5 * unc.get(cid, 0.0) + 0.5 * margin_unc, 0.0, 1.0))

        # Expected regret vs hindsight when realized utilities provided;
        # otherwise estimate vs best combined score among actionable.
        exp_regret = {}
        if realized_utilities:
            best_r = max(realized_utilities.values()) if realized_utilities else 0.0
            for cid in combined:
                exp_regret[cid] = float(
                    best_r - realized_utilities.get(cid, best_r))
        else:
            best_c = max((combined[c] for c in actionable), default=0.0)
            for cid in combined:
                exp_regret[cid] = float(best_c - combined.get(cid, 0.0))

        return CandidateRanking(
            snapshot_id=snapshot_id,
            ordered_candidate_ids=ordered,
            combined_scores=combined,
            absolute_utilities=abs_u,
            pairwise_scores=pair_scores,
            expected_regret=exp_regret,
            selection_uncertainty=sel_unc,
            top_candidate_id=top,
            second_candidate_id=second,
            top_score_margin=margin,
            model_version=self.model_version,
            diagnostics={
                "absolute_weight": aw,
                "pairwise_weight": pw,
                "n_candidates": len(combined),
                "n_vetoed": len(vetoed),
                "fitted": bool(self.fitted),
                "estimator": self.cfg.estimator,
            },
        )


def pairwise_accuracy(pairs: Sequence, ranker: PairwiseCandidateRanker) -> float:
    if not pairs:
        return float("nan")
    correct = 0
    for p in pairs:
        pred = ranker.predict_pair_proba(p.pair_features)
        correct += int((pred >= 0.5) == bool(p.a_wins))
    return correct / len(pairs)
