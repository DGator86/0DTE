"""
prediction/candidate_ranker.py
==============================
V2 expected-utility candidate ranking
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §14, PR 8).

Legacy multiplicative score remains authoritative until promotion. This
module computes CandidateForecast utilities, ranks by utility, and produces
shadow diagnostics comparing V2 vs legacy tops — without changing the live
TradeDecision.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from prediction.candidate_dataset import (
    build_candidate_feature_row, candidate_id_for,
    fill_uncertainty_from_execution, legacy_metrics_from_candidate,
    legs_as_dicts,
)
from prediction.models.candidate_value import (
    CANDIDATE_VALUE_VERSION, CandidateForecast, CandidateValueModel,
)


@dataclass
class UtilityConfig:
    lambda_shortfall: float = 0.50
    lambda_fill: float = 0.25
    lambda_model: float = 0.25
    lambda_capital: float = 0.10
    portfolio_risk_budget: float = 1.0


@dataclass
class RankerConfig:
    mode: str = "shadow"                 # shadow | advisory | champion
    model_id: Optional[str] = None
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    # During shadow research, also rank candidates outside the routed family
    # (§14.4). Callers generate the set; this flag only documents intent.
    shadow_all_families: bool = True
    shadow_top_n_log: int = 16
    journal_legacy_comparison: bool = True


@dataclass
class SnapshotRankingResult:
    snapshot_id: str
    legacy_top_id: Optional[str]
    v2_top_id: Optional[str]
    rank_disagreement: bool
    forecasts: dict                        # candidate_id -> CandidateForecast
    diagnostics: dict = field(default_factory=dict)
    model_version: str = CANDIDATE_VALUE_VERSION

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "legacy_top_id": self.legacy_top_id,
            "v2_top_id": self.v2_top_id,
            "rank_disagreement": self.rank_disagreement,
            "forecasts": {k: v.to_dict() for k, v in self.forecasts.items()},
            "diagnostics": dict(self.diagnostics),
            "model_version": self.model_version,
        }

    def signals(self) -> dict:
        """Observation-only keys for journal signals_json."""
        v2 = self.forecasts.get(self.v2_top_id) if self.v2_top_id else None
        leg = self.diagnostics.get("legacy_top_score")
        out = {
            "v2_rank_disagreement": bool(self.rank_disagreement),
            "v2_top_candidate_id": self.v2_top_id,
            "legacy_top_candidate_id": self.legacy_top_id,
            "candidate_model_version": self.model_version,
            "legacy_top_score": leg,
        }
        if v2 is not None:
            out.update({
                "v2_utility_score": v2.utility_score,
                "v2_expected_net_pnl": v2.expected_net_pnl,
                "v2_p_profit": v2.p_profit,
                "v2_expected_shortfall": v2.expected_shortfall,
            })
        if "spearman_within_snapshot" in self.diagnostics:
            out["v2_legacy_spearman"] = self.diagnostics["spearman_within_snapshot"]
        if "expected_net_pnl_delta" in self.diagnostics:
            out["v2_vs_legacy_pnl_delta"] = self.diagnostics["expected_net_pnl_delta"]
        return out


def candidate_utility(
    forecast: CandidateForecast,
    *,
    capital: float = 0.0,
    cfg: Optional[UtilityConfig] = None,
) -> float:
    """
    §14.2 utility. Monotonicity (PR 8 AC):
      higher expected_shortfall / fill_uncertainty / model_uncertainty
      → lower utility (holding other terms fixed).
    """
    cfg = cfg or UtilityConfig()
    budget = max(cfg.portfolio_risk_budget, 1e-9)
    capital_penalty = float(capital) / budget
    return (
        float(forecast.expected_net_pnl)
        - cfg.lambda_shortfall * float(forecast.expected_shortfall)
        - cfg.lambda_fill * float(forecast.fill_uncertainty)
        - cfg.lambda_model * float(forecast.model_uncertainty)
        - cfg.lambda_capital * capital_penalty
    )


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def rank_candidates(
    candidates: Sequence,
    forecasts: dict,
    *,
    cfg: Optional[UtilityConfig] = None,
    require_veto_pass: bool = True,
) -> list:
    """
    Return [(candidate, forecast), ...] sorted by utility descending.
    Hard constraints (§14.3): optionally require passes_vetoes first.
    """
    cfg = cfg or UtilityConfig()
    paired = []
    for c in candidates:
        cid = None
        # Prefer matching via forecast keys already assigned
        for k, fc in forecasts.items():
            # identity by object attribute if present
            if getattr(c, "_v2_candidate_id", None) == k:
                cid = k
                break
        if cid is None:
            # fall through: caller should have set _v2_candidate_id
            continue
        if require_veto_pass and not getattr(c, "passes_vetoes", True):
            continue
        fc = forecasts[cid]
        paired.append((c, fc))
    paired.sort(key=lambda pair: pair[1].utility_score, reverse=True)
    return paired


def run_shadow_ranking(
    candidates: Sequence,
    model: CandidateValueModel,
    *,
    snapshot_id: str,
    spot: float,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    net_gex: Optional[float] = None,
    bundle: Optional[dict] = None,
    cfg: Optional[RankerConfig] = None,
    store=None,
) -> SnapshotRankingResult:
    """
    Score every candidate with the V2 model, compute utilities, compare to
    legacy ranking. Optionally persist candidate snapshots to PredictionStore.

    Does NOT mutate candidate.score — legacy ranking stays authoritative.
    """
    cfg = cfg or RankerConfig()
    util_cfg = cfg.utility

    # Assign stable ids
    ids = []
    rows = []
    fills = []
    caps = []
    legacy_scores = []
    feasible = []
    for c in candidates:
        cid = candidate_id_for(snapshot_id, c)
        setattr(c, "_v2_candidate_id", cid)
        ids.append(cid)
        rows.append(build_candidate_feature_row(
            c, snapshot_id=snapshot_id, spot=spot,
            call_wall=call_wall, put_wall=put_wall, gamma_flip=gamma_flip,
            minutes_to_close=minutes_to_close, net_gex=net_gex, bundle=bundle))
        fills.append(fill_uncertainty_from_execution(
            getattr(c, "execution", None)))
        caps.append(float(getattr(c, "capital", 0.0)
                          or getattr(c, "max_loss", 0.0) or 0.0))
        legacy_scores.append(float(getattr(c, "score", 0.0) or 0.0))
        if getattr(c, "passes_vetoes", False):
            feasible.append(c)

    def _util(fc: CandidateForecast, capital: float = 0.0) -> float:
        return candidate_utility(fc, capital=capital, cfg=util_cfg)

    forecasts_list = model.predict(
        rows, candidate_ids=ids, fill_uncertainty=fills,
        capital=caps, utility_fn=_util) if rows else []
    forecasts = {fc.candidate_id: fc for fc in forecasts_list}

    # Legacy top among veto-passing candidates
    legacy_pool = feasible or list(candidates)
    legacy_top = (max(legacy_pool, key=lambda c: float(c.score or 0.0))
                  if legacy_pool else None)
    legacy_top_id = (getattr(legacy_top, "_v2_candidate_id", None)
                     if legacy_top is not None else None)

    # V2 top among veto-passing (hard constraints before ranking)
    v2_ranked = rank_candidates(
        candidates, forecasts, cfg=util_cfg, require_veto_pass=True)
    if not v2_ranked and forecasts:
        v2_ranked = rank_candidates(
            candidates, forecasts, cfg=util_cfg, require_veto_pass=False)
    v2_top = v2_ranked[0][0] if v2_ranked else None
    v2_top_id = (getattr(v2_top, "_v2_candidate_id", None)
                 if v2_top is not None else None)

    # Diagnostics (§14.5)
    spearman = None
    if len(ids) >= 2:
        util_scores = [forecasts[i].utility_score for i in ids]
        spearman = _spearman(legacy_scores, util_scores)

    v2_fc = forecasts.get(v2_top_id) if v2_top_id else None
    leg_fc = forecasts.get(legacy_top_id) if legacy_top_id else None
    pnl_delta = None
    if v2_fc is not None and leg_fc is not None:
        pnl_delta = v2_fc.expected_net_pnl - leg_fc.expected_net_pnl

    diagnostics = {
        "n_candidates": len(candidates),
        "n_feasible": len(feasible),
        "legacy_top_score": (float(legacy_top.score)
                             if legacy_top is not None else None),
        "legacy_top_family": (legacy_top.family
                              if legacy_top is not None else None),
        "v2_top_family": (v2_top.family if v2_top is not None else None),
        "v2_top_utility": (v2_fc.utility_score if v2_fc else None),
        "expected_net_pnl_delta": pnl_delta,
        "spearman_within_snapshot": spearman,
        "mode": cfg.mode,
    }

    result = SnapshotRankingResult(
        snapshot_id=snapshot_id,
        legacy_top_id=legacy_top_id,
        v2_top_id=v2_top_id,
        rank_disagreement=(legacy_top_id != v2_top_id),
        forecasts=forecasts,
        diagnostics=diagnostics,
        model_version=getattr(model, "metadata", {}).get(
            "model_version", CANDIDATE_VALUE_VERSION),
    )

    # Persist shadow candidate set (top-N by utility + legacy top)
    if store is not None and candidates:
        keep_ids = set()
        for c, fc in v2_ranked[:cfg.shadow_top_n_log]:
            keep_ids.add(fc.candidate_id)
        if legacy_top_id:
            keep_ids.add(legacy_top_id)
        for c in candidates:
            cid = getattr(c, "_v2_candidate_id", None)
            if cid not in keep_ids:
                continue
            fc = forecasts.get(cid)
            try:
                store.log_candidate_snapshot(
                    cid, snapshot_id, c.family, legs_as_dicts(c.legs),
                    quote={"mid_credit": c.credit,
                           "expected": (c.execution or {}).get(
                               "net_expected_credit") if c.execution else None},
                    geometry=build_candidate_feature_row(
                        c, snapshot_id=snapshot_id, spot=spot,
                        call_wall=call_wall, put_wall=put_wall,
                        gamma_flip=gamma_flip,
                        minutes_to_close=minutes_to_close, net_gex=net_gex,
                        bundle=bundle),
                    legacy_metrics=legacy_metrics_from_candidate(c),
                    execution_estimate=getattr(c, "execution", None),
                    prediction=(fc.to_dict() if fc else None),
                )
            except Exception:
                # Observation-only: never break the tick
                pass

    return result
