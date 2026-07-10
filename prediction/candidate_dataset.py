"""
prediction/candidate_dataset.py
===============================
Candidate-level training frames and snapshot-grouped folds
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11.6, PR 8).

Candidates that share a snapshot_id (or session_date) must never be divided
between training and test. This module builds feature rows from
SpreadCandidate-like objects and provides fold helpers that keep whole
snapshots together.

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from prediction.storage import make_candidate_id


@dataclass
class CandidateTrainingFrame:
    candidate_ids: list = field(default_factory=list)
    snapshot_ids: list = field(default_factory=list)
    session_dates: list = field(default_factory=list)
    rows: list = field(default_factory=list)          # feature dicts
    y_pnl: list = field(default_factory=list)         # primary label
    y_profit: list = field(default_factory=list)      # 1 if pnl > 0
    fill_uncertainty: list = field(default_factory=list)
    capital: list = field(default_factory=list)
    labels: list = field(default_factory=list)        # full outcome dicts

    def __len__(self) -> int:
        return len(self.rows)


def legs_as_dicts(legs) -> list:
    """Normalize Leg objects / dicts / tuples into [{strike, kind, qty}]."""
    out = []
    for lg in legs:
        if isinstance(lg, dict):
            out.append({"strike": float(lg["strike"]), "kind": lg["kind"],
                        "qty": int(lg["qty"])})
        else:
            out.append({"strike": float(lg.strike), "kind": lg.kind,
                        "qty": int(lg.qty)})
    return out


def legacy_metrics_from_candidate(c) -> dict:
    """Serialize the current multiplicative score panel for audit."""
    return {
        "score": float(getattr(c, "score", 0.0) or 0.0),
        "ev": float(getattr(c, "ev", 0.0) or 0.0),
        "ev_per_risk": float(getattr(c, "ev_per_risk", 0.0) or 0.0),
        "prob_profit": float(getattr(c, "prob_profit", 0.0) or 0.0),
        "prob_touch_short": float(getattr(c, "prob_touch_short", 0.0) or 0.0),
        "liquidity_score": float(getattr(c, "liquidity_score", 0.0) or 0.0),
        "wall_safety": float(getattr(c, "wall_safety", 0.0) or 0.0),
        "gamma_safety": float(getattr(c, "gamma_safety", 0.0) or 0.0),
        "touch_safety": float(getattr(c, "touch_safety", 0.0) or 0.0),
        "credit": float(getattr(c, "credit", 0.0) or 0.0),
        "max_loss": float(getattr(c, "max_loss", 0.0) or 0.0),
        "passes_vetoes": bool(getattr(c, "passes_vetoes", False)),
        "touch_source": getattr(c, "touch_source", "reflection"),
    }


def fill_uncertainty_from_execution(execution: Optional[dict]) -> float:
    """
    Higher when mid→natural gap is large relative to |mid| credit, or when
    expected fill fraction is low. Bounded to [0, 1].
    """
    if not execution:
        return 0.5
    mid = execution.get("mid_credit")
    nat = execution.get("natural_credit")
    frac = execution.get("fill_fraction_expected")
    if mid is None:
        return 0.5
    mid = float(mid)
    gap = abs(float(mid) - float(nat if nat is not None else mid))
    denom = max(abs(mid), 0.05)
    u_spread = min(gap / denom, 1.0)
    u_fill = 1.0 - float(frac) if frac is not None else 0.5
    return float(np.clip(0.5 * u_spread + 0.5 * u_fill, 0.0, 1.0))


def build_candidate_feature_row(
    candidate,
    *,
    snapshot_id: str,
    spot: float,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    gamma_flip: Optional[float] = None,
    minutes_to_close: Optional[float] = None,
    net_gex: Optional[float] = None,
    bundle: Optional[dict] = None,
    data_quality: Optional[float] = None,
) -> dict:
    """
    Flat feature dict for CandidateValueModel (§11.6 inputs).

    `candidate` is a SpreadCandidate (duck-typed). `bundle` is an optional
    PredictionBundle.to_dict()-like mapping of forecast summaries.
    """
    legs = legs_as_dicts(candidate.legs)
    shorts = [lg["strike"] for lg in legs if lg["qty"] < 0]
    longs = [lg["strike"] for lg in legs if lg["qty"] > 0]
    width = 0.0
    if shorts and longs:
        width = abs(float(shorts[0]) - float(longs[0]))
    elif len(shorts) >= 2:
        width = abs(float(shorts[0]) - float(shorts[1]))

    credit = float(getattr(candidate, "credit", 0.0) or 0.0)
    is_debit = credit < 0
    execution = getattr(candidate, "execution", None) or {}
    row = {
        "is_debit": float(is_debit),
        "credit": credit,
        "width": width,
        "width_pct": width / spot if spot else None,
        "max_loss": float(getattr(candidate, "max_loss", 0.0) or 0.0),
        "capital": float(getattr(candidate, "capital", 0.0)
                         or getattr(candidate, "max_loss", 0.0) or 0.0),
        "theta": float(getattr(candidate, "theta", 0.0) or 0.0),
        "gamma": float(getattr(candidate, "gamma", 0.0) or 0.0),
        "prob_profit": float(getattr(candidate, "prob_profit", 0.0) or 0.0),
        "prob_touch_short": float(
            getattr(candidate, "prob_touch_short", 0.0) or 0.0),
        "distance_to_wall": float(
            getattr(candidate, "distance_to_wall", 0.0) or 0.0),
        "liquidity_score": float(
            getattr(candidate, "liquidity_score", 0.0) or 0.0),
        "wall_safety": float(getattr(candidate, "wall_safety", 0.0) or 0.0),
        "gamma_safety": float(getattr(candidate, "gamma_safety", 0.0) or 0.0),
        "touch_safety": float(getattr(candidate, "touch_safety", 0.0) or 0.0),
        "legacy_candidate_score": float(getattr(candidate, "score", 0.0) or 0.0),
        "ev": float(getattr(candidate, "ev", 0.0) or 0.0),
        "ev_per_risk": float(getattr(candidate, "ev_per_risk", 0.0) or 0.0),
        "n_legs": float(len(legs)),
        "n_short": float(len(shorts)),
    }
    if execution:
        row["expected_fill_credit"] = execution.get("net_expected_credit")
        row["mid_credit"] = execution.get("mid_credit")
        row["natural_credit"] = execution.get("natural_credit")
        row["fill_fraction_expected"] = execution.get("fill_fraction_expected")
        row["mid_to_natural"] = None
        if (execution.get("mid_credit") is not None
                and execution.get("natural_credit") is not None):
            row["mid_to_natural"] = (
                float(execution["mid_credit"])
                - float(execution["natural_credit"]))
    if call_wall is not None and spot:
        row["dist_call_wall"] = (call_wall - spot) / spot
    if put_wall is not None and spot:
        row["dist_put_wall"] = (spot - put_wall) / spot
    if gamma_flip is not None and spot:
        row["dist_gamma_flip"] = (spot - gamma_flip) / spot
    if minutes_to_close is not None:
        row["minutes_to_close"] = float(minutes_to_close)
    if net_gex is not None:
        row["net_gex_sign"] = float(np.sign(net_gex))
        row["net_gex"] = float(net_gex)
    if data_quality is not None:
        row["data_quality"] = float(data_quality)
    if bundle:
        # Lift a few PredictionBundle summaries when present
        for key in ("direction_p_up_30m", "direction_p_up_close",
                    "expected_realized_move", "vol_uncertainty",
                    "model_uncertainty"):
            if key in bundle:
                row[f"bundle_{key}"] = bundle[key]
        # nested common shapes
        dirs = bundle.get("direction") or {}
        if isinstance(dirs, dict):
            for h, p in dirs.items():
                if isinstance(p, (int, float)):
                    row[f"bundle_dir_{h}"] = float(p)
        vol = bundle.get("volatility") or {}
        if isinstance(vol, dict) and "expected_realized_move" in vol:
            row["bundle_expected_realized_move"] = vol["expected_realized_move"]
    # One-hot-ish family marker via hashed numeric (vectorizer handles strings
    # poorly — encode as categorical float hash of family name length + ord)
    fam = getattr(candidate, "family", "") or ""
    row["family_code"] = float(sum(ord(ch) for ch in fam) % 97) if fam else None
    return row


def append_settled_candidate(
    frame: CandidateTrainingFrame,
    *,
    candidate_id: str,
    snapshot_id: str,
    session_date: str,
    feature_row: dict,
    outcome: dict,
    fill_uncertainty: float = 0.0,
    capital: float = 0.0,
    pnl_key: str = "pnl_expected_fill",
) -> None:
    """Append one settled candidate to a training frame."""
    pnl = outcome.get(pnl_key)
    if pnl is None:
        pnl = outcome.get("pnl_mid")
    if pnl is None:
        return
    frame.candidate_ids.append(candidate_id)
    frame.snapshot_ids.append(snapshot_id)
    frame.session_dates.append(session_date)
    frame.rows.append(feature_row)
    frame.y_pnl.append(float(pnl))
    frame.y_profit.append(int(float(pnl) > 0.0))
    frame.fill_uncertainty.append(float(fill_uncertainty))
    frame.capital.append(float(capital))
    frame.labels.append(dict(outcome))


def load_candidate_training_frame(
    store,
    *,
    pnl_key: str = "pnl_expected_fill",
) -> CandidateTrainingFrame:
    """
    Rebuild a CandidateTrainingFrame from PredictionStore candidate tables.
    Only settled candidates with a usable P&L label are included.
    """
    frame = CandidateTrainingFrame()
    rows = store.fetch_candidates()
    for r in rows:
        if not r.get("settled"):
            continue
        outcome = {
            "settled": r.get("settled"),
            "settlement_price": r.get("settlement_price"),
            "pnl_mid": r.get("pnl_mid"),
            "mfe": r.get("mfe"),
            "mae": r.get("mae"),
            "target_hit": r.get("target_hit"),
            "stop_hit": r.get("stop_hit"),
            "first_event": r.get("first_event"),
        }
        extras = r.get("outcome_extras") or {}
        if isinstance(extras, dict):
            outcome.update(extras)
        # Prefer explicit columns when present on the joined row
        for k in ("pnl_expected_fill", "pnl_conservative", "pnl_policy"):
            if r.get(k) is not None:
                outcome[k] = r[k]
        # Feature row: rebuild a minimal one from stored metrics if no
        # geometry features were persisted separately.
        legacy = r.get("legacy_metrics") or {}
        geom = r.get("geometry") or {}
        execution = r.get("execution_estimate") or {}
        feat = dict(geom) if geom else {}
        feat.update({
            "legacy_candidate_score": legacy.get("score"),
            "ev": legacy.get("ev"),
            "ev_per_risk": legacy.get("ev_per_risk"),
            "prob_profit": legacy.get("prob_profit"),
            "prob_touch_short": legacy.get("prob_touch_short"),
            "liquidity_score": legacy.get("liquidity_score"),
            "wall_safety": legacy.get("wall_safety"),
            "gamma_safety": legacy.get("gamma_safety"),
            "touch_safety": legacy.get("touch_safety"),
            "credit": legacy.get("credit"),
            "max_loss": legacy.get("max_loss"),
            "family_code": float(
                sum(ord(ch) for ch in (r.get("family") or "")) % 97)
            if r.get("family") else None,
            "is_debit": 1.0 if (legacy.get("credit") or 0) < 0 else 0.0,
        })
        if execution:
            feat["expected_fill_credit"] = execution.get("net_expected_credit")
            feat["fill_fraction_expected"] = execution.get(
                "fill_fraction_expected")
        # session_date is embedded in snapshot_id as "DATE|..." by convention
        snap = r["snapshot_id"]
        session = snap.split("|", 1)[0] if "|" in snap else snap[:10]
        append_settled_candidate(
            frame,
            candidate_id=r["candidate_id"],
            snapshot_id=snap,
            session_date=session,
            feature_row=feat,
            outcome=outcome,
            fill_uncertainty=fill_uncertainty_from_execution(execution),
            capital=float(legacy.get("max_loss") or 0.0),
            pnl_key=pnl_key,
        )
    return frame


def grouped_snapshot_folds(
    snapshot_ids: Sequence[str],
    session_dates: Sequence[str],
    n_folds: int = 3,
    embargo_sessions: int = 1,
    min_train_sessions: int = 2,
) -> list:
    """
    Expanding walk-forward folds that keep every snapshot wholly on one side.

    Fold membership is decided at the SESSION level (same as
    prediction.training.grouped_session_folds); every snapshot belonging to a
    session moves with that session. This guarantees candidates from one
    snapshot_id never cross the train/test boundary (PR 8 AC).
    """
    from prediction.training import grouped_session_folds
    if len(snapshot_ids) != len(session_dates):
        raise ValueError("snapshot_ids and session_dates length mismatch")
    sessions = list(session_dates)
    session_folds = grouped_session_folds(
        sessions, n_folds=n_folds, embargo_sessions=embargo_sessions,
        min_train_sessions=min_train_sessions)

    out = []
    for fold in session_folds:
        train_s = set(fold["train_sessions"])
        test_s = set(fold["test_sessions"])
        train_idx = [i for i, s in enumerate(sessions) if s in train_s]
        test_idx = [i for i, s in enumerate(sessions) if s in test_s]
        train_snaps = {snapshot_ids[i] for i in train_idx}
        test_snaps = {snapshot_ids[i] for i in test_idx}
        # Hard invariant: no snapshot on both sides
        leaked = train_snaps & test_snaps
        if leaked:
            raise AssertionError(
                f"snapshot leaked across fold: {sorted(leaked)[:3]}")
        out.append({
            "train_sessions": fold["train_sessions"],
            "test_sessions": fold["test_sessions"],
            "embargoed_sessions": fold["embargoed_sessions"],
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_snapshots": sorted(train_snaps),
            "test_snapshots": sorted(test_snaps),
        })
    return out


def assert_snapshots_not_split(
    snapshot_ids: Sequence[str],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """Raise if any snapshot_id appears in both train and test index sets."""
    train_snaps = {snapshot_ids[i] for i in train_indices}
    test_snaps = {snapshot_ids[i] for i in test_indices}
    leaked = train_snaps & test_snaps
    if leaked:
        raise AssertionError(
            f"candidates from snapshot(s) split across folds: {sorted(leaked)}")


def candidate_id_for(snapshot_id: str, candidate) -> str:
    return make_candidate_id(
        snapshot_id, getattr(candidate, "family", "unknown"),
        legs_as_dicts(candidate.legs))
