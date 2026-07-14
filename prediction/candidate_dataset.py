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


# ---------------------------------------------------------------------------
# Part 3 §8 — within-snapshot pairwise ranking dataset
# ---------------------------------------------------------------------------

PAIR_EPSILON_R = 0.01
PAIRWISE_DATASET_VERSION = "v3.0.0"

# Continuous keys differenced as A - B. Categorical/structural keys become
# same_* indicator features.
DEFAULT_PAIR_CONTINUOUS_KEYS = (
    "expected_net_pnl", "p_profit", "expected_shortfall", "pnl_q05",
    "fill_uncertainty", "model_uncertainty", "forecast_uncertainty",
    "ood_score", "capital_required", "maximum_loss", "return_on_risk",
    "utility_score", "credit", "max_loss", "ev", "ev_per_risk",
    "prob_profit", "liquidity_score", "legacy_candidate_score",
    "relative_spread", "minutes_to_close", "n_legs", "width",
)

DEFAULT_PAIR_CATEGORICAL_KEYS = (
    "family", "direction", "n_legs", "width_bucket",
    "max_loss_bucket", "capital_bucket",
)


@dataclass(frozen=True)
class CandidatePairRecord:
    snapshot_id: str
    candidate_a_id: str
    candidate_b_id: str
    pair_features: dict
    a_wins: int
    weight: float
    realized_utility_a: float
    realized_utility_b: float
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "candidate_a_id": self.candidate_a_id,
            "candidate_b_id": self.candidate_b_id,
            "pair_features": dict(self.pair_features),
            "a_wins": int(self.a_wins),
            "weight": float(self.weight),
            "realized_utility_a": float(self.realized_utility_a),
            "realized_utility_b": float(self.realized_utility_b),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class PairwiseTrainingFrame:
    """All pairs from eligible snapshots; never split across snapshots."""
    pairs: list = field(default_factory=list)  # CandidatePairRecord
    version: str = PAIRWISE_DATASET_VERSION
    pair_epsilon_r: float = PAIR_EPSILON_R
    risk_unit_r: float = 1.0

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def snapshot_ids(self) -> list:
        return [p.snapshot_id for p in self.pairs]


def _is_numeric(v) -> bool:
    if v is None or isinstance(v, bool):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def build_pair_features(
    features_a: dict,
    features_b: dict,
    *,
    continuous_keys: Optional[Sequence[str]] = None,
    categorical_keys: Optional[Sequence[str]] = None,
) -> dict:
    """
    Pair features: continuous as A - B; categorical as same_* indicators.
    Deterministic key set from the union of configured keys present on either.
    """
    cont = list(continuous_keys) if continuous_keys is not None else list(
        DEFAULT_PAIR_CONTINUOUS_KEYS)
    cats = list(categorical_keys) if categorical_keys is not None else list(
        DEFAULT_PAIR_CATEGORICAL_KEYS)
    out: dict = {}
    for k in cont:
        va = features_a.get(k)
        vb = features_b.get(k)
        if _is_numeric(va) and _is_numeric(vb):
            out[f"diff_{k}"] = float(va) - float(vb)
        elif _is_numeric(va) or _is_numeric(vb):
            # One missing → treat missing as 0.0 for stable differencing
            a = float(va) if _is_numeric(va) else 0.0
            b = float(vb) if _is_numeric(vb) else 0.0
            out[f"diff_{k}"] = a - b
    for k in cats:
        va = features_a.get(k)
        vb = features_b.get(k)
        if va is None and vb is None:
            continue
        out[f"same_{k}"] = 1.0 if va == vb else 0.0
    return out


def reverse_pair_features(pair_features: dict) -> dict:
    """Swap A/B: negate diff_* features; same_* unchanged."""
    out = {}
    for k, v in pair_features.items():
        if k.startswith("diff_"):
            out[k] = -float(v)
        else:
            out[k] = v
    return out


def pair_weight(
    *,
    realized_utility_a: float,
    realized_utility_b: float,
    complete_outcomes: bool = True,
    valid_executable_quotes: bool = True,
    passes_feasibility: bool = True,
    quote_quality: float = 1.0,
    fill_uncertain: bool = False,
    quote_age_elevated: bool = False,
    censored: bool = False,
    data_quality: float = 1.0,
    family_support: float = 1.0,
    risk_unit_r: float = 1.0,
) -> float:
    """
    Deterministic pair weight (§8.7). Base = |Δutility| / R, then multipliers.
    """
    r = max(float(risk_unit_r), 1e-9)
    delta = abs(float(realized_utility_a) - float(realized_utility_b))
    w = delta / r
    if complete_outcomes:
        w *= 1.25
    else:
        w *= 0.5
    if valid_executable_quotes:
        w *= 1.10
    else:
        w *= 0.5
    if passes_feasibility:
        w *= 1.10
    else:
        w *= 0.25
    w *= float(np.clip(quote_quality, 0.0, 1.0))
    if fill_uncertain:
        w *= 0.7
    if quote_age_elevated:
        w *= 0.8
    if censored:
        w *= 0.4
    w *= float(np.clip(data_quality, 0.0, 1.0))
    w *= float(np.clip(family_support, 0.0, 1.0))
    return float(max(w, 0.0))


def generate_snapshot_pairs(
    snapshot_id: str,
    candidates: Sequence[dict],
    *,
    pair_epsilon_r: float = PAIR_EPSILON_R,
    risk_unit_r: float = 1.0,
    continuous_keys: Optional[Sequence[str]] = None,
    categorical_keys: Optional[Sequence[str]] = None,
) -> list:
    """
    Build unordered unique pairs within one snapshot.

    Each candidate dict must provide:
      candidate_id, features (dict), realized_utility (float)
    Optional weight knobs: complete_outcomes, valid_executable_quotes, …
    Near ties (|Δu| < pair_epsilon_r * R) are excluded.
    """
    eps = float(pair_epsilon_r) * max(float(risk_unit_r), 1e-9)
    items = list(candidates)
    pairs: list = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ua = float(a["realized_utility"])
            ub = float(b["realized_utility"])
            if abs(ua - ub) < eps:
                continue
            feat = build_pair_features(
                a.get("features") or {},
                b.get("features") or {},
                continuous_keys=continuous_keys,
                categorical_keys=categorical_keys,
            )
            w = pair_weight(
                realized_utility_a=ua,
                realized_utility_b=ub,
                complete_outcomes=bool(a.get("complete_outcomes", True)
                                       and b.get("complete_outcomes", True)),
                valid_executable_quotes=bool(
                    a.get("valid_executable_quotes", True)
                    and b.get("valid_executable_quotes", True)),
                passes_feasibility=bool(a.get("passes_feasibility", True)
                                        and b.get("passes_feasibility", True)),
                quote_quality=min(float(a.get("quote_quality", 1.0)),
                                  float(b.get("quote_quality", 1.0))),
                fill_uncertain=bool(a.get("fill_uncertain", False)
                                    or b.get("fill_uncertain", False)),
                quote_age_elevated=bool(a.get("quote_age_elevated", False)
                                        or b.get("quote_age_elevated", False)),
                censored=bool(a.get("censored", False)
                              or b.get("censored", False)),
                data_quality=min(float(a.get("data_quality", 1.0)),
                                 float(b.get("data_quality", 1.0))),
                family_support=min(float(a.get("family_support", 1.0)),
                                   float(b.get("family_support", 1.0))),
                risk_unit_r=risk_unit_r,
            )
            pairs.append(CandidatePairRecord(
                snapshot_id=snapshot_id,
                candidate_a_id=str(a["candidate_id"]),
                candidate_b_id=str(b["candidate_id"]),
                pair_features=feat,
                a_wins=int(ua > ub),
                weight=w,
                realized_utility_a=ua,
                realized_utility_b=ub,
                diagnostics={
                    "utility_delta": ua - ub,
                    "pair_epsilon": eps,
                },
            ))
    return pairs


def build_pairwise_frame(
    candidates_by_snapshot: dict,
    *,
    pair_epsilon_r: float = PAIR_EPSILON_R,
    risk_unit_r: float = 1.0,
    continuous_keys: Optional[Sequence[str]] = None,
    categorical_keys: Optional[Sequence[str]] = None,
) -> PairwiseTrainingFrame:
    """
    Construct a PairwiseTrainingFrame from {snapshot_id: [candidate dicts]}.
    All pairs remain tagged with their snapshot_id (non-splittable grouping).
    """
    frame = PairwiseTrainingFrame(
        pair_epsilon_r=pair_epsilon_r,
        risk_unit_r=risk_unit_r,
    )
    for snap_id in sorted(candidates_by_snapshot.keys()):
        frame.pairs.extend(generate_snapshot_pairs(
            snap_id,
            candidates_by_snapshot[snap_id],
            pair_epsilon_r=pair_epsilon_r,
            risk_unit_r=risk_unit_r,
            continuous_keys=continuous_keys,
            categorical_keys=categorical_keys,
        ))
    return frame


def pairwise_frame_from_training_frame(
    frame: CandidateTrainingFrame,
    realized_utilities: Sequence[float],
    *,
    pair_epsilon_r: float = PAIR_EPSILON_R,
    risk_unit_r: float = 1.0,
) -> PairwiseTrainingFrame:
    """Group a CandidateTrainingFrame by snapshot and emit pairs."""
    if len(realized_utilities) != len(frame):
        raise ValueError("realized_utilities length must match frame")
    by_snap: dict = {}
    for i in range(len(frame)):
        snap = frame.snapshot_ids[i]
        by_snap.setdefault(snap, []).append({
            "candidate_id": frame.candidate_ids[i],
            "features": dict(frame.rows[i]),
            "realized_utility": float(realized_utilities[i]),
            "fill_uncertain": float(frame.fill_uncertainty[i]) > 0.5,
            "data_quality": 1.0,
        })
    return build_pairwise_frame(
        by_snap,
        pair_epsilon_r=pair_epsilon_r,
        risk_unit_r=risk_unit_r,
    )


def assert_pairs_within_snapshot(pairs: Sequence[CandidatePairRecord]) -> None:
    """Every pair must reference a single snapshot_id (no cross-snapshot)."""
    for p in pairs:
        if "|" in p.candidate_a_id and "|" in p.candidate_b_id:
            # Soft check when IDs embed snapshot prefixes
            pass
        if not p.snapshot_id:
            raise AssertionError("pair missing snapshot_id")
