"""
prediction/training.py
======================
Training pipeline for the V2 baseline models
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §11, §18, PR 4).

Everything is session-grouped and time-ordered:

  * training frames come from the canonical PredictionStore dataset
    (feature_snapshots joined to observation_labels);
  * walk-forward folds are built from COMPLETE session dates with an
    embargo — no session ever appears on both sides of a boundary;
  * hyperparameters and probability calibration live INSIDE each fold's
    training sessions (DirectionModel handles its own inner embargoed
    split); test sessions only ever produce out-of-sample predictions;
  * every trained model is compared against the required naive baselines
    (base rate, previous-return sign, legacy composite, random);
  * final models trained on all sessions serve SHADOW predictions only:
    PredictionBundle rows journaled to prediction_outputs with
    mode="shadow". Nothing downstream reads them yet — no policy effect.

NOT financial advice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from prediction.contracts import PredictionBundle
from prediction.dataset import FEATURE_VERSION
from prediction.models.direction import (DIRECTION_HORIZONS, DirectionModel,
                                         DirectionModelConfig,
                                         baseline_base_rate,
                                         baseline_legacy_composite,
                                         baseline_prev_sign, baseline_random,
                                         evaluate_probabilities)
from prediction.models.return_quantiles import (QUANTILE_HORIZONS,
                                                ReturnQuantileConfig,
                                                ReturnQuantileModel)
from prediction.models.volatility import VolatilityModel, VolatilityModelConfig

ET = ZoneInfo("America/New_York")

MODEL_GROUP_VERSION = "v2.0.0-pr4"

# feature name carrying the legacy 0-100 direction composite when the live
# loop captured it (unified_loop signals); absent in offline recordings
LEGACY_COMPOSITE_FEATURE = "regime_bias_value"
PREV_RETURN_FEATURE = "prev_return_1m"


# --------------------------------------------------------------------------- #
# Training frame                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class TrainingFrame:
    snapshot_ids: list
    sessions: list                 # session_date per row (grouping key)
    ts: list                       # ET iso timestamps per row
    rows: list                     # feature dicts (raw + derived time context)
    labels: list                   # label dicts per row
    quality: list                  # quality dicts per row

    def __len__(self) -> int:
        return len(self.rows)

    def target(self, key: str) -> tuple:
        """(mask, y) for one label key; mask excludes rows where the label
        is missing (e.g. horizons past the close)."""
        vals = [lb.get(key) for lb in self.labels]
        mask = np.array([isinstance(v, (int, float)) for v in vals])
        y = np.array([float(v) if isinstance(v, (int, float)) else np.nan
                      for v in vals])
        return mask, y


def _time_context(ts_iso: str) -> dict:
    t = dt.datetime.fromisoformat(ts_iso).astimezone(ET)
    minute_of_day = t.hour * 60 + t.minute
    return {
        "minute_of_day": float(minute_of_day),
        "tod_sin": math.sin(2 * math.pi * minute_of_day / 1440.0),
        "tod_cos": math.cos(2 * math.pi * minute_of_day / 1440.0),
        "day_of_week": float(t.weekday()),
    }


def load_training_frame(store, feature_version: str = FEATURE_VERSION,
                        require_labels: bool = True) -> TrainingFrame:
    """
    Join feature_snapshots to observation_labels and add derived, as-of-safe
    context features: time-of-day encodings, minutes since open / to close,
    and the previous 1-minute spot return within the session (past data only
    — the baseline input for the previous-return-sign comparison).

    require_labels=False keeps unlabeled observations (labels default to {})
    — the shadow-inference path, where settlement has not happened yet.
    """
    snaps = [s for s in store.fetch_feature_snapshots()
             if s["feature_version"] == feature_version]
    labels = {r["snapshot_id"]: r["labels"] for r in store.fetch_labels()}

    frame = TrainingFrame([], [], [], [], [], [])
    prev_spot: dict = {}                       # session -> last seen spot
    for s in snaps:                            # store orders by session, ts
        lb = labels.get(s["snapshot_id"])
        session = s["session_date"]
        spot = s["spot"]
        prev = prev_spot.get(session)
        prev_spot[session] = spot
        if lb is None:
            if require_labels:
                continue
            lb = {}
        row = dict(s["features"])
        row.update(_time_context(s["ts"]))
        row["minutes_since_open"] = s["minutes_since_open"]
        row["minutes_to_close"] = s["minutes_to_close"]
        row[PREV_RETURN_FEATURE] = ((spot / prev - 1.0)
                                    if (prev and spot) else None)
        frame.snapshot_ids.append(s["snapshot_id"])
        frame.sessions.append(session)
        frame.ts.append(s["ts"])
        frame.rows.append(row)
        frame.labels.append(lb)
        frame.quality.append(s["quality"])
    return frame


# --------------------------------------------------------------------------- #
# Session-grouped walk-forward folds                                           #
# --------------------------------------------------------------------------- #
def grouped_session_folds(sessions: Sequence[str], n_folds: int = 3,
                          embargo_sessions: int = 1,
                          min_train_sessions: int = 2) -> list:
    """
    Expanding walk-forward folds over COMPLETE unique sessions:
    the last `n_folds` roughly equal blocks of sessions are test blocks;
    each fold trains on every session that ends at least `embargo_sessions`
    whole sessions before its test block starts. Sessions are never split
    and never appear on both sides (§18.1-18.4).
    """
    uniq = sorted(set(sessions))
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    # size test blocks so the first fold still has a training prefix
    max_test_total = len(uniq) - (min_train_sessions + embargo_sessions)
    if max_test_total < n_folds:
        raise ValueError(
            f"not enough sessions ({len(uniq)}) for {n_folds} folds with "
            f"embargo={embargo_sessions} and min_train={min_train_sessions}")
    block = max(1, max_test_total // n_folds)
    folds = []
    test_start = len(uniq) - n_folds * block
    for i in range(n_folds):
        lo = test_start + i * block
        hi = lo + block if i < n_folds - 1 else len(uniq)
        test = uniq[lo:hi]
        train = uniq[:max(lo - embargo_sessions, 0)]
        embargoed = uniq[max(lo - embargo_sessions, 0):lo]
        if len(train) < min_train_sessions:
            raise ValueError("fold has too few training sessions")
        assert not (set(train) & set(test)), "session leaked across the fold"
        folds.append({"train_sessions": train, "test_sessions": test,
                      "embargoed_sessions": embargoed})
    return folds


def _mask_for(sessions: Sequence[str], keep: Sequence[str]) -> np.ndarray:
    keep_set = set(keep)
    return np.array([s in keep_set for s in sessions])


def _subset(seq, mask) -> list:
    return [x for x, m in zip(seq, mask) if m]


# --------------------------------------------------------------------------- #
# Direction training with baseline comparison                                  #
# --------------------------------------------------------------------------- #
def train_direction_models(frame: TrainingFrame,
                           horizons: Sequence[str] = DIRECTION_HORIZONS,
                           config: Optional[DirectionModelConfig] = None,
                           n_folds: int = 3,
                           embargo_sessions: int = 1) -> dict:
    """
    Walk-forward evaluation + final fit for each direction horizon.

    Per fold: a fresh DirectionModel is fitted on the fold's TRAIN sessions
    only (its inner calibration split also stays inside them) and produces
    out-of-sample probabilities for the TEST sessions, which are scored
    against the required baselines. The returned final models are trained on
    ALL sessions and are intended for shadow inference only.
    """
    base_cfg = config or DirectionModelConfig()
    out = {"folds": grouped_session_folds(frame.sessions, n_folds,
                                          embargo_sessions),
           "horizons": {}, "models": {}}
    for h in horizons:
        mask_lbl, y_all = frame.target(f"up_{h}")
        report = {"n_labeled": int(mask_lbl.sum()), "fold_metrics": [],
                  "oos": None, "baselines": None}
        oos_p, oos_y, oos_prev, oos_comp = [], [], [], []
        train_base_rates = []
        for fold in out["folds"]:
            tr = _mask_for(frame.sessions, fold["train_sessions"]) & mask_lbl
            te = _mask_for(frame.sessions, fold["test_sessions"]) & mask_lbl
            if tr.sum() < 10 or te.sum() < 1:
                report["fold_metrics"].append({"skipped": "too few rows"})
                continue
            cfg = dataclasses.replace(base_cfg, horizon=h)
            model = DirectionModel(config=cfg).fit(
                _subset(frame.rows, tr), y_all[tr].astype(int),
                _subset(frame.sessions, tr))
            p = model.predict_proba(_subset(frame.rows, te))
            m = evaluate_probabilities(y_all[te], p)
            m["test_sessions"] = fold["test_sessions"]
            report["fold_metrics"].append(m)
            oos_p.extend(p)
            oos_y.extend(y_all[te])
            train_base_rates.append(float(np.mean(y_all[tr])))
            oos_prev.extend(r.get(PREV_RETURN_FEATURE)
                            for r in _subset(frame.rows, te))
            oos_comp.extend(r.get(LEGACY_COMPOSITE_FEATURE)
                            for r in _subset(frame.rows, te))
        if oos_y:
            oos_y_arr = np.asarray(oos_y)
            base = np.full(len(oos_y_arr), float(np.mean(train_base_rates)))
            report["oos"] = evaluate_probabilities(oos_y_arr,
                                                   np.asarray(oos_p))
            report["baselines"] = {
                "base_rate": evaluate_probabilities(oos_y_arr, base),
                "prev_sign": evaluate_probabilities(
                    oos_y_arr, baseline_prev_sign(oos_prev, base)),
                "legacy_composite": evaluate_probabilities(
                    oos_y_arr, baseline_legacy_composite(oos_comp, base)),
                "random": evaluate_probabilities(
                    oos_y_arr, baseline_random(len(oos_y_arr))),
            }
        # final model on ALL labeled sessions — shadow inference only
        if mask_lbl.sum() >= 10 and len(np.unique(y_all[mask_lbl])) >= 2:
            cfg = dataclasses.replace(base_cfg, horizon=h)
            out["models"][h] = DirectionModel(config=cfg).fit(
                _subset(frame.rows, mask_lbl),
                y_all[mask_lbl].astype(int),
                _subset(frame.sessions, mask_lbl))
        out["horizons"][h] = report
    return out


def train_quantile_models(frame: TrainingFrame,
                          horizons: Sequence[str] = QUANTILE_HORIZONS,
                          config: Optional[ReturnQuantileConfig] = None,
                          n_folds: int = 3,
                          embargo_sessions: int = 1) -> dict:
    out = {"horizons": {}, "models": {}}
    folds = grouped_session_folds(frame.sessions, n_folds, embargo_sessions)
    base_cfg = config or ReturnQuantileConfig()
    for h in horizons:
        mask_lbl, y_all = frame.target(f"fwd_return_{h}")
        report = {"n_labeled": int(mask_lbl.sum()), "fold_metrics": []}
        for fold in folds:
            tr = _mask_for(frame.sessions, fold["train_sessions"]) & mask_lbl
            te = _mask_for(frame.sessions, fold["test_sessions"]) & mask_lbl
            if tr.sum() < 10 or te.sum() < 1:
                report["fold_metrics"].append({"skipped": "too few rows"})
                continue
            cfg = dataclasses.replace(base_cfg, horizon=h)
            model = ReturnQuantileModel(config=cfg).fit(
                _subset(frame.rows, tr), y_all[tr],
                _subset(frame.sessions, tr))
            m = model.evaluate(_subset(frame.rows, te), y_all[te])
            m["test_sessions"] = fold["test_sessions"]
            report["fold_metrics"].append(m)
        if mask_lbl.sum() >= 10:
            cfg = dataclasses.replace(base_cfg, horizon=h)
            out["models"][h] = ReturnQuantileModel(config=cfg).fit(
                _subset(frame.rows, mask_lbl), y_all[mask_lbl],
                _subset(frame.sessions, mask_lbl))
        out["horizons"][h] = report
    return out


def train_volatility_model(frame: TrainingFrame,
                           config: Optional[VolatilityModelConfig] = None,
                           n_folds: int = 3,
                           embargo_sessions: int = 1) -> dict:
    cfg = config or VolatilityModelConfig()
    mask_lbl, y_all = frame.target(cfg.target)
    out = {"n_labeled": int(mask_lbl.sum()), "fold_metrics": [],
           "model": None}
    for fold in grouped_session_folds(frame.sessions, n_folds,
                                      embargo_sessions):
        tr = _mask_for(frame.sessions, fold["train_sessions"]) & mask_lbl
        te = _mask_for(frame.sessions, fold["test_sessions"]) & mask_lbl
        if tr.sum() < 10 or te.sum() < 1:
            out["fold_metrics"].append({"skipped": "too few rows"})
            continue
        model = VolatilityModel(config=cfg).fit(
            _subset(frame.rows, tr), y_all[tr], _subset(frame.sessions, tr))
        m = model.evaluate(_subset(frame.rows, te), y_all[te])
        m["test_sessions"] = fold["test_sessions"]
        out["fold_metrics"].append(m)
    if mask_lbl.sum() >= 10:
        out["model"] = VolatilityModel(config=cfg).fit(
            _subset(frame.rows, mask_lbl), y_all[mask_lbl],
            _subset(frame.sessions, mask_lbl))
    return out


# --------------------------------------------------------------------------- #
# Model group + PredictionBundle assembly                                      #
# --------------------------------------------------------------------------- #
@dataclass
class PredictionModelGroup:
    """The trained model set behind one model_group_version."""
    direction: dict = field(default_factory=dict)     # horizon -> DirectionModel
    quantiles: dict = field(default_factory=dict)     # horizon -> ReturnQuantileModel
    volatility: Optional[VolatilityModel] = None
    feature_version: str = FEATURE_VERSION
    group_version: str = MODEL_GROUP_VERSION

    def model_versions(self) -> dict:
        out = {}
        for h, m in self.direction.items():
            out[f"direction_{h}"] = m.metadata.get("target", f"up_{h}")
        for h in self.quantiles:
            out[f"quantiles_{h}"] = f"fwd_return_{h}"
        if self.volatility is not None:
            out["volatility"] = self.volatility.metadata.get("target", "")
        out["group"] = self.group_version
        return out

    def uncertainty(self) -> Optional[float]:
        vals = [m.metadata.get("uncertainty")
                for m in self.direction.values()
                if m.metadata.get("uncertainty") is not None]
        return float(np.mean(vals)) if vals else None


def train_model_group(frame: TrainingFrame, *,
                      direction_config: Optional[DirectionModelConfig] = None,
                      quantile_config: Optional[ReturnQuantileConfig] = None,
                      volatility_config: Optional[VolatilityModelConfig] = None,
                      n_folds: int = 3, embargo_sessions: int = 1) -> dict:
    """Train the full PR 4 suite; returns {"group", "reports"}."""
    d = train_direction_models(frame, config=direction_config,
                               n_folds=n_folds,
                               embargo_sessions=embargo_sessions)
    q = train_quantile_models(frame, config=quantile_config,
                              n_folds=n_folds,
                              embargo_sessions=embargo_sessions)
    v = train_volatility_model(frame, config=volatility_config,
                               n_folds=n_folds,
                               embargo_sessions=embargo_sessions)
    group = PredictionModelGroup(direction=d["models"], quantiles=q["models"],
                                 volatility=v["model"])
    return {"group": group,
            "reports": {"direction": d["horizons"], "quantiles": q["horizons"],
                        "volatility": {k: v[k] for k in
                                       ("n_labeled", "fold_metrics")},
                        "folds": d["folds"]}}


def _f(x) -> Optional[float]:
    """numpy scalar -> plain finite float, else None (bundle contract)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def build_prediction_bundle(group: PredictionModelGroup, row: dict, *,
                            snapshot_id: str, ts: str, session_date: str,
                            symbol: str = "SPY",
                            quality: Optional[dict] = None
                            ) -> PredictionBundle:
    """
    Assemble one PredictionBundle from the model group for one observation
    row. Only forecast inputs enter (§6.3) — the caller cannot pass any
    selected structure, family, conviction, or gate result because there is
    nowhere to put them. Fields whose model is missing stay None.
    """
    kw: dict = {}
    for h, model in group.direction.items():
        kw[f"p_up_{h}"] = _f(model.predict_proba([row])[0])
    for h, model in group.quantiles.items():
        q = model.predict([row])
        if h in ("30m", "close"):
            kw[f"return_q10_{h}"] = _f(q["q10"][0])
            kw[f"return_q50_{h}"] = _f(q["q50"][0])
            kw[f"return_q90_{h}"] = _f(q["q90"][0])
        if h in ("30m", "60m", "close"):
            kw[f"expected_return_{h}"] = _f(q["q50"][0])
    if group.volatility is not None:
        v = group.volatility.predict([row])
        kw["expected_realized_move_close"] = _f(v["expected_move"][0])
    cov = (quality or {}).get("feature_coverage")
    return PredictionBundle(
        snapshot_id=snapshot_id, ts=ts, session_date=session_date,
        symbol=symbol,
        uncertainty=_f(group.uncertainty()),
        data_quality=_f(cov), feature_coverage=_f(cov),
        feature_version=group.feature_version,
        model_versions=group.model_versions(),
        **kw)


# --------------------------------------------------------------------------- #
# Shadow inference — journaled, zero policy effect                             #
# --------------------------------------------------------------------------- #
def run_shadow_predictions(store, group: PredictionModelGroup,
                           session_date: Optional[str] = None,
                           mode: str = "shadow") -> int:
    """
    Predict every stored observation (optionally one session) with the model
    group and journal the PredictionBundle to prediction_outputs with
    mode="shadow". Nothing in the live decision path reads these rows —
    the legacy engine remains authoritative (§23.2).
    """
    frame = load_training_frame(store, group.feature_version,
                                require_labels=False)
    n = 0
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for i in range(len(frame)):
        if session_date and frame.sessions[i] != session_date:
            continue
        bundle = build_prediction_bundle(
            group, frame.rows[i],
            snapshot_id=frame.snapshot_ids[i], ts=frame.ts[i],
            session_date=frame.sessions[i], quality=frame.quality[i])
        store.log_prediction(
            snapshot_id=bundle.snapshot_id,
            model_group_version=group.group_version,
            predictions=bundle.to_dict(),
            uncertainty=bundle.uncertainty,
            generated_at=generated_at,
            mode=mode)
        n += 1
    return n
