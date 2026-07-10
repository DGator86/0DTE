"""
tests/test_grouped_training.py
==============================
PR 4 acceptance — session-grouped training pipeline:
  * walk-forward folds are built from COMPLETE sessions with an embargo —
    no session is ever split or shared between train and test;
  * the training frame joins the canonical store and adds as-of-safe
    derived context (time encodings, previous-return);
  * direction training reports out-of-sample metrics against the required
    baselines and the learnable signal beats the base rate;
  * PredictionBundle assembly: probabilities in bounds, quantiles ordered,
    model versions recorded;
  * shadow predictions are journaled to prediction_outputs (mode="shadow")
    and touch nothing else — no live policy effect;
  * the whole pipeline is deterministic.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from prediction.contracts import PredictionBundle
from prediction.dataset import FEATURE_VERSION, LABEL_VERSION, build_observation
from prediction.models.direction import DirectionModelConfig
from prediction.models.return_quantiles import ReturnQuantileConfig
from prediction.models.volatility import VolatilityModelConfig
from prediction.storage import PredictionStore
from prediction.training import (build_prediction_bundle,
                                 grouped_session_folds, load_training_frame,
                                 run_shadow_predictions, train_direction_models,
                                 train_model_group)

ET = ZoneInfo("America/New_York")

# July 2026 NYSE trading days (2026-07-03 is the Independence Day holiday)
SESSIONS = ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09",
            "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15",
            "2026-07-16", "2026-07-17", "2026-07-20", "2026-07-21"]

D_CFG = DirectionModelConfig(c_grid=(1.0,), l1_ratio_grid=(0.0,),
                             class_weight_options=(None,), max_iter=300)
Q_CFG = ReturnQuantileConfig(max_iter=50, min_samples_leaf=20)
V_CFG = VolatilityModelConfig(max_iter=50, min_samples_leaf=20)


def _make_store(tmp_path, per_session=30, seed=3, unlabeled_tail=0):
    """Synthetic canonical dataset: feature 'signal' drives forward returns."""
    store = PredictionStore(db_path=str(tmp_path / "pred.sqlite"))
    rng = np.random.default_rng(seed)
    for date in SESSIONS:
        base = dt.datetime.fromisoformat(f"{date}T10:00:00").replace(tzinfo=ET)
        for i in range(per_session):
            ts = base + dt.timedelta(minutes=i)
            sig = rng.standard_normal()
            spot = 600.0 + rng.standard_normal() * 2.0
            obs = build_observation(
                "SPY", ts, spot,
                features={"signal": sig,
                          "noise": rng.standard_normal(),
                          "implied_remaining_move": 0.006,
                          "sometimes_missing": (rng.standard_normal()
                                                if rng.uniform() > 0.25
                                                else None)},
                quality={"feature_coverage": 0.9},
                source_seq=i)
            store.log_feature_snapshot(obs)
            if unlabeled_tail and i >= per_session - unlabeled_tail:
                continue                          # simulate unsettled ticks
            labels = {}
            for h in ("5m", "15m", "30m", "60m", "close"):
                ret = 0.002 * sig + rng.standard_normal() * 0.0005
                labels[f"fwd_return_{h}"] = ret
                labels[f"up_{h}"] = int(ret > 0)
            labels["remaining_realized_move"] = abs(
                0.004 * sig) + 0.001 + rng.uniform() * 0.001
            store.log_labels(obs.snapshot_id, labels, LABEL_VERSION)
    return store


class TestGroupedSessionFolds:
    def test_no_overlap_no_split_embargo(self):
        folds = grouped_session_folds(
            [s for s in SESSIONS for _ in range(5)], n_folds=3,
            embargo_sessions=1)
        assert len(folds) == 3
        for f in folds:
            train, test = set(f["train_sessions"]), set(f["test_sessions"])
            assert train and test
            assert not train & test
            # embargoed sessions sit strictly between train and test
            for e in f["embargoed_sessions"]:
                assert e not in train and e not in test
                assert max(f["train_sessions"]) < e < min(f["test_sessions"])
            # strict time ordering: every train session precedes every test
            assert max(f["train_sessions"]) < min(f["test_sessions"])

    def test_expanding_train_windows(self):
        folds = grouped_session_folds(SESSIONS, n_folds=3, embargo_sessions=1)
        sizes = [len(f["train_sessions"]) for f in folds]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_all_test_blocks_disjoint_and_cover_tail(self):
        folds = grouped_session_folds(SESSIONS, n_folds=3, embargo_sessions=1)
        seen = []
        for f in folds:
            assert not set(seen) & set(f["test_sessions"])
            seen.extend(f["test_sessions"])
        assert seen[-1] == SESSIONS[-1]

    def test_too_few_sessions_raises(self):
        with pytest.raises(ValueError):
            grouped_session_folds(["a", "b", "c"], n_folds=3,
                                  embargo_sessions=1)


class TestTrainingFrame:
    def test_join_and_derived_features(self, tmp_path):
        store = _make_store(tmp_path, per_session=5)
        frame = load_training_frame(store)
        assert len(frame) == len(SESSIONS) * 5
        assert sorted(set(frame.sessions)) == SESSIONS
        row = frame.rows[1]
        for key in ("signal", "tod_sin", "tod_cos", "minute_of_day",
                    "day_of_week", "minutes_since_open", "minutes_to_close",
                    "prev_return_1m"):
            assert key in row
        # first row of each session has no previous return (as-of safety)
        first_idx = [frame.sessions.index(s) for s in SESSIONS]
        for i in first_idx:
            assert frame.rows[i]["prev_return_1m"] is None
        assert isinstance(frame.rows[1]["prev_return_1m"], float)

    def test_unlabeled_rows_excluded_by_default(self, tmp_path):
        store = _make_store(tmp_path, per_session=6, unlabeled_tail=2)
        frame = load_training_frame(store)
        assert len(frame) == len(SESSIONS) * 4
        frame_all = load_training_frame(store, require_labels=False)
        assert len(frame_all) == len(SESSIONS) * 6

    def test_target_mask_skips_missing_labels(self, tmp_path):
        store = _make_store(tmp_path, per_session=5)
        frame = load_training_frame(store)
        mask, y = frame.target("up_30m")
        assert mask.all()
        assert set(np.unique(y[mask])) <= {0.0, 1.0}
        mask_none, _ = frame.target("does_not_exist")
        assert not mask_none.any()


class TestDirectionTraining:
    def test_oos_beats_base_rate_with_full_provenance(self, tmp_path):
        store = _make_store(tmp_path)
        frame = load_training_frame(store)
        out = train_direction_models(frame, horizons=("30m",), config=D_CFG)
        rep = out["horizons"]["30m"]
        assert rep["oos"] is not None
        assert rep["baselines"] is not None
        # the learnable signal must beat every naive baseline on Brier
        for name, m in rep["baselines"].items():
            assert rep["oos"]["brier"] < m["brier"], name
        # every fold's test sessions are disjoint from its train sessions
        for fold, metrics in zip(out["folds"], rep["fold_metrics"]):
            if "skipped" in metrics:
                continue
            assert metrics["test_sessions"] == fold["test_sessions"]
            assert not set(fold["train_sessions"]) & set(fold["test_sessions"])
        # final model saw all sessions (shadow use only)
        final = out["models"]["30m"]
        assert final.metadata["train_sessions"] == SESSIONS

    def test_calibration_sessions_never_touch_test(self, tmp_path):
        store = _make_store(tmp_path)
        frame = load_training_frame(store)
        out = train_direction_models(frame, horizons=("30m",), config=D_CFG)
        # rebuild each fold's model to inspect its inner split provenance
        from prediction.models.direction import DirectionModel
        import dataclasses as dc
        for fold in out["folds"]:
            tr = [i for i, s in enumerate(frame.sessions)
                  if s in set(fold["train_sessions"])]
            m = DirectionModel(config=dc.replace(D_CFG, horizon="30m")).fit(
                [frame.rows[i] for i in tr],
                [int(frame.labels[i]["up_30m"]) for i in tr],
                [frame.sessions[i] for i in tr])
            cal = set(m.metadata["calibration_sessions"])
            assert cal <= set(fold["train_sessions"])
            assert not cal & set(fold["test_sessions"])


class TestModelGroupAndBundle:
    def test_bundle_bounds_ordering_versions(self, tmp_path):
        store = _make_store(tmp_path)
        frame = load_training_frame(store)
        out = train_model_group(frame, direction_config=D_CFG,
                                quantile_config=Q_CFG,
                                volatility_config=V_CFG)
        group = out["group"]
        assert set(group.direction) == {"5m", "15m", "30m", "60m", "close"}
        assert set(group.quantiles) == {"30m", "60m", "close"}
        assert group.volatility is not None

        bundle = build_prediction_bundle(
            group, frame.rows[0], snapshot_id=frame.snapshot_ids[0],
            ts=frame.ts[0], session_date=frame.sessions[0],
            quality=frame.quality[0])
        assert isinstance(bundle, PredictionBundle)   # __post_init__ validates
        for h in ("5m", "15m", "30m", "60m", "close"):
            p = getattr(bundle, f"p_up_{h}")
            assert p is not None and 0.0 <= p <= 1.0
        for h in ("30m", "close"):
            q10 = getattr(bundle, f"return_q10_{h}")
            q50 = getattr(bundle, f"return_q50_{h}")
            q90 = getattr(bundle, f"return_q90_{h}")
            assert q10 <= q50 <= q90                   # ordered quantiles
        assert bundle.expected_return_60m is not None
        assert bundle.expected_realized_move_close is not None
        assert bundle.expected_realized_move_close >= 0.0
        assert bundle.uncertainty is not None
        assert 0.0 <= bundle.uncertainty <= 1.0
        assert bundle.feature_version == FEATURE_VERSION
        assert bundle.model_versions["group"] == group.group_version
        assert bundle.model_versions["direction_30m"] == "up_30m"
        assert bundle.feature_coverage == pytest.approx(0.9)

    def test_signal_moves_direction_probability(self, tmp_path):
        store = _make_store(tmp_path)
        frame = load_training_frame(store)
        out = train_model_group(frame, direction_config=D_CFG,
                                quantile_config=Q_CFG,
                                volatility_config=V_CFG)
        group = out["group"]
        up_row = dict(frame.rows[0], signal=2.5)
        dn_row = dict(frame.rows[0], signal=-2.5)
        p_up = group.direction["30m"].predict_proba([up_row])[0]
        p_dn = group.direction["30m"].predict_proba([dn_row])[0]
        assert p_up > p_dn

    def test_deterministic_end_to_end(self, tmp_path):
        store = _make_store(tmp_path)
        frame = load_training_frame(store)
        bundles = []
        for _ in range(2):
            out = train_model_group(frame, direction_config=D_CFG,
                                    quantile_config=Q_CFG,
                                    volatility_config=V_CFG)
            bundles.append(build_prediction_bundle(
                out["group"], frame.rows[5],
                snapshot_id=frame.snapshot_ids[5], ts=frame.ts[5],
                session_date=frame.sessions[5], quality=frame.quality[5]))
        assert bundles[0].to_dict() == bundles[1].to_dict()


class TestShadowPredictions:
    def test_written_to_prediction_outputs_only(self, tmp_path):
        store = _make_store(tmp_path, per_session=10, unlabeled_tail=2)
        frame = load_training_frame(store)
        out = train_model_group(frame, direction_config=D_CFG,
                                quantile_config=Q_CFG,
                                volatility_config=V_CFG)
        n = run_shadow_predictions(store, out["group"])
        # every observation gets a shadow prediction, labeled or not
        assert n == len(SESSIONS) * 10
        rows = store.fetch_predictions(mode="shadow")
        assert len(rows) == n
        first = rows[0]
        assert first["model_group_version"] == out["group"].group_version
        assert first["predictions"]["snapshot_id"] == first["snapshot_id"]
        p30 = first["predictions"]["p_up_30m"]
        assert p30 is not None and 0.0 <= p30 <= 1.0
        # no policy effect: candidate/outcome tables remain untouched
        assert store.conn.execute(
            "SELECT COUNT(*) FROM candidate_snapshots").fetchone()[0] == 0
        assert store.conn.execute(
            "SELECT COUNT(*) FROM candidate_outcomes").fetchone()[0] == 0

    def test_single_session_filter(self, tmp_path):
        store = _make_store(tmp_path, per_session=5)
        frame = load_training_frame(store)
        out = train_model_group(frame, direction_config=D_CFG,
                                quantile_config=Q_CFG,
                                volatility_config=V_CFG)
        n = run_shadow_predictions(store, out["group"],
                                   session_date=SESSIONS[0])
        assert n == 5
        assert all(r["predictions"]["session_date"] == SESSIONS[0]
                   for r in store.fetch_predictions(mode="shadow"))
