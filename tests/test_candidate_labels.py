"""
tests/test_candidate_labels.py
==============================
PR 3 acceptance — candidate outcome records:
  * settlement P&L from midpoint entry economics (per-share intrinsic math,
    same convention as journal.realized_pnl);
  * expected/conservative fill P&L stays None until PR 6 (absent, not faked);
  * path MFE/MAE and target/stop first-passage from intrinsic bar marks,
    with same-bar ambiguity resolved conservatively;
  * stable candidate ids and SQLite round-trip via PredictionStore.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from journal import realized_pnl
from prediction.labels import SessionLabeler, candidate_outcome_labels
from prediction.storage import PredictionStore, make_candidate_id

UTC = dt.timezone.utc
SESSION_START = dt.datetime(2026, 7, 6, 13, 30)

# short 599P / long 598P put credit spread, 0.30 credit
PCS_LEGS = [{"strike": 599.0, "kind": "P", "qty": -1},
            {"strike": 598.0, "kind": "P", "qty": 1}]
PCS_CREDIT = 0.30


def _labeler(closes, highs=None, lows=None) -> SessionLabeler:
    closes = np.asarray(closes, dtype=float)
    ts = (np.datetime64(SESSION_START) +
          np.arange(len(closes)) * np.timedelta64(1, "m"))
    return SessionLabeler(
        ts=ts.astype("datetime64[ns]"),
        high=np.asarray(highs if highs is not None else closes, dtype=float),
        low=np.asarray(lows if lows is not None else closes, dtype=float),
        close=closes,
    )


def _obs(minute: int) -> dt.datetime:
    return (SESSION_START + dt.timedelta(minutes=minute)).replace(tzinfo=UTC)


class TestSettlementPnl:
    def test_otm_settle_keeps_credit(self):
        out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 602.0)
        assert out["pnl_mid"] == pytest.approx(0.30)
        assert out["settled"] == 1
        assert out["settlement_price"] == 602.0

    def test_max_loss_settle(self):
        out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 590.0)
        assert out["pnl_mid"] == pytest.approx(0.30 - 1.0)

    def test_matches_journal_realized_pnl(self):
        for settle in (590.0, 598.5, 599.0, 602.0):
            out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, settle)
            assert out["pnl_mid"] == pytest.approx(
                realized_pnl(PCS_LEGS, PCS_CREDIT, settle))

    def test_fill_adjusted_pnl_absent_until_pr6(self):
        out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 602.0)
        assert out["pnl_expected_fill"] is None
        assert out["pnl_conservative"] is None
        assert out["pnl_policy"] is None

    def test_return_on_risk_and_capital(self):
        out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 602.0,
                                       max_loss=0.70, capital=0.70)
        assert out["return_on_risk"] == pytest.approx(0.30 / 0.70)
        assert out["return_on_capital"] == pytest.approx(0.30 / 0.70)


class TestPathOutcomes:
    def test_mfe_mae_from_path(self):
        # price dips to 597 mid-session (spread deep ITM = worst mark),
        # recovers to 602 (spread worthless = best mark, +credit)
        closes = [600.0] * 120
        lows = [600.0] * 120
        lows[40] = 597.0
        closes[-1] = 602.0
        lab = _labeler(closes, None, lows)
        out = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 602.0,
                                       labeler=lab, entry_ts=_obs(0))
        assert out["mfe"] == pytest.approx(0.30)               # full credit
        # at S=597: short 599P intrinsic -2.0, long 598P intrinsic +1.0
        assert out["mae"] == pytest.approx(0.30 - 1.0)
        assert out["time_in_trade_min"] == 119.0

    def test_stop_then_recovery_is_stop_first(self):
        # intrinsic marks: at 598.7 the spread P&L is ~0.0; the dip to 597
        # marks -0.70 (stop); the recovery to 602 would mark full credit —
        # but the stop happened FIRST and must be the recorded event
        closes = [598.7] * 120
        lows = [598.7] * 120
        lows[40] = 597.0                                        # stop territory
        closes[-1] = 602.0                                      # recovers late
        lab = _labeler(closes, None, lows)
        out = candidate_outcome_labels(
            PCS_LEGS, PCS_CREDIT, 602.0, labeler=lab, entry_ts=_obs(0),
            target_pnl=0.25, stop_pnl=-0.40)
        assert out["stop_hit"] == 1
        assert out["first_event"] == "stop"
        assert out["time_in_trade_min"] == 40.0                 # exits at stop

    def test_target_first(self):
        closes = [602.0] * 60                                   # spread worthless
        lab = _labeler(closes)
        out = candidate_outcome_labels(
            PCS_LEGS, PCS_CREDIT, 602.0, labeler=lab, entry_ts=_obs(0),
            target_pnl=0.25, stop_pnl=-0.40)
        assert out["target_hit"] == 1
        assert out["first_event"] == "target"
        assert out["time_in_trade_min"] == 1.0

    def test_same_bar_ambiguity_is_conservative(self):
        # one wide bar where the high-mark makes target and the low-mark
        # makes stop: never assume the favorable order.
        # Base 598.7 marks ~0.0 P&L (short 599P intrinsic = credit).
        closes = [598.7] * 30
        highs = [598.7] * 30
        lows = [598.7] * 30
        highs[5] = 602.0                                        # spread worthless
        lows[5] = 596.0                                         # spread deep ITM
        lab = _labeler(closes, highs, lows)
        out = candidate_outcome_labels(
            PCS_LEGS, PCS_CREDIT, 602.0, labeler=lab, entry_ts=_obs(0),
            target_pnl=0.25, stop_pnl=-0.40)
        assert out["ambiguous_same_bar"] == 1
        assert out["first_event"] == "stop"

    def test_neither_event(self):
        # flat at 598.7 marks ~0.0 P&L: neither the 0.25 target nor the
        # -0.40 stop is ever touched
        closes = [598.7] * 60
        lab = _labeler(closes)
        out = candidate_outcome_labels(
            PCS_LEGS, PCS_CREDIT, 598.7, labeler=lab, entry_ts=_obs(0),
            target_pnl=0.25, stop_pnl=-0.40)
        assert out["first_event"] == "neither"
        assert out["target_hit"] == 0
        assert out["stop_hit"] == 0


class TestCandidateStorage:
    def test_candidate_id_stable_and_geometry_sensitive(self):
        a = make_candidate_id("snap1", "put_credit", PCS_LEGS)
        b = make_candidate_id("snap1", "put_credit", PCS_LEGS)
        assert a == b and len(a) == 64
        other = [{"strike": 598.0, "kind": "P", "qty": -1},
                 {"strike": 597.0, "kind": "P", "qty": 1}]
        assert make_candidate_id("snap1", "put_credit", other) != a
        assert make_candidate_id("snap2", "put_credit", PCS_LEGS) != a

    def test_round_trip(self, tmp_path):
        store = PredictionStore(str(tmp_path / "pred.sqlite"))
        cid = make_candidate_id("snap1", "put_credit", PCS_LEGS)
        store.log_candidate_snapshot(
            cid, "snap1", "put_credit", PCS_LEGS,
            quote={"mid_credit": 0.30},
            legacy_metrics={"score": 1.2, "ev": 0.05})
        outcome = candidate_outcome_labels(PCS_LEGS, PCS_CREDIT, 602.0,
                                           max_loss=0.70)
        store.log_candidate_outcome(cid, outcome)

        rows = store.fetch_candidates(snapshot_id="snap1")
        assert len(rows) == 1
        r = rows[0]
        assert r["candidate_id"] == cid
        assert r["family"] == "put_credit"
        assert r["legs"] == PCS_LEGS
        assert r["pnl_mid"] == pytest.approx(0.30)
        assert r["settled"] == 1
        assert r["outcome_extras"]["return_on_risk"] == pytest.approx(0.30 / 0.70)
        store.close()

    def test_idempotent_rewrite(self, tmp_path):
        store = PredictionStore(str(tmp_path / "pred.sqlite"))
        cid = make_candidate_id("snap1", "put_credit", PCS_LEGS)
        for _ in range(3):
            store.log_candidate_snapshot(cid, "snap1", "put_credit", PCS_LEGS)
        assert len(store.fetch_candidates()) == 1
        store.close()
