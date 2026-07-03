"""
Predictive-power measurement: journal readouts, readiness gates, chain
recording/replay, and the coupled synthetic world.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from journal import Journal

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# journal fixtures                                                             #
# --------------------------------------------------------------------------- #
def _row(session="2026-07-01", ts="2026-07-01T10:00:00-04:00", spot=600.0,
         direction="none", was_traded=0, prob_profit=None, legs=None,
         credit=None, ev=None):
    return {
        "session_date": session, "ts": ts, "spot": spot,
        "net_gex": 2e9, "gex_regime": "long", "gex_pct_rank": 0.5,
        "zero_gamma_dist": 1.0, "zero_gamma_dist_pct": 0.001, "adx": 15.0,
        "call_wall": spot + 5, "put_wall": spot - 5,
        "selected_family": "put_credit" if legs else None,
        "short_strikes": None, "long_strikes": None,
        "legs_json": json.dumps(legs) if legs else None,
        "credit": credit, "candidate_score": 0.5 if legs else None,
        "ev": ev, "max_loss": 1.0 if legs else None,
        "ev_per_risk": ev if legs else None, "theta": None, "gamma": None,
        "prob_profit": prob_profit, "prob_touch_short": None,
        "liquidity_score": None, "wall_safety": None,
        "gamma_safety": None, "touch_safety": None,
        "gate_pass": 1 if was_traded else 0, "gate_score": 50.0,
        "gate_failed": json.dumps([]), "veto_reasons": json.dumps([]),
        "decision": "TRADE" if was_traded else "NO_TRADE",
        "no_trade_reason": "" if was_traded else "test",
        "was_traded": was_traded,
        "candidate_present": 1 if legs else 0,
        "regime_direction": direction,
    }


PUT_SPREAD = [{"strike": 599.0, "kind": "P", "qty": -1},
              {"strike": 598.0, "kind": "P", "qty": 1}]


def test_directional_accuracy_scores_no_trades_too():
    j = Journal(":memory:")
    # 3 "call" bias ticks: settle 602 vs spots 600/601/603 -> 2 hits, 1 miss
    for spot in (600.0, 601.0, 603.0):
        j.log(_row(spot=spot, direction="call"))
    # 1 "put" bias tick: settle 602 vs 600 -> miss
    j.log(_row(spot=600.0, direction="put"))
    # unresolved bias must not count
    j.log(_row(spot=600.0, direction="none"))
    j.settle_session("2026-07-01", 602.0)

    d = j.directional_accuracy()
    assert d["overall"]["n"] == 4
    assert d["overall"]["hit_rate"] == pytest.approx(0.5)
    assert d["by_direction"]["call"]["n"] == 3
    assert d["by_direction"]["call"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert d["by_direction"]["put"]["hit_rate"] == 0.0
    assert d["traded_only"]["n"] == 0            # none were traded


def test_prob_calibration_brier_and_skill():
    j = Journal(":memory:")
    # perfectly confident and right: p=0.99 credit spread that keeps its credit
    for _ in range(5):
        j.log(_row(direction="none", was_traded=1, prob_profit=0.99,
                   legs=PUT_SPREAD, credit=0.3, ev=0.1))
    # confident and wrong: p=0.9 but settlement blows through the spread
    j.log(_row(session="2026-07-02", ts="2026-07-02T10:00:00-04:00",
               direction="none", was_traded=1, prob_profit=0.9,
               legs=PUT_SPREAD, credit=0.3, ev=0.1))
    j.settle_session("2026-07-01", 602.0)        # spread expires OTM: win
    j.settle_session("2026-07-02", 590.0)        # deep ITM: loss
    pp = j.prob_calibration(n_bins=5)
    assert pp["n"] == 6
    assert 0 < pp["brier"] < 0.25
    assert pp["bins"][-1]["mean_predicted"] > 0.9


def test_calibration_aggregates_all_three_panels():
    j = Journal(":memory:")
    j.log(_row(direction="call", was_traded=1, prob_profit=0.8,
               legs=PUT_SPREAD, credit=0.3, ev=0.1))
    j.settle_session("2026-07-01", 602.0)
    cal = j.calibration()
    assert set(cal) == {"directional", "prob_profit", "ev"}
    assert cal["ev"]["n"] == 1
    assert cal["ev"]["mean_ev_error"] == pytest.approx(0.3 - 0.1, abs=1e-6)


def test_readiness_includes_predictive_gates(tmp_path):
    from dashboard.queries import readiness_summary
    db = os.path.join(tmp_path, "j.sqlite")
    j = Journal(db)
    j.log(_row(direction="call", was_traded=1, prob_profit=0.8,
               legs=PUT_SPREAD, credit=0.3, ev=0.1))
    j.settle_session("2026-07-01", 602.0)
    j.close()

    r = readiness_summary(db, os.path.join(tmp_path, "nope.sqlite"))
    labels = [c["label"] for c in r["checks"]]
    for expected in ("Directional edge present", "Probabilities calibrated", "EV unbiased"):
        assert expected in labels
    # with n=1 nothing predictive can pass -> not ready
    assert r["ready"] is False
    assert "calibration" in r["facts"]


# --------------------------------------------------------------------------- #
# chain recording + replay                                                     #
# --------------------------------------------------------------------------- #
def test_record_and_replay_roundtrip(tmp_path):
    from chain_store import ChainRecorder, RecordedFeed
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    src = CoupledSyntheticFeed(WorldConfig(days=2, seed=3, tick_stride=30))
    ticks = src.timestamps()
    rec = ChainRecorder(str(tmp_path))
    served = []
    for t in ticks:
        snap = src.snapshot(t)
        if snap is not None:
            rec.record(t, snap)
            served.append((t, snap))
    for date, price in src.day_close.items():
        rec.record_settlement(date, price)

    replay = RecordedFeed(str(tmp_path))
    assert len(replay) == len(served)
    assert replay.timestamps() == [t for t, _ in served]

    for t, orig in served:
        got = replay.snapshot(t)
        assert got is not None
        assert got.market.spot == pytest.approx(orig.market.spot)
        assert got.market.net_gex == pytest.approx(orig.market.net_gex)
        assert got.market.now == orig.market.now
        assert len(got.chain.quotes) == len(orig.chain.quotes)
        assert got.chain.quotes[0].strike == orig.chain.quotes[0].strike
        assert got.chain.t_years == pytest.approx(orig.chain.t_years)
        # bar window reassembled from incremental rows
        assert got.bars is not None
        assert got.bars.close[-1] == pytest.approx(orig.bars.close[-1])

    for date, price in src.day_close.items():
        assert replay.settlement_price(date) == pytest.approx(price)


def test_replay_survives_truncated_tail(tmp_path):
    from chain_store import ChainRecorder, RecordedFeed
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    src = CoupledSyntheticFeed(WorldConfig(days=1, seed=3, tick_stride=60))
    rec = ChainRecorder(str(tmp_path))
    n = 0
    for t in src.timestamps():
        snap = src.snapshot(t)
        if snap:
            rec.record(t, snap)
            n += 1
    # simulate a crash mid-write: append garbage bytes
    path = [p for p in os.listdir(tmp_path) if p.endswith(".jsonl.gz")][0]
    import gzip
    with gzip.open(os.path.join(tmp_path, path), "at") as f:
        f.write('{"t":"tick","broken...')
    replay = RecordedFeed(str(tmp_path))
    assert len(replay) == n                       # bad line dropped, not fatal


# --------------------------------------------------------------------------- #
# coupled synthetic world                                                      #
# --------------------------------------------------------------------------- #
def test_coupled_world_actually_couples():
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    feed = CoupledSyntheticFeed(WorldConfig(days=6, seed=11, tick_stride=15))
    ticks = feed.timestamps()

    settles = list(feed.day_close.values())
    assert len(set(round(s, 2) for s in settles)) > 1   # settlement varies

    spots, chains_atm = [], []
    feed2 = CoupledSyntheticFeed(WorldConfig(days=6, seed=11, tick_stride=15))
    for t in feed2.timestamps():
        s = feed2.snapshot(t)
        spots.append(s.market.spot)
        atm = min(s.chain.quotes, key=lambda q: abs(q.strike - s.market.spot))
        chains_atm.append(atm.strike)
    # chain re-centers as spot moves (unlike the frozen-chain synthetic feed)
    assert np.corrcoef(spots, chains_atm)[0, 1] > 0.95
    # settlement equals the path's own final close, not a constant
    last_day = sorted(feed2.day_close)[-1]
    assert feed2.settlement_price(last_day) == pytest.approx(
        float(feed2._close[-1]), rel=1e-9)


def test_coupled_world_pipeline_produces_scoreable_predictions():
    """End to end: the pipeline trades on the coupled world and every
    predictive readout returns a real number (the thing the frozen synthetic
    could never do)."""
    from backtest import run_backtest
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    feed = CoupledSyntheticFeed(WorldConfig(days=4, seed=11, tick_stride=15))
    jrn = Journal(":memory:")
    ts = run_backtest(feed, feed.timestamps(), journal=jrn)

    assert ts.total_ticks > 0
    assert ts.trade_ticks > 0                      # it actually trades
    cal = jrn.calibration()
    assert cal["prob_profit"]["n"] > 0
    assert cal["prob_profit"]["brier"] is not None
    assert cal["ev"]["n"] > 0
    # outcome variance exists: not every candidate wins
    rows = [r for r in jrn.fetch(settled_only=True) if r["realized_pnl"] is not None]
    assert any(r["realized_pnl"] < 0 for r in rows)
    assert any(r["realized_pnl"] > 0 for r in rows)


def test_walk_forward_runs_on_recordings(tmp_path):
    """The real-data loop: record sessions -> replay -> walk-forward folds."""
    from chain_store import ChainRecorder, RecordedFeed
    from synthetic_world import CoupledSyntheticFeed, WorldConfig
    from walk_forward import run_walk_forward, WalkForwardConfig

    src = CoupledSyntheticFeed(WorldConfig(days=3, seed=3, tick_stride=30))
    rec = ChainRecorder(str(tmp_path))
    for t in src.timestamps():
        snap = src.snapshot(t)
        if snap:
            rec.record(t, snap)
    for date, px in src.day_close.items():
        rec.record_settlement(date, px)

    ticks = RecordedFeed(str(tmp_path)).timestamps()
    result = run_walk_forward(
        feed_factory=lambda: RecordedFeed(str(tmp_path)),
        timestamps=ticks,
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=2, train_frac=0.5),
    )
    assert len(result.folds) == 2
    assert all(f.tearsheet.total_ticks > 0 for f in result.folds)
    # test folds must cover only recorded timestamps, in order
    assert result.folds[0].test_start < result.folds[1].test_start
    assert result.folds[-1].test_end == ticks[-1]


# --------------------------------------------------------------------------- #
# optimizer holdout                                                            #
# --------------------------------------------------------------------------- #
def test_optimizer_holdout_never_seen_by_search():
    from optimizer import run_optimizer, OptimizerConfig
    from walk_forward import WalkForwardConfig
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    def make_feed():
        return CoupledSyntheticFeed(WorldConfig(days=4, seed=5, tick_stride=30))

    ticks = make_feed().timestamps()
    res = run_optimizer(
        feed_factory=make_feed,
        timestamps=ticks,
        param_space={"gate.max_adx": [18.0, 22.0]},
        opt_cfg=OptimizerConfig(search="grid", metric="total_pnl", holdout_frac=0.25),
        wf_cfg=WalkForwardConfig(mode="expanding", n_folds=2, train_frac=0.5),
    )
    assert res.holdout_score is not None
    assert res.holdout_result is not None
    # the holdout fold's test window must start after every search tick
    cut = ticks[int(len(ticks) * 0.75) - 1]
    assert res.holdout_result.folds[0].test_start > cut
    assert res.to_dict()["holdout_score"] == res.holdout_score
