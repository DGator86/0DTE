"""
Observation-only signal tranche: dealer dynamics, expected-move-consumed,
options-flow lite, breadth lite, and the signals_json admission channel.

The contract under test: new signals are journaled and scored on every tick
but have NO gate/veto power — and missing sources drop out instead of
reading as zero.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from market_dynamics import DynamicsWindow, DynamicsConfig, session_open_from_bars

ET = ZoneInfo("America/New_York")
T0 = dt.datetime(2026, 7, 6, 10, 0, tzinfo=ET).timestamp()


def _upd(w, minute, **kw):
    base = dict(spot=600.0, gamma_flip=596.0, call_wall=605.0, put_wall=595.0,
                net_gex=3e9, straddle_be=4.0)
    base.update(kw)
    return w.update(T0 + minute * 60.0, **base)


# --------------------------------------------------------------------------- #
# DynamicsWindow                                                               #
# --------------------------------------------------------------------------- #
def test_no_derivatives_until_history_exists():
    w = DynamicsWindow()
    out = _upd(w, 0)
    assert "flip_velocity" not in out
    assert "wall_rupture" not in out


def test_velocities_measure_change_per_5min():
    w = DynamicsWindow()
    for i in range(6):
        _upd(w, i, gamma_flip=596.0 + 0.2 * i, call_wall=605.0, net_gex=3e9 + 0.1e9 * i)
    out = _upd(w, 6, gamma_flip=596.0 + 0.2 * 6, net_gex=3e9 + 0.6e9)
    # flip moved 0.2/min -> 1.0 per 5 min
    assert out["flip_velocity"] == pytest.approx(1.0, rel=0.25)
    assert out["call_wall_velocity"] == pytest.approx(0.0, abs=1e-9)
    assert out["gex_velocity_bn"] == pytest.approx(0.5, rel=0.25)


def test_flip_chase_positive_when_flip_converges_on_spot():
    w = DynamicsWindow()
    for i in range(7):
        _upd(w, i, spot=600.0, gamma_flip=590.0 + i)      # flip closing the gap
    out = _upd(w, 7, spot=600.0, gamma_flip=597.0)
    assert out["flip_chase"] > 0


def test_wall_rupture_detects_acceptance_beyond_wall():
    w = DynamicsWindow()
    for i in range(7):
        out = _upd(w, i, spot=606.0, call_wall=605.0)     # holding ABOVE call wall
    assert out["wall_rupture"] == 1.0
    w2 = DynamicsWindow()
    for i in range(7):
        out = _upd(w2, i, spot=594.0, put_wall=595.0)     # holding BELOW put wall
    assert out["wall_rupture"] == -1.0
    w3 = DynamicsWindow()
    for i in range(7):
        out = _upd(w3, i, spot=600.0)                      # inside the channel
    assert out["wall_rupture"] == 0.0


def test_expected_move_consumed():
    w = DynamicsWindow()
    out = _upd(w, 0, spot=603.0, straddle_be=4.0, session_open=600.0)
    assert out["expected_move_consumed"] == pytest.approx(0.75)
    out = _upd(w, 1, spot=600.0, straddle_be=4.0, session_open=600.0)
    assert out["expected_move_consumed"] == pytest.approx(0.0)


def test_straddle_ramp_sign():
    w = DynamicsWindow()
    for i in range(6):
        _upd(w, i, straddle_be=4.0 - 0.02 * i)             # normal theta decay
    out = _upd(w, 6, straddle_be=4.0 - 0.12)
    assert out["straddle_ramp"] < 0                        # crush, not ramp


def test_dynamics_persist_across_restart(tmp_path):
    p = str(tmp_path / "dyn.json")
    w = DynamicsWindow(path=p)
    for i in range(6):
        _upd(w, i)
    w2 = DynamicsWindow(path=p)                            # "restart"
    out = _upd(w2, 6)
    assert "flip_velocity" in out                          # history survived


# --------------------------------------------------------------------------- #
# session_open_from_bars: both timestamp conventions                            #
# --------------------------------------------------------------------------- #
def _bars(ts0: dt.datetime, n: int, open0: float = 600.0):
    from resample import RawBars
    ts = np.array([np.datetime64(ts0.replace(tzinfo=None)) + np.timedelta64(60 * i, "s")
                   for i in range(n)], dtype="datetime64[ns]")
    opens = np.full(n, open0); opens[0] = open0
    px = np.linspace(open0, open0 + 1, n)
    return RawBars(ts=ts, open=np.concatenate([[open0], px[:-1]]), high=px + 0.1,
                   low=px - 0.1, close=px, volume=np.full(n, 1000.0))


def test_session_open_et_naive_bars():
    now = dt.datetime(2026, 7, 6, 10, 30, tzinfo=ET)
    bars = _bars(dt.datetime(2026, 7, 6, 9, 30), 60, open0=601.5)   # ET wall time
    assert session_open_from_bars(bars, now) == pytest.approx(601.5)


def test_session_open_utc_naive_bars():
    now = dt.datetime(2026, 7, 6, 10, 30, tzinfo=ET)
    utc0 = dt.datetime(2026, 7, 6, 13, 30)                # 9:30 ET as naive UTC
    bars = _bars(utc0, 60, open0=602.5)
    assert session_open_from_bars(bars, now) == pytest.approx(602.5)


def test_session_open_none_before_open():
    now = dt.datetime(2026, 7, 6, 9, 0, tzinfo=ET)
    bars = _bars(dt.datetime(2026, 7, 6, 4, 0), 60)
    assert session_open_from_bars(bars, now) is None


# --------------------------------------------------------------------------- #
# journal signals_json channel                                                 #
# --------------------------------------------------------------------------- #
def _min_row(**kw):
    from tests.test_predictive import _row
    return _row(**kw)


def test_journal_migrates_legacy_db_and_scores_signals(tmp_path):
    # legacy DB created before signals_json existed
    db = str(tmp_path / "j.sqlite")
    from journal import COLUMNS, _coltype
    legacy_cols = [c for c in COLUMNS if c != "signals_json"]
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        + ", ".join(f"{c} {_coltype(c)}" for c in legacy_cols)
        + ", settle_price REAL, realized_pnl REAL, ev_error REAL, "
          "settled INTEGER NOT NULL DEFAULT 0)")
    conn.commit(); conn.close()

    from journal import Journal
    j = Journal(db)                                       # must migrate, not crash
    legs = [{"strike": 599.0, "kind": "P", "qty": -1},
            {"strike": 598.0, "kind": "P", "qty": 1}]
    for i, (emc, settle) in enumerate([(0.2, 602.0), (0.9, 590.0), (0.4, 602.0)]):
        row = _min_row(session=f"2026-07-0{i+1}", was_traded=1,
                       prob_profit=0.7, legs=legs, credit=0.3, ev=0.1)
        row["ts"] = f"2026-07-0{i+1}T10:00:00-04:00"
        row["signals_json"] = json.dumps({"expected_move_consumed": emc})
        j.log(row)
        j.settle_session(f"2026-07-0{i+1}", settle)

    corr = j.component_correlations()
    assert "sig:expected_move_consumed" in corr
    # high EMC row lost; low EMC rows won -> negative correlation
    assert corr["sig:expected_move_consumed"] < 0


def test_journal_accepts_rows_without_signals(tmp_path):
    from journal import Journal
    j = Journal(":memory:")
    row = _min_row()
    row.pop("signals_json", None)
    j.log(row)                                            # defaults to NULL
    assert j.fetch()[0]["signals_json"] is None


# --------------------------------------------------------------------------- #
# flow lite + breadth lite                                                      #
# --------------------------------------------------------------------------- #
def test_flow_lite_pcr_and_participation():
    from massive_feed import flow_lite
    from spy0dte import OptionRow

    def row(side, vol, oi=100):
        return OptionRow(side=side, strike=600.0, oi=oi, gamma=0.01,
                         bid=1.0, ask=1.1, delta=0.5, volume=vol)

    rows = [row("call", 300), row("put", 600), row("call", 100, oi=200)]
    out = flow_lite(rows)
    assert out["pcr_volume"] == pytest.approx(600 / 400)
    assert out["volume_oi_ratio"] == pytest.approx(1000 / 400)

    # no volume data (e.g. tastytrade stream) -> NaN, never zero
    out2 = flow_lite([row("call", 0), row("put", 0)])
    assert math.isnan(out2["pcr_volume"])
    assert math.isnan(out2["volume_oi_ratio"])


def test_breadth_lite_from_batched_quotes(monkeypatch):
    import tradier_feed

    def fake_get(path, params):
        assert path == "/markets/quotes"
        syms = params["symbols"].split(",")
        quote = []
        for s in syms:
            cp = {"SPY": 0.5, "RSP": 0.9}.get(s)
            if cp is None:
                cp = -1.0 if s.startswith("XL") and s in ("XLE", "XLU") else 1.0
            quote.append({"symbol": s, "change_percentage": cp,
                          "last": 100.0, "prevclose": 99.0})
        return {"quotes": {"quote": quote}}

    monkeypatch.setattr(tradier_feed, "_get", fake_get)
    out = tradier_feed.get_breadth_lite()
    assert out["rsp_spy_div"] == pytest.approx(0.004)      # equal-weight leading
    assert out["sector_align"] == pytest.approx(9 / 11)    # 2 of 11 sectors red
    assert out["top10_pressure"] == pytest.approx(0.01)    # all mega-caps +1%


def test_breadth_lite_failure_reads_nan(monkeypatch):
    import tradier_feed
    monkeypatch.setattr(tradier_feed, "_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = tradier_feed.get_breadth_lite()
    assert all(math.isnan(v) for v in out.values())


# --------------------------------------------------------------------------- #
# matrix + snapshot integration: present when finite, absent when NaN          #
# --------------------------------------------------------------------------- #
def test_market_snapshot_omits_nan_signals():
    from tests.test_directional_path import _trend_down_market
    m = _trend_down_market()                               # defaults: all NaN
    snap = m.mtf_snapshot()
    assert "pcr_volume" not in snap and "rsp_spy_div" not in snap

    import dataclasses
    m2 = dataclasses.replace(m, pcr_volume=1.4, sector_align=0.7)
    snap2 = m2.mtf_snapshot()
    assert snap2["pcr_volume"] == 1.4 and snap2["sector_align"] == 0.7


def test_new_matrix_rows_are_observation_only():
    """The admission rule, enforced: new rows exist in VARS but are consumed
    by NO regime blend, decision cell, gate, or veto."""
    import inspect
    from mtf_matrix import VARS, _REGIME_DEF
    import decision_matrix

    new = {"flip_velocity", "flip_chase", "call_wall_velocity", "put_wall_velocity",
           "gex_velocity_bn", "wall_rupture", "straddle_ramp",
           "expected_move_consumed", "pcr_volume", "volume_oi_ratio",
           "rsp_spy_div", "sector_align", "top10_pressure"}
    names = {v.name for v in VARS}
    assert new <= names                                    # present in the matrix

    # entries are (variable, weight, invert) with an optional 4th "fold" flag
    consumed = {spec[0] for weights in _REGIME_DEF.values() for spec in weights}
    assert not (new & consumed)                            # no regime blend uses them
    assert not (new & set(decision_matrix.DIR_VARS))       # no direction bias either
    src = inspect.getsource(decision_matrix)
    for n in new:
        assert f'"{n}"' not in src                         # and no decision-table use


def test_end_to_end_signals_reach_journal_and_matrix():
    from journal import Journal
    from unified_loop import UnifiedOrchestrator
    from synthetic_world import CoupledSyntheticFeed, WorldConfig

    feed = CoupledSyntheticFeed(WorldConfig(days=1, seed=3, tick_stride=5))
    jrn = Journal(":memory:")
    orch = UnifiedOrchestrator(feed=feed, journal=jrn)
    orch.run_replay(feed.timestamps()[:12])

    sig = json.loads(jrn.fetch()[-1]["signals_json"])
    assert "expected_move_consumed" in sig
    assert "flip_velocity" in sig
    assert "wall_rupture" in sig
    # observation-only regime time series for the dashboard visualizations
    assert 0.0 <= sig["regime_bias_value"] <= 100.0
    assert 0.0 <= sig["regime_dominant_conf"] <= 100.0
    # raw fast/slow direction composites (turn-detection channel)
    assert 0.0 <= sig["bias_fast"] <= 100.0
    assert 0.0 <= sig["bias_slow"] <= 100.0


def test_intent_exposes_fast_and_slow_composites():
    """decide_from_matrix must surface the raw composites behind bias_value:
    the fast one is the early-warning channel for RAS and the lag study."""
    from decision_matrix import decide_from_matrix
    from mtf_matrix import build_matrix, demo_input, regime_rows

    rows = build_matrix(demo_input())
    intent = decide_from_matrix(rows, regime_rows(rows))
    assert intent.bias_fast is not None and 0.0 <= intent.bias_fast <= 100.0
    assert intent.bias_slow is not None and 0.0 <= intent.bias_slow <= 100.0
    # the blend is the documented 0.4 fast + 0.6 slow combination
    blend = 0.4 * intent.bias_fast + 0.6 * intent.bias_slow
    assert abs(blend - intent.bias_value) < 0.2


def test_bias_cross_detector_hysteresis():
    """+/-1 only on the tick the fast composite decisively overtakes/loses
    the slow one; deadband holds the prior side; missing data is inert."""
    from unified_loop import UnifiedOrchestrator

    orch = UnifiedOrchestrator(feed=None)
    assert orch._bias_cross(40.0, 55.0) is None    # first reading: sets side
    assert orch._bias_cross(45.0, 55.0) is None    # same side, no event
    assert orch._bias_cross(55.2, 55.0) is None    # inside deadband: hold
    assert orch._bias_cross(60.0, 55.0) == 1.0     # decisive cross above
    assert orch._bias_cross(62.0, 55.0) is None    # no repeat while above
    assert orch._bias_cross(40.0, 55.0) == -1.0    # cross back below
    assert orch._bias_cross(None, 55.0) is None    # missing data: inert
    assert orch._bias_cross(41.0, 55.0) is None    # still below, no event
