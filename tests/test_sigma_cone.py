"""
tests/test_sigma_cone.py
========================
MTF outward-looking sigma cones: build, journal, settle vs true spot.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from prediction.sigma_cone import (
    SIGMA_LEVELS,
    build_cone_for_timeframe,
    build_mtf_cones,
    cones_to_signals,
    settle_band,
)
from prediction.storage import PredictionStore

ET = ZoneInfo("America/New_York")


def test_cone_horizons_grow_with_sigma():
    ts = dt.datetime(2026, 7, 13, 11, 0, tzinfo=ET)
    cone = build_cone_for_timeframe(
        snapshot_id="snap-a",
        ts=ts,
        session_date="2026-07-13",
        timeframe="5m",
        spot=100.0,
        sigma_per_sqrt_min=0.0015,
        drift_per_min=0.0,
        minutes_to_close=240.0,
    )
    assert cone.timeframe == "5m"
    assert len(cone.bands) == len(SIGMA_LEVELS)
    horizons = [b.horizon_min for b in cone.bands]
    assert horizons[0] < horizons[1] < horizons[2]
    # Wider σ → wider band
    widths = [b.hi - b.lo for b in cone.bands]
    assert widths[0] < widths[1] < widths[2]
    # Spot inside every band at emission (zero drift)
    for b in cone.bands:
        assert b.lo < 100.0 < b.hi


def test_example_shape_endogenous_horizons():
    """Spot 100 on 5m: 0.5σ looks nearer than 2σ; bands are price intervals."""
    ts = dt.datetime(2026, 7, 13, 10, 30, tzinfo=ET)
    cone = build_cone_for_timeframe(
        snapshot_id="snap-ex",
        ts=ts,
        session_date="2026-07-13",
        timeframe="5m",
        spot=100.0,
        sigma_per_sqrt_min=0.002,
        drift_per_min=0.00005,
        minutes_to_close=300.0,
    )
    by = {b.sigma: b for b in cone.bands}
    assert by[0.5].horizon_min < by[1.0].horizon_min < by[2.0].horizon_min
    assert by[0.5].lo < by[0.5].hi
    assert by[2.0].lo < by[0.5].lo
    assert by[2.0].hi > by[0.5].hi


def test_settle_band_inside_outside():
    from prediction.sigma_cone import ConeBand
    band = ConeBand(sigma=1.0, lo=98.0, hi=107.0, horizon_min=15.0,
                    mid=102.5, settle_by="2026-07-13T11:15:00-04:00")
    s = settle_band(band, realized_spot=101.0, realized_ts="2026-07-13T11:15:00-04:00")
    assert s.inside is True
    assert s.coverage_note == "inside"
    s2 = settle_band(band, realized_spot=110.0, realized_ts="2026-07-13T11:15:00-04:00")
    assert s2.inside is False
    assert s2.coverage_note == "above"
    assert s2.error_mid > 0


def test_journal_and_match_true_spot(tmp_path):
    store = PredictionStore(db_path=str(tmp_path / "pred.sqlite"))
    ts = dt.datetime(2026, 7, 13, 10, 0, tzinfo=ET)
    market = SimpleNamespace(
        spot=100.0, expected_range=2.5, straddle_breakeven=None,
        bb_width=None, tick_abs_mean=None,
    )
    cones = build_mtf_cones(
        snapshot_id="snap-1",
        ts=ts,
        session_date="2026-07-13",
        spot=100.0,
        market=market,
        signals={"regime_bias_value": 55.0},
        minutes_to_close=200.0,
    )
    assert len(cones) == 4  # 1m 5m 15m 30m
    n = store.log_sigma_cones(cones)
    assert n == 4 * 3

    sig = cones_to_signals(cones)
    assert sig["cone_primary_tf"] == "5m"
    assert "cone_0p5_lo" in sig

    # Nothing due yet at emission time
    assert store.settle_sigma_cones(ts.isoformat(), 100.0) == 0

    # Advance past all horizons with a spot still inside the tightest bands
    far = (ts + dt.timedelta(hours=6)).isoformat()
    settled = store.settle_sigma_cones(far, 101.0, realized_ts=far)
    assert settled == n
    rows = store.fetch_sigma_cones(settled=True, limit=100)
    assert len(rows) == n
    assert all(r["settled"] == 1 for r in rows)
    assert all(r["realized_spot"] == 101.0 for r in rows)

    cov = store.sigma_cone_coverage()
    assert cov["n_settled"] == n
    assert cov["hit_rate"] is not None
    assert "0.5" in cov["by_sigma"] or "0.5" in {str(float(k)) for k in cov["by_sigma"]}


def test_dashboard_sigma_cones_api(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    from fastapi.testclient import TestClient
    from dashboard.server import app, _configure
    from dashboard.state import write_live_state

    store = PredictionStore(db_path=str(tmp_path / "prediction_store.sqlite"))
    ts = dt.datetime(2026, 7, 13, 10, 0, tzinfo=ET)
    cones = build_mtf_cones(
        snapshot_id="snap-api",
        ts=ts,
        session_date="2026-07-13",
        spot=100.0,
        market=SimpleNamespace(
            spot=100.0, expected_range=3.0, straddle_breakeven=None,
            bb_width=None, tick_abs_mean=None,
        ),
        minutes_to_close=180.0,
    )
    store.log_sigma_cones(cones)
    far = (ts + dt.timedelta(hours=5)).isoformat()
    store.settle_sigma_cones(far, 99.5, realized_ts=far)
    store.conn.close()

    live = str(tmp_path / "live.json")
    write_live_state(live, {
        "ts": ts.isoformat(),
        "doing": {},
        "sigma_cones": {"model_version": "sigma-cone-v1", "panes": []},
    })
    _configure(str(tmp_path / "shadow.db"), str(tmp_path / "paper.sqlite"), live,
               prediction_db=str(tmp_path / "prediction_store.sqlite"))

    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}
    r = c.get("/api/sigma-cones?limit=50", headers=hdrs)
    assert r.status_code == 200
    body = r.json()
    assert body["coverage"]["n_settled"] > 0
    assert len(body["rows"]) > 0
    assert body["rows"][0]["realized_spot"] == 99.5
