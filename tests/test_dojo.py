"""Tests for the dojo matrix-training orchestrator + its dashboard surface."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from dojo import DojoConfig, _archetype_matrix, run_dojo
from journal import Journal
from matrix_universe import ARCHETYPES, MarkovWorldFeed, UniverseSpec


def _tiny_cfg(tmp: str, **kw) -> DojoConfig:
    base = dict(
        db_path=os.path.join(tmp, "shadow.db"),
        reports_dir=os.path.join(tmp, "reports"),
        skip_learner=True,           # learner has its own test module
        universes_per_gen=2, generations=1,
        universe_days=1, tick_stride=30,
        report_date="2026-07-23",
    )
    base.update(kw)
    return DojoConfig(**base)


# --------------------------------------------------------------------------- #
# end-to-end                                                                  #
# --------------------------------------------------------------------------- #
def test_run_dojo_persists_report_and_degrades_honestly():
    with tempfile.TemporaryDirectory() as tmp:
        out = run_dojo(_tiny_cfg(tmp))

        # no recorded tape -> honest insufficient_data, not a crash
        phases = out["metrics"]["phases"]
        assert phases["recorded"]["status"] == "insufficient_data"
        assert phases["learner"]["status"] == "skipped"
        assert phases["universe"]["status"] == "ok"
        assert any(f["flag"] == "no_recorded_tape" for f in out["flags"])

        # robustness matrix covers every archetype
        matrix = phases["universe"]["archetype_matrix"]
        assert set(matrix) == set(ARCHETYPES)
        sessions = sum(m["n_sessions"] for m in matrix.values())
        assert sessions == 2  # 2 universes x 1 day

        # coverage map has the right shape
        cov = phases["universe"]["coverage"]
        assert phases["universe"]["coverage_cells_total"] == len(ARCHETYPES) * 5
        assert 0 < phases["universe"]["coverage_cells_visited"] <= \
            phases["universe"]["coverage_cells_total"]
        assert set(cov) == set(ARCHETYPES)

        # persisted to validation_reports with report_type='dojo'
        jrn = Journal(os.path.join(tmp, "shadow.db"))
        reports = jrn.fetch_validation_reports(report_type="dojo")
        jrn.close()
        assert len(reports) == 1
        assert reports[0]["id"] == out["report_id"]
        assert reports[0]["metrics"]["phases"]["universe"]["status"] == "ok"

        # JSON artifact written
        assert os.path.isfile(out["json_path"])


def test_run_dojo_skip_universe():
    with tempfile.TemporaryDirectory() as tmp:
        out = run_dojo(_tiny_cfg(tmp, skip_universe=True))
        assert out["metrics"]["phases"]["universe"]["status"] == "skipped"
        assert "universe sparring: skipped" in out["summary"]


# --------------------------------------------------------------------------- #
# attribution                                                                 #
# --------------------------------------------------------------------------- #
def test_archetype_matrix_attributes_sessions_to_day_archetype():
    spec = UniverseSpec(universe_id="a", seed=1, days=2,
                        start_archetype="calm_pin", tick_stride=30)
    feed = MarkovWorldFeed(spec)
    days = list(feed.day_archetype)
    row = {
        **spec.to_dict(),
        "daily_pnl": {days[0]: 2.0},
        "session_stats": {days[0]: {"trades": 3, "wins": 2, "pnl": 2.0}},
        "trades": 3, "dir_hit": 0.6, "dir_n": 10,
        "total_pnl": 2.0, "win_rate": None, "sharpe": None,
    }
    matrix = _archetype_matrix([row], [feed])
    arch0 = feed.day_archetype[days[0]]
    assert matrix[arch0]["trades"] == 3
    assert matrix[arch0]["total_pnl"] == pytest.approx(2.0)
    assert matrix[arch0]["win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    # every session is an observation, traded or not
    assert sum(m["n_sessions"] for m in matrix.values()) == len(days)
    # directional stats charge to the start archetype
    assert matrix["calm_pin"]["dir_hit"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# dashboard surface                                                           #
# --------------------------------------------------------------------------- #
def test_api_dojo_serves_persisted_reports(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "dojo-test-token")
    from dashboard.server import app, _configure

    with tempfile.TemporaryDirectory() as tmp:
        out = run_dojo(_tiny_cfg(tmp))
        _configure(os.path.join(tmp, "shadow.db"),
                   os.path.join(tmp, "paper.sqlite"),
                   os.path.join(tmp, "live_state.json"))
        client = TestClient(
            app, headers={"Authorization": "Bearer dojo-test-token"})

        r = client.get("/api/dojo")
        assert r.status_code == 200
        reports = r.json()["reports"]
        assert len(reports) == 1
        assert reports[0]["report_type"] == "dojo"

        rid = out["report_id"]
        detail = client.get(f"/api/dojo/{rid}")
        assert detail.status_code == 200
        uni = detail.json()["metrics"]["phases"]["universe"]
        assert set(uni["archetype_matrix"]) == set(ARCHETYPES)

        assert client.get("/api/dojo/999999").status_code == 404
        # non-dojo reports are not served on the dojo route
        jrn = Journal(os.path.join(tmp, "shadow.db"))
        other = jrn.log_validation_report("2026-07-23", "daily", {}, "x")
        jrn.close()
        assert client.get(f"/api/dojo/{other}").status_code == 404
