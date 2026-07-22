"""
tests/test_competition.py
=========================
0DTE-vs-SPY-DER head-to-head scoreboard (dashboard.queries.competition_view)
and the /api/competition endpoint.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from dashboard.queries import competition_view
from dashboard.server import app, _configure


def _summary(by_track):
    return {"by_track": by_track}


def test_competition_aggregates_zerodte_and_spyder_sides():
    comp = competition_view(_summary({
        "legacy": {"trades": 4, "total_pnl": 100.0, "win_rate": 0.5, "open_positions": 0},
        "v2": {"trades": 2, "total_pnl": -50.0, "win_rate": 0.5, "open_positions": 1},
        "v3": {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "open_positions": 0},
        "spy_der": {"trades": 3, "total_pnl": 90.0, "win_rate": 0.667, "open_positions": 1},
    }))
    z, s = comp["zerodte"], comp["spyder"]
    assert z["name"] == "0DTE" and s["name"] == "SPY-DER"
    # 0DTE = 3 tracks @ $1000 start = $3000; pnl 100-50 = 50.
    assert z["starting_capital"] == 3000.0
    assert z["total_pnl"] == 50.0
    assert z["trades"] == 6
    # SPY-DER = 1 track @ $1000; pnl 90 => 9%.
    assert s["starting_capital"] == 1000.0
    assert s["return_pct"] == 0.09
    assert s["open_positions"] == 1


def test_competition_leader_is_by_return_on_capital():
    comp = competition_view(_summary({
        "legacy": {"trades": 10, "total_pnl": 120.0, "win_rate": 0.6},
        "v2": {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0},
        "v3": {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0},
        "spy_der": {"trades": 5, "total_pnl": 90.0, "win_rate": 0.8},
    }))
    # 0DTE: 120/3000 = 4%; SPY-DER: 90/1000 = 9% => SPY-DER leads.
    assert comp["leader"] == "SPY-DER"
    assert comp["margin_pct"] > 0


def test_competition_prefers_live_equity_when_present():
    comp = competition_view(_summary({
        "spy_der": {"trades": 1, "total_pnl": 50.0, "win_rate": 1.0, "equity": 1234.0},
        "legacy": {"trades": 1, "total_pnl": 10.0, "win_rate": 1.0},
        "v2": {}, "v3": {},
    }))
    assert comp["spyder"]["equity"] == 1234.0


def test_competition_handles_empty_summary():
    comp = competition_view(None)
    assert comp["leader"] == "tie"
    assert comp["zerodte"]["trades"] == 0
    assert comp["spyder"]["trades"] == 0


def test_api_competition_endpoint(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    with tempfile.TemporaryDirectory() as tmp:
        _configure(os.path.join(tmp, "shadow.db"),
                   os.path.join(tmp, "paper.sqlite"),
                   os.path.join(tmp, "live_state.json"))
        c = TestClient(app)
        r = c.get("/api/competition",
                  headers={"Authorization": "Bearer test-secret-token"})
        assert r.status_code == 200
        data = r.json()
        assert data["zerodte"]["tracks"] == ["legacy", "v2", "v3"]
        assert data["spyder"]["tracks"] == ["spy_der"]
        assert data["leader"] in ("0DTE", "SPY-DER", "tie")
