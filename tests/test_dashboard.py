"""Tests for read-only observability dashboard."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from dashboard.server import app, _configure
from dashboard.state import serialize_tick_result, write_live_state, read_live_state
from decision_matrix import Decision, TradeIntent
from gate_scorer import MarketSnapshot
from regime_classifier import RegimeState
from unified_loop import TickResult, TickSnapshot

ET = ZoneInfo("America/New_York")


def _market():
    return MarketSnapshot(
        spot=600.0, net_gex=4e9, gamma_flip=595.0,
        call_wall=605.0, put_wall=595.0, gex_pct_rank=0.85,
        vix9d=12.0, vix=13.0, vix3m=15.0, vvix=92.0, vvix_baseline=95.0,
        straddle_breakeven=4.0, expected_range=3.2,
        adx=14.0, rsi=50.0, bb_width=1.4, bb_width_baseline=2.0,
        vwap=600.0, vwap_reversion_count=2,
        tick_abs_mean=400.0, cvd_slope=0.01,
        now=dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET),
        has_catalyst=False,
    )


def _tick_result():
    regime = RegimeState(
        confidences={"compression": 72, "trend": 18},
        reliabilities={"compression": 0.8, "trend": 0.3},
        dominant_regime="compression",
        permitted_engine="premium_selling",
        vetoes=[],
        global_information_gain=12.0,
        standardized={},
        stand_down=False,
    )
    intent = TradeIntent(
        exec_regime="compression",
        context_regime="compression",
        direction_bias="neutral",
        bias_value=0.0,
        decision=Decision("IC", "both", "HIGH", "theta", "shorts inside walls", "15m"),
        size_mult=1.0,
        vetoes=[],
        note="",
    )
    snap = TickSnapshot(market=_market(), bars=None, chain=None)
    return TickResult(
        ts=dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET),
        regime=regime,
        intent=intent,
        decision=None,
        final_size_mult=1.0,
        vetoes=[],
        snapshot=snap,
    )


def test_serialize_tick_result_sections():
    payload = serialize_tick_result(
        _tick_result(),
        feed_source="Tradier",
        paper_summary={"trades": 0, "equity": 1000},
        market_status={"is_open": True},
    )
    assert "doing" in payload
    assert "inputs" in payload
    assert "why" in payload
    assert payload["doing"]["dominant_regime"] == "compression"
    assert payload["inputs"]["spot"] == 600.0
    assert payload["why"]["matrix_cell"] == ["compression", "compression", "neutral"]
    assert payload["feed_source"] == "Tradier"
    # continuous direction bias for the four-way quadrant / regime shading
    assert payload["doing"]["direction_bias"] == "neutral"
    assert payload["doing"]["bias_value"] == 0.0


def test_write_read_live_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "live_state.json")
        write_live_state(path, {"ts": "test", "doing": {}})
        data = read_live_state(path)
        assert data["ts"] == "test"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "shadow.db")
        paper = os.path.join(tmp, "paper.sqlite")
        live = os.path.join(tmp, "live_state.json")
        write_live_state(live, serialize_tick_result(
            _tick_result(),
            feed_source="Yahoo",
            market_status={"is_open": True, "label_open": "Market Open"},
        ))
        _configure(db, paper, live)
        yield TestClient(app)


def test_api_requires_auth(client):
    r = client.get("/api/live")
    assert r.status_code == 401


def test_api_rejects_query_param_token(client):
    # tokens in query strings end up in access logs — header only
    r = client.get("/api/live?token=test-secret-token")
    assert r.status_code == 401


def test_api_rejects_wrong_bearer(client):
    r = client.get("/api/live", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_api_live_with_auth(client):
    r = client.get("/api/live", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["doing"]["structure"] == "IC"


def test_api_market_status(client):
    r = client.get("/api/market-status", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    data = r.json()
    assert "is_open" in data
    assert "seconds_until_open" in data or "seconds_until_close" in data


def test_post_returns_405(client):
    r = client.post("/api/live", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 405


def test_journal_fetch_decodes_signals_json(tmp_path):
    """The ticks query hands the frontend a decoded signals dict (regime
    shading + quadrant read regime_bias_value / regime_dominant_conf)."""
    from dashboard.queries import journal_fetch
    from journal import COLUMNS, Journal

    db = str(tmp_path / "shadow.db")
    jrn = Journal(db)
    row = {c: None for c in COLUMNS}
    row.update({
        "ts": "2026-07-08T10:30:00-04:00", "session_date": "2026-07-08",
        "decision": "NO_TRADE", "spot": 600.0,
        "signals_json": json.dumps({"regime_bias_value": 63.4,
                                    "regime_dominant_conf": 71.0}),
    })
    jrn.log(row)
    jrn.close()

    ticks = journal_fetch(db, limit=5)
    assert len(ticks) == 1
    sig = ticks[0]["signals_json"]
    assert isinstance(sig, dict)
    assert sig["regime_bias_value"] == 63.4
    assert sig["regime_dominant_conf"] == 71.0


def test_api_ras_history(monkeypatch, tmp_path):
    """/api/ras serves the per-position RAS timeline with full components."""
    from journal import Journal
    from regime_alignment import RASComponent, RASResult

    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    db = str(tmp_path / "shadow.db")
    jrn = Journal(db)
    for i, (score, action) in enumerate([(20.0, "ok"), (-35.0, "warning"),
                                         (-75.0, "exit")]):
        jrn.log_ras(
            f"2026-07-08T10:3{i}:00-04:00", "2026-07-08",
            RASResult(
                score=score,
                components=[RASComponent("gamma_alignment", -0.5, 1.5, -0.75,
                                         "below flip")],
                action=action, position_id="posX", ema_score=score,
            ))
    jrn.close()
    _configure(db, str(tmp_path / "paper.sqlite"), str(tmp_path / "live.json"))

    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}
    r = c.get("/api/ras?position_id=posX", headers=hdrs)
    assert r.status_code == 200
    evals = r.json()["evaluations"]
    assert len(evals) == 3
    assert evals[-1]["action"] == "exit"
    assert evals[-1]["score"] == -75.0
    assert evals[0]["components"][0]["note"] == "below flip"

    # filters
    r = c.get("/api/ras?position_id=nope", headers=hdrs)
    assert r.json()["evaluations"] == []

    # missing / legacy DB degrades gracefully
    _configure(str(tmp_path / "absent.db"), str(tmp_path / "p.sqlite"),
               str(tmp_path / "live.json"))
    r = c.get("/api/ras", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["evaluations"] == []
