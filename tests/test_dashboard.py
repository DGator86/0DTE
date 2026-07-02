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
