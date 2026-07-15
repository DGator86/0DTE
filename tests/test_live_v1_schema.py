"""
tests/test_live_v1_schema.py
============================
PR C — versioned /api/live (live.v1): schema validation, feed truth table,
no V2/V3 label confusion. Authority unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from dashboard.live_schema import (
    LIVE_SCHEMA_VERSION,
    LiveSchemaError,
    assert_live_v1,
    feeds_payload_from_statuses,
    synthesize_feed_statuses,
    validate_live_v1,
)
from dashboard.server import app, _configure
from dashboard.state import heartbeat_state, serialize_tick_result, write_live_state
from decision_matrix import Decision, TradeIntent
from gate_scorer import MarketSnapshot
from prediction.feed_status import build_feed_status
from regime_classifier import RegimeState
from unified_loop import TickResult, TickSnapshot

ET = ZoneInfo("America/New_York")


def _tick():
    regime = RegimeState(
        confidences={"compression": 72}, reliabilities={"compression": 0.8},
        dominant_regime="compression", permitted_engine="premium_selling",
        vetoes=[], global_information_gain=12.0, standardized={},
        stand_down=False,
    )
    intent = TradeIntent(
        exec_regime="compression", context_regime="compression",
        direction_bias="neutral", bias_value=0.0,
        decision=Decision("IC", "both", "HIGH", "theta", "note", "15m"),
        size_mult=1.0, vetoes=[], note="",
    )
    market = MarketSnapshot(
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
    return TickResult(
        ts=dt.datetime(2026, 6, 30, 10, 0, tzinfo=ET),
        regime=regime, intent=intent, decision=None,
        final_size_mult=1.0, vetoes=[],
        snapshot=TickSnapshot(market=market, bars=None, chain=None),
    )


def test_serialize_emits_live_v1_sections():
    payload = serialize_tick_result(
        _tick(), feed_source="Tradier",
        paper_summary={"equity": 1000},
        market_status={"is_open": True, "session_type": "regular"},
        feed_ages_seconds={
            "spot": 0.5, "bars": 4.0, "option_chain": 1.0, "settlement": 20.0,
        },
    )
    assert payload["schema_version"] == LIVE_SCHEMA_VERSION
    assert validate_live_v1(payload) == []
    assert_live_v1(payload)
    assert payload["feeds"]["overall_status"] == "LIVE"
    assert payload["legacy"]["source_version"] == "v1"
    assert payload["forecast"]["source_version"] == "v2"
    assert payload["v3"]["source_version"] == "v3"
    # No ambiguous V2-under-V3
    assert payload["v3"].get("source_version") != "v2"
    assert payload["forecast"].get("source_version") != "v3"
    # Compat: market session still exposes is_open under market section
    assert payload["market"]["is_open"] is True
    assert payload["legacy"]["doing"]["dominant_regime"] == "compression"
    assert "doing" not in payload
    assert payload["system"]["compat_flat_keys"] is False


def test_truthy_feed_source_alone_is_not_overall_live():
    """Age-unknown synthesis must not claim overall LIVE."""
    payload = serialize_tick_result(
        _tick(), feed_source="Tradier",
        market_status={"is_open": True},
    )
    assert payload["snapshot"]["feed_source"] == "Tradier"
    assert payload["feeds"]["overall_status"] != "LIVE"
    assert payload["feeds"]["overall_status"] in ("DEGRADED", "MISSING", "STALE")


def test_feed_status_truth_table_stale_and_missing():
    stale = serialize_tick_result(
        _tick(), feed_source="Tradier",
        feed_ages_seconds={
            "spot": 0.5, "bars": 4.0, "option_chain": 90.0, "settlement": 20.0,
        },
    )
    assert stale["feeds"]["option_chain"]["status"] == "STALE"
    assert stale["feeds"]["overall_status"] == "STALE"

    missing = serialize_tick_result(
        _tick(), feed_source="Tradier",
        feed_ages_seconds={"spot": 0.5, "bars": 4.0, "settlement": 20.0},
        # chain_available False from tick (no chain) → option_chain MISSING
    )
    assert missing["feeds"]["option_chain"]["status"] == "MISSING"
    assert missing["feeds"]["overall_status"] == "MISSING"


def test_explicit_feed_statuses_override_synthesis():
    statuses = {
        "spot": build_feed_status(
            source="spot", age_seconds=0.2, freshness_limit_seconds=5.0,
            provider="Tradier", required=True),
        "bars": build_feed_status(
            source="bars", age_seconds=3.0, freshness_limit_seconds=90.0,
            provider="Massive", required=True),
        "option_chain": build_feed_status(
            source="option_chain", age_seconds=1.0, freshness_limit_seconds=15.0,
            provider="Tradier", required=True),
        "settlement": build_feed_status(
            source="settlement", age_seconds=10.0,
            freshness_limit_seconds=86_400.0, provider="Yahoo", required=True),
    }
    payload = serialize_tick_result(_tick(), feed_statuses=statuses)
    assert payload["feeds"]["overall_status"] == "LIVE"


def test_no_v2_values_under_v3_section():
    result = _tick()
    result.signals = {
        "v2_fc_p_up_30m": 0.55,
        "v2_policy_structure": "PCS",
    }
    payload = serialize_tick_result(result, feed_source="Tradier")
    assert "p_up_30m" in (payload["forecast"].get("summary") or {}) or \
           payload["forecast"].get("p_up_30m") == 0.55
    # V3 decision block is part3 shadow, not the V2 policy structure
    assert payload["v3"]["decision"].get("mode") == "shadow" or \
           payload["v3"]["mode"] == "shadow"
    assert payload["forecast"]["parallel"] is None or \
           payload["forecast"]["parallel"].get("structure") == "PCS"


def test_heartbeat_is_live_v1_and_not_overall_live():
    now = dt.datetime(2026, 6, 30, 7, 0, tzinfo=ET)
    hb = heartbeat_state(now, status="feed_not_ready", note="waiting",
                         feed_source=None)
    assert hb["schema_version"] == LIVE_SCHEMA_VERSION
    assert validate_live_v1(hb) == []
    assert hb["feeds"]["overall_status"] != "LIVE"
    assert hb["system"]["status"] == "feed_not_ready"


def test_validate_rejects_wrong_schema_and_missing_feeds():
    assert validate_live_v1({"schema_version": "nope"}) != []
    with pytest.raises(LiveSchemaError):
        assert_live_v1({"schema_version": LIVE_SCHEMA_VERSION})


def test_api_live_serves_live_v1(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "live-v1-token")
    payload = serialize_tick_result(
        _tick(), feed_source="Tradier",
        market_status={"is_open": True},
        feed_ages_seconds={
            "spot": 0.5, "bars": 4.0, "option_chain": 1.0, "settlement": 20.0,
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        live = os.path.join(tmp, "live_state.json")
        write_live_state(live, payload)
        _configure(os.path.join(tmp, "shadow.db"),
                   os.path.join(tmp, "paper.sqlite"), live)
        client = TestClient(app)
        r = client.get("/api/live",
                       headers={"Authorization": "Bearer live-v1-token"})
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == LIVE_SCHEMA_VERSION
        assert body["feeds"]["overall_status"] == "LIVE"
        assert body == json.loads(Path(live).read_text())


def test_feeds_payload_includes_required_sources():
    statuses = synthesize_feed_statuses(feed_source="X", chain_available=True)
    feeds = feeds_payload_from_statuses(statuses)
    for src in ("spot", "bars", "option_chain", "settlement"):
        assert src in feeds
        assert "status" in feeds[src]
    assert "overall_status" in feeds
