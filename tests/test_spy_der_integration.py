"""Phase 5: packet publishers, decision client, dashboard reader, providers."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from integrations.spy_der.contracts import (
    DASHBOARD_SCHEMA,
    DECISION_RESPONSE_SCHEMA,
    MARKET_PACKET_SCHEMA,
    OUTCOME_PACKET_SCHEMA,
    build_market_packet,
    build_outcome_packet,
    dashboard_to_parallel_payload,
    validate_dashboard_packet,
    validate_market_packet,
)
from integrations.spy_der.dashboard_reader import (
    read_dashboard_bundle,
    read_dojo_latest,
    read_live_state,
)
from integrations.spy_der.decision_client import DecisionClient, unavailable_result
from integrations.spy_der.experience import MarketExperienceProvider
from integrations.spy_der.market_publisher import (
    build_market_packet_from_tick,
    publish_market_packet,
)
from integrations.spy_der.outcome_publisher import (
    publish_outcome_packet,
    publish_settlement_outcomes,
)
from integrations.spy_der.synthetic import SyntheticUniverseProvider


def test_build_and_validate_market_packet():
    packet = build_market_packet(
        snapshot_id="snap-1",
        session_date=date(2026, 7, 24),
        underlying_price="600.25",
        forecast_uncertainty=0.2,
        candidates=[],
        generated_at=datetime(2026, 7, 24, 14, 0, tzinfo=timezone.utc),
    )
    assert packet["schema_version"] == MARKET_PACKET_SCHEMA
    validate_market_packet(packet)


def test_publish_market_and_outcome_round_trip(tmp_path):
    packet = build_market_packet_from_tick(
        snapshot_id="snap-rt",
        session_date=date(2026, 7, 24),
        symbol="SPY",
        underlying_price=601.0,
        shadow_candidates=[],
        forecast={"median": 602.0},
        forecast_uncertainty=0.1,
        generated_at=datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc),
    )
    path = publish_market_packet(packet, root=tmp_path)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["snapshot_id"] == "snap-rt"
    assert (tmp_path / "sessions.json").is_file()

    outcome = build_outcome_packet(
        snapshot_id="snap-rt",
        session_date=date(2026, 7, 24),
        candidate_id="c1",
        action="TRADE",
        realized_pnl=12.5,
        settled=True,
        settled_at=datetime(2026, 7, 24, 20, 0, tzinfo=timezone.utc),
    )
    assert outcome["schema_version"] == OUTCOME_PACKET_SCHEMA
    opath = publish_outcome_packet(outcome, root=tmp_path)
    assert opath.is_file()

    provider = MarketExperienceProvider(tmp_path)
    sessions = list(provider.sessions())
    assert date(2026, 7, 24) in sessions
    snaps = list(provider.snapshots(date(2026, 7, 24)))
    assert len(snaps) == 1
    assert provider.outcome("snap-rt")["realized_pnl"] == "12.5"


def test_publish_settlement_outcomes(tmp_path):
    rows = [
        {
            "snapshot_id": "s-a",
            "decision": "TRADE",
            "candidate_id": "cand-1",
            "realized_pnl": 3.0,
            "track": "spy_der",
        },
        {"decision": "NO_TRADE"},  # skipped — no snapshot_id
    ]
    written = publish_settlement_outcomes(
        session_date="2026-07-24",
        symbol="SPY",
        rows=rows,
        settled_at=datetime(2026, 7, 24, 20, 15, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(written) == 1
    assert (tmp_path / "outcomes" / "s-a.json").is_file()


def test_decision_client_success():
    market = build_market_packet(
        snapshot_id="s1",
        session_date="2026-07-24",
        underlying_price=600,
    )
    decision = {
        "schema_version": DASHBOARD_SCHEMA,
        "generated_at": "2026-07-24T15:00:00+00:00",
        "mode": "shadow",
        "action": "TRADE",
        "candidate_id": "c1",
        "confidence": 0.7,
        "uncertainty": 0.3,
        "size_scalar": 0.5,
        "structure": "put_credit",
        "direction": "bearish",
        "provider": "spy_der",
        "available": True,
        "rationale": "ok",
        "reason_codes": ["edge"],
    }

    class OkTransport:
        def post_json(self, url, payload, timeout):
            assert payload["schema_version"].startswith("spyder.decision")
            assert payload["market"]["snapshot_id"] == "s1"
            return {
                "schema_version": DECISION_RESPONSE_SCHEMA,
                "request_id": payload["request_id"],
                "decision": decision,
            }

    client = DecisionClient(transport=OkTransport(), retries=0)
    result = client.decide(market, request_id="req-1")
    assert result.action == "TRADE"
    assert result.candidate_id == "c1"
    payload = result.as_parallel_payload()
    assert payload["track"] == "spy_der"
    assert payload["action"] == "TRADE"


def test_decision_client_timeout_fallback_and_retry():
    calls = {"n": 0}

    class Flaky:
        def post_json(self, url, payload, timeout):
            calls["n"] += 1
            raise TimeoutError("slow")

    sleeps: list[float] = []
    client = DecisionClient(
        transport=Flaky(),
        retries=2,
        retry_backoff_s=0.01,
        sleeper=lambda s: sleeps.append(s),
    )
    market = build_market_packet(
        snapshot_id="s2", session_date="2026-07-24", underlying_price=1
    )
    result = client.decide(market)
    assert result.action == "UNAVAILABLE"
    assert result.available is False
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_dashboard_reader_missing_and_valid(tmp_path):
    missing = read_live_state(tmp_path / "nope.json")
    assert "note" in missing

    live = {
        "schema_version": DASHBOARD_SCHEMA,
        "generated_at": "2026-07-24T15:00:00+00:00",
        "mode": "shadow",
        "action": "ABSTAIN",
        "candidate_id": None,
        "confidence": 0.0,
        "uncertainty": 1.0,
        "provider": "spy_der",
        "available": True,
        "dojo": {"latest_status": "OK", "summary": "fine"},
    }
    live_path = tmp_path / "live_state.json"
    live_path.write_text(json.dumps(live), encoding="utf-8")
    validate_dashboard_packet(live)
    got = read_live_state(live_path)
    assert got["action"] == "ABSTAIN"

    dojo_path = tmp_path / "latest.json"
    dojo_path.write_text(
        json.dumps({"report_date": "2026-07-24", "summary": "ok", "flags": []}),
        encoding="utf-8",
    )
    assert read_dojo_latest(dojo_path)["summary"] == "ok"
    bundle = read_dashboard_bundle(live_path=live_path, dojo_path=dojo_path)
    assert bundle["live"]["action"] == "ABSTAIN"
    assert bundle["dojo"]["summary"] == "ok"


def test_dashboard_reader_schema_mismatch_soft_fail(tmp_path):
    bad = {"schema_version": "not.a.schema", "generated_at": "2026-07-24T15:00:00+00:00"}
    path = tmp_path / "live.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    out = read_live_state(path)
    assert "note" in out
    assert out.get("raw") == bad


def test_dashboard_to_parallel_payload():
    payload = dashboard_to_parallel_payload(
        {
            "schema_version": DASHBOARD_SCHEMA,
            "provider": "spy_der",
            "mode": "shadow",
            "action": "TRADE",
            "candidate_id": "x",
            "size_scalar": 0.25,
            "confidence": 0.8,
            "uncertainty": 0.2,
            "rationale": "r",
            "reason_codes": ["a"],
            "available": True,
        }
    )
    assert payload["size_cap"] == 0.25
    assert payload["label"] == "SPY-DER"


def test_unavailable_result():
    r = unavailable_result("down")
    assert r.action == "UNAVAILABLE"
    assert r.as_parallel_payload()["available"] is False


def test_synthetic_universe_provider_emits_packets():
    from matrix_universe import UniverseCatalog

    catalog = UniverseCatalog(seed=1)
    specs = catalog.sample(1)
    provider = SyntheticUniverseProvider(max_ticks=3)
    packets = list(provider.generate(specs[0]))
    assert len(packets) == 3
    assert packets[0]["schema_version"] == MARKET_PACKET_SCHEMA
    assert packets[0]["snapshot_id"]


def test_champion_reader_build_engine_cfg():
    from decision_engine import EngineConfig
    from integrations.spy_der.champion_reader import build_engine_cfg, load_config

    base = EngineConfig()
    cfg = build_engine_cfg(base, {"gate.max_adx": 22.0})
    assert cfg.gate.max_adx == 22.0

    # round-trip a minimal champion file
    import tempfile
    import os

    rec = {
        "config_id": "abc123def456",
        "created_at": "2026-07-24T00:00:00+00:00",
        "label": "test",
        "overrides": {"gate.max_adx": 19.0},
        "regime_overrides": {},
        "status": "promoted",
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "champion.json")
        Path(path).write_text(json.dumps(rec), encoding="utf-8")
        loaded = load_config(path)
        eng, _ = loaded.engine_cfg()
        assert eng.gate.max_adx == 19.0


def test_unified_loop_spy_der_fallback_offline(tmp_path, monkeypatch):
    """When HTTP is down, spy_der track reports UNAVAILABLE without raising."""
    from zoneinfo import ZoneInfo

    monkeypatch.setenv("ZERODTE_SPYDER_EXPERIENCE_ROOT", str(tmp_path))

    class Boom:
        def post_json(self, url, payload, timeout):
            raise RuntimeError("down")

    client = DecisionClient(transport=Boom(), retries=0, sleeper=lambda _s: None)
    ET = ZoneInfo("America/New_York")
    now = datetime(2026, 7, 24, 11, 0, tzinfo=ET)
    packet = build_market_packet_from_tick(
        snapshot_id="tick-test-1",
        session_date=now.date(),
        symbol="SPY",
        underlying_price=600.0,
        shadow_candidates=[],
        generated_at=now,
    )
    publish_market_packet(packet, root=tmp_path)
    result = client.decide(packet)
    payload = result.as_parallel_payload()
    assert payload["action"] == "UNAVAILABLE"
    assert payload["available"] is False
    assert (tmp_path / "snapshots" / "tick-test-1.json").is_file()
