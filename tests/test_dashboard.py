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
    assert payload["schema_version"] == "live.v1"
    assert "doing" not in payload
    assert "inputs" not in payload
    assert payload["legacy"]["doing"]["dominant_regime"] == "compression"
    assert payload["market"]["inputs"]["spot"] == 600.0
    assert payload["legacy"]["why"]["matrix_cell"] == [
        "compression", "compression", "neutral"]
    assert payload["snapshot"]["feed_source"] == "Tradier"
    assert payload["system"]["compat_flat_keys"] is False
    # continuous direction bias for the four-way quadrant / regime shading
    assert payload["legacy"]["doing"]["direction_bias"] == "neutral"
    assert payload["legacy"]["doing"]["bias_value"] == 0.0


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
    assert data["schema_version"] == "live.v1"
    assert data["legacy"]["doing"]["structure"] == "IC"
    assert "doing" not in data
    assert "feeds" in data and "overall_status" in data["feeds"]


def test_api_market_status(client):
    r = client.get("/api/market-status", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    data = r.json()
    assert "is_open" in data
    assert "seconds_until_open" in data or "seconds_until_close" in data


def test_api_trade_insights(client):
    """Trade-journal learning + validation endpoint: full payload shape even
    on a fresh install with no paper database yet."""
    r = client.get("/api/trade-insights",
                   headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["n_trades"] == 0
    assert "note" in data
    for panel in ("ev", "prob_profit", "gate_score"):
        assert panel in data["validation"]
    assert "lessons" in data["lessons"]
    assert "exit_discipline" in data["lessons"]
    assert "segments" in data["lessons"]


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


def test_api_validation_reports(monkeypatch, tmp_path):
    """/api/validation serves report history (filterable) and /api/validation/{id}
    the full decoded metrics; both degrade gracefully on legacy/missing DBs."""
    from journal import Journal

    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    db = str(tmp_path / "shadow.db")
    jrn = Journal(db)
    daily_id = jrn.log_validation_report(
        "2026-07-08", "daily", {"journal": {"win_rate": 0.6}},
        "Daily validation — healthy",
        flags=[{"flag": "insufficient_data", "severity": "info", "detail": "x"}])
    jrn.log_validation_report(
        "2026-07-06", "weekly", {"per_regime": {}}, "Weekly validation — ok")
    jrn.close()
    _configure(db, str(tmp_path / "paper.sqlite"), str(tmp_path / "live.json"))

    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}

    # unauthenticated -> 401
    assert c.get("/api/validation").status_code == 401

    r = c.get("/api/validation", headers=hdrs)
    assert r.status_code == 200
    reports = r.json()["reports"]
    assert len(reports) == 2
    assert reports[0]["report_type"] == "daily"     # newest first
    assert reports[0]["metrics"]["journal"]["win_rate"] == 0.6
    assert reports[0]["flags"][0]["flag"] == "insufficient_data"

    # type filter
    r = c.get("/api/validation?report_type=weekly", headers=hdrs)
    assert [x["report_type"] for x in r.json()["reports"]] == ["weekly"]

    # invalid type rejected
    assert c.get("/api/validation?report_type=bogus", headers=hdrs).status_code == 422

    # single report
    r = c.get(f"/api/validation/{daily_id}", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["summary"] == "Daily validation — healthy"
    assert c.get("/api/validation/9999", headers=hdrs).status_code == 404

    # missing DB degrades gracefully
    _configure(str(tmp_path / "absent.db"), str(tmp_path / "p.sqlite"),
               str(tmp_path / "live.json"))
    r = c.get("/api/validation", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["reports"] == []


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


def test_api_gex_variants_and_predictions(monkeypatch, tmp_path):
    """V2 dashboard endpoints: GEX variant comparison + PredictionBundle join."""
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    from journal import COLUMNS, Journal
    from prediction.storage import PredictionStore

    db = str(tmp_path / "shadow.db")
    pred_db = str(tmp_path / "prediction_store.sqlite")
    jrn = Journal(db)
    # Need >=3 settled rows for gex_variant_comparison. Settlement columns
    # are not part of Journal.log()'s insert set — fill them via SQL.
    for i, (pnl, disagree) in enumerate([(10.0, 0), (-5.0, 1), (3.0, 0)]):
        row = {c: None for c in COLUMNS}
        row.update({
            "ts": f"2026-07-10T10:{i:02d}:00-04:00",
            "session_date": "2026-07-10",
            "decision": "TRADE",
            "spot": 600.0 + i,
            "snapshot_id": f"snap-{i}",
            "signals_json": json.dumps({
                "gex_oi_net_gex": 1e9,
                "gex_weekly_net_gex": -1e9 if disagree else 1e9,
                "gex_volume_net_gex": 1e9,
                "gex_hybrid_net_gex": 1e9,
                "gex_disagree_sign": disagree,
                "policy_mode": "shadow",
                "policy_disagreement": disagree,
            }),
        })
        rid = jrn.log(row)
        jrn.conn.execute(
            "UPDATE evaluations SET realized_pnl=?, settled=1 WHERE id=?",
            (pnl, rid))
    jrn.conn.commit()
    jrn.close()

    store = PredictionStore(pred_db)
    store.log_prediction(
        snapshot_id="snap-2",
        model_group_version="test-v1",
        predictions={
            "p_up_30m": 0.62,
            "p_range_survive_30m": 0.55,
            "uncertainty": 0.2,
            "data_quality": 0.9,
        },
        uncertainty=0.2,
        generated_at="2026-07-10T10:02:00-04:00",
        mode="shadow",
    )
    store.conn.close()

    live = str(tmp_path / "live.json")
    write_live_state(live, {"ts": "x", "doing": {}})
    _configure(db, str(tmp_path / "paper.sqlite"), live,
               prediction_db=pred_db)

    c = TestClient(app)
    hdrs = {"Authorization": "Bearer test-secret-token"}

    r = c.get("/api/gex-variants", headers=hdrs)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] >= 3
    assert "oi" in body["variants"]
    assert body["variants"]["weekly"]["sign_disagree_vs_oi"] is not None

    r = c.get("/api/report", headers=hdrs)
    assert r.status_code == 200
    assert "gex_variant_comparison" in r.json()

    r = c.get("/api/predictions?snapshot_id=snap-2", headers=hdrs)
    assert r.status_code == 200
    pred = r.json()["prediction"]
    assert pred is not None
    assert pred["predictions"]["p_up_30m"] == 0.62

    r = c.get("/api/predictions?snapshot_id=missing", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["prediction"] is None

    # missing prediction DB still degrades
    _configure(db, str(tmp_path / "paper.sqlite"), live,
               prediction_db=str(tmp_path / "absent_pred.sqlite"))
    r = c.get("/api/predictions?snapshot_id=snap-2", headers=hdrs)
    assert r.status_code == 200
    assert r.json()["prediction"] is None


def test_api_live_missing_file_returns_live_v1_heartbeat(monkeypatch, tmp_path):
    """Absent live_state.json must still be a valid live.v1 envelope."""
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    live = str(tmp_path / "missing_live_state.json")
    _configure(str(tmp_path / "shadow.db"), str(tmp_path / "paper.sqlite"), live)
    c = TestClient(app)
    r = c.get("/api/live", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "live.v1"
    assert body["system"]["status"] == "no_live_state"
    assert "feeds" in body and "overall_status" in body["feeds"]
    assert body["feeds"]["overall_status"] != "LIVE"


def test_api_paper_merges_live_open_positions(monkeypatch, tmp_path):
    """Closed-SQL paper summary must surface live open holdings/counts."""
    from dashboard.queries import enrich_paper_summary_with_live, paper_summary
    from paper_broker import PaperBroker, PaperConfig

    monkeypatch.setenv("DASHBOARD_TOKEN", "test-secret-token")
    paper_db = str(tmp_path / "paper.sqlite")
    # Ensure the paper_trades table exists (empty closed book).
    PaperBroker(db_path=paper_db, cfg=PaperConfig(starting_cash=10_000))

    live = str(tmp_path / "live_state.json")
    live_paper = {
        "trades": 0,
        "equity": 10_000.0,
        "open_positions": 1,
        "open": [{
            "id": "pos1", "family": "PCS", "contracts": 2,
            "unrealized_pnl_dollars": -52.34, "strikes": "590/585",
        }],
        "by_track": {
            "legacy": {"trades": 0, "total_pnl": 0.0, "open_positions": 1,
                       "equity": 9947.66, "win_rate": 0.0},
            "v2": {"trades": 0, "total_pnl": 0.0, "open_positions": 0,
                   "equity": 10000.0, "win_rate": 0.0},
            "v3": {"trades": 0, "total_pnl": 0.0, "open_positions": 0,
                   "equity": 10000.0, "win_rate": 0.0},
        },
    }
    write_live_state(live, serialize_tick_result(
        _tick_result(),
        feed_source="Tradier",
        paper_summary=live_paper,
        market_status={"is_open": True},
        feed_ages_seconds={
            "spot": 0.5, "bars": 4.0, "option_chain": 1.0, "settlement": 20.0,
        },
    ))
    _configure(str(tmp_path / "shadow.db"), paper_db, live)

    closed_only = paper_summary(paper_db)
    assert closed_only.get("open_positions", 0) == 0

    merged = enrich_paper_summary_with_live(closed_only, live_paper)
    assert merged["open_positions"] == 1
    assert merged["open"][0]["id"] == "pos1"
    assert merged["by_track"]["legacy"]["open_positions"] == 1

    c = TestClient(app)
    r = c.get("/api/paper", headers={"Authorization": "Bearer test-secret-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["open_positions"] == 1
    assert body["open"][0]["unrealized_pnl_dollars"] == -52.34
    assert body["by_track"]["legacy"]["open_positions"] == 1
    assert body["source"] == "closed_sql+live_open"