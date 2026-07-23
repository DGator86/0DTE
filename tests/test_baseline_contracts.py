"""
Post-PR119 baseline contract locks.

PR #119 restored main to the PR #118 baseline tree. These tests turn the
runtime-contract inventory in docs/BASELINE_POST_PR119.md into regression
tests, so that later integration PRs (feed status, canonical snapshot,
versioned /api/live, dashboard migration, prediction runtime, decision stack)
change each contract *explicitly*: an intentional contract change must update
the corresponding snapshot here in the same PR.

No test here exercises new behavior — they only pin down what exists.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from dashboard.server import app, _configure
from dashboard.state import heartbeat_state, serialize_tick_result, write_live_state
from decision_matrix import Decision, TradeIntent
from gate_scorer import MarketSnapshot
from regime_classifier import RegimeState
from unified_loop import TickResult, TickSnapshot, UnifiedOrchestrator

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "live_state_baseline.json"
APP_JS = ROOT / "dashboard" / "static" / "app.js"


# --------------------------------------------------------------------------- #
# Synthetic tick — deterministic inputs so the serialized shape is stable.    #
# --------------------------------------------------------------------------- #
def _market() -> MarketSnapshot:
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


def _tick_result() -> TickResult:
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
        decision=Decision("IC", "both", "HIGH", "theta",
                          "shorts inside walls", "15m"),
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


def baseline_live_payload() -> dict:
    """The production-shaped /api/live payload for the synthetic tick,
    passed through the same JSON round trip write_live_state performs."""
    payload = serialize_tick_result(
        _tick_result(),
        feed_source="Tradier",
        paper_summary={"trades": 0, "equity": 1000.0},
        market_status={"is_open": True, "session_type": "regular"},
    )
    return json.loads(json.dumps(payload, default=str))


def _shape(obj):
    """Recursive key/type skeleton — values vary, structure must not."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return ["list"]
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, (int, float)):
        return "number"
    return "str"


# --------------------------------------------------------------------------- #
# 1. Serializer contract                                                      #
# --------------------------------------------------------------------------- #
def test_live_state_top_level_keys():
    keys = set(baseline_live_payload().keys())
    # live.v1 sections only (PR D removed flat aliases)
    assert keys == {
        "schema_version", "generated_at", "snapshot", "feeds", "market",
        "legacy", "forecast", "v3", "accounts", "risk", "paper", "system",
    }
    assert baseline_live_payload()["system"]["compat_flat_keys"] is False


def test_live_state_shape_matches_committed_fixture():
    """The full recursive shape of serialize_tick_result output is pinned by
    tests/fixtures/live_state_baseline.json. An intentional serializer change
    must regenerate the fixture (see fixture header) in the same PR."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert _shape(baseline_live_payload()) == _shape(fixture["payload"])


def test_live_state_has_versioned_live_v1_contract():
    """PR C/D: schema_version + per-source feeds; no flat feed aliases."""
    from dashboard.live_schema import LIVE_SCHEMA_VERSION, validate_live_v1
    payload = baseline_live_payload()
    assert payload["schema_version"] == LIVE_SCHEMA_VERSION
    assert "feeds" in payload
    assert "overall_status" in payload["feeds"]
    assert validate_live_v1(payload) == []
    assert "feed_source" not in payload
    assert "chain_available" not in payload
    assert payload["snapshot"]["feed_source"] == "Tradier"
    assert payload["snapshot"]["chain_available"] is False
    assert payload["feeds"]["option_chain"]["status"] == "MISSING"


def test_heartbeat_state_contract():
    from dashboard.live_schema import LIVE_SCHEMA_VERSION, validate_live_v1
    now = dt.datetime(2026, 6, 30, 7, 0, tzinfo=ET)
    hb = heartbeat_state(now, status="market_closed", note="closed",
                         feed_source=None, paper_summary=None,
                         market_status={"is_open": False})
    assert hb["schema_version"] == LIVE_SCHEMA_VERSION
    assert validate_live_v1(hb) == []
    assert hb["system"]["status"] == "market_closed"
    assert "status" not in hb
    assert hb["feeds"]["overall_status"] != "LIVE"
    # The three no-tick statuses shadow_runner emits (real ticks use "live").
    for status in ("market_closed", "feed_not_ready", "feed_error"):
        assert heartbeat_state(now, status=status, note="n")["system"]["status"] == status


# --------------------------------------------------------------------------- #
# 2. /api/live serves the file verbatim (no envelope, no validation)          #
# --------------------------------------------------------------------------- #
def test_api_live_serves_serialized_file_verbatim(monkeypatch):
    monkeypatch.setenv("DASHBOARD_TOKEN", "baseline-test-token")
    payload = baseline_live_payload()
    with tempfile.TemporaryDirectory() as tmp:
        live = os.path.join(tmp, "live_state.json")
        write_live_state(live, payload)
        _configure(os.path.join(tmp, "shadow.db"),
                   os.path.join(tmp, "paper.sqlite"), live)
        client = TestClient(app)
        r = client.get("/api/live",
                       headers={"Authorization": "Bearer baseline-test-token"})
        assert r.status_code == 200
        assert r.json() == json.loads(Path(live).read_text())


# --------------------------------------------------------------------------- #
# 3. Dashboard server route inventory                                         #
# --------------------------------------------------------------------------- #
def test_dashboard_route_inventory():
    routes = {r.path for r in app.routes if isinstance(r, APIRoute)}
    assert routes == {
        "/",
        "/api/health",
        "/api/market-status",
        "/api/live",
        "/api/ticks",
        "/api/ticks/{row_id}",
        "/api/paper",
        "/api/competition",
        "/api/trades",
        "/api/ras",
        "/api/report",
        "/api/gex-variants",
        "/api/predictions",
        "/api/sigma-cones",
        "/api/validation",
        "/api/validation/{report_id}",
        "/api/learning",
        "/api/candidates",
        "/api/promotions",
        "/api/feature-scores",
        "/api/drift",
        "/api/readiness",
        "/api/stream",
    }
    # Read-only server: every API route is GET-only.
    for r in app.routes:
        if isinstance(r, APIRoute):
            assert r.methods <= {"GET", "HEAD"}, r.path


# --------------------------------------------------------------------------- #
# 4. Dashboard frontend inventory (app.js)                                    #
# --------------------------------------------------------------------------- #
def _appjs() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_appjs_render_and_refresh_inventory():
    found = set(re.findall(r"function ((?:render|refresh)[A-Za-z0-9_]*)\(",
                           _appjs()))
    assert found == {
        "refresh", "refreshJournal", "refreshLearning", "refreshPrediction",
        "refreshValidation",
        "renderCompetition",
        "renderConeCoverage", "renderConeJournal", "renderDynamics",
        "renderEdge", "renderFeatureImpactDetail", "renderForecast",
        "renderFunnel", "renderGexVariants", "renderJournal",
        "renderLearningBadge", "renderLearningCandidates",
        "renderLearningDiagnostics", "renderLearningDrift",
        "renderLearningFeatures", "renderLearningPromotions",
        "renderLearningRuns", "renderLearningV2Status", "renderLiveCones",
        "renderMarketPill", "renderOpenPositions", "renderPaper",
        "renderParallel", "renderPart3", "renderPhysDensity", "renderPin",
        "renderPlaybook", "renderPolicy", "renderPredict", "renderRanker",
        "renderReadiness", "renderReason", "renderRegime", "renderSigCorr",
        "renderSignal", "renderSpyder", "renderSpyderContext", "renderSpyderOpen",
        "renderSpyderPrediction", "renderSpyderTrades",
        "renderSpyderUsage", "renderSpyderVs", "renderTech", "renderTimeline",
        "renderTopbar",
        "renderV2Funnel", "renderV2OpenPositions", "renderV2Paper",
        "renderV2Playbook", "renderV2Regime", "renderV2Signal",
        "renderV2Timeline", "renderV2Why", "renderValDetail", "renderValList",
        "renderValidationBadge", "renderVol", "renderWhy",
    }


def test_appjs_api_endpoint_inventory():
    found = set(re.findall(r'api\("(/api/[a-z-]+)', _appjs()))
    assert found == {
        "/api/candidates", "/api/drift", "/api/feature-scores",
        "/api/learning", "/api/live", "/api/market-status", "/api/paper",
        "/api/predictions", "/api/promotions", "/api/readiness",
        "/api/report", "/api/sigma-cones", "/api/ticks", "/api/trades",
        "/api/validation",
    }


def test_appjs_single_polling_loop():
    """One setInterval(refresh, ...) drives the page; refresh() schema-validates
    live.v1 once before the render cycle."""
    assert len(re.findall(r"setInterval\(refresh\b", _appjs())) == 1
    assert "requireLiveV1" in _appjs()
    assert "showLiveUnavailable" in _appjs()


def test_appjs_no_ambiguous_cross_version_fallbacks():
    """PR D: no v2_policy_* || policy_* cross-version reads in serializer or UI."""
    state = (ROOT / "dashboard" / "state.py").read_text(encoding="utf-8")
    assert 'or raw_signals.get("policy_structure")' not in state
    assert 'or raw_signals.get("policy_action")' not in state
    assert 'or raw_signals.get("policy_direction")' not in state
    assert 'or raw_signals.get("policy_confidence")' not in state
    assert 'or raw_signals.get("policy_uncertainty")' not in state
    js = _appjs()
    assert "v2_policy_action || s.policy_action" not in js
    assert "v2_policy_structure || s.policy_structure" not in js
    assert "v2_policy_direction || s.policy_direction" not in js
    assert "live.chain_available" not in js
    assert "live.feed_source" not in js
    assert "live.doing" not in js
    assert "live.v2_signals" not in js
    assert "meta-feeds" in js
    assert "liveParallel" in js
    assert "liveInputs" in js
    assert "liveDoing" in js


def test_appjs_consumes_feeds_section():
    js = _appjs()
    assert "feeds.overall_status" in js
    assert "option_chain" in js
    assert "feedStatusCls" in js
    assert "LIVE_SCHEMA_VERSION" in js
    assert "live.v1" in js

# --------------------------------------------------------------------------- #
# 5. Paper accounts and risk-manager wiring                                   #
# --------------------------------------------------------------------------- #
def test_paper_tracks_inventory():
    from paper_broker import PAPER_TRACKS, PaperBroker, PaperConfig
    assert PAPER_TRACKS == ("legacy", "v2", "v3", "spy_der")
    with tempfile.TemporaryDirectory() as tmp:
        broker = PaperBroker(db_path=os.path.join(tmp, "paper.sqlite"),
                             cfg=PaperConfig())
        # One broker, one sqlite, one ledger per track.
        assert set(broker.ledgers.keys()) == set(PAPER_TRACKS)
        assert broker.cfg.parallel_tracks is True
        assert broker.cfg.max_open_positions == 1  # per-track cap


def test_single_risk_manager_wiring():
    """UnifiedOrchestrator holds exactly one optional RiskManager, shared by
    every track — candidate-mode isolation (separate ledgers + risk managers
    per account) does not exist yet and must be added explicitly."""
    from risk_manager import RiskManager
    field = UnifiedOrchestrator.__dataclass_fields__["risk_manager"]
    assert field.default is None
    # The interface tick() depends on:
    for method in ("check", "record_trade", "close_positions", "status"):
        assert callable(getattr(RiskManager, method))


# --------------------------------------------------------------------------- #
# 6. Model registry fail-closed baseline                                      #
# --------------------------------------------------------------------------- #
def test_registry_load_fails_closed_on_missing_model():
    from prediction.registry import ModelRegistry, RegistryError
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(directory=tmp)
        with pytest.raises(RegistryError):
            reg.load("does-not-exist")


def test_registry_load_fails_closed_on_tampered_artifact():
    from prediction.registry import ModelRegistry, RegistryError
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(directory=tmp)
        model_id = reg.save(
            {"kind": "baseline-lock"}, model_type="test", target="t",
            horizon=None, feature_version="fv1", label_version="lv1",
        )
        artifact = os.path.join(tmp, f"{model_id}.joblib")
        with open(artifact, "ab") as f:
            f.write(b"tamper")
        with pytest.raises(RegistryError, match="hash mismatch"):
            reg.load(model_id)
