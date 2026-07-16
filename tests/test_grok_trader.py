from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from grok_trader.config import GrokConfig
from grok_trader.evidence import EvidenceTerminal, assert_decision_blind
from grok_trader.integration import register_grok_track
from grok_trader.risk import RiskFirewall
from rnd_extractor import ChainQuote, ChainSnapshot

ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2026, 7, 16, 11, 0, tzinfo=ET)


def _chain():
    def q(k, cmid, pmid):
        return ChainQuote(k, cmid - .05, cmid + .05, pmid - .05, pmid + .05)
    return ChainSnapshot(
        [q(620, 6.0, .35), q(622, 4.2, .55), q(624, 2.7, 1.05),
         q(626, 1.4, 2.2), q(628, .65, 4.0), q(630, .30, 6.0)],
        spot=625.0,
        t_years=2e-4,
    )


def _broker(tmp_path):
    from paper_broker import PaperBroker, PaperConfig
    register_grok_track()
    return PaperBroker(
        db_path=str(tmp_path / "paper.sqlite"),
        cfg=PaperConfig(starting_cash=10_000, risk_per_trade_frac=.50),
        symbol="SPY",
    )


def _result():
    market = SimpleNamespace(spot=625.0, gamma_flip=624.0, call_wall=630.0,
                             put_wall=620.0, quote_age_seconds=1.0)
    snap = SimpleNamespace(
        market=market,
        bars=SimpleNamespace(close=[623.0, 624.0, 625.0]),
        chain=_chain(),
        option_rows=[{"strike": 624, "delta": .45}],
        weekly_option_rows=[],
    )
    return SimpleNamespace(
        snapshot=snap,
        regime=SimpleNamespace(dominant_regime="range_compression",
                               permitted_engine="premium", stand_down=True),
        signals={
            "legacy_policy_structure": "PCS",
            "v2_probability_up": .61,
            "v2_policy_action": "TRADE",
            "v3_action": "NO_TRADE",
            "rnd_variance_ratio": 1.12,
            "session_warmup": 0.0,
        },
        part3={"decision_summary": {"action": "TRADE"},
               "scenario_probabilities": {"up": .57}},
        vetoes=[],
        sigma_cones=None,
    )


def test_config_fails_closed_without_key(monkeypatch):
    monkeypatch.setenv("GROK_ENABLED", "1")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        GrokConfig.from_env()


def test_evidence_removes_engine_decisions(tmp_path):
    terminal = EvidenceTerminal(
        now=NOW,
        result=_result(),
        broker=_broker(tmp_path),
        symbol="SPY",
        max_rows=100,
    )
    data = terminal.summary()
    assert_decision_blind(data)
    text = str(data)
    assert "v2_policy_action" not in text
    assert "decision_summary" not in text
    assert "legacy_policy_structure" not in text
    assert "stand_down" not in text
    assert data["v2_analysis"]["v2_probability_up"] == pytest.approx(.61)


def test_firewall_accepts_defined_risk_put_credit(tmp_path):
    cfg = GrokConfig(enabled=False, submission_enabled=True, paper_only=True)
    broker = _broker(tmp_path)
    fw = RiskFirewall(cfg).validate_entry(
        now=NOW,
        result=_result(),
        broker=broker,
        allow_new_entry=True,
        plan={
            "symbol": "SPY",
            "expiration": "0DTE",
            "family": "put_credit",
            "direction": "bullish",
            "legs": [
                {"kind": "P", "strike": 622, "side": "sell"},
                {"kind": "P", "strike": 620, "side": "buy"},
            ],
            "limit_price": .15,
            "risk_fraction": .02,
            "confidence": .66,
            "thesis": "Support above the put wall.",
            "invalidation_conditions": ["acceptance below 620"],
            "mandatory_exit_time": "15:45",
        },
    )
    assert fw.approved, fw.reasons
    assert fw.paper_intent["track"] == "grok"
    assert fw.paper_intent["size_mult"] == pytest.approx(.04)


def test_firewall_rejects_naked_leg(tmp_path):
    cfg = GrokConfig(enabled=False, submission_enabled=True, paper_only=True)
    fw = RiskFirewall(cfg).validate_entry(
        now=NOW,
        result=_result(),
        broker=_broker(tmp_path),
        allow_new_entry=True,
        plan={
            "symbol": "SPY",
            "expiration": "0DTE",
            "family": "put_credit",
            "direction": "bullish",
            "legs": [{"kind": "P", "strike": 622, "side": "sell"}],
            "limit_price": .50,
            "risk_fraction": .02,
            "confidence": .5,
            "thesis": "bad",
        },
    )
    assert not fw.approved
    assert "legs_do_not_match_defined_risk_family" in fw.reasons


def test_firewall_rejects_invalid_mandatory_exit(tmp_path):
    cfg = GrokConfig(enabled=False, submission_enabled=True, paper_only=True)
    fw = RiskFirewall(cfg).validate_entry(
        now=NOW, result=_result(), broker=_broker(tmp_path), allow_new_entry=True,
        plan={
            "symbol": "SPY", "expiration": "0DTE", "family": "put_credit",
            "direction": "bullish",
            "legs": [
                {"kind": "P", "strike": 622, "side": "sell"},
                {"kind": "P", "strike": 620, "side": "buy"},
            ],
            "limit_price": .15, "risk_fraction": .02, "confidence": .66,
            "thesis": "Support", "mandatory_exit_time": "16:05",
        },
    )
    assert not fw.approved
    assert "mandatory_exit_after_broker_eod" in fw.reasons
