from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from grok_trader.config import GrokConfig
from grok_trader.risk import RiskFirewall
from paper_broker import PaperBroker, PaperConfig
from rnd_extractor import ChainQuote, ChainSnapshot

ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2026, 7, 16, 11, 0, tzinfo=ET)


def _chain() -> ChainSnapshot:
    rows = []
    for strike in (615.0, 617.0, 618.0, 620.0, 622.0, 623.0, 625.0):
        distance = abs(strike - 620.0)
        call_mid = max(0.20, 2.00 - 0.20 * distance)
        put_mid = max(0.20, 2.00 - 0.20 * distance)
        rows.append(ChainQuote(
            strike=strike,
            call_bid=call_mid - 0.05,
            call_ask=call_mid + 0.05,
            put_bid=put_mid - 0.05,
            put_ask=put_mid + 0.05,
        ))
    return ChainSnapshot(rows, spot=620.0, t_years=2e-4)


def _result():
    return SimpleNamespace(
        snapshot=SimpleNamespace(
            chain=_chain(),
            market=SimpleNamespace(quote_age_seconds=1.0),
        ),
        signals={"session_warmup": 0.0},
    )


def _cfg() -> GrokConfig:
    return GrokConfig(
        enabled=True,
        submission_enabled=True,
        paper_only=True,
        api_key="test",
        max_leg_relative_spread=2.0,
    )


def _broker(tmp_path):
    return PaperBroker(
        db_path=str(tmp_path / "paper.sqlite"),
        cfg=PaperConfig(risk_per_trade_frac=0.50),
        symbol="SPY",
    )


def _plan(family: str, legs: list[dict]) -> dict:
    return {
        "symbol": "SPY",
        "expiration": "0DTE",
        "family": family,
        "direction": "neutral",
        "legs": legs,
        "limit_price": 5.00,
        "risk_fraction": 0.05,
        "confidence": 0.70,
        "thesis": "synthetic defined-risk structure validation",
        "supporting_evidence": [],
        "contradictory_evidence": [],
        "invalidation_conditions": [],
        "mandatory_exit_time": "15:30",
    }


def test_requested_premium_families_are_included_with_simple_directionals():
    families = set(_cfg().allowed_families)
    assert {
        "put_credit", "call_credit", "iron_condor", "iron_fly", "broken_wing"
    } <= families
    assert {
        "long_call", "long_put", "long_call_spread", "long_put_spread"
    } <= families


def test_iron_fly_is_approved_and_preserves_family(tmp_path):
    plan = _plan("iron_fly", [
        {"kind": "P", "strike": 617.0, "side": "buy"},
        {"kind": "P", "strike": 620.0, "side": "sell"},
        {"kind": "C", "strike": 620.0, "side": "sell"},
        {"kind": "C", "strike": 623.0, "side": "buy"},
    ])
    plan["limit_price"] = 0.01
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert out.approved, out.reasons
    assert out.paper_intent["candidate"].family == "iron_fly"
    assert out.paper_intent["structure"] == "iron_fly"


@pytest.mark.parametrize("kind,strikes", [
    ("P", (615.0, 618.0, 622.0)),
    ("C", (618.0, 622.0, 625.0)),
])
def test_broken_wing_is_approved_with_two_short_body_units(tmp_path, kind, strikes):
    low, body, high = strikes
    plan = _plan("broken_wing", [
        {"kind": kind, "strike": low, "side": "buy", "quantity": 1},
        {"kind": kind, "strike": body, "side": "sell", "quantity": 2},
        {"kind": kind, "strike": high, "side": "buy", "quantity": 1},
    ])
    plan["limit_price"] = 0.01
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert out.approved, out.reasons
    assert out.paper_intent["candidate"].family == "broken_wing"
    quantities = sorted(abs(leg.qty) for leg in out.paper_intent["candidate"].legs)
    assert quantities == [1, 1, 2]


def test_equal_wing_butterfly_is_not_accepted_as_broken_wing(tmp_path):
    plan = _plan("broken_wing", [
        {"kind": "P", "strike": 617.0, "side": "buy"},
        {"kind": "P", "strike": 620.0, "side": "sell", "quantity": 2},
        {"kind": "P", "strike": 623.0, "side": "buy"},
    ])
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert not out.approved
    assert "legs_do_not_match_defined_risk_family" in out.reasons


@pytest.mark.parametrize("family,kind,direction", [
    ("long_call", "C", "bullish"),
    ("long_put", "P", "bearish"),
])
def test_simple_long_option_directional_is_approved(tmp_path, family, kind, direction):
    plan = _plan(family, [
        {"kind": kind, "strike": 620.0, "side": "buy"},
    ])
    plan["direction"] = direction
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert out.approved, out.reasons
    candidate = out.paper_intent["candidate"]
    assert candidate.family == family
    assert len(candidate.legs) == 1
    assert candidate.legs[0].qty == 1
    assert out.diagnostics["max_loss_per_share"] > 0


def test_single_short_option_is_rejected_as_undefined_risk(tmp_path):
    plan = _plan("long_call", [
        {"kind": "C", "strike": 620.0, "side": "sell"},
    ])
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert not out.approved
    assert "legs_do_not_match_defined_risk_family" in out.reasons


def test_bullish_debit_spread_remains_available(tmp_path):
    plan = _plan("long_call_spread", [
        {"kind": "C", "strike": 620.0, "side": "buy"},
        {"kind": "C", "strike": 623.0, "side": "sell"},
    ])
    out = RiskFirewall(_cfg()).validate_entry(
        now=NOW, plan=plan, result=_result(), broker=_broker(tmp_path),
        allow_new_entry=True,
    )
    assert out.approved, out.reasons
    assert out.paper_intent["candidate"].family == "long_call_spread"
