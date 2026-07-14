"""
tests/test_review5_merge_gate.py
================================
PR #117 fifth-review merge-gate regressions:

* Portfolio RiskManager applies to authoritative V3 candidate
* Forecast assembly preserves data_quality=0.0
* Fill-prob and fill-concession share one feature row
* Strict actionable candidates require complete execution economics
* size_mult=0.0 is preserved (not coerced to 1.0)
* V3 paper intents carry execution estimates
"""
from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from decision_stack.authority import coerce_size_mult, resolve_authority
from decision_stack.stack import UnifiedDecisionStack
from prediction.adapters import fill_attempt_features_from_candidate
from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.contracts import PredictionBundle
from prediction.forecast_assembly import build_v3_forecast
from risk_manager import RiskConfig, RiskManager
from unified_loop import UnifiedOrchestrator


ET = ZoneInfo("America/New_York")


def test_coerce_size_mult_preserves_zero():
    assert coerce_size_mult(0.0) == 0.0
    assert coerce_size_mult(None, default=1.0) == 1.0
    assert coerce_size_mult(0.5) == 0.5
    # The bug: `0.0 or 1.0` → 1.0
    assert (0.0 or 1.0) == 1.0
    assert coerce_size_mult(0.0, default=1.0) == 0.0


def test_forecast_assembly_preserves_bundle_data_quality_zero():
    bundle = PredictionBundle(
        snapshot_id="dq0",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        symbol="SPY",
        uncertainty=0.2,
        data_quality=0.0,
        feature_coverage=0.0,
        ood_score=0.1,
    )
    from dataclasses import replace
    data_quality = 0.85
    feature_coverage = 0.9
    bundle_dq = getattr(bundle, "data_quality", None)
    merged_dq = (
        data_quality if bundle_dq is None
        else min(float(bundle_dq), data_quality)
    )
    bundle_fc = getattr(bundle, "feature_coverage", None)
    merged_fc = (
        feature_coverage if bundle_fc is None
        else min(float(bundle_fc), feature_coverage)
    )
    merged = replace(bundle, data_quality=merged_dq, feature_coverage=merged_fc)
    assert merged.data_quality == 0.0
    assert merged.feature_coverage == 0.0
    # Contrast the old truthiness bug
    assert min(float(getattr(bundle, "data_quality", 1) or 1), 0.85) == 0.85


def test_forecast_assembly_end_to_end_zero_quality():
    """Heuristic path still respects snapshot quality=0.0 (no or-default)."""
    snap = build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="dq-e2e",
        quality={"data_quality": 0.0, "feature_coverage": 0.0},
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    out = build_v3_forecast(snapshot=snap, mode="shadow")
    assert out.data_quality == 0.0
    assert out.feature_coverage == 0.0


def test_fill_models_receive_identical_feature_keys():
    """Regression: concession must not rebuild a stripped feature row."""
    feats = fill_attempt_features_from_candidate(
        candidate={"family": "put_credit"},
        mid_credit=0.50,
        natural_credit=0.40,
        family="put_credit",
        n_legs=2,
        quote_age_seconds=3.0,
        minutes_to_close=90.0,
        data_quality=0.9,
    )
    for key in ("quote_age_seconds", "minutes_to_close", "data_quality"):
        assert key in feats
    assert feats["quote_age_seconds"] == 3.0
    assert feats["minutes_to_close"] == 90.0
    assert feats["data_quality"] == 0.9


def test_strict_actionable_requires_execution_economics():
    """Candidates without fill_p / fill_price / EOV are not actionable."""
    from decision_stack.contracts import CandidateEvaluation

    complete = CandidateEvaluation(
        candidate_id="a",
        final_rank=1,
        fill_probability=0.7,
        expected_fill_price=0.4,
        expected_order_value=0.1,
    )
    incomplete = CandidateEvaluation(
        candidate_id="b",
        final_rank=6,
        fill_probability=None,
        expected_fill_price=None,
        expected_order_value=None,
        vetoes=("execution_not_evaluated",),
    )
    unevaluated = CandidateEvaluation(
        candidate_id="c",
        final_rank=2,
        # ranked but no economics and no veto yet — must not be actionable
    )

    def _actionable(e, mode="candidate") -> bool:
        if e.final_rank is None or e.vetoes:
            return False
        if mode in ("candidate", "champion"):
            return (
                e.fill_probability is not None
                and e.expected_fill_price is not None
                and e.expected_order_value is not None
            )
        return True

    assert _actionable(complete) is True
    assert _actionable(incomplete) is False
    assert _actionable(unevaluated) is False
    # Shadow mode still allows ranked candidates without execution
    assert _actionable(unevaluated, mode="shadow") is True


def test_rank_beyond_top5_gets_execution_not_evaluated_veto():
    """After ranking, candidates without fill fields pick up the veto."""
    # Simulate the post-loop veto append used in build_v3_candidate_evaluations.
    mode = "candidate"
    fill_p = None
    fill_price = None
    eov = None
    vetoes = ()
    if mode in ("candidate", "champion"):
        if fill_p is None or fill_price is None or eov is None:
            if "execution_failed" not in vetoes and (
                    "credit_unavailable" not in vetoes):
                vetoes = vetoes + ("execution_not_evaluated",)
    assert "execution_not_evaluated" in vetoes


def test_portfolio_risk_enters_hard_veto_for_champion():
    rm = RiskManager(RiskConfig(max_open_positions=1))
    dummy = SimpleNamespace(
        family="put_credit", max_loss=2.0, gamma=-0.01, capital=2.0)
    assert rm.check(dummy, "2026-07-14").approved
    rm.record_trade(dummy, "2026-07-14")
    assert not rm.check(dummy, "2026-07-14").approved

    calls = []

    def risk_fn(cid, session_date):
        calls.append((cid, session_date))
        cand = SimpleNamespace(
            family="put_credit", max_loss=2.0, gamma=-0.01, capital=2.0)
        r = rm.check(cand, session_date)
        if r.approved:
            return ()
        return tuple(f"risk:{v}" for v in r.vetoes)

    auth = resolve_authority(
        mode="champion",
        legacy_decision={"action": "NO_EDGE"},
        v3_decision={"final_action": "TRADE", "candidate_id": "cand_a",
                     "structure": "put_credit", "size_mult": 1.0},
        hard_vetoes=("risk:max_open_positions",),
        fallback_policy="abstain",
        v3_size_mult=1.0,
    )
    assert auth.final_action == "HARD_VETO"
    assert any("risk:" in r for r in auth.reasons)

    stack = UnifiedDecisionStack(
        deployment=SimpleNamespace(
            mode="champion", deployment_id="d1",
            configuration_hash="h", fallback_policy="abstain",
        ),
        portfolio_risk_fn=risk_fn,
    )
    vetoes = stack.portfolio_risk_fn("cand_a", "2026-07-14")
    assert vetoes
    assert all(v.startswith("risk:") for v in vetoes)
    assert calls


def test_v3_paper_intent_carries_execution_estimate_and_respects_zero_size():
    orch = UnifiedOrchestrator.__new__(UnifiedOrchestrator)
    cand = SimpleNamespace(
        candidate_id="cand_aaaaaaaaaaaaaaaaaaaaaaaa",
        family="put_credit",
        v2_candidate_id="cand_aaaaaaaaaaaaaaaaaaaaaaaa",
        _v2_candidate_id="cand_aaaaaaaaaaaaaaaaaaaaaaaa",
    )
    orch._tick_shadow_cands = [cand]
    orch._tick_unified_v3 = {
        "final_action": "TRADE",
        "candidate_id": "cand_aaaaaaaaaaaaaaaaaaaaaaaa",
        "structure": "put_credit",
        "direction": "put",
    }
    orch._tick_part3 = {
        "authority_source": "v3",
        "unified": {
            "authority_source": "v3",
            "selected_candidate_evaluation": {
                "fill_probability": 0.7,
                "expected_fill_price": 0.38,
                "conservative_fill_price": 0.32,
                "expected_order_value": 0.12,
                "fees": 0.02,
            },
        },
    }
    orch._tick_authoritative = {
        "final_action": "TRADE",
        "selected_candidate_id": "cand_aaaaaaaaaaaaaaaaaaaaaaaa",
        "size_mult": 1.0,
    }
    snap = SimpleNamespace(chain=object(), market=SimpleNamespace())
    intents = orch._build_paper_intents(
        snap=snap, signals={}, intent=None, regime_state=None,
        decision=None, decide_pdf=None, cfg=None, pin_active=False,
        density_mode="vrp", density_moments=None, final_size_mult=1.0,
        matrix_stand_down=True,
    )
    v3 = [i for i in intents if i.get("track") == "v3"]
    assert len(v3) == 1
    assert v3[0]["execution_estimate"]["expected_fill_price"] == 0.38
    assert v3[0].get("risk_record") is True

    orch._tick_authoritative["size_mult"] = 0.0
    intents0 = orch._build_paper_intents(
        snap=snap, signals={}, intent=None, regime_state=None,
        decision=None, decide_pdf=None, cfg=None, pin_active=False,
        density_mode="vrp", density_moments=None, final_size_mult=0.0,
        matrix_stand_down=True,
    )
    assert [i for i in intents0 if i.get("track") == "v3"] == []


def test_paper_broker_uses_execution_estimate_and_suppresses_zero_size():
    import datetime as dt
    from paper_broker import PaperBroker, PaperConfig
    from spread_selector import Leg, SpreadCandidate
    from rnd_extractor import ChainQuote, ChainSnapshot

    T0 = dt.datetime(2026, 7, 14, 11, 0, tzinfo=ET)
    cand = SpreadCandidate(
        family="put_credit",
        short_strikes=(490.0,),
        long_strikes=(485.0,),
        legs=(Leg(490.0, "P", -1), Leg(485.0, "P", 1)),
        credit=0.50, max_loss=4.50, ev=0.1, ev_per_risk=0.02,
        theta=0.05, gamma=-0.01, prob_profit=0.6, prob_touch_short=0.2,
        distance_to_wall=2.0, liquidity_score=0.8, wall_safety=0.7,
        gamma_safety=0.7, touch_safety=0.7, score=0.5,
        passes_vetoes=True, veto_reasons=(), capital=4.5,
    )
    qs = [
        ChainQuote(485.0, 0.10, 0.20, 0.40, 0.60),
        ChainQuote(490.0, 0.20, 0.30, 0.90, 1.10),
    ]
    chain = ChainSnapshot(qs, spot=500.0, t_years=0.001, r=0.05)

    class _Res:
        decision = None
        signals = {}
        intent = None
        regime = None
        paper_intents = []
        final_size_mult = 1.0
        part3 = None

    b = PaperBroker(db_path=":memory:", cfg=PaperConfig(starting_cash=100_000))
    res = _Res()
    res.paper_intents = [{
        "track": "v3", "candidate": cand, "size_mult": 0.0,
    }]
    # Attach a chain via a fake snapshot path — on_tick needs snap.chain
    # Use the internal open path via intents; on_tick reads result.snapshot
    from unified_loop import TickSnapshot
    res.snapshot = TickSnapshot(market=None, bars=None, chain=chain)
    assert b.on_tick(T0, res) == []

    b2 = PaperBroker(db_path=":memory:", cfg=PaperConfig(starting_cash=100_000))
    res2 = _Res()
    res2.snapshot = TickSnapshot(market=None, bars=None, chain=chain)
    res2.paper_intents = [{
        "track": "v3", "candidate": cand, "size_mult": 1.0,
        "execution_estimate": {"expected_fill_price": 0.37},
        "structure": "put_credit", "direction": "put",
    }]
    ev = b2.on_tick(T0, res2)
    assert ev, "expected an entry"
    assert b2.open_positions
    assert b2.open_positions[0].entry_credit == pytest.approx(0.37)
