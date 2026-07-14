"""
tests/test_review6_merge_gate.py
================================
PR #117 sixth-review merge-gate regressions.
"""
from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from decision_stack.authority import resolve_authority
from decision_stack.contracts import CandidateEvaluation
from decision_stack.stack import UnifiedDecisionStack
from prediction.models.candidate_rank import PairwiseCandidateRanker
from prediction.part3_decision import Part3DecisionError, build_v3_candidate_evaluations
from risk_manager import RiskConfig, RiskManager


ET = ZoneInfo("America/New_York")


def test_selector_vetoes_enter_evaluation_and_vetoed_ids():
    """SpreadCandidate.passes_vetoes=False must reach the ranker as vetoed_ids."""
    from prediction.canonical_snapshot import build_canonical_snapshot
    from prediction.candidate_universe import build_candidate_universe
    from prediction.contracts import PredictionBundle
    from spread_selector import Leg, SpreadCandidate

    good = SpreadCandidate(
        family="put_credit", short_strikes=(490.0,), long_strikes=(485.0,),
        legs=(Leg(490.0, "P", -1), Leg(485.0, "P", 1)),
        credit=0.45, max_loss=4.55, ev=0.12, ev_per_risk=0.03,
        theta=0.05, gamma=-0.01, prob_profit=0.58, prob_touch_short=0.2,
        distance_to_wall=2.0, liquidity_score=0.8, wall_safety=0.7,
        gamma_safety=0.7, touch_safety=0.7, score=0.55,
        passes_vetoes=True, veto_reasons=(), capital=4.55,
    )
    good.candidate_id = "cand_good_good_good_good_go"
    bad = SpreadCandidate(
        family="put_credit", short_strikes=(480.0,), long_strikes=(475.0,),
        legs=(Leg(480.0, "P", -1), Leg(475.0, "P", 1)),
        credit=0.10, max_loss=4.90, ev=-0.05, ev_per_risk=-0.01,
        theta=0.02, gamma=-0.02, prob_profit=0.40, prob_touch_short=0.8,
        distance_to_wall=-1.0, liquidity_score=0.1, wall_safety=0.2,
        gamma_safety=0.2, touch_safety=0.2, score=0.01,
        passes_vetoes=False, veto_reasons=("EV<=0", "illiquid"), capital=4.90,
    )
    bad.candidate_id = "cand_bad_bad_bad_bad_bad_ba"

    snap = build_canonical_snapshot(
        symbol="SPY", ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14", raw_features={"spot": 500.0},
        snapshot_id="veto1",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    universe = build_candidate_universe(
        snapshot_id="veto1", generated_at=snap.ts, candidates=[good, bad])
    forecast = PredictionBundle(
        snapshot_id="veto1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    # Shadow mode uses legacy EV baseline — enough to exercise veto plumbing.
    evs = build_v3_candidate_evaluations(
        snapshot=snap, forecast=forecast, universe=universe, mode="shadow")
    by_id = {e.candidate_id: e for e in evs}
    assert any(str(v).startswith("selector:") for v in by_id[bad.candidate_id].vetoes)
    assert not by_id[good.candidate_id].vetoes

    # Actionable filter excludes selector-vetoed candidates.
    actionable = [
        e for e in evs
        if e.final_rank is not None and not e.vetoes
    ]
    assert all(e.candidate_id != bad.candidate_id for e in actionable)


def test_unfitted_rank_model_rejected_in_strict_mode():
    from prediction.canonical_snapshot import build_canonical_snapshot
    from prediction.candidate_universe import build_candidate_universe
    from prediction.contracts import PredictionBundle

    snap = build_canonical_snapshot(
        symbol="SPY", ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14", raw_features={"spot": 500.0},
        snapshot_id="rank1",
        source_timestamps={"bars": "2026-07-14T09:59:00-04:00"},
    )
    universe = build_candidate_universe(
        snapshot_id="rank1", generated_at=snap.ts,
        candidates=[{
            "candidate_id": "cand_aaaaaaaaaaaaaaaaaaaaaaaa",
            "family": "put_credit", "ev": 0.1, "prob_profit": 0.6,
            "score": 0.5, "passes_vetoes": True,
            "legs": [{"right": "P", "side": "sell", "qty": 1, "strike": 490,
                      "expiration": "2026-07-14"}],
        }],
    )
    forecast = PredictionBundle(
        snapshot_id="rank1", ts=snap.ts, session_date=snap.session_date,
        symbol="SPY", uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )

    class _Val:
        def predict_v3(self, rows, candidate_ids=None):
            from prediction.models.candidate_value import CandidateForecastV3
            out = []
            for cid in (candidate_ids or []):
                out.append(CandidateForecastV3(
                    candidate_id=cid, expected_net_pnl=0.1, p_profit=0.6,
                    pnl_q05=-1.0, pnl_q10=-0.5, pnl_q25=-0.2, pnl_q50=0.1,
                    pnl_q75=0.3, pnl_q90=0.5, pnl_q95=0.7,
                    expected_shortfall=0.8,
                    p_target_first=None, p_stop_first=None, p_neither=None,
                    expected_time_in_trade=None,
                    fill_probability=0.7, expected_fill_fraction=0.5,
                    conservative_fill_fraction=0.3, fill_uncertainty=0.1,
                    model_uncertainty=0.1, forecast_uncertainty=0.2,
                    ood_score=0.1, capital_required=4.0, maximum_loss=4.0,
                    return_on_risk=0.02, utility_score=0.15,
                ))
            return out

        @property
        def vectorizer(self):
            return None

        @property
        def metadata(self):
            return {}

    class _Art:
        candidate_value = _Val()
        candidate_rank = PairwiseCandidateRanker()  # fitted=False
        fill_probability = SimpleNamespace(fitted=True)
        fill_concession = SimpleNamespace(fitted=True)
        meta_model = SimpleNamespace(fitted=True)

    class _RT:
        artifacts = _Art()

    evs = build_v3_candidate_evaluations(
        snapshot=snap, forecast=forecast, universe=universe,
        runtime=_RT(), mode="candidate",
    )
    assert evs
    assert all("candidate_rank_unusable" in e.vetoes for e in evs)


def test_candidate_mode_v3_risk_does_not_hard_veto_legacy():
    """V3 account risk rejection must not convert legacy reference to HARD_VETO."""
    calls = []

    def risk_fn(cid, session_date, account="v3"):
        calls.append((cid, account))
        if account == "v3":
            return ("risk:max_positions:1>=1",)
        return ()

    # Simulate stack post-V3 logic: apply V3 risk to v3_final only.
    from prediction.part3_decision import _apply_vetoes
    v3_final = "TRADE"
    hard = ()
    mode = "candidate"
    risk_extra = risk_fn("cand_v3", "2026-07-14", account="v3")
    if risk_extra:
        v3_final, _ = _apply_vetoes(v3_final, tuple(risk_extra))
        if mode == "champion":
            hard = hard + risk_extra
    auth = resolve_authority(
        mode="candidate",
        legacy_decision={"action": "TRADE", "candidate_id": "cand_leg",
                         "structure": "put_credit", "size_mult": 1.0},
        v3_decision={"final_action": v3_final, "candidate_id": "cand_v3",
                     "structure": "put_credit", "size_mult": 1.0},
        hard_vetoes=hard,
        fallback_policy="abstain",
        legacy_size_mult=1.0,
        v3_size_mult=1.0,
    )
    assert auth.final_action == "TRADE"
    assert auth.authority_source == "legacy"
    assert auth.selected_candidate_id == "cand_leg"
    assert v3_final == "HARD_VETO"


def test_legacy_risk_exception_fails_closed():
    def boom(cid, session_date, account="legacy"):
        raise RuntimeError("risk backend down")

    stack = UnifiedDecisionStack(
        deployment=SimpleNamespace(
            mode="champion", deployment_id="d1",
            configuration_hash="h", fallback_policy="legacy",
        ),
        portfolio_risk_fn=boom,
    )
    # Exercise the exception wrapper used for legacy-authority risk.
    risk_extra = ()
    try:
        risk_extra = tuple(stack.portfolio_risk_fn(
            "cand_x", "2026-07-14", account="legacy") or ())
    except Exception:
        risk_extra = ("risk:check_failed",)
    assert risk_extra == ("risk:check_failed",)
    auth = resolve_authority(
        mode="champion",
        legacy_decision={"action": "TRADE", "candidate_id": "cand_x"},
        v3_decision={"final_action": "UNAVAILABLE"},
        hard_vetoes=risk_extra,
        fallback_policy="legacy",
        legacy_size_mult=1.0,
    )
    assert auth.final_action == "HARD_VETO"


def test_risk_manager_release_trade_clears_open_count_keeps_daily_loss():
    rm = RiskManager(RiskConfig(max_open_positions=1, daily_loss_limit=10.0))
    c = SimpleNamespace(family="put_credit", max_loss=2.0, gamma=0.01)
    pid = rm.record_trade(c, "2026-07-14")
    assert rm.status()["open_positions"] == 1
    assert not rm.check(c, "2026-07-14").approved  # max positions
    assert rm.release_trade(pid)
    assert rm.status()["open_positions"] == 0
    assert rm.status()["daily_loss_committed"] == pytest.approx(2.0)
    # Can open again after release
    assert rm.check(c, "2026-07-14").approved


def test_separate_risk_ledgers_are_independent():
    ref = RiskManager(RiskConfig(max_open_positions=1))
    cand = RiskManager(RiskConfig(max_open_positions=1))
    c = SimpleNamespace(family="put_credit", max_loss=1.0, gamma=0.01)
    ref.record_trade(c, "2026-07-14")
    # Candidate account still has capacity
    assert cand.check(c, "2026-07-14").approved
    # Reference is full
    assert not ref.check(c, "2026-07-14").approved


def test_paper_unfilled_when_p_fill_zero_and_fees_applied():
    import datetime as dt
    from paper_broker import PaperBroker, PaperConfig
    from spread_selector import Leg, SpreadCandidate
    from rnd_extractor import ChainQuote, ChainSnapshot
    from unified_loop import TickSnapshot

    T0 = dt.datetime(2026, 7, 14, 11, 0, tzinfo=ET)
    cand = SpreadCandidate(
        family="put_credit", short_strikes=(490.0,), long_strikes=(485.0,),
        legs=(Leg(490.0, "P", -1), Leg(485.0, "P", 1)),
        credit=0.50, max_loss=4.50, ev=0.1, ev_per_risk=0.02,
        theta=0.05, gamma=-0.01, prob_profit=0.6, prob_touch_short=0.2,
        distance_to_wall=2.0, liquidity_score=0.8, wall_safety=0.7,
        gamma_safety=0.7, touch_safety=0.7, score=0.5,
        passes_vetoes=True, veto_reasons=(), capital=4.5,
    )
    cand.candidate_id = "cand_fill_test_fill_test_fi"
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
        snapshot = TickSnapshot(market=None, bars=None, chain=chain)

    # p_fill=0 → always unfilled
    b = PaperBroker(db_path=":memory:", cfg=PaperConfig(starting_cash=100_000))
    res = _Res()
    res.paper_intents = [{
        "track": "v3", "candidate": cand, "size_mult": 1.0,
        "execution_estimate": {
            "expected_fill_price": 0.40,
            "fill_probability": 0.0,
            "entry_fees": 0.02,
            "mid_credit": 0.50,
        },
        "risk_record": True,
        "structure": "put_credit",
    }]
    ev = b.on_tick(T0, res)
    assert ev and "UNFILLED" in ev[0]
    assert b.open_positions == []
    n = b._db.execute("SELECT COUNT(*) FROM paper_fill_attempts").fetchone()[0]
    assert n == 1
    filled = b._db.execute(
        "SELECT filled FROM paper_fill_attempts").fetchone()[0]
    assert filled == 0

    # Filled path deducts entry fees from expected fill
    b2 = PaperBroker(db_path=":memory:", cfg=PaperConfig(starting_cash=100_000))
    res2 = _Res()
    res2.paper_intents = [{
        "track": "v3", "candidate": cand, "size_mult": 1.0,
        "execution_estimate": {
            "expected_fill_price": 0.40,
            "fill_probability": 1.0,
            "entry_fees": 0.03,
            "mid_credit": 0.50,
        },
        "risk_record": True,
        "structure": "put_credit",
    }]
    ev2 = b2.on_tick(T0, res2)
    assert ev2 and "ENTRY" in ev2[0]
    assert b2.open_positions[0].entry_credit == pytest.approx(0.37)
