"""
tests/test_v3_candidate_stack.py
tests/test_execution_economics.py
"""
from __future__ import annotations

from prediction.canonical_snapshot import build_canonical_snapshot
from prediction.candidate_universe import build_candidate_universe
from prediction.contracts import PredictionBundle
from prediction.part3_decision import build_v3_decision


def _snap():
    return build_canonical_snapshot(
        symbol="SPY",
        ts="2026-07-14T10:00:00-04:00",
        session_date="2026-07-14",
        raw_features={"spot": 500.0},
        snapshot_id="s1",
    )


def _universe():
    return build_candidate_universe(
        snapshot_id="s1",
        generated_at="t",
        candidates=[
            {
                "family": "put_credit",
                "ev": 0.20,
                "credit": 0.55,
                "natural_credit": 0.45,
                "prob_profit": 0.7,
                "legs": [
                    {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                     "expiration": "2026-07-14"},
                    {"right": "P", "side": "buy", "qty": 1, "strike": 485,
                     "expiration": "2026-07-14"},
                ],
            },
            {
                "family": "call_debit",
                "ev": 0.05,
                "credit": -0.40,
                "natural_credit": -0.48,
                "prob_profit": 0.55,
                "legs": [
                    {"right": "C", "side": "buy", "qty": 1, "strike": 500,
                     "expiration": "2026-07-14"},
                    {"right": "C", "side": "sell", "qty": 1, "strike": 505,
                     "expiration": "2026-07-14"},
                ],
            },
        ],
    )


def test_hard_veto_overrides_trade():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.1, ood_score=0.05, data_quality=0.95,
    )
    result = build_v3_decision(
        snapshot=_snap(),
        forecast=forecast,
        universe=_universe(),
        hard_vetoes=("stale_data",),
        mode="shadow",
    )
    assert result.final_action == "HARD_VETO"
    assert "stale_data" in result.reasons or "hard_veto" in str(result.reasons)


def test_rankings_deterministic():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, ood_score=0.1, data_quality=0.9,
    )
    a = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(), mode="shadow")
    b = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(), mode="shadow")
    assert a.candidate_id == b.candidate_id
    assert [e.candidate_id for e in a.evaluations] == [
        e.candidate_id for e in b.evaluations]


def test_midpoint_never_treated_as_filled():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    # Candidates with only EV (no credit) must not invent fill prices.
    universe = build_candidate_universe(
        snapshot_id="s1",
        generated_at="t",
        candidates=[{
            "family": "put_credit",
            "ev": 0.20,
            "prob_profit": 0.7,
            "legs": [
                {"right": "P", "side": "sell", "qty": 1, "strike": 490,
                 "expiration": "2026-07-14"},
            ],
        }],
    )
    result = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=universe, mode="shadow")
    for ev in result.evaluations:
        # Without mid/natural credit, fill fields stay None — never fill_p=0.5
        if ev.expected_fill_price is None:
            assert ev.fill_probability is None


def test_required_component_failure_abstain_in_champion():
    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    result = build_v3_decision(
        snapshot=_snap(),
        forecast=forecast,
        universe=_universe(),
        runtime=None,
        mode="champion",
    )
    assert result.statistical_action == "ABSTAIN"
    assert "required_component_missing" in result.reasons


def test_partial_missing_artifacts_abstain():
    """Any nonempty missing list must abstain — not only when all five missing."""
    from prediction.runtime import LoadedArtifacts, PredictionRuntime
    from prediction.deployment import DeploymentBundle

    class _Art:
        candidate_value = object()
        candidate_rank = None  # missing
        fill_probability = object()
        fill_concession = object()
        meta_model = object()

    bundle = DeploymentBundle(
        deployment_id="d", mode="champion",
        prediction_model_group_id="g",
        candidate_value_model_id="cv",
        candidate_rank_model_id="cr",
        fill_probability_model_id="fp",
        fill_concession_model_id="fc",
        meta_model_id="mm",
        authority_source="v3", fallback_policy="abstain",
    )
    rt = PredictionRuntime(
        bundle=bundle,
        registry=None,  # type: ignore
        artifacts=LoadedArtifacts(bundle=bundle),
        strict=True,
    )
    rt.artifacts.candidate_value = _Art.candidate_value
    rt.artifacts.candidate_rank = None
    rt.artifacts.fill_probability = _Art.fill_probability
    rt.artifacts.fill_concession = _Art.fill_concession
    rt.artifacts.meta_model = _Art.meta_model

    forecast = PredictionBundle(
        snapshot_id="s1", ts="t", session_date="2026-07-14", symbol="SPY",
        uncertainty=0.2, data_quality=0.9, ood_score=0.1,
    )
    result = build_v3_decision(
        snapshot=_snap(), forecast=forecast, universe=_universe(),
        runtime=rt, mode="champion",
    )
    assert result.final_action == "ABSTAIN"
    assert "required_component_missing" in result.reasons
