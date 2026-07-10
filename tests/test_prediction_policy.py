"""
tests/test_prediction_policy.py
================================
PR 10 — PredictionPolicy consumes PredictionBundle and emits PolicyDecision
per docs/PREDICTION_ENGINE_V2_HANDOFF.md §17.4.
"""
from __future__ import annotations

import pytest

from prediction.contracts import PredictionBundle
from policy.contracts import (
    SOURCE_V2, PolicyInput, StructuralState,
)
from policy.prediction_policy import (
    PredictionPolicy, PredictionPolicyConfig, PredictionUnavailable,
    bundle_is_usable,
)


def _bundle(**kw) -> PredictionBundle:
    base = dict(
        snapshot_id="s1",
        ts="2026-07-10T10:00:00-04:00",
        session_date="2026-07-10",
        symbol="SPY",
        uncertainty=0.20,
        data_quality=0.80,
        feature_coverage=0.90,
    )
    base.update(kw)
    return PredictionBundle(**base)


def _input(bundle, *, vetoes=(), stand_down=False, implied=0.01) -> PolicyInput:
    return PolicyInput(
        predictions=bundle,
        structural_state=StructuralState(spot=600.0, net_gex=1e8,
                                         gamma_flip=595.0,
                                         call_wall=610.0, put_wall=590.0),
        operational_risk_state={
            "hard_vetoes": list(vetoes),
            "stand_down": stand_down,
            "implied_remaining_move": implied,
        },
    )


class TestBundleUsable:
    def test_none_not_usable(self):
        assert not bundle_is_usable(None)

    def test_empty_not_usable(self):
        assert not bundle_is_usable(_bundle())

    def test_direction_usable(self):
        assert bundle_is_usable(_bundle(p_up_30m=0.62))

    def test_range_usable(self):
        assert bundle_is_usable(_bundle(p_range_survive_30m=0.70))


class TestPredictionPolicy:
    def test_missing_bundle_raises(self):
        pol = PredictionPolicy()
        with pytest.raises(PredictionUnavailable):
            pol.decide(_input(None))

    def test_premium_path(self):
        b = _bundle(
            p_range_survive_30m=0.72,
            expected_realized_move_30m=0.006,
            p_up_30m=0.52,
        )
        dec = PredictionPolicy().decide(_input(b, implied=0.012))
        assert dec.source == SOURCE_V2
        assert dec.action == "TRADE"
        assert dec.structure_code in {"IC", "PCS", "CCS"}
        assert "iron_condor" in dec.eligible_families or dec.structure_code != "IC"
        assert dec.size_cap > 0

    def test_directional_bull(self):
        b = _bundle(
            p_up_30m=0.70,
            expected_return_30m=0.002,
            return_q50_30m=0.0015,
            p_range_survive_30m=0.40,  # not premium
            expected_realized_move_30m=0.015,
        )
        dec = PredictionPolicy().decide(_input(b, implied=0.012))
        assert dec.action == "TRADE"
        assert dec.direction == "call"
        assert dec.structure_code == "LCS"
        assert "long_call_spread" in dec.eligible_families

    def test_directional_bear(self):
        b = _bundle(
            p_up_30m=0.28,
            expected_return_30m=-0.002,
            return_q50_30m=-0.0015,
            p_range_survive_30m=0.35,
        )
        dec = PredictionPolicy().decide(_input(b, implied=0.012))
        assert dec.action == "TRADE"
        assert dec.direction == "put"
        assert dec.structure_code == "LPS"

    def test_long_vol(self):
        b = _bundle(
            p_up_30m=0.50,
            p_range_survive_30m=0.30,
            expected_realized_move_30m=0.025,
            uncertainty=0.30,
        )
        dec = PredictionPolicy().decide(_input(b, implied=0.010))
        assert dec.action == "TRADE"
        assert dec.structure_code == "STG"
        assert "long_strangle" in dec.eligible_families

    def test_high_uncertainty_no_trade(self):
        b = _bundle(p_up_30m=0.70, expected_return_30m=0.002,
                    uncertainty=0.90, data_quality=0.90)
        dec = PredictionPolicy().decide(_input(b))
        assert dec.action == "NO_TRADE"
        assert any("uncertainty" in r for r in dec.rationale)

    def test_low_data_quality_no_trade(self):
        b = _bundle(p_range_survive_30m=0.80, data_quality=0.10,
                    uncertainty=0.20)
        dec = PredictionPolicy().decide(_input(b))
        assert dec.action == "NO_TRADE"
        assert any("data_quality" in r for r in dec.rationale)

    def test_operational_stand_down(self):
        b = _bundle(p_range_survive_30m=0.80, expected_realized_move_30m=0.005)
        dec = PredictionPolicy().decide(_input(b, stand_down=True))
        assert dec.action == "NO_TRADE"
        assert dec.source == SOURCE_V2

    def test_catalyst_veto(self):
        b = _bundle(p_range_survive_30m=0.80, expected_realized_move_30m=0.005)
        dec = PredictionPolicy().decide(
            _input(b, vetoes=["catalyst:FOMC"]))
        assert dec.action == "NO_TRADE"
        assert "catalyst:FOMC" in dec.hard_vetoes

    def test_replay_from_stored_bundle(self):
        """Acceptance: policy replayable from stored PredictionBundle."""
        b = _bundle(
            p_up_30m=0.68,
            expected_return_30m=0.0015,
            return_q50_30m=0.001,
            uncertainty=0.25,
            data_quality=0.85,
        )
        stored = b.to_dict()
        restored = PredictionBundle.from_dict(stored)
        d1 = PredictionPolicy().decide(_input(b))
        d2 = PredictionPolicy().decide(_input(restored))
        assert d1.to_dict() == d2.to_dict()

    def test_config_thresholds_bite(self):
        cfg = PredictionPolicyConfig(min_direction_prob=0.80)
        b = _bundle(p_up_30m=0.65, expected_return_30m=0.002,
                    return_q50_30m=0.001, p_range_survive_30m=0.30)
        dec = PredictionPolicy(cfg).decide(_input(b, implied=0.02))
        # 0.65 < 0.80 and not premium/vol -> NO_TRADE
        assert dec.action == "NO_TRADE"
