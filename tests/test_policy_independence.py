"""
tests/test_policy_independence.py
=================================
PR 10 acceptance — forecast independent of policy; structural vetoes
separate; PredictionBundle has no policy fields.
"""
from __future__ import annotations

import dataclasses

from prediction.contracts import PredictionBundle
from policy.contracts import PolicyDecision, PolicyInput, StructuralState
from policy.prediction_policy import PredictionPolicy
from policy.router import PolicyRouter, PolicyRouterConfig


def _bundle(**kw) -> PredictionBundle:
    base = dict(
        snapshot_id="indep-1",
        ts="2026-07-10T11:00:00-04:00",
        session_date="2026-07-10",
        symbol="SPY",
        p_up_30m=0.66,
        expected_return_30m=0.0012,
        return_q50_30m=0.0010,
        p_range_survive_30m=0.40,
        expected_realized_move_30m=0.012,
        uncertainty=0.22,
        data_quality=0.88,
        feature_coverage=0.91,
        model_versions={"direction": "test"},
    )
    base.update(kw)
    return PredictionBundle(**base)


POLICY_FIELD_NAMES = {
    "action", "direction", "eligible_families", "confidence", "size_cap",
    "hard_vetoes", "rationale", "policy_version", "structure_code",
    "selected_structure", "selected_family", "conviction", "gate_pass",
    "candidate_score", "trade_intent",
}


class TestForecastIndependence:
    def test_bundle_has_no_policy_fields(self):
        names = {f.name for f in dataclasses.fields(PredictionBundle)}
        overlap = names & POLICY_FIELD_NAMES
        assert not overlap, f"PredictionBundle leaked policy fields: {overlap}"

    def test_policy_does_not_mutate_bundle(self):
        b = _bundle()
        before = b.to_dict()
        pin = PolicyInput(
            predictions=b,
            structural_state=StructuralState(spot=600.0),
            operational_risk_state={
                "hard_vetoes": [],
                "stand_down": False,
                "implied_remaining_move": 0.01,
            },
        )
        dec = PredictionPolicy().decide(pin)
        assert isinstance(dec, PolicyDecision)
        assert b.to_dict() == before
        # frozen dataclass — assignment must fail
        try:
            b.p_up_30m = 0.99  # type: ignore[misc]
            raised = False
        except dataclasses.FrozenInstanceError:
            raised = True
        assert raised

    def test_same_bundle_same_decision_regardless_of_legacy_intent(self):
        """Policy path from bundle alone — legacy intent must not alter V2."""
        from dataclasses import dataclass

        @dataclass
        class _D:
            structure: str
            direction: str
            conviction: str = "HIGH"
            capture: str = ""
            strike_rule: str = ""
            anchor_tf: str = ""

        @dataclass
        class _I:
            decision: _D
            size_mult: float = 1.0
            vetoes: list = None
            note: str = ""

            def __post_init__(self):
                if self.vetoes is None:
                    self.vetoes = []

        b = _bundle()
        structural = StructuralState(spot=600.0, net_gex=1e8)
        op = {"hard_vetoes": [], "stand_down": False,
              "implied_remaining_move": 0.01}
        d1 = PredictionPolicy().decide(PolicyInput(
            predictions=b, structural_state=structural,
            operational_risk_state=op,
            legacy_matrix_intent=_I(_D("IC", "both")),
        ))
        d2 = PredictionPolicy().decide(PolicyInput(
            predictions=b, structural_state=structural,
            operational_risk_state=op,
            legacy_matrix_intent=_I(_D("LPS", "put"), size_mult=0.3),
        ))
        assert d1.to_dict() == d2.to_dict()
        assert d1.source == "v2"

    def test_structural_vetoes_remain_separate(self):
        b = _bundle(p_range_survive_30m=0.80,
                    expected_realized_move_30m=0.004,
                    p_up_30m=0.51)
        # Without veto — premium path
        ok = PredictionPolicy().decide(PolicyInput(
            predictions=b,
            structural_state=StructuralState(spot=600.0),
            operational_risk_state={
                "hard_vetoes": [], "stand_down": False,
                "implied_remaining_move": 0.012,
            },
        ))
        assert ok.action == "TRADE"
        # With short_gamma veto — premium blocked; may fall through to
        # directional/vol/no-trade, but hard_vetoes must surface.
        blocked = PredictionPolicy().decide(PolicyInput(
            predictions=b,
            structural_state=StructuralState(spot=600.0),
            operational_risk_state={
                "hard_vetoes": ["short_gamma"], "stand_down": False,
                "implied_remaining_move": 0.012,
            },
        ))
        assert "short_gamma" in blocked.hard_vetoes
        # Bundle unchanged and veto not written into forecast
        assert b.diagnostics == {} or "short_gamma" not in str(b.diagnostics)

    def test_shadow_provenance_complete(self):
        from dataclasses import dataclass

        @dataclass
        class _D:
            structure: str = "PCS"
            direction: str = "put"
            conviction: str = "HIGH"
            capture: str = ""
            strike_rule: str = ""
            anchor_tf: str = ""

        @dataclass
        class _I:
            decision: _D = None
            size_mult: float = 1.0
            vetoes: list = None
            note: str = ""

            def __post_init__(self):
                if self.decision is None:
                    self.decision = _D()
                if self.vetoes is None:
                    self.vetoes = []

        pin = PolicyInput(
            predictions=_bundle(p_range_survive_30m=0.75,
                                expected_realized_move_30m=0.004),
            structural_state=StructuralState(spot=600.0),
            operational_risk_state={
                "hard_vetoes": [], "stand_down": False,
                "implied_remaining_move": 0.012,
            },
            legacy_matrix_intent=_I(),
        )
        out = PolicyRouter(PolicyRouterConfig(mode="shadow")).route(pin)
        sig = out.journal_signals()
        required = {
            "policy_mode", "policy_source", "policy_action",
            "legacy_policy_action", "legacy_policy_structure",
            "v2_policy_action", "v2_policy_structure",
            "policy_disagreement", "policy_fallback_used",
            "policy_version",
        }
        missing = required - set(sig)
        assert not missing, f"incomplete provenance: {missing}"
