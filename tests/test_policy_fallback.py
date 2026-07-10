"""
tests/test_policy_fallback.py
=============================
PR 10 — explicit legacy fallback (§17.5) and router mode behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

from prediction.contracts import PredictionBundle
from policy.contracts import (
    SOURCE_FALLBACK_LEGACY, SOURCE_LEGACY, SOURCE_V2,
    PolicyInput, PolicyMode, StructuralState,
)
from policy.legacy_matrix import LegacyMatrixPolicy, intent_to_decision
from policy.router import PolicyRouter, PolicyRouterConfig


@dataclass
class _Decision:
    structure: str
    direction: str
    conviction: str
    capture: str = "test"
    strike_rule: str = "x"
    anchor_tf: str = "15m"


@dataclass
class _Intent:
    decision: _Decision
    size_mult: float
    vetoes: list
    note: str = ""
    exec_regime: str = "compression"
    context_regime: str = "compression"
    direction_bias: str = "neutral"
    bias_value: float = 50.0


@dataclass
class _Regime:
    stand_down: bool = False
    vetoes: list = None
    dominant_regime: str = "compression"

    def __post_init__(self):
        if self.vetoes is None:
            self.vetoes = []


def _intent(structure="IC", direction="both", size=1.0, vetoes=None):
    return _Intent(
        decision=_Decision(structure, direction,
                           "HIGH" if size >= 0.9 else "MED"),
        size_mult=size,
        vetoes=list(vetoes or []),
    )


def _bundle(**kw) -> PredictionBundle:
    base = dict(
        snapshot_id="s1", ts="2026-07-10T10:00:00-04:00",
        session_date="2026-07-10", symbol="SPY",
        uncertainty=0.20, data_quality=0.80, feature_coverage=0.9,
        p_range_survive_30m=0.70, expected_realized_move_30m=0.005,
        p_up_30m=0.52,
    )
    base.update(kw)
    return PredictionBundle(**base)


def _pin(bundle=None, intent=None, regime=None) -> PolicyInput:
    return PolicyInput(
        predictions=bundle,
        structural_state=StructuralState(spot=600.0),
        operational_risk_state={
            "hard_vetoes": list(getattr(regime, "vetoes", None) or []),
            "stand_down": bool(getattr(regime, "stand_down", False)),
            "implied_remaining_move": 0.012,
        },
        legacy_regime_state=regime or _Regime(),
        legacy_matrix_intent=intent or _intent(),
    )


class TestLegacyAdapter:
    def test_maps_trade_intent(self):
        dec = intent_to_decision(_intent("PCS", "put", 0.6))
        assert dec.source == SOURCE_LEGACY
        assert dec.action == "TRADE"
        assert dec.structure_code == "PCS"
        assert "put_credit" in dec.eligible_families
        assert dec.size_cap == 0.6

    def test_stand_down_forces_no_trade(self):
        dec = intent_to_decision(
            _intent("IC", "both", 1.0),
            regime_state=_Regime(stand_down=True, dominant_regime="expansion"),
        )
        assert dec.action == "NO_TRADE"
        assert dec.size_cap == 0.0
        assert any("stand_down" in r for r in dec.rationale)

    def test_nt_structure(self):
        dec = intent_to_decision(_intent("NT", "none", 0.0))
        assert dec.action == "NO_TRADE"
        assert dec.eligible_families == ()

    def test_as_fallback_marks_source(self):
        fb = LegacyMatrixPolicy().as_fallback(_pin())
        assert fb.source == SOURCE_FALLBACK_LEGACY
        assert fb.action == "TRADE"  # default IC intent


class TestRouterModes:
    def test_legacy_mode_skips_v2(self):
        r = PolicyRouter(PolicyRouterConfig(mode=PolicyMode.LEGACY.value))
        out = r.route(_pin(bundle=_bundle()))
        assert out.mode == "legacy"
        assert out.v2 is None
        assert out.authoritative.source == SOURCE_LEGACY
        assert out.fallback_used is False

    def test_shadow_legacy_authoritative(self):
        r = PolicyRouter(PolicyRouterConfig(mode=PolicyMode.SHADOW.value))
        out = r.route(_pin(bundle=_bundle(), intent=_intent("IC")))
        assert out.authoritative.source == SOURCE_LEGACY
        assert out.v2 is not None
        assert out.v2.source == SOURCE_V2
        assert out.fallback_used is False
        sig = out.journal_signals()
        assert sig["policy_mode"] == "shadow"
        assert "v2_policy_action" in sig
        assert "legacy_policy_structure" in sig

    def test_shadow_missing_bundle_explicit_fallback_flag(self):
        r = PolicyRouter(PolicyRouterConfig(mode="shadow"))
        out = r.route(_pin(bundle=None))
        assert out.authoritative.source == SOURCE_LEGACY
        assert out.v2 is None
        assert out.fallback_used is True
        assert out.journal_signals()["policy_fallback_used"] == 1.0

    def test_champion_uses_v2(self):
        r = PolicyRouter(PolicyRouterConfig(mode="champion"))
        out = r.route(_pin(bundle=_bundle(), intent=_intent("NT", "none", 0.0)))
        assert out.authoritative.source == SOURCE_V2
        assert out.authoritative.action == "TRADE"
        assert out.fallback_used is False
        # Disagreement vs legacy NT
        assert out.disagreement is True

    def test_champion_fallback_never_silent(self):
        r = PolicyRouter(PolicyRouterConfig(mode="champion"))
        out = r.route(_pin(bundle=None, intent=_intent("PCS", "put", 0.6)))
        assert out.fallback_used is True
        assert out.authoritative.source == SOURCE_FALLBACK_LEGACY
        assert out.authoritative.action == "TRADE"
        assert out.authoritative.structure_code == "PCS"
        assert out.v2 is None
        # Must not look like a native V2 decision
        assert out.authoritative.source != SOURCE_V2
        sig = out.journal_signals()
        assert sig["policy_source"] == SOURCE_FALLBACK_LEGACY
        assert sig["policy_fallback_used"] == 1.0

    def test_promotion_is_config_pointer(self):
        """Acceptance: promotion changes one config pointer, not code."""
        pin = _pin(bundle=_bundle(), intent=_intent("NT", "none", 0.0))
        shadow = PolicyRouter(PolicyRouterConfig(mode="shadow")).route(pin)
        champ = PolicyRouter(PolicyRouterConfig(mode="champion")).route(pin)
        assert shadow.authoritative.source == SOURCE_LEGACY
        assert champ.authoritative.source == SOURCE_V2
        assert shadow.authoritative.action == "NO_TRADE"
        assert champ.authoritative.action == "TRADE"

    def test_disagreement_journaled(self):
        # Legacy wants PCS; V2 premium with neutral p_up -> IC
        pin = _pin(bundle=_bundle(p_up_30m=0.50, p_range_survive_30m=0.75,
                                  expected_realized_move_30m=0.004),
                   intent=_intent("PCS", "put", 1.0))
        out = PolicyRouter(PolicyRouterConfig(mode="shadow")).route(pin)
        assert out.v2 is not None
        if out.v2.structure_code != "PCS":
            assert out.disagreement is True
            assert out.journal_signals()["policy_disagreement"] == 1.0
