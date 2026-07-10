"""
policy/router.py
================
PolicyRouter — dual-run legacy + V2 with explicit promotion modes
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17, PR 10).

Modes (single config pointer):
  legacy    — matrix only
  shadow    — both; legacy authoritative; disagreement journaled
  champion  — V2 authoritative; explicit fallback_legacy on failure

NOT financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from policy.contracts import (
    SOURCE_FALLBACK_LEGACY, SOURCE_LEGACY, SOURCE_V2,
    PolicyDecision, PolicyInput, PolicyMode,
)
from policy.legacy_matrix import LegacyMatrixPolicy
from policy.prediction_policy import (
    PredictionPolicy, PredictionPolicyConfig, PredictionUnavailable,
)


@dataclass
class PolicyRouterConfig:
    mode: str = PolicyMode.SHADOW.value   # legacy | shadow | champion
    prediction: Optional[PredictionPolicyConfig] = None
    journal_disagreement: bool = True


@dataclass
class PolicyRouteResult:
    authoritative: PolicyDecision
    legacy: PolicyDecision
    v2: Optional[PolicyDecision]
    mode: str
    disagreement: bool
    fallback_used: bool
    diagnostics: dict = field(default_factory=dict)

    def journal_signals(self) -> dict:
        """Observation-only keys for journal signals_json."""
        auth = self.authoritative
        out = {
            "policy_mode": self.mode,
            "policy_source": auth.source,
            "policy_action": auth.action,
            "policy_direction": auth.direction,
            "policy_structure": auth.structure_code or "",
            "policy_confidence": float(auth.confidence),
            "policy_uncertainty": float(auth.uncertainty),
            "policy_size_cap": float(auth.size_cap),
            "policy_version": auth.policy_version,
            "policy_fallback_used": 1.0 if self.fallback_used else 0.0,
            "policy_disagreement": 1.0 if self.disagreement else 0.0,
            "legacy_policy_action": self.legacy.action,
            "legacy_policy_direction": self.legacy.direction,
            "legacy_policy_structure": self.legacy.structure_code or "",
        }
        if self.v2 is not None:
            out.update({
                "v2_policy_action": self.v2.action,
                "v2_policy_direction": self.v2.direction,
                "v2_policy_structure": self.v2.structure_code or "",
                "v2_policy_confidence": float(self.v2.confidence),
                "v2_policy_uncertainty": float(self.v2.uncertainty),
                "v2_policy_version": self.v2.policy_version,
                "v2_policy_source": self.v2.source,
            })
        if auth.hard_vetoes:
            out["policy_hard_vetoes"] = ",".join(auth.hard_vetoes)
        if auth.rationale:
            out["policy_rationale"] = " | ".join(auth.rationale)[:240]
        return out


def _decisions_disagree(a: PolicyDecision, b: PolicyDecision) -> bool:
    if a.action != b.action:
        return True
    if a.action == "NO_TRADE":
        return False
    if a.direction != b.direction:
        return True
    if (a.structure_code or "") != (b.structure_code or ""):
        return True
    return False


class PolicyRouter:
    """§17 transitional dual-policy runner."""

    def __init__(self, cfg: Optional[PolicyRouterConfig] = None,
                 legacy: Optional[LegacyMatrixPolicy] = None,
                 v2: Optional[PredictionPolicy] = None):
        self.cfg = cfg or PolicyRouterConfig()
        self.legacy = legacy or LegacyMatrixPolicy()
        self.v2 = v2 or PredictionPolicy(self.cfg.prediction)

    @property
    def mode(self) -> str:
        return str(self.cfg.mode or PolicyMode.SHADOW.value).lower()

    def route(self, inp: PolicyInput) -> PolicyRouteResult:
        mode = self.mode
        legacy_dec = self.legacy.decide(inp, source=SOURCE_LEGACY)

        if mode == PolicyMode.LEGACY.value:
            return PolicyRouteResult(
                authoritative=legacy_dec,
                legacy=legacy_dec,
                v2=None,
                mode=mode,
                disagreement=False,
                fallback_used=False,
            )

        v2_dec: Optional[PolicyDecision] = None
        fallback_used = False
        try:
            v2_dec = self.v2.decide(inp)
        except PredictionUnavailable as exc:
            fallback_used = True
            v2_dec = None
            diag_reason = str(exc)
        except Exception as exc:  # noqa: BLE001 — never break the live tick
            fallback_used = True
            v2_dec = None
            diag_reason = f"v2_policy_error:{type(exc).__name__}:{exc}"
        else:
            diag_reason = ""

        if mode == PolicyMode.SHADOW.value:
            disagreement = (
                _decisions_disagree(legacy_dec, v2_dec)
                if v2_dec is not None else False
            )
            # Missing V2 in shadow is still an explicit fallback signal for
            # journaling, but legacy remains authoritative.
            if v2_dec is None:
                fallback_used = True
            return PolicyRouteResult(
                authoritative=legacy_dec,
                legacy=legacy_dec,
                v2=v2_dec,
                mode=mode,
                disagreement=disagreement if self.cfg.journal_disagreement
                else False,
                fallback_used=fallback_used,
                diagnostics={"v2_unavailable_reason": diag_reason}
                if diag_reason else {},
            )

        # champion
        if v2_dec is not None:
            disagreement = _decisions_disagree(legacy_dec, v2_dec)
            return PolicyRouteResult(
                authoritative=v2_dec,
                legacy=legacy_dec,
                v2=v2_dec,
                mode=mode,
                disagreement=disagreement,
                fallback_used=False,
            )

        # Explicit fallback — never silent (§17.5).
        fb = self.legacy.as_fallback(inp)
        assert fb.source == SOURCE_FALLBACK_LEGACY
        return PolicyRouteResult(
            authoritative=fb,
            legacy=legacy_dec,
            v2=None,
            mode=mode,
            disagreement=False,
            fallback_used=True,
            diagnostics={"v2_unavailable_reason": diag_reason
                         or "prediction unavailable"},
        )
