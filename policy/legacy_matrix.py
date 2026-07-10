"""
policy/legacy_matrix.py
=======================
LegacyMatrixPolicy — wraps the existing regime classifier + 27-cell
matrix path into the unified PolicyDecision contract
(docs/PREDICTION_ENGINE_V2_HANDOFF.md §17.1, PR 10).

Does not re-run the matrix; it adapts an already-computed TradeIntent
(+ optional RegimeState) so the live loop stays a single evaluation.

NOT financial advice.
"""
from __future__ import annotations

from typing import Optional

from policy.contracts import (
    SOURCE_FALLBACK_LEGACY, SOURCE_LEGACY, PolicyDecision, PolicyInput,
)
from spread_selector import STRUCTURE_TO_FAMILIES

LEGACY_POLICY_VERSION = "legacy-matrix-v1"


def _families_for(structure_code: str) -> tuple[str, ...]:
    fams = STRUCTURE_TO_FAMILIES.get(structure_code)
    if not fams:
        return ()
    return tuple(sorted(fams))


def intent_to_decision(
    intent: object,
    *,
    regime_state: Optional[object] = None,
    source: str = SOURCE_LEGACY,
    policy_version: str = LEGACY_POLICY_VERSION,
) -> PolicyDecision:
    """
    Adapt a decision_matrix.TradeIntent (+ optional RegimeState) into
    PolicyDecision. Pure function — safe to call from tests / replay.
    """
    decision = getattr(intent, "decision", None)
    structure = str(getattr(decision, "structure", "NT") or "NT")
    direction = str(getattr(decision, "direction", "none") or "none")
    size_mult = float(getattr(intent, "size_mult", 0.0) or 0.0)
    note = str(getattr(intent, "note", "") or "")
    intent_vetoes = tuple(getattr(intent, "vetoes", None) or ())

    stand_down = bool(getattr(regime_state, "stand_down", False)) if regime_state else False
    regime_vetoes = tuple(getattr(regime_state, "vetoes", None) or ()) if regime_state else ()
    hard_vetoes = tuple(dict.fromkeys([*intent_vetoes, *regime_vetoes]))

    rationale: list[str] = []
    if note:
        rationale.append(note)
    if stand_down:
        dom = getattr(regime_state, "dominant_regime", "") or "unknown"
        rationale.append(f"regime stand_down:{dom}")
    capture = getattr(decision, "capture", None)
    if capture:
        rationale.append(str(capture))

    no_trade = stand_down or structure == "NT" or size_mult <= 0.0
    action = "NO_TRADE" if no_trade else "TRADE"

    # Confidence from matrix conviction label when present.
    conv = str(getattr(decision, "conviction", "NONE") or "NONE").upper()
    conf_map = {"HIGH": 0.85, "MED": 0.65, "LOW": 0.40, "NONE": 0.0}
    confidence = conf_map.get(conv, 0.5)
    if stand_down:
        confidence = 0.0

    # Legacy path has no calibrated model uncertainty — surface 0 when
    # trading, 1 when standing down so downstream size logic stays sane.
    uncertainty = 1.0 if no_trade else 0.0

    return PolicyDecision(
        action=action,
        direction="none" if no_trade else direction,
        eligible_families=() if no_trade else _families_for(structure),
        confidence=float(confidence),
        uncertainty=float(uncertainty),
        size_cap=0.0 if no_trade else max(0.0, size_mult),
        hard_vetoes=hard_vetoes,
        rationale=tuple(rationale) or (("legacy matrix NT",) if no_trade
                                       else ("legacy matrix route",)),
        policy_version=policy_version,
        source=source,
        structure_code="" if no_trade else structure,
    )


class LegacyMatrixPolicy:
    """§17.1 — deterministic matrix / classifier adapter."""

    version: str = LEGACY_POLICY_VERSION

    def decide(self, inp: PolicyInput, *,
               source: str = SOURCE_LEGACY) -> PolicyDecision:
        intent = inp.legacy_matrix_intent
        if intent is None:
            return PolicyDecision(
                action="NO_TRADE",
                direction="none",
                eligible_families=(),
                confidence=0.0,
                uncertainty=1.0,
                size_cap=0.0,
                hard_vetoes=tuple(inp.operational_risk_state.get("hard_vetoes")
                                  or ()),
                rationale=("legacy matrix intent missing",),
                policy_version=self.version,
                source=source,
                structure_code="",
            )
        return intent_to_decision(
            intent,
            regime_state=inp.legacy_regime_state,
            source=source,
            policy_version=self.version,
        )

    def as_fallback(self, inp: PolicyInput) -> PolicyDecision:
        """Explicit §17.5 fallback — never silent."""
        return self.decide(inp, source=SOURCE_FALLBACK_LEGACY)
